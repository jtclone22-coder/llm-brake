"""HTTP streaming that can actually be interrupted.

A plain blocking read on a socket cannot be cancelled from another thread, and a socket timeout leaves
http.client unable to continue. Here every read waits with select() in short slices, checking the
budget between slices, so a Stop or deadline closes the connection within ~0.25 s even while the
server is still thinking (before the first token). Closing the connection is what makes llama.cpp,
Ollama and most OpenAI-compatible servers cancel the generation; whether they really did is checked
separately (see verify.py), never assumed.
"""
from __future__ import annotations

import http.client
import io
import json
import select
import socket
from collections.abc import Iterator
from urllib.parse import urlsplit

from .budget import Budget

POLL_SECONDS = 0.25


class Interrupted(Exception):
    """The budget stopped the request mid-flight; the connection has been closed."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _BudgetReader(io.RawIOBase):
    def __init__(self, sock: socket.socket, budget: Budget, owner: "_BudgetSocket"):
        self.sock, self.budget, self.owner = sock, budget, owner

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        if not self.closed:
            super().close()
            self.owner._reader_closed()

    def readinto(self, buffer) -> int:
        while True:
            if self.budget():
                raise Interrupted(self.budget.reason)
            if getattr(self.sock, "pending", None) and self.sock.pending():  # TLS bytes select() can't see
                return self.sock.recv_into(buffer)
            ready, _, _ = select.select([self.sock], [], [], min(POLL_SECONDS, max(0.0, self.budget.remaining())))
            if ready:
                return self.sock.recv_into(buffer)


class _BudgetSocket:
    """Wraps the connected socket so http.client's response reader checks the budget.

    Like socket.makefile, it keeps the socket open while a reader is still using it. With "Connection: close"
    http.client closes the connection as soon as the headers are in and leaves the body to the response; closing
    the real socket then would break reading a body that arrives a moment later.
    """

    def __init__(self, sock: socket.socket, budget: Budget):
        self._sock, self._budget = sock, budget
        self._readers, self._close_wanted = 0, False

    def makefile(self, mode, *args, **kwargs):
        if "r" not in mode:
            return self._sock.makefile(mode, *args, **kwargs)
        self._readers += 1
        return io.BufferedReader(_BudgetReader(self._sock, self._budget, self))

    def close(self) -> None:
        self._close_wanted = True
        if self._readers <= 0:
            self._sock.close()

    def _reader_closed(self) -> None:
        self._readers -= 1
        if self._readers <= 0 and self._close_wanted:
            self._sock.close()

    def __getattr__(self, name):
        return getattr(self._sock, name)


class HTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


def open_stream(url: str, payload: dict, budget: Budget, *, headers: dict | None = None,
                connect_timeout: float = 10.0) -> Iterator[bytes]:
    """POST JSON and yield response lines (bytes, without the newline) until the body ends.

    Raises Interrupted when the budget stops; the connection is closed before the exception leaves.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL: {url}")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    connection_class = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parts.hostname, port, timeout=budget.timeout(connect_timeout))
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    response = None
    try:
        if budget():
            raise Interrupted(budget.reason)
        connection.connect()
        connection.sock = _BudgetSocket(connection.sock, budget)
        body = json.dumps(payload).encode("utf-8")
        connection.request("POST", path, body=body, headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "Connection": "close", **(headers or {})})
        response = connection.getresponse()
        if response.status >= 400:
            raise HTTPError(response.status, response.read(4096).decode("utf-8", "replace"))
        while True:
            line = response.readline()
            if not line:
                return
            yield line.rstrip(b"\r\n")
    finally:
        if response is not None:
            response.close()
        connection.close()
