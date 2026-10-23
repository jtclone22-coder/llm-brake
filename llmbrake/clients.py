"""Streaming chat clients for local servers, with a working Stop.

    budget = Budget(90)
    try:
        for text in OllamaClient().chat({"model": "qwen3:14b", "messages": msgs}, budget):
            print(text, end="", flush=True)
    except GenerationStopped as stop:
        print(stop.reason, stop.verified)       # e.g. "cancelled", True

Call budget.cancel() from anywhere (a UI button, a disconnect handler) to stop. The client closes the
connection, then asks the verifier whether the server really went idle. With a Gate, an unverified
stop puts the gate on HOLD so the next request doesn't stack onto a GPU that may still be busy.
"""
from __future__ import annotations

import json
from collections.abc import Iterator

from .budget import Budget
from .gate import Gate
from .transport import Interrupted, open_stream
from .verify import NullVerifier, OllamaRunnerVerifier, SlotsVerifier, StopReport


class GenerationStopped(Exception):
    def __init__(self, reason: str, report: StopReport, partial: str):
        state = "verified idle" if report.verified else "NOT verified"
        super().__init__(f"generation stopped ({reason}); server {state}: {report.detail}")
        self.reason, self.report, self.partial = reason, report, partial

    @property
    def verified(self) -> bool:
        return self.report.verified


class ServerError(Exception):
    pass


class _StreamingClient:
    path = ""

    def __init__(self, base_url: str, *, verifier=None, gate: Gate | None = None, stop_wait: float = 10.0,
                 headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.verifier = verifier if verifier is not None else self._default_verifier()
        self.gate = gate
        self.stop_wait = stop_wait
        self.headers = headers or {}

    def _default_verifier(self):
        return NullVerifier()

    def _texts(self, line: bytes) -> tuple[list[str], bool]:
        """(text pieces, finished) for one response line."""
        raise NotImplementedError

    def _prepare(self, payload: dict) -> dict:
        return {**payload, "stream": True}

    def chat(self, payload: dict, budget: Budget | None = None) -> Iterator[str]:
        budget = budget or Budget()
        if self.gate is not None:
            with self.gate.slot(budget):
                yield from self._stream(payload, budget)
        else:
            yield from self._stream(payload, budget)

    def complete(self, payload: dict, budget: Budget | None = None) -> str:
        return "".join(self.chat(payload, budget))

    def _stream(self, payload: dict, budget: Budget) -> Iterator[str]:
        parts: list[str] = []
        finished = False
        lines = open_stream(self.base_url + self.path, self._prepare(payload), budget, headers=self.headers)
        try:
            for line in lines:
                if not line.strip():
                    continue
                texts, done = self._texts(line)
                for text in texts:
                    parts.append(text)
                    yield text
                if done:
                    finished = True
                    return
            finished = True
        except Interrupted as stop:
            raise self._stopped(stop.reason, "".join(parts)) from None
        except GeneratorExit:
            # The caller abandoned the iterator mid-generation: close the connection first (that is what
            # makes the server cancel), then check honestly whether it went idle.
            lines.close()
            if not finished:
                self._stopped("abandoned", "".join(parts))
            raise
        finally:
            lines.close()

    def _stopped(self, reason: str, partial: str) -> GenerationStopped:
        report = self.verifier.verify_idle(self.stop_wait)
        if not report.verified and self.gate is not None:
            self.gate.hold(f"stop after '{reason}' not verified: {report.detail}")
        return GenerationStopped(reason, report, partial)


class OllamaClient(_StreamingClient):
    """Ollama's native /api/chat (NDJSON stream)."""

    path = "/api/chat"

    def __init__(self, base_url: str = "http://127.0.0.1:11434", **kwargs):
        super().__init__(base_url, **kwargs)

    def _default_verifier(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            return NullVerifier()
        return OllamaRunnerVerifier(self.base_url)

    def _texts(self, line: bytes) -> tuple[list[str], bool]:
        data = json.loads(line)
        if data.get("error"):
            raise ServerError(str(data["error"]))
        content = (data.get("message") or {}).get("content") or data.get("response") or ""
        return ([content] if content else []), bool(data.get("done"))


class OpenAICompatClient(_StreamingClient):
    """/v1/chat/completions with stream=true: llama.cpp server, LM Studio, vLLM, Ollama's /v1."""

    path = "/v1/chat/completions"

    def __init__(self, base_url: str = "http://127.0.0.1:8080", **kwargs):
        super().__init__(base_url, **kwargs)

    def _default_verifier(self):
        candidate = SlotsVerifier(self.base_url)
        return candidate if candidate.available() else NullVerifier()

    def _texts(self, line: bytes) -> tuple[list[str], bool]:
        text = line.decode("utf-8", "replace")
        if not text.startswith("data:"):
            return [], False
        body = text[5:].strip()
        if body == "[DONE]":
            return [], True
        data = json.loads(body)
        if data.get("error"):
            raise ServerError(str(data["error"]))
        pieces = []
        for choice in data.get("choices") or []:
            content = (choice.get("delta") or {}).get("content")
            if content:
                pieces.append(content)
        return pieces, False

