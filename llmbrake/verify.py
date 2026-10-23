"""Prove a stopped generation really stopped, instead of assuming it.

A verifier looks at the inference server itself and answers: is anything still generating?
  * SlotsVerifier          llama.cpp server (and anything exposing its GET /slots): every slot's
                           `is_processing` must be false.
  * OllamaRunnerVerifier   Ollama starts a llama.cpp server child per loaded model on a random port;
                           this finds those children (needs `psutil`) and asks their /slots.
  * NullVerifier           when nothing can be checked: always "unverified", never a guess.
"""
from __future__ import annotations

import http.client
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass
class StopReport:
    verified: bool
    detail: str
    polls: int = 0
    waited: float = 0.0
    evidence: list = field(default_factory=list)


def slot_activity(host: str, port: int, timeout: float = 1.0) -> list[bool] | None:
    """is_processing for each slot, or None when the server gives no usable answer."""
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        connection.request("GET", "/slots")
        response = connection.getresponse()
        if response.status != 200:
            return None
        data = json.loads(response.read(1 << 20).decode("utf-8"))
    except (OSError, ValueError, http.client.HTTPException):
        return None
    finally:
        connection.close()
    if not isinstance(data, list) or not data:
        return None
    if any(not isinstance(slot, dict) or type(slot.get("is_processing")) is not bool for slot in data):
        return None
    return [slot["is_processing"] for slot in data]


class _PollingVerifier:
    poll_seconds = 0.2

    def __init__(self, *, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep

    def endpoints(self) -> list[tuple[str, int]]:
        raise NotImplementedError

    def available(self) -> bool:
        """Can this verifier answer right now? (Used to pick abort-and-verify over waiting it out.)"""
        try:
            targets = self.endpoints()
            return bool(targets) and all(slot_activity(h, p) is not None for h, p in targets)
        except Exception:
            return False

    def verify_idle(self, wait: float = 10.0) -> StopReport:
        started = self.clock()
        polls = 0
        evidence: list = []
        while True:
            polls += 1
            try:
                targets = self.endpoints()
                states = {f"{h}:{p}": slot_activity(h, p) for h, p in targets}
            except Exception as exc:  # discovery failure is never proof
                targets, states = [], {}
                evidence = [{"error": type(exc).__name__}]
            else:
                evidence = [{"server": key, "slots": None if s is None else len(s),
                             "busy": None if s is None else sum(s)} for key, s in states.items()]
                if targets and all(s is not None and not any(s) for s in states.values()):
                    return StopReport(True, "every slot reports idle", polls, round(self.clock() - started, 3), evidence)
            if self.clock() - started >= wait:
                return StopReport(False, "could not confirm the server is idle", polls,
                                  round(self.clock() - started, 3), evidence)
            self.sleep(self.poll_seconds)


class SlotsVerifier(_PollingVerifier):
    """For a llama.cpp server you started yourself (llama-server --port N)."""

    def __init__(self, base_url: str, **kwargs):
        super().__init__(**kwargs)
        parts = urlsplit(base_url)
        self.host, self.port = parts.hostname or "127.0.0.1", parts.port or 80

    def endpoints(self) -> list[tuple[str, int]]:
        return [(self.host, self.port)]


class OllamaRunnerVerifier(_PollingVerifier):
    """Finds the llama.cpp runner processes behind an Ollama server and checks their slots."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", **kwargs):
        super().__init__(**kwargs)
        self.port = urlsplit(base_url).port or 11434

    def endpoints(self) -> list[tuple[str, int]]:
        import psutil  # optional dependency: pip install psutil

        servers = {c.pid for c in psutil.net_connections(kind="tcp")
                   if c.pid and c.laddr and c.laddr.port == self.port and c.status == psutil.CONN_LISTEN}
        found: set[tuple[str, int]] = set()
        for pid in servers:
            for child in psutil.Process(pid).children(recursive=False):
                if "llama-server" not in child.name().lower():
                    continue
                args = child.cmdline()
                for index, arg in enumerate(args):
                    if arg == "--port" and index + 1 < len(args) and args[index + 1].isdigit():
                        found.add(("127.0.0.1", int(args[index + 1])))
                    elif arg.startswith("--port=") and arg[7:].isdigit():
                        found.add(("127.0.0.1", int(arg[7:])))
        return sorted(found)


class NullVerifier:
    def available(self) -> bool:
        return False

    def verify_idle(self, wait: float = 0.0) -> StopReport:
        return StopReport(False, "no way to check this server; assume it may still be generating")
