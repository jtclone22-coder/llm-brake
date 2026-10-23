"""One generation at a time per GPU/server, with a bounded queue and an honest HOLD.

If a stop could not be verified, the server may still be generating, so the gate HOLDs: new requests
are refused instead of piling onto a busy GPU. Unlike a permanent lockout, the hold clears itself
once the verifier sees the server idle (checked on the next request, at most every `recheck_seconds`),
or when an operator calls reset() after checking by hand.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

from .budget import Budget
from .transport import Interrupted


class GateBusy(Exception):
    """The waiting line is full."""


class GateHeld(Exception):
    """An earlier stop could not be verified; the server may still be busy."""


class Gate:
    def __init__(self, verifier=None, *, max_waiting: int = 8, recheck_seconds: float = 5.0,
                 clock=time.monotonic):
        self.verifier = verifier
        self.max_waiting = max_waiting
        self.recheck_seconds = recheck_seconds
        self.clock = clock
        self._cond = threading.Condition()
        self._queue: deque[int] = deque()
        self._active: int | None = None
        self._ticket = 0
        self._held = ""
        self._last_check = 0.0
        self.completed = 0

    def hold(self, detail: str) -> None:
        with self._cond:
            self._held = detail or "unverified stop"
            self._last_check = self.clock()
            self._cond.notify_all()

    def reset(self) -> None:
        """Clear a hold by hand (after confirming the server is idle yourself)."""
        with self._cond:
            self._held = ""
            self._cond.notify_all()

    def _maybe_clear_hold(self) -> None:
        if not self._held or self.verifier is None or self.clock() - self._last_check < self.recheck_seconds:
            return
        self._last_check = self.clock()
        report = self.verifier.verify_idle(0)
        if report.verified:
            self._held = ""

    def snapshot(self) -> dict:
        with self._cond:
            return {"state": "HOLD" if self._held else "BUSY" if self._active is not None else "READY",
                    "held": self._held, "waiting": len(self._queue), "completed": self.completed}

    @contextmanager
    def slot(self, budget: Budget):
        with self._cond:
            self._maybe_clear_hold()
            if self._held:
                raise GateHeld(self._held)
            if len(self._queue) >= self.max_waiting:
                raise GateBusy("too many requests waiting")
            self._ticket += 1
            ticket = self._ticket
            self._queue.append(ticket)
            try:
                while True:
                    if self._held:
                        raise GateHeld(self._held)
                    if budget():
                        raise Interrupted(budget.reason)
                    if self._active is None and self._queue[0] == ticket:
                        self._queue.popleft()
                        self._active = ticket
                        break
                    self._cond.wait(min(0.05, max(0.0, budget.remaining())))
            finally:
                if ticket in self._queue:
                    self._queue.remove(ticket)
                    self._cond.notify_all()
        try:
            yield
        finally:
            with self._cond:
                self._active = None
                self.completed += 1
                self._cond.notify_all()
