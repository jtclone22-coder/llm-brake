"""A Budget bounds one request: a monotonic deadline plus an explicit stop signal, with the reason recorded.

    budget = Budget(60)            # at most 60 s
    budget.cancel("user_pressed_stop")
    budget()                       # True: stop now
    budget.reason                  # "user_pressed_stop"

Child budgets never outlive their parent: a tool call inside a chat turn gets min(own limit, time left).
"""
from __future__ import annotations

import math
import threading
import time
from typing import Callable

CANCELLED = "cancelled"
DEADLINE = "deadline"
DISCONNECTED = "client_disconnected"


class Budget:
    def __init__(self, seconds: float | None = None, *, parent: "Budget | None" = None,
                 clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.started = clock()
        own = math.inf if seconds is None else self.started + max(0.0, float(seconds))
        self.deadline = min(own, parent.deadline) if parent is not None else own
        self.parent = parent
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()

    def __call__(self) -> bool:
        """True when the request should stop now (cancelled, parent stopped, or out of time)."""
        if self._event.is_set():
            return True
        if self.parent is not None and self.parent():
            self._note(self.parent.reason or CANCELLED)
            return True
        if self.clock() >= self.deadline:
            self._note(DEADLINE)
            return True
        return False

    def _note(self, reason: str) -> None:
        with self._lock:
            if not self._reason:
                self._reason = reason

    def cancel(self, reason: str = CANCELLED) -> None:
        self._note(reason)
        self._event.set()

    @property
    def reason(self) -> str:
        """Why it stopped, or "" while it may continue. The first reason wins."""
        self()
        return self._reason

    @property
    def stopped(self) -> bool:
        return self()

    def remaining(self) -> float:
        return self.deadline - self.clock()

    def timeout(self, cap: float | None = None, floor: float = 0.05) -> float:
        """A socket/HTTP timeout for the next step: what is left, optionally capped."""
        left = self.remaining()
        if cap is not None:
            left = min(left, cap)
        return max(floor, left) if math.isfinite(left) else (cap if cap is not None else 3600.0)

    def child(self, seconds: float | None = None) -> "Budget":
        return Budget(seconds, parent=self, clock=self.clock)

    def wait(self, seconds: float) -> bool:
        """Sleep up to `seconds`, waking early on cancel. Returns True if the budget stopped."""
        end = min(self.clock() + seconds, self.deadline)
        while not self():
            left = end - self.clock()
            if left <= 0:
                break
            self._event.wait(min(left, 0.05))
        return self()

    def __repr__(self) -> str:
        left = self.remaining()
        return f"Budget(remaining={'inf' if not math.isfinite(left) else round(left, 2)}, reason={self._reason!r})"
