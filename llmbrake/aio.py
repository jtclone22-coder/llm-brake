"""Use a blocking generation from asyncio web frameworks without losing the Stop.

Frameworks like Starlette/FastAPI drive a sync generator one next() at a time on a worker thread.
When the client disconnects they simply stop asking, so the generator is only closed when the garbage
collector gets to it, and the model keeps generating meanwhile. `pump()` runs the generator to
completion on its own thread and cancels the budget the moment the consumer goes away; the
generation then stops through the normal, verified path.

    @app.post("/chat")
    def chat(req: ChatRequest):
        budget = Budget(120)
        events = (f"data: {json.dumps({'text': t})}\\n\\n" for t in client.chat(payload, budget))
        return StreamingResponse(pump(events, budget), media_type="text/event-stream")
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterable

from .budget import DISCONNECTED, Budget

_END = object()


class _Failure:
    def __init__(self, error: BaseException):
        self.error = error


async def pump(items: Iterable, budget: Budget, *, thread_name: str = "llmbrake-pump") -> AsyncIterator:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    finished = False

    def put(item) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # loop already closed
            pass

    def drive() -> None:
        try:
            for item in items:
                put(item)
        except BaseException as error:  # handed to the consumer, not lost on the thread
            put(_Failure(error))
        finally:
            put(_END)

    threading.Thread(target=drive, name=thread_name, daemon=True).start()
    try:
        while True:
            item = await queue.get()
            if item is _END:
                finished = True
                return
            if isinstance(item, _Failure):
                finished = True
                raise item.error
            yield item
    finally:
        if not finished:
            budget.cancel(DISCONNECTED)
