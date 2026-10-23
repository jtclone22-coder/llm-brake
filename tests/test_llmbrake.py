"""llm-brake tests against a fake local LLM server (loopback only; no real model)."""
from __future__ import annotations

import asyncio
import json
import select
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmbrake import (  # noqa: E402
    DISCONNECTED, Budget, Gate, GateBusy, GateHeld, GenerationStopped, HTTPError, Interrupted, NullVerifier,
    OllamaClient, OpenAICompatClient, ServerError, SlotsVerifier, pump)


class FakeServer:
    """Streams `tokens` words, one every `interval` s, after `prefill` s. Cancels on disconnect unless stubborn."""

    def __init__(self, tokens=20, interval=0.05, prefill=0.0, stubborn_for=0.0, error=None, status=200):
        self.tokens, self.interval, self.prefill = tokens, interval, prefill
        self.stubborn_for, self.error, self.status = stubborn_for, error, status
        self.processing = False
        self.disconnects = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path != "/slots":
                    self.send_error(404)
                    return
                body = json.dumps([{"id": 0, "is_processing": owner.processing}]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def client_gone(self) -> bool:
                ready, _, _ = select.select([self.connection], [], [], 0)
                if not ready:
                    return False
                try:
                    return self.connection.recv(1, socket.MSG_PEEK) == b""
                except OSError:
                    return True

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                json.loads(self.rfile.read(length) or b"{}")
                if owner.status != 200:
                    body = b'{"error": "model not found"}'
                    self.send_response(owner.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.flush()
                    time.sleep(0.2)  # the body arrives after the headers, as it often does on a real network
                    self.wfile.write(body)
                    return
                openai = self.path.startswith("/v1/")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if openai else "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                owner.processing = True
                try:
                    end = time.monotonic() + owner.prefill
                    while time.monotonic() < end:
                        if self.client_gone():
                            raise ConnectionResetError
                        time.sleep(0.02)
                    if owner.error:
                        self.chunk(json.dumps({"error": owner.error}) + "\n")
                        self.finish_chunks()
                        return
                    for index in range(owner.tokens):
                        word = f"w{index} "
                        if openai:
                            self.chunk("data: " + json.dumps({"choices": [{"delta": {"content": word}}]}) + "\n\n")
                        else:
                            self.chunk(json.dumps({"message": {"role": "assistant", "content": word}, "done": False}) + "\n")
                        time.sleep(owner.interval)
                        if self.client_gone():
                            raise ConnectionResetError
                    if openai:
                        self.chunk("data: [DONE]\n\n")
                    else:
                        self.chunk(json.dumps({"message": {"content": ""}, "done": True}) + "\n")
                    self.finish_chunks()
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                    owner.disconnects += 1
                    if owner.stubborn_for:
                        threading.Timer(owner.stubborn_for, lambda: setattr(owner, "processing", False)).start()
                        return
                finally:
                    if not owner.stubborn_for:
                        owner.processing = False

            def chunk(self, text: str) -> None:
                data = text.encode()
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            def finish_chunks(self) -> None:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def wait_until(condition, seconds=3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


class BudgetTests(unittest.TestCase):
    def test_reasons_children_and_waiting(self) -> None:
        now = [0.0]
        budget = Budget(10, clock=lambda: now[0])
        child = budget.child(60)
        self.assertEqual(child.deadline, 10)  # never outlives the parent
        self.assertFalse(budget())
        now[0] = 11
        self.assertTrue(child())
        self.assertEqual(child.reason, "deadline")
        parent = Budget()
        kid = parent.child(5)
        parent.cancel("user_pressed_stop")
        self.assertEqual(kid.reason, "user_pressed_stop")
        waiter = Budget()
        threading.Timer(0.1, waiter.cancel).start()
        started = time.monotonic()
        self.assertTrue(waiter.wait(5))
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(Budget().timeout(cap=7), 7)


class StreamingStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.servers = []

    def tearDown(self) -> None:
        for server in self.servers:
            server.close()

    def server(self, **kwargs) -> FakeServer:
        server = FakeServer(**kwargs)
        self.servers.append(server)
        return server

    def client(self, server, gate=None, cls=OllamaClient) -> OllamaClient:
        return cls(server.url, verifier=SlotsVerifier(server.url), gate=gate, stop_wait=2.0)

    def test_full_stream(self) -> None:
        server = self.server(tokens=5, interval=0.01)
        text = self.client(server).complete({"model": "m", "messages": []}, Budget(10))
        self.assertEqual(text, "w0 w1 w2 w3 w4 ")

    def test_cancel_mid_stream_stops_and_is_verified(self) -> None:
        server = self.server(tokens=200, interval=0.05)
        budget = Budget(30)
        got = []
        started = time.monotonic()
        with self.assertRaises(GenerationStopped) as caught:
            for text in self.client(server).chat({"model": "m", "messages": []}, budget):
                got.append(text)
                if len(got) == 3:
                    threading.Timer(0.01, budget.cancel).start()
        stop = caught.exception
        self.assertEqual(stop.reason, "cancelled")
        self.assertTrue(stop.verified, stop.report)
        self.assertTrue(stop.partial.startswith("w0 w1 w2 "))
        self.assertLess(time.monotonic() - started, 3)
        self.assertTrue(wait_until(lambda: server.disconnects == 1))

    def test_cancel_before_the_first_token(self) -> None:
        server = self.server(prefill=10)
        budget = Budget(30)
        threading.Timer(0.3, budget.cancel).start()
        started = time.monotonic()
        with self.assertRaises(GenerationStopped) as caught:
            list(self.client(server).chat({"model": "m", "messages": []}, budget))
        self.assertLess(time.monotonic() - started, 2)  # not 10 s of prefill
        self.assertTrue(caught.exception.verified)

    def test_deadline(self) -> None:
        server = self.server(tokens=200, interval=0.05)
        with self.assertRaises(GenerationStopped) as caught:
            list(self.client(server).chat({"model": "m", "messages": []}, Budget(0.4)))
        self.assertEqual(caught.exception.reason, "deadline")

    def test_unverified_stop_holds_the_gate_until_the_server_is_idle(self) -> None:
        server = self.server(tokens=200, interval=0.05, stubborn_for=0.8)
        gate = Gate(SlotsVerifier(server.url), recheck_seconds=0.2)
        client = OllamaClient(server.url, verifier=SlotsVerifier(server.url), gate=gate, stop_wait=0.3)
        budget = Budget(30)
        threading.Timer(0.2, budget.cancel).start()
        with self.assertRaises(GenerationStopped) as caught:
            list(client.chat({"model": "m", "messages": []}, budget))
        self.assertFalse(caught.exception.verified)
        self.assertEqual(gate.snapshot()["state"], "HOLD")
        with self.assertRaises(GateHeld):
            list(client.chat({"model": "m", "messages": []}, Budget(5)))
        server.stubborn_for = 0  # from now on it behaves
        server.tokens = 2
        self.assertTrue(wait_until(lambda: not server.processing, 3))
        time.sleep(0.25)
        self.assertEqual(client.complete({"model": "m", "messages": []}, Budget(5)), "w0 w1 ")
        self.assertEqual(gate.snapshot()["state"], "READY")

    def test_abandoned_iterator_closes_the_connection(self) -> None:
        server = self.server(tokens=200, interval=0.05)
        gate = Gate(SlotsVerifier(server.url))
        stream = self.client(server, gate).chat({"model": "m", "messages": []}, Budget(30))
        next(stream)
        next(stream)
        stream.close()
        self.assertTrue(wait_until(lambda: server.disconnects == 1))
        self.assertEqual(gate.snapshot()["state"], "READY")  # verified idle, so no hold

    def test_openai_compatible_stream(self) -> None:
        server = self.server(tokens=3, interval=0.01)
        client = self.client(server, cls=OpenAICompatClient)
        self.assertEqual(client.complete({"model": "m", "messages": []}, Budget(10)), "w0 w1 w2 ")

    def test_errors(self) -> None:
        with self.assertRaises(HTTPError):
            list(self.client(self.server(status=404)).chat({"model": "nope", "messages": []}, Budget(5)))
        with self.assertRaises(ServerError):
            list(self.client(self.server(error="out of memory")).chat({"model": "m", "messages": []}, Budget(5)))

    def test_null_verifier_never_claims_success(self) -> None:
        report = NullVerifier().verify_idle()
        self.assertFalse(report.verified)


class GateTests(unittest.TestCase):
    def test_queue_limit_order_and_budget(self) -> None:
        gate = Gate(max_waiting=1)
        order = []
        release = threading.Event()

        def first():
            with gate.slot(Budget(5)):
                order.append("first")
                release.wait(2)

        worker = threading.Thread(target=first)
        worker.start()
        self.assertTrue(wait_until(lambda: gate.snapshot()["state"] == "BUSY"))
        second_done = threading.Event()

        def second():
            with gate.slot(Budget(5)):
                order.append("second")
            second_done.set()

        threading.Thread(target=second).start()
        self.assertTrue(wait_until(lambda: gate.snapshot()["waiting"] == 1))
        with self.assertRaises(GateBusy):
            with gate.slot(Budget(5)):
                pass
        release.set()
        self.assertTrue(second_done.wait(2))
        self.assertEqual(order, ["first", "second"])
        worker.join(1)
        with gate.slot(Budget(5)):  # occupy the gate
            with self.assertRaises(Interrupted):
                with gate.slot(Budget(0.2)):  # waits in line, then its own budget runs out
                    pass

    def test_manual_reset(self) -> None:
        gate = Gate()
        gate.hold("checked by hand")
        with self.assertRaises(GateHeld):
            with gate.slot(Budget(1)):
                pass
        gate.reset()
        with gate.slot(Budget(1)):
            pass


class PumpTests(unittest.TestCase):
    def test_consumer_leaving_cancels_the_budget(self) -> None:
        budget = Budget(30)
        produced = []

        def slow():
            for index in range(100):
                if budget():
                    return
                produced.append(index)
                time.sleep(0.02)
                yield index

        async def consume():
            agen = pump(slow(), budget)
            got = [await agen.__anext__(), await agen.__anext__()]
            await agen.aclose()
            return got

        self.assertEqual(asyncio.run(consume()), [0, 1])
        self.assertEqual(budget.reason, DISCONNECTED)
        self.assertTrue(wait_until(lambda: len(produced) < 100 and budget()))

    def test_complete_run_and_errors(self) -> None:
        async def collect(items):
            return [item async for item in pump(items, Budget(5))]

        self.assertEqual(asyncio.run(collect(iter([1, 2, 3]))), [1, 2, 3])

        def broken():
            yield 1
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            asyncio.run(collect(broken()))


if __name__ == "__main__":
    unittest.main()
