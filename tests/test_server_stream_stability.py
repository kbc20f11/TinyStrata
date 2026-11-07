"""Real socket backpressure and log races; no listener, native engine or GPU.

Run: python -m unittest discover -s tests -p test_server_stream_stability.py -v
The socketpair drives the production handler directly with a CPU MockEngine.
"""
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

from serve import server as api
from serve.frontend import ChatTemplate


ROOT = Path(__file__).resolve().parents[1]


class ConvCachePolling(unittest.TestCase):
    PARK = ("strata serve: conversation cache: parked 5000 tokens in 12.0 ms; "
            "parked=1 bytes=104857600 evictions=0\n")

    def test_overlapping_polls_count_each_line_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.log"
            path.write_text(self.PARK, encoding="utf-8")
            cache = api.ConvCacheLog()
            self.assertEqual(cache.poll(str(path), 0)["parks"], 1)
            with path.open("a", encoding="utf-8") as file:
                file.write(self.PARK)
            first_sized, proceed = threading.Event(), threading.Event()
            original_size, calls, errors = os.path.getsize, [], []

            def interrupted_size(filename):
                size = original_size(filename)
                if not calls and filename == str(path):
                    calls.append(filename)
                    first_sized.set()
                    if not proceed.wait(3):
                        raise AssertionError("test did not release first poll")
                return size

            def poll():
                try:
                    cache.poll(str(path), 0)
                except BaseException as error:
                    errors.append(error)

            first = threading.Thread(target=poll, daemon=True)
            second = threading.Thread(target=poll, daemon=True)
            with mock.patch.object(api.os.path, "getsize", interrupted_size):
                try:
                    first.start()
                    self.assertTrue(first_sized.wait(3))
                    with path.open("a", encoding="utf-8") as file:
                        file.write(self.PARK * 2)
                    second.start()
                    second.join(.1)  # without the lock it advances pos past first's captured size
                finally:
                    proceed.set()
                    first.join(3)
                    if second.ident is not None:
                        second.join(3)
            self.assertFalse(first.is_alive() or second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(cache.poll(str(path), 0)["parks"], 4)

    def test_partial_line_and_repeated_poll_do_not_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.log"
            path.write_text(self.PARK[:-1], encoding="utf-8")
            cache = api.ConvCacheLog()
            self.assertEqual(cache.poll(str(path), 0)["parks"], 0)
            with path.open("a", encoding="utf-8") as file:
                file.write("\n")
            result = cache.poll(str(path), 0)
            self.assertEqual(result["parks"], 1)
            for _ in range(3):
                self.assertEqual(cache.poll(str(path), 0), result)

    def test_new_run_offset_resets_and_truncation_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.log"
            path.write_text(self.PARK * 2, encoding="utf-8")
            cache = api.ConvCacheLog()
            self.assertEqual(cache.poll(str(path), 0)["parks"], 2)
            offset = path.stat().st_size
            with path.open("a", encoding="utf-8") as file:
                file.write(self.PARK)
            self.assertEqual(cache.poll(str(path), offset)["parks"], 1)
            path.write_text(self.PARK, encoding="utf-8")
            self.assertEqual(cache.poll(str(path), offset)["parks"], 2)
            self.assertEqual(cache.poll(None, None)["parks"], 2)


def handler_fixture(script):
    tok = api.ByteTokenizer()
    engine = api.MockEngine(tok, script, max_context=2_000_000)
    svc = api.Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    # No server/listener or telemetry is started. The handler still exercises
    # its actual preparation, FIFO, generator, socket writes and cancellation.
    svc.start_telemetry = lambda: (_ for _ in ()).throw(AssertionError("telemetry initialized"))
    Handler = api.make_handler(svc)
    handler = object.__new__(Handler)
    writer, reader = socket.socketpair()
    writer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    reader.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    handler.connection = writer
    handler.wfile = writer.makefile("wb", buffering=0)
    handler.headers = {}
    handler.sent_headers = []
    handler.send_response = lambda status: setattr(handler, "response_status", status)
    handler.send_header = lambda key, value: handler.sent_headers.append((key, value))
    handler.end_headers = lambda: None
    handler._cors = lambda: None
    handler._json = lambda status, value: setattr(handler, "json_response", (status, value))
    request = {"messages": [{"role": "user", "content": "go"}], "stream": True,
               "max_tokens": 400000, "reasoning_effort": "none"}
    return handler, reader, svc, engine, request


def close_fixture(handler, reader):
    if hasattr(handler, "watch_done"):
        handler.watch_done.set()
    reader.close()
    handler.wfile.close()
    handler.connection.close()


class StreamBackpressure(unittest.TestCase):
    def test_nonreading_open_client_releases_fifo_and_next_request_works(self):
        # Distinct tokens prevent the existing repetition guard from ending the
        # request before the real OS send buffer fills.
        handler, reader, svc, engine, request = handler_fixture(" ".join(map(str, range(80000))))
        errors = []

        def run():
            try:
                handler._openai(request)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run, daemon=True)
        try:
            with mock.patch.object(api, "STREAM_WRITE_TIMEOUT_S", .1, create=True), redirect_stdout(StringIO()):
                worker.start()
                end = time.monotonic() + 3
                while not svc.fifo.locked() and worker.is_alive() and time.monotonic() < end:
                    time.sleep(.005)
                self.assertTrue(svc.fifo.locked(), "the real stream must acquire its FIFO turn")
                worker.join(2)
                self.assertFalse(worker.is_alive(), "a connected client that never reads retained the turn")
                self.assertFalse(svc.fifo.locked())
                self.assertEqual(errors, [])
                # A subsequent ordinary request can immediately own the same
                # Service. No mocked FIFO release or engine restart is used.
                engine.script = svc.tok.encode("next", parse_special=True)
                handler._openai({**request, "stream": False, "max_tokens": 4})
                self.assertEqual(handler.json_response[0], 200)
                self.assertEqual(handler.json_response[1]["choices"][0]["message"]["content"], "next")
        finally:
            reader.close()  # also releases a baseline blocked writer on failure
            worker.join(3)
            close_fixture(handler, reader)

    def test_zero_timeout_keeps_existing_unbounded_stream_until_client_closes(self):
        handler, reader, svc, _, request = handler_fixture(" ".join(map(str, range(80000))))
        errors = []

        def run():
            try:
                handler._openai(request)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run, daemon=True)
        try:
            with mock.patch.object(api, "STREAM_WRITE_TIMEOUT_S", 0, create=True), redirect_stdout(StringIO()):
                worker.start()
                end = time.monotonic() + 3
                while not svc.fifo.locked() and worker.is_alive() and time.monotonic() < end:
                    time.sleep(.005)
                self.assertTrue(svc.fifo.locked())
                worker.join(.25)
                self.assertTrue(worker.is_alive())
                self.assertIsNone(handler.connection.gettimeout())
                reader.close()
                worker.join(3)
                self.assertFalse(worker.is_alive())
                self.assertFalse(svc.fifo.locked())
                self.assertEqual(errors, [])
        finally:
            reader.close()
            worker.join(3)
            close_fixture(handler, reader)

    def test_reading_client_gets_unchanged_text_usage_and_sse_headers(self):
        handler, reader, svc, _, request = handler_fixture("hello 日本語")
        received = bytearray()
        finished = threading.Event()

        def drain():
            reader.settimeout(.1)
            while not finished.is_set():
                try:
                    received.extend(reader.recv(65536))
                except socket.timeout:
                    continue

        consumer = threading.Thread(target=drain, daemon=True)
        try:
            with mock.patch.object(api, "STREAM_WRITE_TIMEOUT_S", .1, create=True), redirect_stdout(StringIO()):
                consumer.start()
                handler._openai({**request, "max_tokens": 100, "return_live_stats": True})
                deadline = time.monotonic() + 2
                while b"[DONE]" not in received and time.monotonic() < deadline:
                    time.sleep(.005)
                finished.set()
                consumer.join(1)
            chunks = [json.loads(line[6:]) for line in received.decode().splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            text = "".join(chunk.get("choices", [{}])[0].get("delta", {}).get("content", "") for chunk in chunks)
            self.assertEqual(text, "hello 日本語")
            self.assertIn(b"data: [DONE]", received)
            self.assertTrue(any("strata_stats" in chunk for chunk in chunks))
            self.assertTrue(any(chunk.get("usage", {}).get("completion_tokens", 0) > 0 for chunk in chunks))
            self.assertFalse(svc.fifo.locked())
            self.assertEqual(handler.response_status, 200)
            self.assertIn(("Content-Type", "text/event-stream"), handler.sent_headers)
            self.assertIn(("X-Accel-Buffering", "no"), handler.sent_headers)
        finally:
            finished.set()
            consumer.join(1)
            close_fixture(handler, reader)

    def test_nonstreaming_request_leaves_socket_read_timeout_unchanged(self):
        handler, reader, _, _, request = handler_fixture("ok")
        try:
            handler.connection.settimeout(2.5)
            with mock.patch.object(api, "STREAM_WRITE_TIMEOUT_S", .1, create=True), redirect_stdout(StringIO()):
                handler._openai({**request, "stream": False, "max_tokens": 2})
            self.assertEqual(handler.connection.gettimeout(), 2.5)
            self.assertEqual(handler.json_response[0], 200)
        finally:
            close_fixture(handler, reader)


if __name__ == "__main__":
    unittest.main()
