"""Real local socket deadlines; CPU MockEngine only, no listener/model/GPU.

Behavior ported from aflin/momoe e7fe3200 (serve/test_connection_deadline.py).
Use socketpair so the production handler and OS buffering run without opening
an application port or launching a server process.
"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from serve import server
from serve.frontend import ChatTemplate
from test_server_stream_stability import handler_fixture, close_fixture


class PeerTimeout(unittest.TestCase):
    def start_handler(self, timeout=.15, script="ok"):
        tok = server.ByteTokenizer()
        service = server.Service(server.MockEngine(tok, script, max_context=4096), tok,
                                 ChatTemplate(ROOT / "serve/chat_template.jinja"))
        with mock.patch.object(server, "PEER_STALL_S", timeout):
            handler = server.make_handler(service)
        upstream, client = socket.socketpair()
        errors = []

        def serve_one():
            try:
                handler(upstream, ("127.0.0.1", 1), SimpleNamespace())
            except BaseException as error:
                errors.append(error)
            finally:
                upstream.close()

        thread = threading.Thread(target=serve_one, daemon=True)
        thread.start()

        def cleanup():
            client.close()
            upstream.close()
            thread.join(2)
            self.assertFalse(thread.is_alive(), "fixture left an owned handler running")

        self.addCleanup(cleanup)
        client.settimeout(2)
        return client, thread, errors, service, handler

    def test_silent_peer_and_partial_headers_end_without_admission(self):
        for data in (b"", b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"):
            with self.subTest(data=data):
                client, thread, errors, service, _ = self.start_handler()
                if data:
                    client.sendall(data)
                started = time.monotonic()
                self.assertEqual(client.recv(1), b"")
                thread.join(1)
                self.assertGreater(time.monotonic() - started, .05)
                self.assertFalse(thread.is_alive())
                self.assertFalse(service.fifo.locked())
                self.assertEqual(errors, [])

    def test_incomplete_fixed_and_chunked_bodies_do_not_hold_a_handler(self):
        for transfer, body in ((b"Content-Length: 200\r\n", b'{"messages":'),
                               (b"Transfer-Encoding: chunked\r\n", b"20\r\npartial")):
            with self.subTest(transfer=transfer):
                client, thread, errors, service, _ = self.start_handler()
                client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                               b"Content-Type: application/json\r\n" + transfer + b"\r\n" + body)
                self.assertEqual(client.recv(1), b"")
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertFalse(service.fifo.locked())
                self.assertEqual(errors, [])

    def test_working_peer_can_send_for_longer_than_one_timeout(self):
        client, thread, errors, _, _ = self.start_handler(timeout=.2)
        chunks = [b"GET /health HTTP/1.1\r\n", b"Host: 127.0.0.1\r\n",
                  b"X-Progress: one\r\n", b"X-Progress-2: two\r\n", b"\r\n"]
        started = time.monotonic()
        for chunk in chunks:
            client.sendall(chunk)
            time.sleep(.06)
        wire = bytearray()
        while piece := client.recv(4096):
            wire.extend(piece)
        self.assertGreater(time.monotonic() - started, .2)
        self.assertTrue(wire.startswith(b"HTTP/1.0 200"), wire)
        self.assertEqual(json.loads(wire.split(b"\r\n\r\n", 1)[1])["status"], "ok")
        thread.join(1)
        self.assertEqual(errors, [])

    def test_explicit_zero_environment_disables_peer_deadline(self):
        with mock.patch.dict(server.os.environ, {"STRATA_PEER_STALL_S": "0"}):
            self.assertIsNone(server._timeout_env("STRATA_PEER_STALL_S", 120))
        client, thread, errors, _, handler = self.start_handler(timeout=None)
        self.assertIsNone(handler.timeout)
        time.sleep(.2)
        self.assertTrue(thread.is_alive())
        client.sendall(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        self.assertTrue(client.recv(4096).startswith(b"HTTP/1.0 200"))
        thread.join(1)
        self.assertEqual(errors, [])

    def test_sse_keeps_the_shorter_peer_or_write_limit(self):
        for peer, write, expected in ((.1, 2., .1), (2., .1, .1), (None, .1, .1),
                                      (.1, 0., .1), (None, 0., None), (.1, -1., .1), (None, -1., None)):
            with self.subTest(peer=peer, write=write):
                handler, client, _, _, _ = handler_fixture("ok")
                try:
                    with mock.patch.object(type(handler), "timeout", peer), \
                         mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", write):
                        handler._sse()
                    self.assertEqual(handler.connection.gettimeout(), expected)
                finally:
                    close_fixture(handler, client)

    def test_peer_limit_releases_fifo_when_stream_write_timeout_is_disabled(self):
        handler, client, service, engine, request = handler_fixture(" ".join(map(str, range(80000))))
        errors = []

        def write_stream():
            try:
                handler._openai(request)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=write_stream, daemon=True)
        try:
            with mock.patch.object(type(handler), "timeout", .1), \
                 mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0), redirect_stdout(StringIO()):
                worker.start()
                deadline = time.monotonic() + 2
                while not service.fifo.locked() and worker.is_alive() and time.monotonic() < deadline:
                    time.sleep(.005)
                self.assertTrue(service.fifo.locked(), "a real generation must acquire its turn")
                worker.join(2)
                self.assertFalse(worker.is_alive(), "a non-reading peer retained the engine turn")
                self.assertFalse(service.fifo.locked())
                self.assertEqual(errors, [])
                engine.script = service.tok.encode("next", parse_special=True)
                handler._openai({**request, "stream": False, "max_tokens": 4})
                self.assertEqual(handler.json_response[0], 200)
                self.assertEqual(handler.json_response[1]["choices"][0]["message"]["content"], "next")
        finally:
            client.close()
            worker.join(2)
            close_fixture(handler, client)


if __name__ == "__main__":
    unittest.main()
