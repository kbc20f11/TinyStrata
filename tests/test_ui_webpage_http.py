"""Exercise the browser web reader's real HTTP boundary without any network."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest import mock

from ui.server import Handler, UIServer
from ui.web_page import PageError, PageService
from ui.web_search import SearchService


class WebpageHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tinystrata-webpage-http-")
        self.logs = mock.patch.object(Handler, "log_message"); self.logs.start()
        self.no_relay = mock.patch.object(Handler, "relay", side_effect=AssertionError("UI tool reached inference relay"))
        self.relay = self.no_relay.start()
        resolve = socket.getaddrinfo
        def loopback_only(host, *args, **kwargs):
            if host != "127.0.0.1":
                raise AssertionError("External network forbidden in HTTP fixture")
            return resolve(host, *args, **kwargs)
        self.no_network = mock.patch("socket.getaddrinfo", side_effect=loopback_only)
        self.no_network.start()
        self.server = UIServer(0, 9, Path(self.temp.name))
        self.server.search.close(); self.server.pages.close()
        self.page_calls, self.search_calls, self.failures = [], [], {}
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.search_started, self.page_started = threading.Event(), threading.Event()
        self.release = threading.Event(); self.release.set()
        self.block_search = False
        self.server.pages = PageService(runner=self.page_runner, interval=0)
        self.server.search = SearchService(runner=self.search_runner, intervals={"ddgs": 0, "brave": 0})
        self.page_spy = mock.patch.object(self.server.pages, "read", wraps=self.server.pages.read)
        self.page_read = self.page_spy.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.release.set()
        self.server.shutdown(); self.thread.join(3); self.server.server_close()
        self.page_spy.stop(); self.no_network.stop(); self.no_relay.stop(); self.logs.stop()
        self.temp.cleanup()
        self.assertFalse(self.thread.is_alive())

    @contextmanager
    def running(self):
        with self.lock:
            self.active += 1; self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            with self.lock:
                self.active -= 1

    def page_runner(self, url, cancel, timeout):
        with self.running():
            self.page_started.set(); self.page_calls.append(url)
            failures = self.failures.get(url)
            if failures:
                raise PageError(failures.pop(0))
            return {"url": url, "requested_url": url, "title": "Fixture", "text": "承認済み26台、午前指定",
                    "truncated": False, "content_type": "text/html", "source": "public_web_page", "javascript_executed": False}

    def search_runner(self, query, settings, cancel, timeout):
        with self.running():
            self.search_calls.append(query); self.search_started.set()
            if self.block_search:
                while not self.release.wait(.01):
                    if cancel.is_set():
                        raise AssertionError("Fixture was cancelled unexpectedly")
            return {"results": [{"title": "Fixture", "url": "https://example.org/reference", "snippet": "public fixture"}]}

    def http(self, body=None, *, method="POST", path="/local-ui/webpage", headers=None, raw=None):
        request_headers = {"Origin": self.origin, "Content-Type": "application/json", "Connection": "close", **(headers or {})}
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=json.dumps(body) if raw is None and body is not None else raw, headers=request_headers)
            response = connection.getresponse(); data = response.read()
            return response.status, json.loads(data) if "application/json" in response.getheader("Content-Type", "") else data
        finally:
            connection.close()

    def test_public_page_uses_current_global_settings_without_inference_or_downloads(self):
        self.server.tool_settings.update({"max_concurrency": 2})
        with mock.patch.object(self.server.sandbox_runtime, "prepare", side_effect=AssertionError("No Python bootstrap")):
            status, result = self.http({"url": "https://example.org/reference#section"})
        self.assertEqual(status, 200)
        self.assertEqual(result["text"], "承認済み26台、午前指定")
        self.assertEqual(result["url"], "https://example.org/reference")
        self.assertFalse(result["cached"])
        self.assertEqual(self.page_read.call_args.kwargs["settings"]["max_concurrency"], 2)
        self.assertEqual(self.page_read.call_args.kwargs["settings"]["provider"], "ddgs")
        self.assertFalse(self.server.sandbox_runtime.path.exists())
        self.relay.assert_not_called()

    def test_bad_arguments_and_private_urls_do_not_start_a_fetch(self):
        bodies = [{}, {"url": "https://example.org", "headers": {}}, {"url": "https://example.org", "provider": "brave"},
                  {"url": None}, {"url": 7}, {"url": True}, {"url": []}, {"url": {}}, {"url": ""},
                  {"url": "file:///private.txt"}, {"url": "http://127.0.0.1/private"}, {"url": "https://user:password@example.org/"}]
        for body in bodies:
            with self.subTest(body=body):
                status, value = self.http(body)
                self.assertEqual(status, 400); self.assertIn("message", value["error"])
                self.assertNotIn("password", value["error"]["message"])
        self.assertEqual(self.page_calls, [])
        self.relay.assert_not_called()

    def test_host_origin_and_json_content_type_checks_precede_tool_work(self):
        for headers, expected in [({"Host": "attacker.invalid"}, 403), ({"Origin": "https://attacker.invalid"}, 403), ({"Content-Type": "text/plain"}, 415)]:
            with self.subTest(headers=headers):
                self.assertEqual(self.http({"url": "https://example.org"}, headers=headers)[0], expected)
        self.page_read.assert_not_called(); self.assertEqual(self.page_calls, [])

    def test_invalid_json_oversize_and_wrong_method_never_reach_tool_or_api(self):
        for raw, headers in [(b"[]", {}), (b"{broken", {}), (b"", {"Content-Length": "12289"}), (b"", {"Transfer-Encoding": "chunked"})]:
            with self.subTest(raw=raw, headers=headers):
                self.assertEqual(self.http(raw=raw, headers=headers)[0], 400)
        self.assertEqual(self.http(method="GET")[0], 404)
        self.page_read.assert_not_called(); self.relay.assert_not_called()

    def test_service_failures_map_to_http_status_and_retry_is_available(self):
        for code, expected in [("blocked_address", 400), ("busy", 429), ("fetch_error", 503)]:
            url = "https://example.org/" + code
            self.failures[url] = [code]
            with self.subTest(code=code):
                status, value = self.http({"url": url})
                self.assertEqual(status, expected); self.assertEqual(value["error"]["code"], code)
                status, value = self.http({"url": url})
                self.assertEqual(status, 200); self.assertEqual(value["url"], url)
                self.assertFalse(value["cached"])
        self.relay.assert_not_called()

    def test_global_limit_serializes_search_and_page_requests_from_different_clients(self):
        # Two real HTTP clients are deliberately used: a per-reply browser
        # mapLimit cannot enforce the saved setting across clients.
        self.server.tool_settings.update({"max_concurrency": 1})
        self.block_search = True; self.release.clear()
        with ThreadPoolExecutor(2) as pool:
            search = pool.submit(self.http, {"query": "hold search"}, path="/local-ui/search")
            self.assertTrue(self.search_started.wait(2))
            page = pool.submit(self.http, {"url": "https://example.org/queued"})
            try:
                self.assertFalse(self.page_started.wait(.2), "Saved concurrency1 was bypassed between web tools")
            finally:
                self.release.set()
            self.assertEqual(search.result(timeout=3)[0], 200)
            self.assertEqual(page.result(timeout=3)[0], 200)
        self.assertEqual(self.peak, 1)
        self.assertEqual(self.active, 0)


if __name__ == "__main__":
    unittest.main()
