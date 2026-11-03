"""HTTP boundary regressions using only a temp UI, fake search and temp data.

No production port/profile, real provider, model process or GPU is accessed.
Run: python -B tests/test_ui_server_tools.py
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui import sandbox_runtime
from ui.chat_history import MAX_REQUEST_BYTES
from ui.server import Handler, UIServer
from ui.web_search import SearchError

Connection = http.client.HTTPConnection


def chat(id="cfixture", text="question"):
    return {"id": id, "time": 1234, "title": "", "messages": [{"role": "user", "text": text}]}


class FakeSearch:
    def __init__(self):
        self.calls = []; self.closed = 0
        self.started, self.cancelled, self.ended = threading.Event(), threading.Event(), threading.Event()
    def search(self, query, *, settings, cancel):
        self.calls.append({"query": query, "settings": dict(settings)})
        if query == "rate-limit":
            raise SearchError("rate_limited", retry_after=2.1, retryable=True)
        if query == "blocked":
            self.started.set()
            try:
                if cancel.wait(4): self.cancelled.set()
                raise SearchError("cancelled")
            finally: self.ended.set()
        return {"query": query, "provider": settings["provider"], "results": [{"title": "Fixture", "url": "https://example.test/reference", "snippet": "untrusted fixture text"}]}
    def close(self): self.closed += 1


class ServerToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tinystrata-ui-http-test-")
        self.data = Path(self.temp.name)
        # Provider and runtime downloads are forbidden even if a route regresses.
        self.no_network = mock.patch("ui.sandbox_runtime.urllib.request.build_opener", side_effect=AssertionError("Runtime network forbidden in HTTP fixture"))
        self.no_network.start(); self.addCleanup(self.no_network.stop)
        self.no_relay = mock.patch.object(Handler, "relay", side_effect=AssertionError("Inference relay forbidden in HTTP fixture"))
        self.no_relay.start(); self.addCleanup(self.no_relay.stop)
        self.logs = mock.patch.object(Handler, "log_message")
        self.log_spy = self.logs.start(); self.addCleanup(self.logs.stop)
        self.server = None; self.thread = None; self.start_server()
    def tearDown(self):
        self.stop_server(); self.temp.cleanup()
    def start_server(self):
        self.server = UIServer(0, 9, self.data)
        self.server.search.close()
        self.fake = FakeSearch(); self.server.search = self.fake
        self.port = self.server.server_port; self.origin = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
    def stop_server(self):
        if self.server is not None:
            self.server.shutdown(); self.thread.join(3); self.server.server_close()
            self.assertFalse(self.thread.is_alive()); self.server = None
    def http(self, method, path, value=None, *, headers=None, raw=None):
        body = json.dumps(value).encode() if value is not None else raw
        request_headers = {"Origin": self.origin, "Connection": "close"}
        if method in {"POST", "PUT", "PATCH", "DELETE"}: request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        connection = Connection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            response = connection.getresponse(); data = response.read()
            meta = dict(response.getheaders())
            return response.status, meta, json.loads(data) if data and "application/json" in meta.get("Content-Type", "") else data
        finally: connection.close()
    def settings(self): return self.http("GET", "/local-ui/settings")[2]
    def put_settings(self, **values):
        return self.http("PUT", "/local-ui/settings", values)

    def test_history_http_migration_attachments_and_server_restart(self):
        original = chat(); original["messages"][0].update(images=[{"name": "x.png", "url": "data:image/png;base64,cGl4ZWxz"}], files=[{"name": "note.txt", "text": "保存する内容\nline2"}])
        status, _, receipt = self.http("POST", "/local-ui/chats/import", {"chats": [original]})
        self.assertEqual(status, 200); self.assertTrue(receipt["confirmed"])
        self.assertEqual(self.http("GET", "/local-ui/chats/imports/" + receipt["backup_id"])[2]["chats"], [original])
        self.assertTrue(self.http("POST", "/local-ui/chats/import", {"chats": [original]})[2]["replayed"])
        saved = self.http("GET", "/local-ui/chats")[2]["chats"][0]
        self.assertEqual(saved["messages"], original["messages"]); self.assertEqual(saved["_revision"], 1)
        changed = {**saved, "title": "renamed"}
        self.assertEqual(self.http("PUT", "/local-ui/chats/cfixture", {"chat": changed, "expected_revision": 1})[0], 200)
        self.assertEqual(self.http("PUT", "/local-ui/chats/cfixture", {"chat": changed, "expected_revision": 1})[0], 409)
        old_search = self.fake; self.stop_server(); self.assertEqual(old_search.closed, 1); self.start_server()
        saved = self.http("GET", "/local-ui/chats")[2]["chats"][0]
        self.assertEqual(saved["title"], "renamed"); self.assertEqual(saved["messages"], original["messages"])
        self.assertEqual(self.http("DELETE", "/local-ui/chats/cfixture", {"expected_revision": 2})[0], 200)
        # A different import batch cannot resurrect an old tombstoned browser ID.
        self.http("POST", "/local-ui/chats/import", {"chats": [original, chat("cother")]})
        self.assertEqual([c["id"] for c in self.http("GET", "/local-ui/chats")[2]["chats"]], ["cother"])

    def test_host_origin_and_content_type_boundaries(self):
        for headers in ({"Host": "attacker.invalid"}, {"Origin": "https://attacker.invalid"}):
            self.assertEqual(self.http("GET", "/local-ui/settings", headers=headers)[0], 403)
            self.assertEqual(self.http("PUT", "/local-ui/settings", {}, headers=headers)[0], 403)
        self.assertEqual(self.http("PUT", "/local-ui/settings", {}, headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.http("PUT", "/local-ui/settings", {}, headers={"Content-Type": "Application/JSON; charset=utf-8"})[0], 200)
        self.assertFalse(self.server.tool_settings.path.parent.joinpath("attacker.invalid").exists())

    def test_oversize_invalid_json_and_traversal_are_rejected_before_work(self):
        self.assertEqual(self.http("PUT", "/local-ui/chats/cfixture", raw=b"", headers={"Content-Length": str(MAX_REQUEST_BYTES + 1)})[0], 413)
        self.assertEqual(self.http("POST", "/local-ui/search", raw=b"", headers={"Content-Length": "4097"})[0], 400)
        self.assertEqual(self.http("PUT", "/local-ui/settings", raw=b"{broken")[0], 400)
        self.assertEqual(self.http("POST", "/local-ui/search", raw=b"[]")[0], 400)
        self.assertEqual(self.http("POST", "/local-ui/search", raw=b"", headers={"Transfer-Encoding": "chunked"})[0], 400)
        for path in ("/web/%2e%2e/server.py", "/web/%252e%252e/server.py", "/local-ui/chats/../../escape", "/sandbox-runtime/../search-settings.json"):
            self.assertGreaterEqual(self.http("GET", path)[0], 400)
        self.assertEqual(self.fake.calls, [])

    def test_settings_are_global_private_persisted_and_key_omission_keeps_it(self):
        key = "fixture-private-key-4c4f3dfd"
        self.assertEqual(self.settings(), {"provider": "ddgs", "max_concurrency": 1, "has_brave_key": False, "revision": 0})
        status, _, public = self.put_settings(provider="brave", max_concurrency=3, brave_api_key=key, revision=0)
        self.assertEqual(status, 200); self.assertTrue(public["has_brave_key"]); self.assertNotIn(key, json.dumps(public))
        status, _, public = self.put_settings(max_concurrency=2, revision=1)
        self.assertEqual(status, 200); self.assertTrue(public["has_brave_key"])
        self.assertEqual(self.server.tool_settings.read()["brave_api_key"], key)
        for id, model in (("cqwen", "qwen"), ("cgemma", "gemma")):
            self.http("PUT", "/local-ui/chats/" + id, {"chat": {**chat(id), "model": model}, "expected_revision": 0})
        self.assertEqual(self.settings()["provider"], "brave"); self.assertEqual(self.settings()["max_concurrency"], 2)
        saved = self.server.tool_settings.path.read_bytes(); decoded = json.loads(saved)
        if os.name == "nt":
            self.assertEqual(decoded["credential"]["encoding"], "dpapi"); self.assertNotIn(key.encode(), saved)
        else: self.assertEqual(decoded["credential"]["encoding"], "user-file")
        self.stop_server(); self.start_server()
        self.assertTrue(self.settings()["has_brave_key"]); self.assertEqual(self.server.tool_settings.read()["brave_api_key"], key)
        self.assertNotIn(key, json.dumps(self.settings()))
        self.assertEqual(self.put_settings(brave_api_key="", revision=2)[0], 200)
        self.assertFalse(self.settings()["has_brave_key"])

    def test_concurrent_settings_updates_and_invalid_values_preserve_original(self):
        def write(concurrency): return self.put_settings(max_concurrency=concurrency, revision=0)[0]
        with ThreadPoolExecutor(2) as pool: statuses = list(pool.map(write, [2, 3]))
        self.assertEqual(sorted(statuses), [200, 400]); self.assertEqual(self.settings()["revision"], 1)
        before = self.server.tool_settings.path.read_bytes()
        for values in ({"revision": 0}, {"revision": True}, {"provider": "untrusted"}, {"max_concurrency": True}, {"max_concurrency": 0}, {"max_concurrency": 5}, {"brave_api_key": "secret\r\nheader"}, {"unknown": "secret"}):
            status, _, data = self.http("PUT", "/local-ui/settings", values)
            self.assertEqual(status, 400); self.assertNotIn("secret", json.dumps(data)); self.assertEqual(self.server.tool_settings.path.read_bytes(), before)

    def test_corrupt_settings_fail_closed_without_replacing_the_file(self):
        self.put_settings(max_concurrency=2)
        good = json.loads(self.server.tool_settings.path.read_bytes())
        bad_values = [{**good, "version": 2}, {**good, "version": "1"}, {**good, "revision": True}, {**good, "revision": 0},
                      {**good, "credential": {"encoding": "invalid", "value": "fixture-secret"}}, {**good, "credential": {"encoding": "dpapi", "value": "not base64 !"}}]
        for value in bad_values:
            raw = json.dumps(value).encode(); self.server.tool_settings.path.write_bytes(raw)
            for method, body in (("GET", None), ("PUT", {"provider": "ddgs"})):
                status, _, data = self.http(method, "/local-ui/settings", body)
                self.assertEqual(status, 400); self.assertNotIn("fixture-secret", json.dumps(data))
                self.assertEqual(self.server.tool_settings.path.read_bytes(), raw)

    def test_search_key_stays_server_side_and_client_cannot_override_provider(self):
        key = "server-only-key-313f110e"
        self.put_settings(provider="brave", brave_api_key=key, max_concurrency=2)
        status, _, result = self.http("POST", "/local-ui/search", {"query": "local test"})
        self.assertEqual(status, 200); self.assertEqual(self.fake.calls[-1]["settings"]["brave_api_key"], key)
        self.assertEqual(self.fake.calls[-1]["settings"]["max_concurrency"], 2)
        self.assertNotIn(key, json.dumps(result)); self.assertNotIn(key, str(self.log_spy.call_args_list))
        for extra in ({"provider": "ddgs"}, {"brave_api_key": "client key"}, {"max_concurrency": 4}):
            self.assertEqual(self.http("POST", "/local-ui/search", {"query": "local test", **extra})[0], 400)
        self.assertEqual(len(self.fake.calls), 1)

    def test_search_rate_limit_status_and_retry_after_survive_http(self):
        status, headers, value = self.http("POST", "/local-ui/search", {"query": "rate-limit"})
        self.assertEqual(status, 429); self.assertEqual(headers["Retry-After"], "3")
        self.assertEqual(value["error"]["code"], "rate_limited"); self.assertEqual(value["error"]["retry_after"], 3)

    def test_disconnected_search_request_cancels_provider_work(self):
        body = json.dumps({"query": "blocked"}).encode()
        client = socket.create_connection(("127.0.0.1", self.port), timeout=3)
        request = (f"POST /local-ui/search HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nOrigin: {self.origin}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
        try:
            client.sendall(request); self.assertTrue(self.fake.started.wait(2))
            client.shutdown(socket.SHUT_RDWR)
        finally: client.close()
        self.assertTrue(self.fake.cancelled.wait(2)); self.assertTrue(self.fake.ended.wait(2))

    def test_runtime_get_never_downloads_and_fixed_asset_hash_is_enforced(self):
        version = sandbox_runtime.VERSION
        self.assertFalse(self.http("GET", "/local-ui/sandbox")[2]["ready"])
        for name in sandbox_runtime.FILES:
            self.assertEqual(self.http("GET", f"/sandbox-runtime/{version}/{name}")[0], 404)
        for path in (f"/sandbox-runtime/{version}/arbitrary.py", f"/sandbox-runtime/{version}/%2e%2e%2fsecret", f"/sandbox-runtime/wrong/{next(iter(sandbox_runtime.FILES))}"):
            self.assertEqual(self.http("GET", path)[0], 404)
        self.assertFalse(self.server.sandbox_runtime.path.exists())
        data = b"fixture runtime bytes"; name = "pyodide.js"
        self.server.sandbox_runtime.path.mkdir(parents=True)
        asset = self.server.sandbox_runtime.path / name; asset.write_bytes(data)
        with mock.patch.dict(sandbox_runtime.FILES, {name: (len(data), hashlib.sha256(data).hexdigest())}):
            status, headers, body = self.http("GET", f"/sandbox-runtime/{version}/{name}")
            self.assertEqual(status, 200); self.assertEqual(body, data); self.assertEqual(headers["Cross-Origin-Resource-Policy"], "same-origin")
            self.assertEqual(self.http("HEAD", f"/sandbox-runtime/{version}/{name}")[2], b"")
            asset.write_bytes(b"X" + data[1:])
            self.assertEqual(self.http("GET", f"/sandbox-runtime/{version}/{name}")[0], 404)

    def test_runtime_download_requires_explicit_prepare_and_server_close_closes_search(self):
        calls = []
        def prepare(cancel): calls.append(cancel); return {"version": sandbox_runtime.VERSION, "ready": True}
        with mock.patch.object(self.server.sandbox_runtime, "prepare", side_effect=prepare):
            self.http("GET", "/local-ui/sandbox"); self.assertEqual(calls, [])
            self.assertEqual(self.http("POST", "/local-ui/sandbox/prepare", {})[0], 200); self.assertEqual(len(calls), 1)
        fake = self.fake; self.stop_server(); self.assertEqual(fake.closed, 1)


if __name__ == "__main__": unittest.main()
