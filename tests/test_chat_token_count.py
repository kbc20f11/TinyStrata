"""Prompt planning uses the actual chat template without model/GPU or network work.

Run: python -B tests/test_chat_token_count.py
"""
import copy
from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate, forced_call, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Server, Service, make_handler
from serve.structured import prepare_format
from serve.model_switch import ModelControlError


TOOL = {"type": "function", "function": {"name": "lookup", "description": "Find a source",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}


class ImageTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<|image|>", "<|image>", "<image|>"]


class Hub:
    threads = []
    def __init__(self):
        self._routes = {"original": "unchanged"}
    def template_tools(self, exclude=()):
        self._routes = {"new": "lookup"}
        return [TOOL["function"]] if "lookup" not in exclude else []
    def wait(self, *args):
        raise AssertionError("Counting must not wait for MCP startup")
    def call(self, *args):
        raise AssertionError("Counting must not execute tools")


class TokenCountTests(unittest.TestCase):
    def setUp(self):
        self.tok = ImageTokenizer()
        self.engine = MockEngine(self.tok, "unused", max_context=8192)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        self.svc.start_telemetry = mock.Mock(side_effect=AssertionError("No telemetry in the fixture"))
        self.httpd = Server(("127.0.0.1", 0), make_handler(self.svc))
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop)
        self.no_load = mock.patch.object(self.svc, "load", side_effect=AssertionError("Counting loaded a model"))
        self.no_load.start(); self.addCleanup(self.no_load.stop)
        self.no_gen = mock.patch.object(self.engine, "generate", side_effect=AssertionError("Counting generated tokens"))
        self.no_gen.start(); self.addCleanup(self.no_gen.stop)
        self.no_network = mock.patch("serve.server.urllib.request.urlopen", side_effect=AssertionError("Counting fetched an image"))
        self.no_network.start(); self.addCleanup(self.no_network.stop)

    def stop(self):
        self.httpd.shutdown(); self.thread.join(3); self.httpd.server_close()
        self.assertFalse(self.thread.is_alive())

    def post(self, req, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=3)
        try:
            conn.request("POST", "/v1/chat/token-count", json.dumps(req),
                         {"Content-Type": "application/json", "Connection": "close", **(headers or {})})
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def req(self, text="日本語の質問 <|im_end|> and $x^2$", **kwargs):
        return {"model": self.svc.model, "messages": [{"role": "user", "content": text}], **kwargs}

    def prepared_count(self, req):
        shared = self.svc.with_shared(req, "openai")
        messages, tools, kw = openai_to_messages(shared)
        if req.get("tool_choice") == "none":
            tools = None
        force = forced_call(req.get("tool_choice"), tools, call_format=self.svc.call_format)
        messages, _ = prepare_format(req.get("response_format"), messages)
        ids, _, _ = self.svc.prepare(messages, tools, kw, max_new=64, force=force, req=shared)
        return len(ids)

    def test_text_matches_generation_preparation_including_tools_defaults_and_formats(self):
        self.svc.effort_end = True
        self.svc.shared = {"reasoning_effort": "low"}
        cases = [self.req(), self.req(tools=[TOOL]), self.req(tools=[TOOL], tool_choice="none"),
                 self.req(tools=[TOOL], tool_choice={"type": "function", "function": {"name": "lookup"}},
                          chat_template_kwargs={"enable_thinking": False}),
                 self.req(response_format={"type": "json_object"}),
                 self.req(reasoning_effort="high")]
        for req in cases:
            with self.subTest(req=req):
                before = copy.deepcopy(req)
                expected = self.prepared_count(req)
                code, body = self.post(req)
                self.assertEqual(code, 200, body)
                self.assertEqual(body, {"prompt_tokens": expected, "max_context": 8192,
                                       "exact": True, "images_estimated": 0, "model": self.svc.model})
                self.assertEqual(req, before)

    def test_overflow_is_counted_while_generation_is_busy_without_state_changes(self):
        self.engine.max_context = 256
        req = self.req("long text " * 1000)
        before = copy.deepcopy(self.svc.status)
        history = list(self.svc.history)
        self.svc.fifo.acquire()
        try:
            code, body = self.post(req)
        finally:
            self.svc.fifo.release()
        self.assertEqual(code, 200, body)
        self.assertGreater(body["prompt_tokens"], body["max_context"])
        self.assertEqual(self.svc.status, before)
        self.assertEqual(list(self.svc.history), history)
        self.assertEqual(self.engine.last_embeddings if hasattr(self.engine, "last_embeddings") else None, None)

    def test_unloaded_model_uses_known_context_without_loading(self):
        self.engine.max_context = 0
        self.engine.known_ctx = 65536
        self.engine.unloaded = True
        code, body = self.post(self.req())
        self.assertEqual(code, 200, body)
        self.assertEqual(body["max_context"], 65536)

    def test_images_use_enforced_capacity_and_do_not_read_or_encode_image_data(self):
        for limit, expected in [(37, 37), (0, 4096)]:
            with self.subTest(limit=limit):
                self.svc.vision = SimpleNamespace(max_tokens=limit, cache={"keep": ("file", 7)})
                req = self.req([{"type": "text", "text": "What is this?"},
                                {"type": "image_url", "image_url": {"url": "http://invalid.example/image.png"}},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-even-decoded"}}])
                messages, tools, kw = openai_to_messages(req)
                raw = len(self.svc.encode_prompt(messages, tools, kw))
                code, body = self.post(req)
                self.assertEqual(code, 200, body)
                self.assertEqual(body["prompt_tokens"], raw + 2 * (expected - 1))
                self.assertFalse(body["exact"])
                self.assertEqual(body["images_estimated"], 2)
                self.assertEqual(self.svc.vision.cache, {"keep": ("file", 7)})

    def test_gemma_image_bound_accounts_for_added_boundary_tokens(self):
        self.svc.call_format = "gemma"
        self.svc.vision = SimpleNamespace(max_tokens=0)
        pad = self.tok.encode("<|image|>", parse_special=True)[0]
        with mock.patch.object(self.svc, "encode_prompt", return_value=[1, pad, 2]):
            code, body = self.post(self.req([{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]))
        self.assertEqual(code, 200, body)
        self.assertEqual(body["prompt_tokens"], 3 + 1120 + 2 - 1)
        self.assertEqual(body["images_estimated"], 1)

    def test_wrong_model_bad_request_missing_vision_and_sampling_are_rejected(self):
        for req in [self.req(model="another-model"), self.req(messages="invalid"), self.req(temperature="hot"),
                    self.req([{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]),
                    self.req(tools=[TOOL], response_format={"type": "json_object"})]:
            with self.subTest(req=req):
                code, body = self.post(req)
                self.assertEqual(code, 400, body)
                self.assertIn("error", body)

    def test_auth_and_model_availability_guards_apply(self):
        self.svc.api_key = "fixture-key"
        code, _ = self.post(self.req())
        self.assertEqual(code, 401)
        code, body = self.post(self.req(), {"Authorization": "Bearer fixture-key"})
        self.assertEqual(code, 200, body)
        self.svc.model_available = False
        code, body = self.post(self.req(), {"Authorization": "Bearer fixture-key"})
        self.assertEqual(code, 503, body)

    def test_model_switch_lease_and_profile_validation_apply(self):
        svc = self.svc
        class Guard:
            busy = False
            leases = 0
            @contextmanager
            def admit(self):
                if self.busy:
                    raise ModelControlError(503, "a model switch is in progress")
                self.leases += 1
                try:
                    yield
                finally:
                    self.leases -= 1
            def check_model(self, req):
                if req.get("model") not in (None, svc.model, "profile-alias"):
                    raise ModelControlError(409, "model is not active")
        guard = self.svc.models = Guard()
        code, body = self.post(self.req(model="profile-alias"))
        self.assertEqual(code, 200, body)
        self.assertEqual(body["model"], svc.model)
        self.assertEqual(guard.leases, 0)
        code, body = self.post(self.req(model="inactive"))
        self.assertEqual(code, 409, body)
        guard.busy = True
        code, body = self.post(self.req())
        self.assertEqual(code, 503, body)

    def test_mcp_tool_snapshot_does_not_change_routing_cache_or_wait(self):
        self.svc.mcp = Hub()
        plain = self.svc.chat_token_count(self.req())["prompt_tokens"]
        code, body = self.post(self.req(strata_mcp=True))
        self.assertEqual(code, 200, body)
        self.assertGreater(body["prompt_tokens"], plain)
        self.assertEqual(self.svc.mcp._routes, {"original": "unchanged"})
        self.svc.mcp.threads = [SimpleNamespace(is_alive=lambda: True)]
        code, body = self.post(self.req(strata_mcp=True))
        self.assertEqual(code, 503, body)


if __name__ == "__main__":
    unittest.main()
