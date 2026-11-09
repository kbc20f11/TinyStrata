"""Model-native thinking controls; CPU/mock HTTP only, never loads a model."""
import copy
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import responses
from serve.frontend import ChatTemplate, anthropic_to_messages, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Server, Service, clean_shared_defaults, make_handler
from serve.thinking import capabilities, normalize_effort, normalize_request, template_kwargs

QWEN = "qwen3.6-35b-a3b"
CLOVER = "clover-35b-a3b-preview"
GEMMA = "gemma-4-26b-a4b-it"
LFM = "lfm2.5-24b-a2b"
TIEL = "tiel-coder-35b-a3b"
MODELS = (QWEN, CLOVER, GEMMA, LFM, TIEL)


class ThinkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.template_path = Path(cls.tmp.name) / "template.jinja"
        cls.template_path.write_text("BOOL={{ enable_thinking }};EFFORT={{ reasoning_effort|default('missing') }}\n"
                                     "{% for m in messages %}{{ m.content }}{% endfor %}"
                                     "{% if enable_thinking %}<think>{% endif %}", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def service(self, model):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "</think>OK", max_context=8192)
        engine.seen_prompts = []
        original = engine.generate
        def generate(ids, *args, **kwargs):
            engine.seen_prompts.append(tok.decode(ids))
            yield from original(ids, *args, **kwargs)
        engine.generate = generate
        return Service(engine, tok, ChatTemplate(self.template_path), model_name=model)

    def kwargs(self, svc, api="openai", **controls):
        raw = {"messages": [{"role": "user", "content": "hello"}], **controls}
        before = copy.deepcopy(raw)
        req = svc.with_shared(raw, api)
        self.assertEqual(raw, before)
        if api == "openai":
            _, _, kwargs = openai_to_messages(req)
        elif api == "anthropic":
            _, _, kwargs = anthropic_to_messages(req, False)
        else:
            kwargs = responses.template_kwargs(req, {"reasoning_effort": svc.effort_default()})
        return svc.template_defaults(kwargs, req)

    def test_capabilities_are_model_specific_and_not_mutable_globals(self):
        for model in MODELS:
            with self.subTest(model=model):
                cap = capabilities(model)
                self.assertEqual(cap["default"], "medium" if model == TIEL else "on")
                self.assertEqual(cap["mode"], "effort" if model == TIEL else "always" if model == LFM else "toggle")
                cap["options"].append("bad")
                self.assertNotIn("bad", capabilities(model)["options"])
        self.assertEqual(capabilities("tiel-coder"), capabilities(TIEL))
        self.assertIsNone(capabilities("custom-model", "qwen35moe"))

    def test_gpt_oss_has_effort_without_an_off_switch(self):
        policy = capabilities("gpt-oss-20b")
        self.assertEqual(policy, {"mode": "effort_always", "default": "medium", "options": ["low", "medium", "xhigh"]})
        self.assertEqual(capabilities("gpt-oss"), policy)
        self.assertEqual(capabilities("gptoss"), policy)
        # off and minimal ask for the least reasoning it has; high is the template's high (xhigh here)
        self.assertEqual([normalize_effort(v, policy) for v in ("none", False, "minimal", "low", "medium", "high", "xhigh")],
                         ["low", "low", "low", "low", "medium", "xhigh", "xhigh"])
        # the switch cannot turn it off: the template always gets an effort
        self.assertEqual(template_kwargs({"enable_thinking": False}, policy, True),
                         {"enable_thinking": True, "reasoning_effort": "medium"})
        self.assertEqual(template_kwargs({"reasoning_effort": "xhigh"}, policy, True),
                         {"enable_thinking": True, "reasoning_effort": "xhigh"})
        self.assertEqual(template_kwargs({}, policy, False), {"enable_thinking": True, "reasoning_effort": "medium"})

    def test_defaults_all_protocols(self):
        for model in MODELS:
            for api in ("openai", "responses", "anthropic"):
                with self.subTest(model=model, api=api):
                    kw = self.kwargs(self.service(model), api)
                    self.assertIs(kw["enable_thinking"], True)
                    self.assertEqual(kw.get("reasoning_effort"), "medium" if model == TIEL else None)

    def test_tiel_aliases_in_every_effort_location_before_generic_parser(self):
        fields = (lambda v: {"reasoning_effort": v}, lambda v: {"reasoning": {"effort": v}},
                  lambda v: {"output_config": {"effort": v}},
                  lambda v: {"chat_template_kwargs": {"reasoning_effort": v}})
        for make in fields:
            for value, expected in (("minimal", "low"), ("low", "low"), ("medium", "medium"),
                                    ("high", "xhigh"), ("xhigh", "xhigh"), ("max", "xhigh"), ("maximum", "xhigh")):
                raw = make(value)
                normalized = normalize_request(raw, capabilities(TIEL))
                self.assertIn(expected, str(normalized))
                self.assertEqual(raw, make(value))
        for api, field in (("openai", "reasoning_effort"), ("responses", "reasoning"),
                           ("anthropic", "output_config")):
            for value, expected in (("minimal", "low"), ("high", "xhigh"), ("max", "xhigh")):
                control = {field: value if field == "reasoning_effort" else {"effort": value}}
                with self.subTest(api=api, value=value):
                    self.assertEqual(self.kwargs(self.service(TIEL), api, **control)["reasoning_effort"], expected)
            self.assertEqual(self.kwargs(self.service(TIEL), api,
                                         chat_template_kwargs={"reasoning_effort": "minimal"})["reasoning_effort"], "low")

    def test_binary_legacy_levels_become_only_boolean(self):
        for model in (QWEN, CLOVER, GEMMA):
            svc = self.service(model)
            for value in ("on", "low", "medium", "high", "xhigh", "max"):
                kw = self.kwargs(svc, reasoning_effort=value)
                self.assertIs(kw["enable_thinking"], True)
                self.assertNotIn("reasoning_effort", kw)
            for value in ("none", "off", "minimal", False):
                self.assertIs(self.kwargs(svc, reasoning_effort=value)["enable_thinking"], False)

    def test_explicit_boolean_precedes_shared_and_effort_all_protocols(self):
        for model in (QWEN, CLOVER, GEMMA, TIEL):
            svc = self.service(model)
            for api in ("openai", "responses", "anthropic"):
                with self.subTest(model=model, api=api):
                    svc.shared = {"reasoning_effort": "xhigh" if model == TIEL else "high"}
                    kw = self.kwargs(svc, api, enable_thinking=False)
                    self.assertIs(kw["enable_thinking"], False)
                    self.assertNotIn("reasoning_effort", kw)
                    svc.shared = {"enable_thinking": False}
                    kw = self.kwargs(svc, api, enable_thinking=True)
                    self.assertIs(kw["enable_thinking"], True)
                    self.assertEqual(kw.get("reasoning_effort"), "medium" if model == TIEL else None)
                    kw = self.kwargs(svc, api, enable_thinking=False,
                                     chat_template_kwargs={"enable_thinking": True, "reasoning_effort": "none"})
                    self.assertIs(kw["enable_thinking"], True)
                    self.assertEqual(kw.get("reasoning_effort"), "medium" if model == TIEL else None)

    def test_lfm_explicit_off_rejected_saved_legacy_off_keeps_other_settings(self):
        svc = self.service(LFM)
        for api in ("openai", "responses", "anthropic"):
            for controls in ({"enable_thinking": False}, {"chat_template_kwargs": {"enable_thinking": False}},
                             {"thinking": {"type": "disabled"}}):
                with self.subTest(api=api, controls=controls), self.assertRaisesRegex(ValueError, "always uses thinking"):
                    self.kwargs(svc, api, **controls)
        svc.shared = clean_shared_defaults({"reasoning_effort": "none", "temperature": .7}, LFM)
        self.assertEqual(svc.shared, {"enable_thinking": True, "temperature": .7})
        self.assertTrue(self.kwargs(svc)["enable_thinking"])
        for api in ("openai", "responses", "anthropic"):
            self.assertTrue(self.kwargs(svc, api, reasoning_effort="none",
                                        chat_template_kwargs={"enable_thinking": True})["enable_thinking"])

    def test_shared_settings_validation_and_model_safe_aliases(self):
        self.assertEqual(clean_shared_defaults({"enable_thinking": False}, QWEN), {"enable_thinking": False})
        self.assertEqual(clean_shared_defaults({"reasoning_effort": "high"}, GEMMA), {"enable_thinking": True})
        self.assertEqual(clean_shared_defaults({"reasoning_effort": "minimal"}, TIEL), {"reasoning_effort": "low"})
        self.assertEqual(clean_shared_defaults({"reasoning_effort": "maximum"}, TIEL), {"reasoning_effort": "xhigh"})
        for value in (0, 1, "true", "off"):
            with self.assertRaisesRegex(ValueError, "enable_thinking"):
                clean_shared_defaults({"enable_thinking": value}, QWEN)
        raw = {"reasoning_effort": "minimal", "reasoning_budget_tokens": 257}
        before = copy.deepcopy(raw)
        clean_shared_defaults(raw, TIEL)
        self.assertEqual(raw, before)
        self.assertEqual(normalize_request(raw, capabilities(TIEL))["reasoning_budget_tokens"], 257)

    def test_anthropic_explicit_on_request_config_preserved_but_request_shared_and_config_win(self):
        for model in MODELS:
            with self.subTest(model=model):
                svc = self.service(model)
                svc.anthropic_think_unasked = False
                self.assertEqual(self.kwargs(svc, "anthropic")["enable_thinking"], model == LFM)
                self.assertTrue(self.kwargs(svc, "anthropic", thinking={"type": "enabled"})["enable_thinking"])
                self.assertTrue(self.kwargs(svc, "anthropic", reasoning_budget_tokens=257)["enable_thinking"])
                svc.shared = {"enable_thinking": True}
                self.assertTrue(self.kwargs(svc, "anthropic")["enable_thinking"])
                svc.shared = {}
                svc.config_effort = "medium" if model == TIEL else "high"
                self.assertTrue(self.kwargs(svc, "anthropic")["enable_thinking"])

    def test_unknown_models_keep_generic_effort_behavior(self):
        svc = self.service("custom-model")
        self.assertEqual(self.kwargs(svc, reasoning_effort="low")["reasoning_effort"], "low")
        self.assertFalse(self.kwargs(svc, reasoning_effort="minimal")["enable_thinking"])
        for off in ("none", "off", "minimal", "disabled", "false", False):
            with self.subTest(off=off):
                self.assertFalse(self.kwargs(svc, reasoning_effort=off,
                                             chat_template_kwargs={"reasoning_effort": "low"})["enable_thinking"])

    def test_actual_http_defaults_aliases_count_and_settings(self):
        for model in MODELS:
            with self.subTest(model=model):
                svc = self.service(model)
                server = Server(("127.0.0.1", 0), make_handler(svc))
                thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
                thread.start()
                def post(path, body):
                    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=4)
                    try:
                        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
                        response = conn.getresponse()
                        return response.status, json.loads(response.read())
                    finally:
                        conn.close()
                try:
                    base = {"model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32}
                    for path, body in (("/v1/chat/completions", base), ("/v1/messages", base),
                                       ("/v1/responses", {"model": model, "input": "hello", "max_output_tokens": 32})):
                        status, result = post(path, body)
                        self.assertEqual(status, 200, result)
                        self.assertIn("BOOL=True;EFFORT=" + ("medium" if model == TIEL else "missing"),
                                      svc.engine.seen_prompts[-1])
                    if model == TIEL:
                        for path, body in (("/v1/chat/completions", {**base, "reasoning_effort": "minimal"}),
                                           ("/v1/messages", {**base, "output_config": {"effort": "minimal"}}),
                                           ("/v1/responses", {"model": model, "input": "hello", "max_output_tokens": 32,
                                                              "reasoning": {"effort": "minimal"}})):
                            status, result = post(path, body)
                            self.assertEqual(status, 200, result)
                            self.assertIn("BOOL=True;EFFORT=low", svc.engine.seen_prompts[-1])
                    count = {**base, "chat_template_kwargs": {"enable_thinking": True}}
                    if model == TIEL:
                        count["reasoning_effort"] = "minimal"
                    status, result = post("/v1/chat/token-count", count)
                    self.assertEqual(status, 200, result)
                    self.assertGreater(result["prompt_tokens"], 0)
                    svc.anthropic_think_unasked = False
                    status, result = post("/v1/messages", base)
                    self.assertEqual(status, 200, result)
                    self.assertIn("BOOL=" + ("True" if model == LFM else "False"), svc.engine.seen_prompts[-1])
                    status, result = post("/v1/messages", {**base, "thinking": {"type": "enabled"}})
                    self.assertEqual(status, 200, result)
                    self.assertIn("BOOL=True;EFFORT=" + ("medium" if model == TIEL else "missing"),
                                  svc.engine.seen_prompts[-1])
                    if model == LFM:
                        status, result = post("/v1/chat/token-count", {**base, "enable_thinking": False})
                        self.assertEqual(status, 400, result)
                    else:
                        status, result = post("/settings", {"defaults": {"enable_thinking": False}})
                        self.assertEqual(status, 200, result)
                        self.assertFalse(svc.shared["enable_thinking"])
                        self.assertFalse(self.kwargs(svc)["enable_thinking"])
                finally:
                    server.shutdown(); thread.join(3); server.server_close()


if __name__ == "__main__":
    unittest.main()
