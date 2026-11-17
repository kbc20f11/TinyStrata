"""ID-matched parallel tool results must render in their calls' order.

HTTP coverage uses MockEngine and captures the complete template prompt; it
does not claim to exercise a GPU model.
"""
import copy
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import ChatTemplate, _answers_in_call_order, anthropic_to_messages, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Server, Service, make_handler


ROOT = Path(__file__).resolve().parents[1]
CALLS = [
    {"id": "north", "type": "function", "function": {"name": "read_north", "arguments": {}}},
    {"id": "south", "type": "function", "function": {"name": "read_south", "arguments": {}}},
]
RESULTS = [
    {"role": "tool", "tool_call_id": "south", "content": "SOUTH_RESULT_22"},
    {"role": "tool", "tool_call_id": "north", "content": "NORTH_RESULT_11"},
]
MESSAGES = [{"role": "user", "content": "Compare the two inventory reports"},
            {"role": "assistant", "content": "", "tool_calls": CALLS}, *RESULTS]
MODELS = ("qwen3.6-35b-a3b", "tiel-coder-35b-a3b", "clover-35b-a3b-preview")


def request(api, model):
    if api == "openai":
        messages = copy.deepcopy(MESSAGES)
    else:
        messages = [{"role": "user", "content": MESSAGES[0]["content"]},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": call["id"], "name": call["function"]["name"], "input": {}}
                        for call in CALLS]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": result["tool_call_id"], "content": result["content"]}
                        for result in RESULTS]}]
    return {"model": model, "messages": messages, "max_tokens": 32,
            "thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": False}}


class ToolResultOrder(unittest.TestCase):
    def test_complete_unique_id_mapping_reorders_without_mutating_history(self):
        before = copy.deepcopy(MESSAGES)
        got = _answers_in_call_order(MESSAGES)
        self.assertEqual([m.get("tool_call_id") for m in got[-2:]], ["north", "south"])
        self.assertEqual(MESSAGES, before)

    def test_ambiguous_incomplete_or_unmatched_history_keeps_its_order(self):
        cases = []
        missing = copy.deepcopy(MESSAGES); missing[-1].pop("tool_call_id"); cases.append(missing)
        unknown = copy.deepcopy(MESSAGES); unknown[-1]["tool_call_id"] = "unknown"; cases.append(unknown)
        duplicate = copy.deepcopy(MESSAGES); duplicate[-1]["tool_call_id"] = "south"; cases.append(duplicate)
        duplicate_calls = copy.deepcopy(MESSAGES); duplicate_calls[1]["tool_calls"][0]["id"] = "south"; cases.append(duplicate_calls)
        no_call_id = copy.deepcopy(MESSAGES); no_call_id[1]["tool_calls"][0].pop("id"); cases.append(no_call_id)
        invalid_id = copy.deepcopy(MESSAGES); invalid_id[-1]["tool_call_id"] = ["north"]; cases.append(invalid_id)
        partial = copy.deepcopy(MESSAGES); partial[1]["tool_calls"].insert(0, {
            "id": "third", "function": {"name": "read_third", "arguments": {}}}); cases.append(partial)
        separated = copy.deepcopy(MESSAGES); separated.insert(2, {"role": "user", "content": "Continue"}); cases.append(separated)
        for history in cases:
            with self.subTest(history=history):
                self.assertEqual(_answers_in_call_order(history), history)

    def test_both_apis_preserve_ids_then_native_qwen_template_orders_results(self):
        template = ChatTemplate(ROOT / "serve" / "chat_template.jinja")
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                req = request(api, MODELS[0])
                before = copy.deepcopy(req)
                normalize = openai_to_messages if api == "openai" else anthropic_to_messages
                messages, tools, kw = normalize(req)
                self.assertEqual([c["id"] for c in messages[1]["tool_calls"]], ["north", "south"])
                self.assertEqual([m["tool_call_id"] for m in messages[-2:]], ["south", "north"])
                prompt = template.render(messages, tools, **kw)
                self.assertLess(prompt.index("NORTH_RESULT_11"), prompt.index("SOUTH_RESULT_22"))
                self.assertEqual(req, before)

    def test_each_completed_tool_run_is_ordered_independently(self):
        second = copy.deepcopy(MESSAGES[1:])
        for call in second[0]["tool_calls"]:
            call["id"] += "2"
        for result in second[1:]:
            result["tool_call_id"] += "2"
        got = _answers_in_call_order(MESSAGES + [{"role": "user", "content": "Again"}] + second)
        self.assertEqual([m["tool_call_id"] for m in got if m["role"] == "tool"],
                         ["north", "south", "north2", "south2"])

    def test_unrelated_custom_template_keeps_explicit_result_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.jinja"
            path.write_text("{% for m in messages %}{{ m.content }}{% endfor %}", encoding="utf-8")
            prompt = ChatTemplate(path).render(copy.deepcopy(MESSAGES))
            self.assertLess(prompt.index("SOUTH_RESULT_22"), prompt.index("NORTH_RESULT_11"))

    def test_gemma_remains_ordered_and_lfm_template_keeps_its_named_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.jinja"
            for marker, expected in (("<|turn>", ("NORTH_RESULT_11", "SOUTH_RESULT_22")),
                                     ("<|tool_call_start|>", ("SOUTH_RESULT_22", "NORTH_RESULT_11"))):
                path.write_text(marker + "{% for m in messages %}{{ m.content }}{% endfor %}", encoding="utf-8")
                prompt = ChatTemplate(path).render(copy.deepcopy(MESSAGES))
                self.assertLess(prompt.index(expected[0]), prompt.index(expected[1]))

    def test_qwen_family_http_routes_render_id_matched_parallel_history(self):
        for model in MODELS:
            tok = ByteTokenizer()
            engine = MockEngine(tok, "matched", max_context=65536)
            captured = []
            generate = engine.generate

            def capture(ids, *args, **kwargs):
                captured.append(tok.decode(ids))
                yield from generate(ids, *args, **kwargs)

            engine.generate = capture
            svc = Service(engine, tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"), model_name=model)
            httpd = Server(("127.0.0.1", 0), make_handler(svc))
            thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            try:
                for api, route in (("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages")):
                    for stream in (False, True):
                        with self.subTest(model=model, api=api, stream=stream):
                            req = request(api, model)
                            req["stream"] = stream
                            before = len(captured)
                            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
                            try:
                                conn.request("POST", route, json.dumps(req), {"Content-Type": "application/json"})
                                response = conn.getresponse()
                                body = response.read().decode("utf-8")
                                self.assertEqual(response.status, 200, body)
                                self.assertEqual(len(captured), before + 1, body)
                                prompt = captured[-1]
                                self.assertLess(prompt.index("NORTH_RESULT_11"), prompt.index("SOUTH_RESULT_22"))
                            finally:
                                conn.close()
            finally:
                httpd.shutdown(); thread.join(3); httpd.server_close()
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
