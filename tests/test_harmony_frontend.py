"""gpt-oss's Harmony format: the request's messages, the template's options, and the streamed output."""
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import frontend
from serve.frontend import (HM_ANALYSIS, ChatTemplate, OutputParser, _harmony_kwargs, _harmony_messages,
                            forced_call, forced_call_prefill)
from serve.server import ByteTokenizer, MockEngine, Service


def parse(chunks, tools=None, stream_tools=True):
    parser = OutputParser(thinking=True, tools=tools, stream_tools=stream_tools, call_format="harmony")
    events = []
    for chunk in chunks:
        events += parser.feed(chunk)
    events += parser.finish()
    return events


def kinds(events, kind):
    return "".join(e.text for e in events if e.kind == kind)


ANSWER = ("<|channel|>analysis<|message|>The user says hi. Reply.<|end|>"
          "<|start|>assistant<|channel|>final<|message|>Hello there!<|return|>")
CALL = ("<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant<|channel|>commentary "
        "to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Tokyo\"}<|call|>")
TOOLS = [{"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]


class HarmonyOutput(unittest.TestCase):
    def test_reasoning_and_answer_are_separate(self):
        events = parse([ANSWER])
        self.assertEqual(kinds(events, "reasoning"), "The user says hi. Reply.")
        self.assertEqual(kinds(events, "content"), "Hello there!")

    def test_any_split_of_the_text_gives_the_same_result(self):
        for size in (1, 2, 3, 7, 11):
            events = parse([ANSWER[i:i + size] for i in range(0, len(ANSWER), size)])
            self.assertEqual(kinds(events, "reasoning"), "The user says hi. Reply.", size)
            self.assertEqual(kinds(events, "content"), "Hello there!", size)
            self.assertNotIn("<|", kinds(events, "content") + kinds(events, "reasoning"), size)

    def test_a_tool_call_is_one_event_with_its_arguments(self):
        for size in (1, 5, len(CALL)):
            events = parse([CALL[i:i + size] for i in range(0, len(CALL), size)], tools=TOOLS)
            calls = [e.call for e in events if e.kind == "tool_call"]
            self.assertEqual([(c.name, c.arguments) for c in calls], [("get_weather", {"city": "Tokyo"})], size)
            self.assertEqual(kinds(events, "reasoning"), "Need the weather.")
            self.assertEqual(kinds(events, "content"), "")

    def test_a_streamed_call_announces_its_name_and_keeps_its_id(self):
        events = parse([CALL], tools=TOOLS)
        start = [e for e in events if e.kind == "tool_start"]
        done = [e for e in events if e.kind == "tool_call"]
        self.assertEqual(len(start), 1)
        self.assertEqual(start[0].call.name, "get_weather")
        self.assertEqual(start[0].call.id, done[0].call.id)

    def test_a_stop_token_that_is_not_fed_still_finishes_the_call(self):
        events = parse([CALL[:-len("<|call|>")]], tools=TOOLS)
        self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [{"city": "Tokyo"}])

    def test_arguments_that_are_not_json_stay_text(self):
        events = parse([CALL.replace('{"city":"Tokyo"}', "not json")], tools=TOOLS)
        self.assertEqual([e for e in events if e.kind == "tool_call"], [])
        self.assertIn("not json", kinds(events, "content"))

    def test_a_commentary_message_without_a_recipient_is_shown(self):
        events = parse(["<|channel|>commentary<|message|>Checking now.<|end|>"
                        "<|start|>assistant<|channel|>final<|message|>Done.<|return|>"])
        self.assertEqual(kinds(events, "content"), "Checking now.Done.")

    def test_text_after_the_last_message_is_not_held_back(self):
        events = parse(["<|channel|>final<|message|>Partial answer"])
        self.assertEqual(kinds(events, "content"), "Partial answer")

    def test_recipient_with_a_format_separator_keeps_its_name_and_json_metadata(self):
        text = ('<|channel|>commentary to=functions.<|constrain|>get_weather '
                '<|constrain|>json<|message|>{"city":"Tokyo"}<|call|>')
        for size in (1, 7, len(text)):
            events = parse([text[i:i + size] for i in range(0, len(text), size)], tools=TOOLS)
            self.assertEqual([(e.call.name, e.call.arguments) for e in events if e.kind == "tool_call"],
                             [("get_weather", {"city": "Tokyo"})])

    def test_blank_and_unoffered_recipients_never_become_tool_calls(self):
        for recipient in ("functions.", "functions.unknown", "functions.<|constrain|>unknown", "unknown"):
            text = f'<|channel|>commentary to={recipient}<|message|>{{"city":"Tokyo"}}<|call|>'
            events = parse([text], tools=TOOLS)
            self.assertFalse([e for e in events if e.kind in ("tool_start", "tool_call")])
            self.assertEqual(kinds(events, "content"), '{"city":"Tokyo"}')


class HarmonyRequest(unittest.TestCase):
    def test_reasoning_effort_uses_the_templates_three_levels(self):
        self.assertEqual(_harmony_kwargs({"reasoning_effort": "xhigh"})["reasoning_effort"], "high")
        self.assertEqual(_harmony_kwargs({"reasoning_effort": "none"})["reasoning_effort"], "low")
        self.assertEqual(_harmony_kwargs({"reasoning_effort": "medium"})["reasoning_effort"], "medium")

    def test_no_thinking_asks_for_the_least_reasoning(self):
        kwargs = _harmony_kwargs({"enable_thinking": False})
        self.assertEqual(kwargs, {"reasoning_effort": "low"})
        self.assertEqual(_harmony_kwargs({"enable_thinking": True}), {})

    def test_a_tool_calls_arguments_become_an_object_and_reasoning_becomes_thinking(self):
        messages = _harmony_messages([
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": None, "reasoning_content": "hmm",
             "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
        ])
        self.assertEqual(messages[0]["content"], "hi")
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertEqual(messages[1]["thinking"], "hmm")
        self.assertNotIn("reasoning_content", messages[1])

    def test_call_history_keeps_analysis_and_preamble_in_one_template_field(self):
        messages = _harmony_messages([{"role": "assistant", "content": "text", "thinking": "more",
                                       "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}])
        self.assertEqual(messages[0]["thinking"], "more\n\ntext")
        self.assertEqual(messages[0]["content"], "")

    def test_parallel_call_history_is_rejected_before_the_template_drops_calls(self):
        with self.assertRaisesRegex(ValueError, "one tool call"):
            _harmony_messages([{"role": "assistant", "tool_calls": [
                {"function": {"name": "first", "arguments": {}}},
                {"function": {"name": "second", "arguments": {}}}]}])

    def test_invalid_history_arguments_do_not_become_an_empty_object(self):
        for arguments in ("invalid JSON", "[]", "null", []):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "JSON object"):
                _harmony_messages([{"role": "assistant", "tool_calls": [
                    {"function": {"name": "tool", "arguments": arguments}}]}])

    def test_the_template_is_recognized_by_its_markers(self):
        source = "{{- '<|start|>system<|message|>x<|end|>' }}{{ '<|channel|>' }}{{ strftime_now('%Y') }}"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "chat_template.jinja")
            path.write_text(source, encoding="utf-8")
            self.assertEqual(ChatTemplate(path).call_format, "harmony")


class HarmonyTokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + ["<|start|>", "<|channel|>", "<|message|>", "<|end|>",
                                        "<|call|>", "<|return|>"]


class HarmonyForcedTools(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "harmony.jinja"
        # The pinned template ends the generation prompt at `assistant`, with
        # no channel yet. Keep that exact contract; no model asset is needed.
        self.path.write_text("{{ '<|start|>system<|message|>tools<|end|>' }}"
                             "{% for m in messages %}{{ '<|start|>' + m.role + '<|message|>' + m.content + '<|end|>' }}{% endfor %}"
                             "{% if add_generation_prompt %}{{ '<|start|>assistant' }}{% endif %}"
                             "{# <|channel|> #}", encoding="utf-8")

    def service(self, scripts, model="gpt-oss-20b"):
        tok = HarmonyTokenizer()
        engine = MockEngine(tok, scripts, max_context=8192)
        engine.prompts = []
        original = engine.generate
        def generate(ids, *args, **kwargs):
            engine.prompts.append(tok.decode(ids))
            yield from original(ids, *args, **kwargs)
        engine.generate = generate
        return Service(engine, tok, ChatTemplate(self.path), model_name=model,
                       generation_defaults={"arch": "gpt-oss"})

    def run_forced(self, choice, *, thinking=True, tools=TOOLS, model="gpt-oss-20b"):
        force = forced_call(choice, tools, call_format="harmony")
        continuation = ('{"city":"Tokyo"}<|call|>' if "<|message|>" in force else
                        'functions.get_weather<|channel|>commentary json<|message|>{"city":"Tokyo"}<|call|>')
        # GPT-OSS always reasons: an explicit OFF maps to its lowest effort.
        effective_thinking = thinking or model == "gpt-oss-20b"
        scripts = ["Need current weather.<|end|>", continuation] if effective_thinking else continuation
        svc = self.service(scripts, model=model)
        messages = [{"role": "user", "content": "Weather in Tokyo?"}]
        ids, actual_thinking, limit = svc.prepare(messages, tools, {"enable_thinking": thinking},
                                                   max_new=512, force=force)
        self.assertEqual(actual_thinking, effective_thinking)
        items = list(svc.run(ids, actual_thinking, tools, limit, {}, threading.Event(), force=force))
        events = [value for kind, value in items if kind == "event"]
        calls = [e.call for e in events if e.kind == "tool_call"]
        starts = [e.call for e in events if e.kind == "tool_start"]
        self.assertEqual([(call.name, call.arguments) for call in calls], [("get_weather", {"city": "Tokyo"})])
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].id, calls[0].id)
        self.assertEqual(kinds(events, "content"), "")
        self.assertEqual(kinds(events, "reasoning"), "Need current weather." if effective_thinking else "")
        if effective_thinking:
            self.assertEqual(len(svc.engine.prompts), 2)
            self.assertTrue(svc.engine.prompts[0].endswith("<|start|>assistant" + HM_ANALYSIS))
            self.assertTrue(svc.engine.prompts[1].endswith(
                "Need current weather.<|end|><|start|>assistant" + force))
        else:
            self.assertEqual(len(svc.engine.prompts), 1)
            self.assertTrue(svc.engine.prompts[0].endswith("<|start|>assistant" + force))
        self.assertNotIn("<tool_call>", "".join(svc.engine.prompts))

    def test_named_choice_preserves_analysis_and_forces_a_harmony_call(self):
        self.run_forced({"type": "function", "function": {"name": "get_weather"}})

    def test_required_choice_with_multiple_tools_leaves_only_the_recipient_name_open(self):
        self.run_forced("required", tools=TOOLS + [{"name": "other", "parameters": {"type": "object"}}])

    def test_immediate_named_and_single_tool_required_prefixes_are_harmony(self):
        for choice in ("required", {"type": "tool", "name": "get_weather"}):
            with self.subTest(choice=choice):
                self.run_forced(choice, thinking=False, model="custom-harmony")

    def test_gpt_oss_low_effort_still_opens_analysis_before_the_forced_call(self):
        self.run_forced("required", thinking=False)

    def test_counting_and_preparation_use_the_same_forced_analysis_or_call_prefix(self):
        svc = self.service("unused")
        messages = [{"role": "user", "content": "Weather in Tokyo?"}]
        choice = {"type": "function", "function": {"name": "get_weather"}}
        force = forced_call(choice, TOOLS, call_format="harmony")
        for thinking in (True, False):
            with self.subTest(thinking=thinking):
                kwargs = {"enable_thinking": thinking}
                ids, _, _ = svc.prepare(messages, TOOLS, kwargs, max_new=512, force=force)
                counted = svc.chat_token_count({"model": svc.model, "messages": messages,
                    "tools": [{"type": "function", "function": t} for t in TOOLS],
                    "tool_choice": choice, "chat_template_kwargs": kwargs})
                self.assertEqual(counted["prompt_tokens"], len(ids))
        self.assertEqual(forced_call_prefill(None, True, "harmony"), "")
        self.assertEqual(forced_call_prefill(force, True, "function"), "")

    def test_text_only_image_error_does_not_suggest_installing_a_vision_encoder(self):
        svc = self.service("unused")
        with self.assertRaisesRegex(ValueError, "gpt-oss-20b is text-only"):
            svc.chat_token_count({"model": svc.model, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}]}]})


if __name__ == "__main__":
    unittest.main()
