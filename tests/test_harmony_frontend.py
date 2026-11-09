"""gpt-oss's Harmony format: the request's messages, the template's options, and the streamed output."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import frontend
from serve.frontend import ChatTemplate, OutputParser, _harmony_kwargs, _harmony_messages


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

    def test_content_and_thinking_do_not_meet_in_a_call(self):
        messages = _harmony_messages([{"role": "assistant", "content": "text", "thinking": "more",
                                       "tool_calls": [{"function": {"name": "f", "arguments": {}}}]}])
        self.assertNotIn("thinking", messages[0])

    def test_the_template_is_recognized_by_its_markers(self):
        source = "{{- '<|start|>system<|message|>x<|end|>' }}{{ '<|channel|>' }}{{ strftime_now('%Y') }}"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "chat_template.jinja")
            path.write_text(source, encoding="utf-8")
            self.assertEqual(ChatTemplate(path).call_format, "harmony")


if __name__ == "__main__":
    unittest.main()
