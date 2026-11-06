"""Literal XML closing tags must survive tool arguments, including every streamed split.

Adapted from Strata 49b880c / 3bbf0f3 (MIT; see NOTICE).
Run: python -m unittest discover -s tests -p test_tool_closing_tags_stream.py -v
"""
import json
import unittest

from serve.frontend import (CALL_END, CLOSE_LOOKAHEAD_CHARS, FUNC_END, OutputParser,
                            _close_verdict, param_end)


TOOLS = [
    {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}},
    {"name": "typed", "parameters": {"type": "object", "properties": {
        "n": {"type": "integer"}, "flag": {"type": "boolean"}, "items": {"type": "array"}}}},
]
CLOSE_DOC = ("# Tool calls\n\nThe closing tags, in order:\n\n</parameter>\n</function>\n"
             "</tool_call>\n\nNothing may follow the last one.")


def qwen(name, **params):
    body = "".join(f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def partitions(text):
    yield "whole", [text]
    for size in (1, 3, 7):
        yield f"chunk{size}", [text[i:i + size] for i in range(0, len(text), size)]
    for cut in range(len(text) + 1):
        yield f"cut{cut}", [text[:cut], text[cut:]]


class ClosingTagsStream(unittest.TestCase):
    def assert_calls(self, events, expected, stream):
        calls = [event.call for event in events if event.kind == "tool_call"]
        self.assertEqual([(call.name, call.arguments) for call in calls], expected)
        if stream:
            fragments = {}
            starts = []
            for event in events:
                if event.kind == "tool_start":
                    starts.append(event.call.id)
                elif event.kind == "tool_args":
                    self.assertIn(event.call.id, starts)
                    fragments[event.call.id] = fragments.get(event.call.id, "") + event.text
            self.assertEqual(starts, [call.id for call in calls])
            self.assertEqual([json.loads(fragments[call.id]) for call in calls],
                             [arguments for _, arguments in expected])

    def test_upstream_four_fixtures_all_1662_stream_partitions(self):
        fixtures = [
            ("I will save the doc.\n\n" + qwen("write_file", path="TOOLS.md", content=CLOSE_DOC),
             False, [("write_file", {"path": "TOOLS.md", "content": CLOSE_DOC})], "I will save the doc."),
            (qwen("write_file", path="TOOLS.md", content=CLOSE_DOC), True,
             [("write_file", {"path": "TOOLS.md", "content": CLOSE_DOC})], ""),
            (qwen("read", path="a.txt") + "\n\nThat is the read.", False,
             [("read", {"path": "a.txt"})], "That is the read."),
            (qwen("read", path="a.txt") + "\n\nAnd the next:\n" + qwen("read", path="b.txt"), False,
             [("read", {"path": "a.txt"}), ("read", {"path": "b.txt"})], "And the next:"),
        ]
        cases = 0
        for index, (text, recover, expected, content) in enumerate(fixtures):
            for stream in (False, True):
                for label, pieces in partitions(text):
                    with self.subTest(fixture=index, stream=stream, partition=label):
                        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=recover)
                        events = []
                        for piece in pieces:
                            events.extend(parser.feed(piece))
                        events.extend(parser.finish())
                        self.assert_calls(events, expected, stream)
                        self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), content)
                        cases += 1
        self.assertEqual(cases, 1662)

    def test_split_closer_held_and_bounded_prose(self):
        for cut in range(len(CALL_END)):
            self.assertEqual(_close_verdict(FUNC_END + "\n" + CALL_END[:cut], False), "hold")
        self.assertEqual(_close_verdict(FUNC_END + CALL_END, True), "end")
        self.assertEqual(_close_verdict(FUNC_END + CALL_END + "x" * (CLOSE_LOOKAHEAD_CHARS + 1), False), "end")
        parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
        events = parser.feed(qwen("read", path="a.txt"))
        self.assertFalse(any(e.kind == "tool_call" for e in events))
        events.extend(parser.feed("x" * (CLOSE_LOOKAHEAD_CHARS + 1)))
        self.assert_calls(events, [("read", {"path": "a.txt"})], True)

    def test_reasoning_keeps_nonholding_close_rule(self):
        value = "a.txt\n</parameter>\n</function>\n</tool_call>"
        self.assertEqual(param_end(value), -2)
        self.assertEqual(param_end(value, hold=False), len("a.txt\n"))
        for reason in (None, "length"):
            parser = OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
            events = parser.feed(qwen("read", path="a.txt"))
            self.assertEqual(len(parser.pending), 1)
            events.extend(parser.finish(reason))
            calls = [e.call.arguments for e in events if e.kind == "tool_call"]
            self.assertEqual(calls, [{"path": "a.txt"}] if reason is None else [])

    def test_incomplete_value_is_never_completed_by_finish(self):
        text = "<tool_call>\n<function=write_file>\n<parameter=content>\nnot finished </par"
        for stream in (False, True):
            parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream)
            events = []
            for char in text:
                events.extend(parser.feed(char))
            events.extend(parser.finish("length"))
            self.assertFalse(any(e.kind == "tool_call" for e in events))
            if stream:
                args = "".join(e.text for e in events if e.kind == "tool_args")
                with self.assertRaises(ValueError):
                    json.loads(args)
            else:
                self.assertEqual("".join(e.text for e in events if e.kind == "content"), text)

    def test_typed_arguments_stay_typed(self):
        text = qwen("typed", n="17", flag="true", items='[1, "two", false]')
        expected = [("typed", {"n": 17, "flag": True, "items": [1, "two", False]})]
        for stream in (False, True):
            parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream)
            events = []
            for char in text:
                events.extend(parser.feed(char))
            self.assert_calls(events + parser.finish(), expected, stream)

    def test_bare_function_recovery_retains_arguments(self):
        text = "<function=write_file><parameter=content>abc</parameter></function>"
        for stream in (False, True):
            for size in (1, len(text)):
                parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, recover=True)
                events = []
                for offset in range(0, len(text), size):
                    events.extend(parser.feed(text[offset:offset + size]))
                self.assert_calls(events + parser.finish(), [("write_file", {"content": "abc"})], stream)

    def test_gemma_and_lfm_formats_are_unchanged(self):
        texts = {
            "lfm2": "<|tool_call_start|>[read(path='</parameter></function></tool_call>')]<|tool_call_end|>",
            "gemma": '<|tool_call>call:read{path:<|"|></parameter></function></tool_call><|"|>}<tool_call|>',
        }
        expected = [("read", {"path": "</parameter></function></tool_call>"})]
        for call_format, text in texts.items():
            for stream in (False, True):
                parser = OutputParser(thinking=False, tools=TOOLS, stream_tools=stream, call_format=call_format)
                events = []
                for char in text:
                    events.extend(parser.feed(char))
                self.assert_calls(events + parser.finish(), expected, stream)


if __name__ == "__main__":
    unittest.main()
