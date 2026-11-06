"""#4078: a string argument that quotes a whole tool call, or reopens its own parameter, stays exact.

    python -m unittest discover -s tests -p test_tool_parser_values.py -v
"""
# Adapted from Strata 77794e34 / b72f56e0 and its wrapped-call scan regression (MIT; see NOTICE).
import json
import unittest
from unittest.mock import patch

from serve import frontend
from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START, OutputParser,
                            parse_tool_call)

SCHEMA = {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}
QUOTED = (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\nx.txt\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")
VALUES = {
    "quoted call": f"example transcript:\n{QUOTED}\nafter block",
    "quoted call at the end": f"example transcript:\n{QUOTED}",
    "heredoc": f"python3 - <<'PY'\nprint('''{QUOTED}''')\nPY",
    "repeated parameter": f"head part\n{PARAM_END}\n{PARAM_START}content>\ntail part",
    "unclosed opener in prose": f"A call starts with `{FUNC_START}NAME>` and its parameters follow.",
    "unclosed opener in code": f'opener = "{FUNC_START}"\n',
}


def call_text(value):
    return (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\nprobe.txt\n{PARAM_END}\n"
            f"{PARAM_START}content>\n{value}\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")


class QuotedCallValues(unittest.TestCase):
    def test_long_value_waits_for_closing_tag_before_structural_walk(self):
        for stream in (False, True):
            with patch.object(frontend, "call_end", wraps=frontend.call_end) as walk:
                for _ in range(2):
                    parser = OutputParser(thinking=False, stream_tools=stream)
                    events = parser.feed("<tool_call><function=write><parameter=text>")
                    before = walk.call_count
                    for _ in range(100):
                        events.extend(parser.feed("abcdefgh"))
                    self.assertEqual(walk.call_count, before)
                    for char in "</parameter></function></tool_call>":
                        events.extend(parser.feed(char))
                    self.assertEqual(walk.call_count, before + 1)
                    # A closed call waits for possible literal closing tags in its value until the turn ends.
                    self.assertFalse(any(event.kind == "tool_call" for event in events))
                    events.extend(parser.finish())
                    self.assertEqual(walk.call_count, before + 2)
                    self.assertEqual(parser._call_end_checked, 0)
                    self.assertEqual([event.call.arguments for event in events if event.kind == "tool_call"],
                                     [{"text": "abcdefgh" * 100}])
                    if stream:
                        self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")),
                                         {"text": "abcdefgh" * 100})

    def test_values_are_exact_whole_streamed_and_chunked(self):
        for label, value in VALUES.items():
            text = call_text(value)
            expected = {"path": "probe.txt", "content": value}
            self.assertEqual(parse_tool_call(text[len(CALL_START):-len(CALL_END)], SCHEMA).arguments, expected)
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(label=label, stream=stream, size=size):
                        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events.extend(parser.feed(text[i:i + size]))
                        events.extend(parser.finish())
                        calls = [e.call for e in events if e.kind == "tool_call"]
                        self.assertEqual([c.arguments for c in calls], [expected])
                        self.assertEqual("".join(e.text for e in events if e.kind == "content").strip(), "")
                        if stream:
                            self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")),
                                             expected)

    def test_prose_between_two_calls_still_ends_the_first(self):
        first, second = call_text("one"), call_text("two")
        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=True)
        events = parser.feed(first + "\nnow the second:\n" + second) + parser.finish()
        self.assertEqual([e.call.arguments["content"] for e in events if e.kind == "tool_call"], ["one", "two"])


if __name__ == "__main__":
    unittest.main()
