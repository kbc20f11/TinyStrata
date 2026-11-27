"""Gemma's malformed list arguments must not block the API request thread.

Regression for PolyStrata 048d3e1aaa74b855a592b02059400c10dc4224b2 (MIT).
Malformed fixtures run in a bounded child process so a regression cannot hang
the test runner while holding Python's GIL.
"""
import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import GEMMA_CALL, GEMMA_CALL_END, GEMMA_QUOTE, OutputParser, parse_gemma_tool_call


TOOLS = [{"name": "save", "parameters": {"type": "object", "properties": {
    "items": {"type": "array"}, "label": {"type": "string"}}}}]


def streamed_call(body, size):
    parser = OutputParser(thinking=False, tools=TOOLS, call_format="gemma")
    text = GEMMA_CALL + body + GEMMA_CALL_END
    events = []
    for at in range(0, len(text), size):
        events.extend(parser.feed(text[at:at + size]))
    events.extend(parser.finish())
    return [(event.call.name, event.call.arguments) for event in events if event.kind == "tool_call"]


MALFORMED = [
    ("call:save{items:[one}", {"items": ["one"]}),
    ("call:save{items:[}", {"items": []}),
    ("call:save{items:[[one}", {"items": [["one"]]}),
    ("call:save{items:[{nested:[one}}", {"items": [{"nested": ["one"]}]}),
    ("call:save{items:[one,two,}", {"items": ["one", "two"]}),
    ("call:save{items:[one", {"items": ["one"]}),
]


def malformed_worker():
    results = []
    for body, expected in MALFORMED:
        assert parse_gemma_tool_call(body).arguments == expected, body
        for size in (1, 3, len(body) + len(GEMMA_CALL) + len(GEMMA_CALL_END)):
            got = streamed_call(body, size)
            assert got == [("save", expected)], (body, size, got)
        results.append(expected)
    print(json.dumps(results))


class GemmaParserProgress(unittest.TestCase):
    def test_malformed_lists_complete_under_hard_timeout(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--malformed-worker"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            encoding="utf-8", timeout=8, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [expected for _, expected in MALFORMED])

    def test_valid_nested_lists_scalars_and_literal_brackets_stay_exact(self):
        quote = GEMMA_QUOTE
        body = (f"call:save{{items:[1,true,null,{{nested:[2,false]}}],"
                f"label:{quote}literal }} ] and 日本語{quote}}}")
        expected = {"items": [1, True, None, {"nested": [2, False]}],
                    "label": "literal } ] and 日本語"}
        self.assertEqual(parse_gemma_tool_call(body).arguments, expected)
        for size in (1, 2, 7, len(body) * 2):
            with self.subTest(size=size):
                self.assertEqual(streamed_call(body, size), [("save", expected)])

    def test_orphaned_channels_and_tool_closers_stay_out_of_answer(self):
        for text, answer, reasoning in (
            ("<|channel>thought<tool_call|>", "", ""),
            ("<|channel>thought<channel|>Answer", "Answer", ""),
            ("<|channel>thoughtPlan<channel|>Answer", "Answer", "Plan"),
            ("<|channel>thought\nPlan<channel|>Answer", "Answer", "Plan"),
            ("Answer<channel|> continues<tool_call|>", "Answer continues", ""),
            ("<|channel>thought", "", ""),
        ):
            for size in (1, 3, len(text)):
                with self.subTest(text=text, size=size):
                    parser = OutputParser(thinking=False, call_format="gemma")
                    events = []
                    for at in range(0, len(text), size):
                        events.extend(parser.feed(text[at:at + size]))
                    events.extend(parser.finish())
                    self.assertEqual("".join(e.text for e in events if e.kind == "content"), answer)
                    self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), reasoning)

    def test_orphaned_opener_does_not_change_a_valid_tool_argument(self):
        text = (f"<|channel>thought{GEMMA_CALL}"
                f"call:save{{label:{GEMMA_QUOTE}literal <channel|>{GEMMA_QUOTE}}}{GEMMA_CALL_END}")
        for size in (1, 4, len(text)):
            with self.subTest(size=size):
                parser = OutputParser(thinking=False, tools=TOOLS, call_format="gemma", stream_tools=True)
                events = []
                for at in range(0, len(text), size):
                    events.extend(parser.feed(text[at:at + size]))
                events.extend(parser.finish())
                self.assertEqual([e.kind for e in events], ["tool_start", "tool_args", "tool_call"])
                self.assertEqual(events[-1].call.arguments, {"label": "literal <channel|>"})


if __name__ == "__main__":
    # Direct execution needs the repository root for the child's serve import.
    if "--malformed-worker" in sys.argv:
        malformed_worker()
    else:
        unittest.main()
