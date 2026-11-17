"""Declared boolean arguments agree in complete and streamed XML tool calls.

Adapted from Strata 1aabfb1ce (MIT; see NOTICE), with chunked streaming checks.
"""
import json
import unittest

from serve.frontend import CALL_END, CALL_START, OutputParser, coerce_tool_value, parse_tool_call


TOOL = {"name": "write_note", "parameters": {"type": "object", "properties": {
    "formatted": {"type": "boolean"}, "title": {"type": "string"},
    "pinned": {"type": ["boolean", "null"]}, "label": {"type": ["boolean", "string"]},
    "count": {"type": "integer"}, "items": {"type": "array"}}}}


def body(values):
    return "<function=write_note>\n" + "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in values.items()) + "</function>"


class BooleanArguments(unittest.TestCase):
    def test_python_case_requires_a_boolean_schema_that_does_not_allow_strings(self):
        for declared in ("boolean", ["boolean", "null"]):
            for text, expected in (("True", True), ("FALSE", False), (" true ", True), ("False", False)):
                with self.subTest(declared=declared, text=text):
                    self.assertIs(coerce_tool_value(text, declared), expected)
        for declared in ("string", ["boolean", "string"], None, "object", "integer"):
            self.assertEqual(coerce_tool_value("True", declared), "True")
        self.assertEqual(coerce_tool_value("yes", "boolean"), "yes")
        self.assertIsNone(coerce_tool_value("null", ["boolean", "null"]))

    def test_full_and_streamed_arguments_match_for_every_chunk_boundary(self):
        values = {"formatted": "True", "title": "True", "pinned": "False", "label": "False",
                  "count": "7", "items": '["日本語", false, 2]'}
        expected = {"formatted": True, "title": "True", "pinned": False, "label": "False",
                    "count": 7, "items": ["日本語", False, 2]}
        raw = body(values)
        self.assertEqual(parse_tool_call(raw, TOOL).arguments, expected)
        text = CALL_START + "\n" + raw + "\n" + CALL_END
        # Exercise every possible split through an argument, its tag, and the closing wrapper.
        for split in range(len(text) + 1):
            with self.subTest(split=split):
                parser = OutputParser(thinking=False, tools=[TOOL], stream_tools=True)
                events = parser.feed(text[:split]) + parser.feed(text[split:]) + parser.finish()
                self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [expected])
                self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")), expected)

    def test_no_schema_preserves_existing_json_decoding(self):
        values = {"formatted": "True", "pinned": "false", "count": "7", "items": "[1,2]"}
        self.assertEqual(parse_tool_call(body(values)).arguments,
                         {"formatted": "True", "pinned": False, "count": 7, "items": [1, 2]})


if __name__ == "__main__":
    unittest.main()
