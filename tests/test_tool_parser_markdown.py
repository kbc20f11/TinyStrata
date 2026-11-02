"""#4702 and #4703: a call quoted in a Markdown code form is text, not a call.  A four-backtick fence is closed only by a fence of
at least four backticks (an inner ``` block stays inside it), and an indented (4-space) code block after a blank line is a quote.
A call the model writes plainly, or indents inside a paragraph, is still a call.

    python -m unittest discover -s tests -p test_tool_parser_markdown.py
"""
# Adapted from Strata 209dcadb (MIT; see NOTICE).
import unittest

from serve.frontend import OutputParser


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


S = {"type": "string"}
TOOLS = [tool("read", path=S), tool("write_file", path=S, content=S)]


def qwen(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def parse(text, piece=0, recover=False):
    p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False, recover=recover)
    if piece:
        evs = []
        for i in range(0, len(text), piece):
            evs += p.feed(text[i:i + piece])
        evs += p.finish("stop")
    else:
        evs = p.feed(text) + p.finish("stop")
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text for e in evs if e.kind == "content"))


class MarkdownCode(unittest.TestCase):
    def each(self, text):
        for recover in (False, True):
            for piece in (0, 1, 7):
                yield parse(text, piece, recover)

    def test_an_inner_fence_does_not_close_a_longer_one(self):
        text = ("Save this as docs/TOOLS.md:\n\n````markdown\n# Tool calls\n\nA call:\n\n```xml\n"
                + qwen("read", path="EXAMPLE.txt") + "\n```\n````\n\nThat is the file.")
        for calls, answer in self.each(text):
            self.assertEqual(calls, [])
            self.assertEqual(answer, text)

    def test_a_call_after_a_closed_longer_fence_is_a_call(self):
        for marker in ("`", "~"):
            for width in (4, 6):
                text = ("Here:\n\n" + marker * width + "md\nexample\n" + marker * width
                        + " \t\n\n" + qwen("read", path="a.txt"))
                for calls, _ in self.each(text):
                    self.assertEqual(calls, [("read", {"path": "a.txt"})])

    def test_a_tilde_fence_is_not_closed_by_backticks(self):
        text = "Quote:\n\n~~~\n```\n" + qwen("read", path="EXAMPLE.txt") + "\n```\n~~~\n"
        for calls, _ in self.each(text):
            self.assertEqual(calls, [])

    def test_six_character_fences_keep_shorter_fences_inside(self):
        for marker in ("`", "~"):
            for inner in ("", marker * 3 + "xml\n"):
                text = ("Example:\n\n" + marker * 6 + "markdown\n" + inner
                        + qwen("read", path="EXAMPLE.txt") + "\n" + marker * 3 + "\n" + marker * 6 + "\nEnd.")
                for calls, answer in self.each(text):
                    self.assertEqual(calls, [])
                    self.assertEqual(answer, text)

    def test_closing_fence_with_trailing_text_does_not_close(self):
        for marker in ("`", "~"):
            text = ("Example:\n\n" + marker * 4 + "\n" + marker * 4 + " still quoted\n"
                    + qwen("read", path="EXAMPLE.txt") + "\n" + marker * 4 + "\n")
            for calls, answer in self.each(text):
                self.assertEqual(calls, [])
                self.assertEqual(answer, text)

    def test_single_line_fence_examples_do_not_open_a_block(self):
        for marker in ("`", "~"):
            for width in (3, 6):
                text = marker * width + "quoted" + marker * width + "\n" + qwen("read", path="a.txt")
                for calls, _ in self.each(text):
                    self.assertEqual(calls, [("read", {"path": "a.txt"})])

    def test_an_indented_code_block_is_a_quote(self):
        block = "\n".join("    " + line for line in qwen("read", path="EXAMPLE.txt").split("\n"))
        text = "The format, indented as code:\n\n" + block + "\n\nIndentation makes it a code block in Markdown."
        for calls, answer in self.each(text):
            self.assertEqual(calls, [])
            self.assertEqual(answer, text)

    def test_a_plain_call_after_a_paragraph_is_still_a_call(self):
        text = "Let me read it.\n" + qwen("read", path="a.txt")
        for calls, _ in self.each(text):
            self.assertEqual(calls, [("read", {"path": "a.txt"})])

    def test_an_indented_call_inside_a_paragraph_is_still_a_call(self):
        text = "Let me read it.\n    " + qwen("read", path="a.txt")
        for calls, _ in self.each(text):
            self.assertEqual(calls, [("read", {"path": "a.txt"})])

    def test_quoted_calls_in_reasoning_stay_reasoning(self):
        call = qwen("read", path="EXAMPLE.txt")
        examples = ["Example:\n\n````md\n```xml\n" + call + "\n```\n````\n",
                    "Example:\n\n" + "\n".join("    " + line for line in call.split("\n")) + "\n"]
        for text in examples:
            for piece in (1, 7, len(text)):
                parser = OutputParser(thinking=True, tools=TOOLS, recover=True, stream_tools=True)
                wire = text + "</think>\nDone."
                events = []
                for i in range(0, len(wire), piece):
                    events.extend(parser.feed(wire[i:i + piece]))
                events.extend(parser.finish("stop"))
                self.assertFalse([e for e in events if e.kind in ("tool_call", "tool_start", "tool_args")])
                self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), text)
                self.assertEqual("".join(e.text for e in events if e.kind == "content"), "Done.")

    def test_a_call_after_an_indented_block_ends_it(self):
        text = "Example:\n\n    x = 1\n\nNow the real one:\n\n" + qwen("read", path="a.txt")
        for calls, _ in self.each(text):
            self.assertEqual(calls, [("read", {"path": "a.txt"})])


if __name__ == "__main__":
    unittest.main()
