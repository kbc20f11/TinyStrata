"""Responses wire regression for tool arguments buffered by the assembler."""
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from serve import server
from serve.frontend import ChatTemplate, Event, ToolCall


class ResponseKeepalive(unittest.TestCase):
    def run_tool(self, kind, stream=True):
        name = "update_document"
        body = "First line.\nSecond line."
        arguments = {"input": body}
        partial = ToolCall(name, {}, "call_test")
        finished = ToolCall(name, arguments, partial.id)
        tokenizer = server.ByteTokenizer()
        template = Path(__file__).resolve().parents[1] / "serve/chat_template.jinja"
        service = server.Service(server.MockEngine(tokenizer, ""), tokenizer, ChatTemplate(template), model_name="test")
        clock = 0.0

        def run(*_args):
            nonlocal clock
            yield "event", Event("tool_start", call=partial)
            clock = 11.0
            yield "event", Event("tool_args", '{"input":', partial)
            clock = 22.0
            yield "event", Event("tool_args", json.dumps(body) + "}", partial)
            clock = 23.0
            yield "event", Event("tool_call", call=finished)
            yield "done", {"finish": "stop", "completion_tokens": 8}

        service.run = run
        handler = object.__new__(server.make_handler(service))
        handler.headers = {}
        handler.wfile = BytesIO()
        handler._watch_client = lambda _cancel: None
        handler._sse = lambda: None
        handler._json = lambda status, response: setattr(handler, "json_response", (status, response))
        tool = {"type": kind, "name": name, "description": "Update the document."}
        if kind == "function":
            tool["parameters"] = {"type": "object", "properties": {"input": {"type": "string"}}}
        request = {"model": "test", "input": "Update the document.", "tools": [tool], "stream": stream}
        with patch.object(server.time, "monotonic", side_effect=lambda: clock):
            handler._responses(request)
        if not stream:
            status, response = handler.json_response
            self.assertEqual(status, 200)
            return body, response
        wire = handler.wfile.getvalue().decode()
        events = [json.loads(line[6:]) for line in wire.splitlines() if line.startswith("data: ")]
        return body, wire, events

    def test_custom_tool_keeps_stream_active_while_input_is_buffered(self):
        body, wire, events = self.run_tool("custom")
        buffer_start = wire.index("event: response.output_item.added\n")
        buffer_end = wire.index("event: response.custom_tool_call_input.delta\n")
        buffered = wire[buffer_start:buffer_end]
        self.assertIn(": keep-alive\n\n", buffered)
        self.assertIn("event: response.in_progress\n", buffered)
        self.assertEqual([event["delta"] for event in events
                          if event["type"] == "response.custom_tool_call_input.delta"], [body])
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual([event["sequence_number"] for event in events], list(range(len(events))))

    def test_function_arguments_continue_to_stream_without_extra_heartbeats(self):
        body, wire, events = self.run_tool("function")
        self.assertNotIn(": keep-alive", wire)
        self.assertEqual([event["type"] for event in events].count("response.in_progress"), 1)
        arguments = "".join(event["delta"] for event in events
                            if event["type"] == "response.function_call_arguments.delta")
        self.assertEqual(json.loads(arguments), {"input": body})
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_nonstreaming_custom_tool_response_keeps_the_complete_input(self):
        body, response = self.run_tool("custom", stream=False)
        self.assertEqual(response["status"], "completed")
        self.assertEqual([item["input"] for item in response["output"] if item["type"] == "custom_tool_call"], [body])


if __name__ == "__main__":
    unittest.main()
