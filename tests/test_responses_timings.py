"""CPU-only regression tests for the optional completed-event timings extension."""
import unittest
from serve.responses import Assembler, collect


class ResponseTimings(unittest.TestCase):
    def assembler(self):
        return Assembler({}, "tiny-test", 12, {}, False)

    def test_completed_event_has_native_timings_at_top_level(self):
        timings = {"prompt_n": 10, "cache_n": 2, "prompt_per_second": 500.0,
                   "predicted_n": 8, "predicted_per_second": 50.0}
        event = self.assembler().finish({"finish": "stop", "completion_tokens": 8, "timings": timings})[-1]
        self.assertEqual(event["type"], "response.completed")
        self.assertEqual(event["timings"], timings)
        self.assertNotIn("timings", event["response"])
        self.assertEqual(event["response"]["usage"]["output_tokens"], 8)
        timings["predicted_n"] = 99
        self.assertEqual(event["timings"]["predicted_n"], 8)

    def test_no_timings_keeps_original_event_shape(self):
        event = self.assembler().finish({"finish": "stop", "completion_tokens": 0})[-1]
        self.assertEqual(set(event), {"type", "sequence_number", "response"})

    def test_length_limit_does_not_add_completed_extension(self):
        event = self.assembler().finish({"finish": "length", "timings": {"predicted_n": 8}})[-1]
        self.assertEqual(event["type"], "response.incomplete")
        self.assertNotIn("timings", event)
        self.assertNotIn("timings", event["response"])

    def test_created_events_are_unchanged(self):
        events = self.assembler().start()
        self.assertEqual([e["type"] for e in events], ["response.created", "response.in_progress"])
        for event in events:
            self.assertNotIn("timings", event)
            self.assertNotIn("timings", event["response"])

    def test_nonstream_collection_retains_response_shape(self):
        asm = self.assembler()
        result = collect(asm.start() + asm.finish({"finish": "stop", "timings": {"cache_n": 2}}))
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("timings", result)


if __name__ == "__main__":
    unittest.main()
