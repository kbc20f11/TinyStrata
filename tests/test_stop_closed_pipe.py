"""Cancellation must survive native close/restart without masking the consumer's error.

Adapted from Strata bbf97633c (MIT; see NOTICE).
"""
import io
import queue
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve import server


def engine():
    instance = server.StrataEngine.__new__(server.StrataEngine)
    instance.alive = lambda: True
    instance.batch, instance.silence_s, instance.can_stop = 0, 0, True
    instance.lines = queue.Queue()
    instance.proc = SimpleNamespace(stdin=io.StringIO())
    instance.sampling_keys = lambda *args, **kwargs: ""
    instance.exit_code = lambda: 1
    return instance


class StopOnClosedPipe(unittest.TestCase):
    def leave_request(self, change, consumer_error=False):
        instance = engine()
        instance.lines.put("T 7")
        with patch.object(server.bias_api, "validate_request"), patch.object(server, "ENGINE_STALL_S", 0):
            gen = instance.generate([1, 2, 3], 8, {}, threading.Event())
            self.assertEqual(next(gen), 7)
            change(instance)
            instance.lines.put(None)
            if consumer_error:
                error = RuntimeError("consumer disconnected")
                with self.assertRaises(RuntimeError) as caught:
                    gen.throw(error)
                self.assertIs(caught.exception, error)
            else:
                gen.close()
        self.assertTrue(instance.lines.empty(), "the engine end marker must still be drained")
        return instance

    def test_closed_pipe_or_removed_process_does_not_mask_cancellation(self):
        def closed(instance):
            instance.proc.stdin.close()

        def removed(instance):
            instance.proc = None

        def broken(instance):
            def write(_):
                raise BrokenPipeError("closed reader")
            instance.proc.stdin.write = write

        def closed_during_flush(instance):
            def flush():
                instance.proc.stdin.close()
                raise ValueError("closed during flush")
            instance.proc.stdin.flush = flush

        for change in (closed, removed, broken, closed_during_flush):
            for consumer_error in (False, True):
                with self.subTest(state=change.__name__, consumer_error=consumer_error):
                    self.leave_request(change, consumer_error)

    def test_healthy_pipe_receives_stop(self):
        instance = self.leave_request(lambda instance: None)
        self.assertTrue(instance.proc.stdin.getvalue().endswith("STOP\n"))

    def test_process_replacement_during_stop_keeps_original_pipe(self):
        original, replacement = None, SimpleNamespace(stdin=io.StringIO())

        def swap(instance):
            nonlocal original
            original = instance.proc.stdin
            write = original.write

            def write_and_swap(text):
                result = write(text)
                instance.proc = replacement
                return result
            original.write = write_and_swap

        self.leave_request(swap)
        self.assertTrue(original.getvalue().endswith("STOP\n"))
        self.assertEqual(replacement.stdin.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
