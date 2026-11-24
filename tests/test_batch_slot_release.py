"""Exercise batch release races without timing or a native GPU process."""
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from serve.server import StrataEngine


class BatchSlotRelease(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine.__new__(StrataEngine)
        self.engine.slot_cv = threading.Condition()
        self.engine.ctl = threading.Lock()
        self.engine.waiting = self.engine.admitting = 0
        self.engine.wait_lens = []
        self.engine._yielded = None
        self.engine._take_control = self.take_control
        self.engine._control = self.admit
        self.engine.pick_slot = self.pick_slot
        self.engine.gen = 1
        self.engine.batch = 2
        self.engine.last = {}
        self.engine.ended = False
        self.engine.starting = False
        self.commands = []
        self.after_stop = lambda: self.engine.slot_q[0].put("BDONE 0 2 cancel\n")
        self.install_slots([False, True], {1}, self.commands, self.on_command)

    def install_slots(self, busy, claims, commands, callback=None):
        engine = self.engine
        engine.slot_q = [queue.Queue() for _ in busy]
        engine.slot_busy = busy
        engine.slot_claims = claims
        engine.slot_releasing = set()
        engine.slot_held = [[] for _ in busy]
        engine.slot_live = [None for _ in busy]
        engine.slot_used = [0.0 for _ in busy]
        engine.wlock = threading.Lock()

        def write(line):
            commands.append(line)
            if callback:
                callback(line)

        engine.proc = SimpleNamespace(poll=lambda: None, stdin=SimpleNamespace(write=write, flush=lambda: None))

    def take_control(self, *_):
        self.engine.ctl.acquire()
        yield from ()
        return True

    def admit(self, _cancel, on_token, **_kwargs):
        self.engine._ctl_result = ("badm", True)
        on_token(71)
        yield False

    def pick_slot(self, _prompt):
        return next((slot for slot, busy in enumerate(self.engine.slot_busy) if not busy), None)

    def on_command(self, line):
        if line.startswith("BGEN 0 "):
            self.engine.slot_q[0].put("BT 0 72\n")
        elif line == "BSTOP 0\n":
            self.after_stop()

    def started_request(self):
        request = self.engine.generate_batched([10, 20, 30], 10, {}, threading.Event())
        self.assertEqual(next(request), 71)
        self.assertEqual(next(request), 72)
        return request

    def run_waiter(self, *, target, daemon):
        return SimpleNamespace(start=target)

    def test_completed_waiter_can_hand_slot_to_next_request_immediately(self):
        request = self.started_request()
        engine = self.engine

        def run_and_claim(*, target, daemon):
            def start():
                target()
                with engine.slot_cv:
                    slot = self.pick_slot([40])
                    self.assertEqual(slot, 0)
                    engine.slot_busy[slot] = True
                    engine.slot_claims.add(slot)

            return SimpleNamespace(start=start)

        with patch("serve.server.threading.Thread", side_effect=run_and_claim):
            request.close()
        self.assertEqual(engine.slot_claims, {0, 1})
        self.assertEqual(engine.slot_releasing, set())
        self.assertEqual(engine.slot_held[0], [10, 20, 30, 71])

    def replacement(self):
        self.engine.starting = True
        commands = []
        self.install_slots([True, False], {0}, commands)
        self.engine.slot_q[0].put("BDONE 0 1 stop\n")
        self.engine.slot_held[0] = [90, 91]
        self.engine.slot_live[0] = {"state": "decoding"}
        self.engine.gen += 1
        self.engine.starting = False
        return commands

    def assert_replacement_untouched(self, commands):
        engine = self.engine
        self.assertEqual(commands, [])
        self.assertEqual(engine.slot_claims, {0})
        self.assertEqual(engine.slot_releasing, set())
        self.assertEqual(engine.slot_busy, [True, False])
        self.assertEqual(engine.slot_held[0], [90, 91])
        self.assertEqual(engine.slot_live[0], {"state": "decoding"})
        self.assertEqual(engine.slot_q[0].get_nowait(), "BDONE 0 1 stop\n")
        self.assertTrue(engine.slot_q[0].empty())

    def test_restart_during_stop_keeps_new_engines_request_and_queue(self):
        request = self.started_request()
        old_queue = self.engine.slot_q[0]
        commands = []

        def restart():
            commands.append(self.replacement())
            old_queue.put(None)

        self.after_stop = restart
        with patch("serve.server.threading.Thread", side_effect=self.run_waiter):
            request.close()
        self.assertEqual(self.commands[-1], "BSTOP 0\n")
        self.assert_replacement_untouched(commands[0])

    def test_consumer_closes_after_restart_without_stopping_replacement(self):
        request = self.started_request()
        commands = self.replacement()
        with patch("serve.server.threading.Thread", side_effect=self.run_waiter):
            request.close()
        self.assertEqual(len(self.commands), 1)
        self.assert_replacement_untouched(commands)

    def test_stop_selected_before_restart_is_sent_only_to_old_pipe(self):
        request = self.started_request()
        old_queue = self.engine.slot_q[0]
        send = self.engine._send
        commands = []
        self.after_stop = lambda: old_queue.put(None)

        def restart_then_send(text, **kwargs):
            commands.append(self.replacement())
            send(text, **kwargs)

        with patch.object(self.engine, "_send", side_effect=restart_then_send), \
                patch("serve.server.threading.Thread", side_effect=self.run_waiter):
            request.close()
        self.assertEqual(self.commands[-1], "BSTOP 0\n")
        self.assert_replacement_untouched(commands[0])


if __name__ == "__main__":
    unittest.main()
