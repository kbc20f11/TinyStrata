"""CPU-only native inbox lifecycle tests over real child stdin pipes.

Build tests/native-command-inbox.cpp with C++17+ / include + Threads, then:
  python tests/test_native_command_inbox.py PATH_TO_NATIVE_INBOX_TEST_EXE
No model, native inference executable, GPU or service is started.
"""
from __future__ import annotations

import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import unittest

EXE = pathlib.Path(os.environ.get("NATIVE_COMMAND_INBOX_TEST_EXE", ""))


def record(text: bytes, cancelled: int = 0) -> str:
    value = 14695981039346656037
    for byte in text:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"CMD {cancelled} {len(text)} {value:016x}"


class Child:
    def __init__(self, case: unittest.TestCase, mode: str):
        self.case = case
        self.proc = subprocess.Popen([str(EXE.resolve()), mode], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.case.addCleanup(self.cleanup)
        self.case.assertEqual(self.line(), "READY")

    def _read(self):
        assert self.proc.stdout
        for line in self.proc.stdout:
            self.lines.put(line.decode("utf-8").rstrip("\r\n"))
        self.lines.put(None)

    def line(self, timeout=5):
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            self.case.fail(f"child stdout timed out; pid={self.proc.pid}")

    def send(self, data: bytes):
        assert self.proc.stdin
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def close_input(self):
        assert self.proc.stdin
        self.proc.stdin.close()

    def exited(self, code=0, timeout=2):
        # stdin deliberately remains open unless the test explicitly closes it.
        self.case.assertEqual(self.proc.wait(timeout=timeout), code)
        self.reader.join(timeout=1)
        self.case.assertFalse(self.reader.is_alive())
        assert self.proc.stderr
        self.case.assertEqual(self.proc.stderr.read(), b"")

    def cleanup(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream:
                try:
                    stream.close()
                except OSError:
                    if self.proc.poll() is None:
                        raise
        self.reader.join(timeout=1)


class NativeInboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not EXE.is_file():
            raise unittest.SkipTest("native inbox CPU test executable is not configured")

    def test_error_unwinds_while_parent_holds_stdin_open(self):
        child = Child(self, "error")
        child.send(b"ERROR\n")
        self.assertEqual(child.line(), "LEAVING")
        self.assertEqual(child.line(), "ERROR simulated model error")
        child.exited(13)
        self.assertFalse(child.proc.stdin.closed)

    def test_failure_before_reader_initialization_unwinds_and_joins(self):
        child = Child(self, "init-error")
        self.assertEqual(child.line(), "ERROR injected reader initialization failure")
        child.exited(13)
        self.assertFalse(child.proc.stdin.closed)

    def test_return_joins_while_parent_holds_stdin_open(self):
        child = Child(self, "return")
        child.send(b"RETURN\n")
        self.assertEqual(child.line(), "LEAVING")
        child.exited()
        self.assertFalse(child.proc.stdin.closed)

    def test_explicit_stop_is_idempotent_and_bounded(self):
        child = Child(self, "stop")
        child.send(b"STOP_READER\npartial request without newline")
        self.assertEqual(child.line(), "STOPPED")
        child.exited()

    def test_data_arriving_during_error_shutdown_is_safe(self):
        child = Child(self, "error")
        stopped = threading.Event()

        def write_racing_data():
            try:
                child.send(b"ERROR\n")
                while not stopped.is_set():
                    child.send(b"GEN arrives during shutdown\n" * 8)
            except (BrokenPipeError, OSError):
                pass  # expected once the child has completed its owned join

        writer = threading.Thread(target=write_racing_data, daemon=True)
        writer.start()
        self.assertEqual(child.line(), "LEAVING")
        self.assertEqual(child.line(), "ERROR simulated model error")
        child.exited(13)
        stopped.set()
        writer.join(timeout=1)
        self.assertFalse(writer.is_alive())

    def test_quit_exits_without_eof_and_discards_following_lines(self):
        child = Child(self, "echo")
        child.send(b"QUIT\nGEN ignored\n")
        self.assertEqual(child.line(), record(b"QUIT"))
        self.assertEqual(child.line(), "END")
        child.exited()

    def test_eof_delivers_partial_line_exactly_once(self):
        child = Child(self, "echo")
        child.send(b"GEN complete\r\nGEN partial\r")
        child.close_input()
        self.assertEqual(child.line(), record(b"GEN complete"))
        self.assertEqual(child.line(), record(b"GEN partial"))
        self.assertEqual(child.line(), "END")
        child.exited()

    def test_empty_eof_and_trailing_newline(self):
        for payload, expected in [(b"", []), (b"\n", [b""]), (b"GEN x\n", [b"GEN x"])]:
            with self.subTest(payload=payload):
                child = Child(self, "echo")
                child.send(payload)
                child.close_input()
                for value in expected:
                    self.assertEqual(child.line(), record(value))
                self.assertEqual(child.line(), "END")
                child.exited()

    def test_fragmented_large_gen_preserves_every_byte(self):
        child = Child(self, "echo")
        payload = b"GEN 64 " + b"12345," * 400000 + "日本語".encode()
        for offset in range(0, len(payload), 3571):
            child.send(payload[offset:offset + 3571])
        child.send(b"\r\nQUIT\n")
        self.assertEqual(child.line(), record(payload))
        self.assertEqual(child.line(), record(b"QUIT"))
        self.assertEqual(child.line(), "END")
        child.exited()

    def test_queued_stop_epoch_and_new_generation(self):
        child = Child(self, "queue")
        child.send(b"GEN first\nSTOP\nGEN second\nSTOP\nGEN third\nQUIT\n")
        for value, cancelled in [(b"GEN first", 1), (b"GEN second", 1), (b"GEN third", 0), (b"QUIT", 0)]:
            self.assertEqual(child.line(), record(value, cancelled))
        self.assertEqual(child.line(), "END")
        child.exited()

    def test_stop_cancels_running_work_but_not_later_gen(self):
        child = Child(self, "cancel")
        child.send(b"GEN first\n")
        self.assertEqual(child.line(), record(b"GEN first"))
        self.assertEqual(child.line(), "START")
        child.send(b"STOP\nGEN second\nQUIT\n")
        self.assertEqual(child.line(), "CANCELLED")
        self.assertEqual(child.line(), record(b"GEN second"))
        self.assertEqual(child.line(), record(b"QUIT"))
        self.assertEqual(child.line(), "END")
        child.exited()

    def test_unknown_cancel_and_single_cr_strip(self):
        child = Child(self, "echo")
        child.send(b"CANCEL\nGEN value\r\r\nQUIT\n")
        for value in (b"CANCEL", b"GEN value\r", b"QUIT"):
            self.assertEqual(child.line(), record(value))
        self.assertEqual(child.line(), "END")
        child.exited()

    def test_line_split_before_newline_does_not_publish_early(self):
        child = Child(self, "echo")
        child.send(b"GEN fragmented\r")
        with self.assertRaises(queue.Empty):
            child.lines.get(timeout=0.05)
        child.send(b"\nQUIT\n")
        self.assertEqual(child.line(), record(b"GEN fragmented"))
        self.assertEqual(child.line(), record(b"QUIT"))
        self.assertEqual(child.line(), "END")
        child.exited()

    def test_repeated_reader_lifetimes_do_not_close_stdin(self):
        child = Child(self, "cycles")
        for i in range(40):
            child.send(b"CYCLE\n")
            self.assertEqual(child.line(), f"CYCLE {i}")
        child.exited()
        self.assertFalse(child.proc.stdin.closed)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXE = pathlib.Path(sys.argv.pop(1))
    if not EXE.is_file():
        raise SystemExit("pass the compiled native-command-inbox test executable path")
    unittest.main(verbosity=2)
