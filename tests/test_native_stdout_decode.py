"""Malformed native diagnostics must not stop protocol delivery or EOF wakeups."""
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from serve import server


CHILD = r'''
import pathlib, struct, sys
mode = sys.argv[1]
out = sys.stdout.buffer
def emit(value):
    out.write(value)
    out.flush()
if mode.startswith("vision"):
    emit(b"READY \xff " + "日本語".encode() + b"\n")
    for line in sys.stdin:
        if line.startswith("QUIT"):
            break
        _, image, output = line.split()
        pathlib.Path(output).write_bytes(struct.pack("<5i4f", 0x31455653, 2, 2, 1, 2, 1, 2, 3, 4))
        emit(b"OK 2 \xfe " + "画像".encode() + b"\n")
else:
    emit(b"diagnostic before READY: \xff\n")
    emit(b"INFO label=" + "日本語".encode() + (b" batch_slots=2" if mode == "slots" else b"") + b"\n")
    emit(b"READY 8192 stop\n")
    for line in sys.stdin:
        if line.startswith("QUIT"):
            break
        emit(b"T 101\n")
        emit(b"diagnostic between tokens: \xfe " + "日本語".encode() + b"\n")
        if mode != "eof":
            emit(b"T 202\nDONE 2 3 1.0 2.0 stop 0 0 0 0 0 0 0 0.0 3 0\n")
        break
'''


class NativeStdoutDecode(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-native-stdout-")
        self.addCleanup(self.temporary.cleanup)
        self.child = Path(self.temporary.name) / "child.py"
        self.child.write_text(CHILD, encoding="utf-8")
        self.processes = []
        self.addCleanup(self.close_processes)

    def close_processes(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None and not stream.closed:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass

    def spawn(self, mode):
        # Keep the production text-pipe options, replacing only the executable.
        def start(description, command, **kwargs):
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen([sys.executable, "-I", "-B", str(self.child), mode], **kwargs)
            self.processes.append(process)
            return process
        return patch.object(server, "popen", side_effect=start)

    def native(self, mode):
        with self.spawn(mode), patch.object(server, "ENGINE_READY_S", 3):
            engine = server.StrataEngine("fixture-native", ["--max-context", "8192"])
        self.addCleanup(engine.close)
        return engine

    def test_bad_utf8_before_ready_and_between_tokens_keeps_request_working(self):
        engine = self.native("native")
        self.assertEqual(engine.max_context, 8192)
        self.assertEqual(engine.info["label"], "日本語")
        self.assertEqual(list(engine.generate([1, 2, 3], 2, {}, threading.Event())), [101, 202])
        engine.pump.join(timeout=3)
        self.assertFalse(engine.pump.is_alive())
        self.assertIsNone(engine.lines.get(timeout=1))

    def test_bad_utf8_then_eof_raises_instead_of_leaving_a_request_waiting(self):
        engine = self.native("eof")
        request = engine.generate([1, 2, 3], 2, {}, threading.Event())
        self.assertEqual(next(request), 101)
        with self.assertRaises(server.EngineDied):
            next(request)
        engine.pump.join(timeout=3)
        self.assertFalse(engine.pump.is_alive())
        self.assertFalse(engine.alive())

    def test_replacement_and_valid_unicode_reach_queue_and_eof_releases_slots(self):
        engine = self.native("slots")
        self.assertEqual(engine.batch, 2)
        engine.proc.stdin.write("TEST\n")
        engine.proc.stdin.flush()
        lines = []
        while True:
            line = engine.lines.get(timeout=3)
            if line is None:
                break
            lines.append(line)
        self.assertEqual(lines[0], "T 101\n")
        self.assertEqual(lines[1], "diagnostic between tokens: \ufffd 日本語\n")
        self.assertTrue(lines[-1].startswith("DONE "))
        for slot in engine.slot_q:
            self.assertIsNone(slot.get(timeout=1))
        engine.pump.join(timeout=3)
        self.assertFalse(engine.pump.is_alive())

    def test_vision_ready_and_vector_reply_tolerate_invalid_utf8_suffixes(self):
        with self.spawn("vision"):
            vision = server.Vision({"exe": "fixture-vision", "mmproj": "fixture.gguf", "model": "fixture.gguf"})
        self.addCleanup(vision.shutdown)
        self.assertTrue(vision.alive())
        path, tokens = vision.encode(b"\x89PNG\r\n\x1a\nfixture")
        self.assertEqual(tokens, 2)
        self.assertEqual(struct.unpack("<5i4f", path.read_bytes()), (0x31455653, 2, 2, 1, 2, 1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
