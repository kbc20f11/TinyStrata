"""Replay Intel's committed-cache protocol through the real stdout pump/metrics.

The child is a CPU-only protocol fixture, not an Intel or model implementation.
Actual GPU commit/rollback coverage lives in the opt-in test_intel_native suite.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve import server


CHILD = r'''
import sys
def emit(text):
    print(text, flush=True)
def gate(expected="NEXT"):
    assert sys.stdin.readline().strip() == expected
emit("INFO backend=sycl memory_model=uma device_integrated=1 kv=int8 kv_used_tokens=0")
emit("READY 8192 stop")
request = 0
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    assert line.startswith("GEN ")
    request += 1
    if request == 1:
        emit("RESUME 0")
        for used in (128, 256, 384):
            emit(f"INFO kv_used_tokens={used}")
            emit(f"PP {used} 384 1.0")
            gate()
        emit("T 101")
        gate()  # The emitted token has not entered the model cache yet.
        emit("INFO kv_used_tokens=385")
        emit("T 102")
        emit("DONE 2 384 1.0 2.0 length 0 0 0 0 0 0 0 0.0 384 0")
    elif request == 2:
        # Saved-turn restore, then a completed prefill and MTP window. Cancellation
        # can leave committed rows ahead of the tokens already sent to the client.
        emit("INFO kv_used_tokens=128")
        emit("RESUME 128")
        emit("INFO kv_used_tokens=390")
        emit("PP 390 390 1.0")
        gate()
        emit("T 103")
        emit("INFO kv_used_tokens=393")
        emit("T 104")
        gate("STOP")
        emit("DONE 2 390 1.0 2.0 cancel 2 3 128 1 0 0 0 0.0 262 0")
    elif request == 3:
        # An invalid GEN must leave the resident cache unchanged.
        emit("ERR Token ID outside vocabulary")
    elif request == 4:
        emit("INFO kv_used_tokens=0")
        emit("RESUME 0")
        emit("INFO kv_used_tokens=1")
        emit("PP 1 1 1.0")
        gate()
        emit("T 105")
        emit("DONE 1 1 1.0 1.0 stop 0 0 0 0 0 0 0 0.0 1 0")
'''


class IntelContextTelemetry(unittest.TestCase):
    def test_auto_uses_each_native_ready_capacity_and_unload_is_unknown(self):
        """The real stdout parser receives two different CPU fixture READYs."""
        with tempfile.TemporaryDirectory(prefix="intel-context-auto-protocol-") as temporary:
            child = Path(temporary) / "fixture.py"
            child.write_text('import sys\nprint("INFO backend=sycl kv_used_tokens=0 max_context=8192", flush=True)\n'
                             'print("READY " + sys.argv[1] + " stop", flush=True)\n'
                             'for line in sys.stdin:\n if line.startswith("QUIT"): break\n', encoding="utf-8")
            capacities, processes = iter((262144, 65536)), []
            def start(description, command, **kwargs):
                self.assertEqual(command[command.index("--max-context") + 1], "auto")
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                process = subprocess.Popen([sys.executable, "-I", "-B", str(child), str(next(capacities))], **kwargs)
                processes.append(process)
                return process
            with patch.object(server, "popen", side_effect=start), patch.object(server, "ENGINE_READY_S", 3):
                engine = server.StrataEngine("cpu-protocol-fixture", ["--max-context", "auto"], lazy=True)
                service = server.Service(engine, server.ByteTokenizer(), SimpleNamespace(call_format="function"))
                service.backend = "sycl"
                try:
                    self.assertEqual(service.reported_ctx(), 0)
                    for actual in (262144, 65536):
                        engine.restart(tries=1)
                        self.assertEqual(engine.max_context, actual)
                        self.assertEqual(service.metrics()["engine"]["max_context"], actual)
                        engine.unload()
                        self.assertEqual(service.reported_ctx(), 0)
                        self.assertIsNone(service.metrics()["engine"]["kv_used_tokens"])
                finally:
                    engine.close()
                    for process in processes:
                        if process.poll() is None:
                            process.kill()  # CPU fixture only, never a model/application process.
                        process.wait(timeout=5)
                        for stream in (process.stdin, process.stdout):
                            if stream is not None and not stream.closed:
                                stream.close()

    def test_committed_usage_reaches_metrics_across_request_lifecycle(self):
        with tempfile.TemporaryDirectory(prefix="intel-kv-protocol-") as temporary:
            child = Path(temporary) / "fixture.py"
            child.write_text(CHILD, encoding="utf-8")
            processes = []

            def start(description, command, **kwargs):
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                process = subprocess.Popen([sys.executable, "-I", "-B", str(child)], **kwargs)
                processes.append(process)
                return process

            with patch.object(server, "popen", side_effect=start), patch.object(server, "ENGINE_READY_S", 3):
                engine = server.StrataEngine("cpu-protocol-fixture", ["--max-context", "8192"])
                try:
                    service = server.Service(engine, server.ByteTokenizer(), SimpleNamespace(call_format="function"))

                    def usage():
                        metrics = service.metrics()
                        self.assertEqual(metrics["engine"]["max_context"], 8192)
                        self.assertEqual(metrics["engine"]["memory_model"], "uma")
                        return metrics["engine"]["kv_used_tokens"]

                    def advance():
                        engine.proc.stdin.write("NEXT\n")
                        engine.proc.stdin.flush()

                    self.assertEqual(usage(), 0)
                    request = engine.generate([1] * 384, 2, {}, threading.Event())
                    for committed in (128, 256, 384):
                        self.assertIsNone(next(request))  # The actual PP heartbeat.
                        self.assertEqual(usage(), committed)
                        advance()
                    self.assertEqual(next(request), 101)
                    self.assertEqual(usage(), 384)  # Not prompt + emitted output.
                    advance()
                    self.assertEqual(list(request), [102])
                    self.assertEqual(usage(), 385)  # Final emitted token is unconsumed.

                    request = engine.generate([2] * 390, 8, {}, threading.Event())
                    self.assertIsNone(next(request))
                    self.assertEqual(engine.reused, 128)
                    self.assertEqual(usage(), 390)
                    advance()
                    self.assertEqual([next(request), next(request)], [103, 104])
                    self.assertEqual(usage(), 393)  # Accepted MTP inputs, not only T lines.
                    request.close()  # Uses the production STOP + drain-to-DONE path.
                    self.assertEqual(engine.last["finish"], "cancel")
                    self.assertEqual(usage(), 393)

                    with self.assertRaisesRegex(ValueError, "Token ID outside vocabulary"):
                        list(engine.generate([999], 1, {}, threading.Event()))
                    self.assertEqual(usage(), 393)

                    request = engine.generate([3], 1, {}, threading.Event())
                    self.assertIsNone(next(request))
                    self.assertEqual(engine.reused, 0)
                    self.assertEqual(usage(), 1)
                    advance()
                    self.assertEqual(list(request), [105])
                    self.assertEqual(usage(), 1)

                    engine.unload()
                    self.assertFalse(service.loaded())
                    engine.restart(tries=1)
                    self.assertTrue(service.loaded())
                    self.assertEqual(usage(), 0)  # Old INFO must not survive restart.
                finally:
                    engine.close()
                    for process in processes:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=5)
                        for stream in (process.stdin, process.stdout):
                            if stream is not None and not stream.closed:
                                stream.close()


if __name__ == "__main__":
    unittest.main()
