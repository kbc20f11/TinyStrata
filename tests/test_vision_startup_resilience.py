"""Image-worker failures stay local to their Service; no native engine or GPU.

The READY tests run pipe-driven Python children, never a listener or model.
Run: python -m unittest discover -s tests -p test_vision_startup_resilience.py -v
"""
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from serve import server as api
from serve import hip_runtime
from serve.frontend import ChatTemplate


ROOT = Path(__file__).resolve().parents[1]


class ReadyProtocol(unittest.TestCase):
    def child(self, body):
        proc = subprocess.Popen([sys.executable, "-u", "-c", body], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8", bufsize=1,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        def cleanup():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=3)
            for stream in (proc.stdin, proc.stdout):
                stream.close()
        self.addCleanup(cleanup)
        return proc

    def worker(self, proc):
        worker = object.__new__(api.Vision)
        worker.proc = proc
        worker.spawn = ([], StringIO(), None)
        return worker

    def test_stdout_banners_do_not_consume_the_later_encode_reply(self):
        proc = self.child("import sys\nprint('driver banner', flush=True)\nprint('READY-made banner', flush=True)\n"
                          "print('ERRoneous driver notice', flush=True)\nprint('READY 128', flush=True)\n"
                          "for line in sys.stdin:\n print('DONE fake.sve 1', flush=True)\n")
        worker, strays = self.worker(proc), []
        self.assertEqual(worker._await_ready(3, strays).strip(), "READY 128")
        self.assertEqual(strays, ["driver banner", "READY-made banner", "ERRoneous driver notice"])
        self.assertIn("driver banner", worker.spawn[1].getvalue())
        proc.stdin.write("ENC image.png out.sve\n")
        proc.stdin.flush()
        self.assertEqual(worker._readline(2, "encode").strip(), "DONE fake.sve 1")

    def test_noisy_stdout_has_one_total_deadline_and_bounded_diagnostics(self):
        proc = self.child("import time\nwhile True:\n print('not READY', flush=True)\n time.sleep(.001)\n")
        worker, strays = self.worker(proc), []
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, "READY was not received"):
            worker._await_ready(.4, strays)
        self.assertLess(time.monotonic() - start, 2)
        self.assertTrue(strays)
        self.assertLessEqual(len(strays), worker.STRAY_KEEP)
        self.assertLessEqual(worker.spawn[1].getvalue().count("before READY"), worker.STRAY_KEEP)
        proc.wait(timeout=2)

    def test_dead_parent_with_held_stdout_does_not_wait_for_full_deadline(self):
        # An in-memory pipe deliberately never returns. No orphan process is
        # required to model an inherited stdout handle held by a descendant.
        release = api.threading.Event()
        stdout = SimpleNamespace(readline=lambda: (release.wait(3) and "") or "")
        proc = SimpleNamespace(stdout=stdout, poll=lambda: 7)
        worker = self.worker(proc)
        worker.READY_GRACE_S = .1
        start = time.monotonic()
        try:
            self.assertEqual(worker._await_ready(2, []), "")
            self.assertLess(time.monotonic() - start, 1)
        finally:
            release.set()

    def test_start_failure_keeps_stdout_and_current_stderr_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "image.log"
            log_path.write_text("stale prior failure\n", encoding="utf-8")
            with log_path.open("a+", encoding="utf-8") as log:
                worker = object.__new__(api.Vision)
                worker.dir = Path(directory)
                worker.spawn = ([], log, None)
                body = "import sys\nprint('driver banner', flush=True)\nprint('current reason', file=sys.stderr, flush=True)\nprint('ERR startup failed', flush=True)\nsys.exit(7)\n"
                def spawn(_what, _args, **kwargs):
                    proc = subprocess.Popen([sys.executable, "-u", "-c", body], **kwargs,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    self.addCleanup(proc.stdin.close)
                    self.addCleanup(proc.stdout.close)
                    return proc
                with mock.patch.object(api, "popen", side_effect=spawn), mock.patch.object(api, "contain"):
                    with self.assertRaisesRegex(RuntimeError, "ERR startup failed") as caught:
                        worker._start()
                self.assertIn("driver banner", str(caught.exception))
                self.assertIn("current reason", str(caught.exception))
                self.assertNotIn("stale prior failure", str(caught.exception))
                self.assertIn("7", str(caught.exception))
                self.assertIsNone(worker.proc)
                self.assertTrue(worker.stopped)


class FakeVision:
    def __init__(self, events, fail=True, gpu=True):
        self.events, self.fail, self.gpu = events, fail, gpu
        self.elastic, self.running, self.startup_error = False, False, ""

    def alive(self):
        return self.running

    def restart(self):
        self.events.append("vision.restart")
        if self.fail:
            raise RuntimeError("image allocation refused")
        self.running = True

    def unload(self):
        self.events.append("vision.unload")
        self.running = False


class RecoveringEngine(api.MockEngine):
    def __init__(self, tok, events, loaded=False):
        super().__init__(tok, "text remains available")
        self.events, self.running = events, loaded
        self.info = {"vision": 1, "images": True}

    def alive(self):
        return self.running

    def restart(self):
        self.events.append("engine.restart")
        self.running = True

    def unload(self):
        self.events.append("engine.unload")
        self.running = False


def service(vision=None, *, loaded=False, events=None):
    events = [] if events is None else events
    tok = api.ByteTokenizer()
    engine = RecoveringEngine(tok, events, loaded)
    return api.Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), vision=vision)


class ServiceResilience(unittest.TestCase):
    def test_eager_child_oserror_keeps_a_retryable_encoder_instance(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(api.Vision, "work_dir", return_value=Path(directory)), \
                mock.patch.object(api, "popen", side_effect=OSError("missing image executable")), \
                redirect_stdout(StringIO()):
            worker = api.Vision({"exe": "missing", "model": "model", "mmproj": "projector"})
        self.assertIn("missing image executable", worker.startup_error)
        self.assertFalse(worker.alive())

    def test_malformed_configuration_still_fails_closed(self):
        with self.assertRaises(KeyError):
            api.Vision({"exe": "missing"})

    def test_failed_lazy_encoder_does_not_disable_text_or_retry_every_turn(self):
        events = []
        vision = FakeVision(events)
        svc = service(vision, events=events)
        with redirect_stdout(StringIO()):
            svc.load()
            svc.load()
        self.assertEqual(events, ["vision.restart", "engine.restart"])
        self.assertTrue(svc.loaded())
        self.assertFalse(svc.vision_available())
        self.assertIn("image allocation refused", svc.vision_off_reason())
        messages = [{"role": "user", "content": "hello"}]
        ids, _, _ = svc.prepare(messages, None, {}, max_new=32)
        self.assertTrue(ids)
        self.assertTrue(list(svc.engine.generate(ids, 32, {}, api.threading.Event())))
        image = [{"role": "user", "content": [{"type": "image", "source": "data:image/png;base64,AA=="}]}]
        with self.assertRaisesRegex(ValueError, "image allocation refused"):
            svc.prepare(image, None, {}, max_new=32)
        # Native INFO still says image-capable; worker availability wins.
        self.assertFalse(svc.metrics()["engine"]["images"])
        self.assertFalse(svc.v1_status()["vision"]["enabled"])
        self.assertEqual(svc.v1_status()["vision"]["error"], "image allocation refused")

    def test_explicit_load_retries_before_engine_and_recovers_capabilities(self):
        events = []
        vision = FakeVision(events)
        svc = service(vision, events=events)
        with redirect_stdout(StringIO()):
            svc.load()
            events.clear()
            vision.fail = False
            svc.load(retry_vision=True)
        self.assertEqual(events, ["engine.unload", "vision.restart", "engine.restart"])
        self.assertEqual(svc.vision_error, "")
        self.assertTrue(svc.vision_available())
        self.assertTrue(svc.v1_status()["vision"]["enabled"])

    def test_failed_retry_reloads_text_engine_after_temporarily_unloading_it(self):
        events = []
        vision = FakeVision(events)
        vision.startup_error = "first failure"
        svc = service(vision, loaded=True, events=events)
        with redirect_stdout(StringIO()):
            svc.load(retry_vision=True)
        self.assertEqual(events, ["engine.unload", "vision.restart", "engine.restart"])
        self.assertTrue(svc.loaded())
        self.assertFalse(svc.vision_available())

    def test_retry_blocked_by_load_policy_keeps_the_failure_reason(self):
        vision = FakeVision([])
        vision.startup_error = "first failure"
        svc = service(vision)
        with mock.patch.object(svc, "check_load_policy", side_effect=api.GpuBusy("GPU busy")):
            with self.assertRaises(api.GpuBusy):
                svc.load(retry_vision=True)
        self.assertEqual(svc.vision_error, "first failure")
        self.assertFalse(svc.vision_available())

    def test_explicit_retry_rearms_an_elastic_encoder_without_allocating_it(self):
        events = []
        vision = FakeVision(events)
        vision.elastic, vision.startup_error = True, "first failure"
        svc = service(vision, loaded=True, events=events)
        svc.load(retry_vision=True)
        self.assertEqual(events, [])
        self.assertEqual(svc.vision_error, "")
        self.assertTrue(svc.vision_available())

    def test_failed_elastic_budget_restore_preserves_the_image_failure(self):
        for restore_error in (ValueError("reserve refused"), api.EngineDied("engine stopped")):
            with self.subTest(error=type(restore_error).__name__):
                events = []
                vision = FakeVision(events)
                vision.elastic, vision.elastic_mib = True, 1024
                svc = service(vision, loaded=True, events=events)
                svc.engine.vram = mock.Mock(side_effect=[{}, restore_error])
                with mock.patch.object(svc, "free_vram_mib", return_value=None), \
                        mock.patch.object(api.time, "sleep"), redirect_stdout(StringIO()), \
                        mock.patch.object(api.sys, "stderr", StringIO()):
                    with self.assertRaisesRegex(ValueError, "image allocation refused"):
                        svc._vision_up()
                self.assertFalse(svc.vision_held)
                self.assertEqual(svc.vision_error, "image allocation refused")
                self.assertEqual(svc.engine.vram.call_count, 2)
                self.assertFalse(svc.vision_available())

    def test_failure_is_not_stale_across_services_or_model_replacement(self):
        bad, good = FakeVision([]), FakeVision([], fail=False)
        bad.startup_error = "one model failed"
        left, right = service(bad), service(good)
        self.assertFalse(left.vision_available())
        self.assertTrue(right.vision_available())
        left.vision = good
        self.assertTrue(left.vision_available())
        self.assertEqual(left.vision_error, "")

    def test_public_capabilities_match_worker_failure(self):
        vision = FakeVision([])
        vision.startup_error = "encoder stopped"
        svc = service(vision, loaded=True)
        handler = object.__new__(api.make_handler(svc))
        handler.headers = {}
        handler._json = lambda status, value: setattr(handler, "result", (status, value))
        for route in ("/health", "/v1/models", "/props"):
            handler.path = route
            handler.do_GET()
            status, value = handler.result
            self.assertEqual(status, 200)
            if route == "/health":
                self.assertFalse(value["images"])
            elif route == "/v1/models":
                self.assertEqual(value["data"][0]["architecture"]["input_modalities"], ["text"])
            else:
                self.assertFalse(value["modalities"]["vision"])


class StartupGuards(unittest.TestCase):
    def test_default_quiet_env_does_not_mutate_or_override_user_setting(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            original = {"PATH": "keep"}
            self.assertEqual(api.quiet_child_env(original)["GGML_NO_BACKTRACE"], "1")
            self.assertNotIn("GGML_NO_BACKTRACE", original)
            for value in ("0", "", "custom"):
                self.assertEqual(api.quiet_child_env({"GGML_NO_BACKTRACE": value})["GGML_NO_BACKTRACE"], value)

    def test_developer_backtrace_escape_and_false_values(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("GGML_NO_BACKTRACE", api.quiet_child_env({"STRATA_BACKTRACE": "1"}))
            for value in ("", "0", "off", "false", "NO"):
                self.assertEqual(api.quiet_child_env({"STRATA_BACKTRACE": value})["GGML_NO_BACKTRACE"], "1")
        with mock.patch.dict(os.environ, {"STRATA_BACKTRACE": "yes"}, clear=True):
            self.assertNotIn("GGML_NO_BACKTRACE", api.quiet_child_env({}))

    def test_vision_sdk_mismatch_remains_an_error_before_any_spawn(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "a.dll", Path(directory) / "b.dll"
            left.write_bytes(b"same core")
            right.write_bytes(b"same core")
            runtimes = [hip_runtime.HipRuntime(str(left), ("sdk-A",), False),
                        hip_runtime.HipRuntime(str(right), ("sdk-B",), False)]
            cfg = {"backend": "hip", "exe": "engine.exe", "vision": {"exe": "image.exe", "gpu": True}}
            with mock.patch.object(hip_runtime, "resolve_hip_runtime", side_effect=runtimes), \
                    mock.patch.object(api, "popen") as spawn:
                with self.assertRaisesRegex(ValueError, "different HIP SDK dependencies"):
                    api.Vision(cfg["vision"], env=api.vision_env(cfg, {}))
            spawn.assert_not_called()

    def test_http_bind_does_not_perform_reverse_dns(self):
        server = object.__new__(api.Server)
        server.server_address = ("127.0.0.1", 18080)
        with mock.patch.object(api.socketserver.TCPServer, "server_bind") as bind, \
                mock.patch.object(api.socket, "getfqdn", side_effect=AssertionError("reverse DNS")):
            server.server_bind()
        bind.assert_called_once_with(server)
        self.assertEqual((server.server_name, server.server_port), ("127.0.0.1", 18080))


if __name__ == "__main__":
    unittest.main()
