"""UI-only update ownership tests; no model, production port or GPU is used."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ui import launch


class LauncherUpgradeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tinystrata-ui-upgrade-")
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.source.mkdir()
        self.work = Path(self.temp.name) / "work"
        args = argparse.Namespace(api_port=9, port=19191, model_family="clover")
        self.launcher = launch.Launcher(args, self.source, self.work)

    def test_gpt_oss_family_alias_has_independent_model_config_and_state(self):
        for selected in ("gpt-oss", "gptoss"):
            args = argparse.Namespace(api_port=9, port=19191, model_family=selected)
            launcher = launch.Launcher(args, self.source, self.work)
            self.assertEqual(launcher.family, "gpt-oss")
            self.assertEqual(launch.PROFILES[launcher.family], ("gpt-oss-20b", "config-gpt-oss.json"))
            launcher.save_settings()
            self.assertEqual(launch.read_json(launcher.state_dir / "settings.json")["model_family"], "gpt-oss")
            config = launcher.runtime / ".tinystrata" / "config-gpt-oss.json"
            with mock.patch.object(launch, "process_started", return_value="created"):
                launcher.save_api_process(argparse.Namespace(pid=123), config)
            state = launch.read_json(launcher.runtime / "server-state-gpt-oss.json")
            self.assertEqual(state["config"], str(config))
            self.assertFalse((launcher.runtime / "server-state.json").exists())

    def test_tiel_family_alias_has_independent_model_and_config(self):
        for selected in ("tiel", "tiel-coder"):
            args = argparse.Namespace(api_port=9, port=19191, model_family=selected)
            launcher = launch.Launcher(args, self.source, self.work)
            self.assertEqual(launcher.family, "tiel")
            self.assertEqual(launch.PROFILES[launcher.family], ("tiel-coder-35b-a3b", "config-tiel-coder.json"))
            launcher.save_settings()
            self.assertEqual(launch.read_json(launcher.state_dir / "settings.json")["model_family"], "tiel")

    def test_existing_tiel_api_is_followed_by_model_identity(self):
        health = {"status": "ok", "service": "strata", "model": "tiel-coder-35b-a3b", "loaded": True}
        with mock.patch.object(launch, "port_occupied", return_value=True), \
             mock.patch.object(launch, "request_json", return_value=health), \
             mock.patch.object(self.launcher, "spawn") as spawn:
            self.launcher.ensure_api()
        self.assertEqual(self.launcher.family, "tiel")
        spawn.assert_not_called()

    def test_tiel_process_state_is_separate_from_qwen(self):
        self.launcher.family = "tiel"
        config = self.launcher.runtime / ".tinystrata" / "config-tiel-coder.json"
        with mock.patch.object(launch, "process_started", return_value="created"):
            self.launcher.save_api_process(argparse.Namespace(pid=123), config)
        state = launch.read_json(self.launcher.runtime / "server-state-tiel-coder.json")
        self.assertEqual(state["config"], str(config))
        self.assertFalse((self.launcher.runtime / "server-state.json").exists())

    def test_owned_current_backend_reused_without_process_action(self):
        with mock.patch.object(self.launcher, "existing_ui", return_value={"pid": 42}), \
             mock.patch.object(self.launcher, "current_ui", return_value=True), \
             mock.patch.object(launch, "stop_owned_ui") as stop, \
             mock.patch.object(self.launcher, "spawn") as spawn:
            self.launcher.ensure_ui()
        stop.assert_not_called()
        spawn.assert_not_called()

    def test_missing_or_changed_process_record_never_stops_process(self):
        for state in (None, [], {"pid": 12, "command": []}, {"pid": True, "command": []}):
            if state is not None:
                self.launcher.write_state("ui-process.json", state)
            with mock.patch.object(launch, "stop_owned_ui") as stop, self.assertRaises(RuntimeError):
                self.launcher.restart_previous_ui({"pid": 42})
            stop.assert_not_called()

    def test_other_python_script_cannot_be_restarted(self):
        command = [str(self.launcher.python), "-B", str(self.source / "serve" / "server.py"),
                   "--port", str(self.launcher.port), "--api-port", "9"]
        self.launcher.write_state("ui-process.json", {"pid": 42, "command": command})
        with mock.patch.object(launch, "stop_owned_ui") as stop, self.assertRaises(RuntimeError):
            self.launcher.restart_previous_ui({"pid": 42})
        stop.assert_not_called()

    def test_backend_match_requires_correct_source_and_data_directory(self):
        valid = {"backend_revision": "revision", "source_dir": str(self.source),
                 "data_dir": str(self.launcher.state_dir)}
        with mock.patch.object(launch, "backend_revision", return_value="revision"):
            self.assertTrue(self.launcher.current_ui(valid))
            for field in valid:
                invalid = {**valid, field: "different"}
                self.assertFalse(self.launcher.current_ui(invalid))

    def test_incomplete_source_update_preserves_old_ui(self):
        with mock.patch.object(self.launcher, "existing_ui", return_value={"pid": 42}), \
             mock.patch.object(self.launcher, "current_ui", return_value=False), \
             mock.patch.object(self.launcher, "restart_previous_ui") as restart, \
             self.assertRaises(OSError):
            self.launcher.ensure_ui()
        restart.assert_not_called()

    def make_fake_python(self):
        self.launcher.python.parent.mkdir(parents=True, exist_ok=True)
        self.launcher.python.write_bytes(b"test-only; never executed")

    def test_spawn_log_failure_stops_only_new_child_and_preserves_record(self):
        self.make_fake_python()
        self.launcher.write_state("api-process.json", {"pid": 12, "command": ["previous"]})
        record = self.launcher.state_dir / "api-process.json"
        previous = record.read_bytes()
        child = mock.Mock(pid=42)
        child.poll.return_value = None
        with mock.patch.object(launch.subprocess, "Popen", return_value=child), \
             mock.patch.object(self.launcher, "log", side_effect=OSError("log disk full")), \
             mock.patch.object(self.launcher, "write_state") as write, \
             self.assertRaisesRegex(OSError, "log disk full"):
            self.launcher.spawn(None, ["-m", "serve.server"], "api")
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=5)
        child.kill.assert_not_called()
        write.assert_not_called()
        self.assertEqual(record.read_bytes(), previous)

    def test_spawn_state_failure_stops_new_child_and_keeps_previous_record(self):
        self.make_fake_python()
        self.launcher.write_state("ui-process.json", {"pid": 12, "command": ["previous"]})
        record = self.launcher.state_dir / "ui-process.json"
        previous = record.read_bytes()
        child = mock.Mock(pid=42)
        child.poll.return_value = None
        with mock.patch.object(launch.subprocess, "Popen", return_value=child), \
             mock.patch.object(launch, "process_started", return_value="new-child-created"), \
             mock.patch.object(self.launcher, "log"), \
             mock.patch.object(self.launcher, "write_state", side_effect=OSError("state denied")), \
             self.assertRaisesRegex(OSError, "state denied"):
            self.launcher.spawn(None, ["ui.py"], "ui")
        child.terminate.assert_called_once_with()
        child.wait.assert_called_once_with(timeout=5)
        self.assertEqual(record.read_bytes(), previous)

    def test_spawn_publication_failure_escalates_only_the_owned_handle(self):
        self.make_fake_python()
        child = mock.Mock(pid=42)
        child.poll.return_value = None
        child.wait.side_effect = [subprocess.TimeoutExpired("owned child", 5), 0]
        with mock.patch.object(launch.subprocess, "Popen", return_value=child), \
             mock.patch.object(self.launcher, "log", side_effect=OSError("log failed")), \
             self.assertRaisesRegex(OSError, "log failed"):
            self.launcher.spawn(None, [], "api")
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()
        self.assertEqual(child.wait.call_count, 2)

    def test_popen_failure_never_publishes_replacement_state(self):
        self.make_fake_python()
        with mock.patch.object(launch.subprocess, "Popen", side_effect=OSError("cannot spawn")), \
             mock.patch.object(self.launcher, "write_state") as write, \
             self.assertRaisesRegex(OSError, "cannot spawn"):
            self.launcher.spawn(None, [], "api")
        write.assert_not_called()

    def test_replacement_import_failure_keeps_old_ui_running(self):
        self.make_fake_python()
        with mock.patch.object(self.launcher, "existing_ui", return_value={"pid": 42}), \
             mock.patch.object(self.launcher, "current_ui", return_value=False), \
             mock.patch.object(launch, "backend_revision", return_value="new"), \
             mock.patch.object(launch.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "SyntaxError: fixture")), \
             mock.patch.object(self.launcher, "restart_previous_ui") as restart, \
             mock.patch.object(self.launcher, "spawn") as spawn, \
             self.assertRaisesRegex(RuntimeError, "previous UI was left running"):
            self.launcher.ensure_ui()
        restart.assert_not_called()
        spawn.assert_not_called()

    def test_replacement_import_timeout_keeps_old_ui_running(self):
        self.make_fake_python()
        with mock.patch.object(self.launcher, "existing_ui", return_value={"pid": 42}), \
             mock.patch.object(self.launcher, "current_ui", return_value=False), \
             mock.patch.object(launch, "backend_revision", return_value="new"), \
             mock.patch.object(launch.subprocess, "run", side_effect=subprocess.TimeoutExpired("import fixture", 30)), \
             mock.patch.object(self.launcher, "restart_previous_ui") as restart, \
             self.assertRaisesRegex(RuntimeError, "previous UI was left running"):
            self.launcher.ensure_ui()
        restart.assert_not_called()

    def test_replacement_import_uses_installed_python_without_starting_server(self):
        self.make_fake_python()
        with mock.patch.object(launch.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            self.launcher.validate_ui_backend()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [str(self.launcher.python), "-I", "-B", "-c"])
        self.assertIn("import_module('ui.server')", command[4])
        self.assertNotIn("serve_forever", command[4])
        self.assertEqual(command[5], str(self.source))
        self.assertEqual(run.call_args.kwargs["timeout"], 30)

    def test_replacement_import_passes_before_previous_ui_stops(self):
        order = []
        with mock.patch.object(self.launcher, "existing_ui", return_value={"pid": 42}), \
             mock.patch.object(self.launcher, "current_ui", return_value=False), \
             mock.patch.object(launch, "backend_revision", return_value="new"), \
             mock.patch.object(self.launcher, "validate_ui_backend", side_effect=lambda: order.append("validated")), \
             mock.patch.object(self.launcher, "restart_previous_ui", side_effect=lambda health: order.append("stopped")), \
             mock.patch.object(self.launcher, "spawn", side_effect=RuntimeError("end mocked path")), \
             self.assertRaisesRegex(RuntimeError, "end mocked path"):
            self.launcher.ensure_ui()
        self.assertEqual(order, ["validated", "stopped"])

    def test_process_match_requires_image_command_parent_and_listener(self):
        command = [sys.executable, "-B", str(self.source / "ui" / "server.py")]
        metadata = {"executable": sys.executable, "command": subprocess.list2cmdline(command),
                    "parent": 44, "owners": [42]}
        self.assertTrue(launch.matching_ui_process(metadata, 42, command, 44))
        for field, value in (("executable", str(self.source / "unknown.exe")), ("command", "other"),
                             ("parent", 99), ("owners", [99]), ("owners", [42, 99])):
            self.assertFalse(launch.matching_ui_process({**metadata, field: value}, 42, command, 44))

    def test_health_reports_frozen_startup_revision(self):
        from ui import server
        with mock.patch.object(server, "backend_revision", return_value="startup-revision"):
            instance = server.UIServer(0, 9, self.work / "http-state")
        thread = threading.Thread(target=instance.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
        thread.start()
        try:
            with mock.patch.object(server, "backend_revision", side_effect=AssertionError("Must not hash edited files during health")), \
                 mock.patch.object(server.Handler, "log_message"):
                health = launch.request_json(instance.server_port, "/local-ui-health")
            self.assertEqual(health["backend_revision"], "startup-revision")
            self.assertEqual(Path(health["source_dir"]), ROOT)
            self.assertEqual(Path(health["data_dir"]), self.work / "http-state")
        finally:
            instance.shutdown()
            thread.join(3)
            instance.server_close()

    @unittest.skipUnless(os.name == "nt", "Windows process ownership integration")
    def test_real_temporary_legacy_ui_upgrade_and_reuse(self):
        ui = self.source / "ui"
        ui.mkdir()
        for name in ("launch.py", "server.py", "chat_history.py", "tool_settings.py", "web_search.py", "sandbox_runtime.py"):
            shutil.copy2(ROOT / "ui" / name, ui / name)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.launcher.port = probe.getsockname()[1]
        self.assertNotIn(self.launcher.port, (3000, 8080, 18080))
        self.launcher.python = Path(sys.executable).resolve()
        script = ui / "server.py"
        new_source = script.read_bytes()
        # This process is the old backend already in memory. Replacing its file
        # below models an installer update without touching a real installation.
        script.write_text('''import argparse, json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
p=argparse.ArgumentParser(); p.add_argument('--port',type=int); p.add_argument('--api-port',type=int); a=p.parse_args()
class H(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  data=json.dumps({'app':'tinystrata-ui','pid':os.getpid(),'api_port':a.api_port}).encode()
  self.send_response(200); self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
class S(HTTPServer):
 allow_reuse_address=False
S(('127.0.0.1',a.port),H).serve_forever()
''', encoding="utf-8")
        command = [str(self.launcher.python), "-B", str(script), "--port", str(self.launcher.port), "--api-port", "9"]
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        new_health = None
        try:
            deadline = time.monotonic() + 10
            last_error = None
            while time.monotonic() < deadline:
                try:
                    health = self.launcher.existing_ui()
                    if health and type(health.get("pid")) is int:
                        break
                except RuntimeError as error:
                    last_error = str(error)
                if child.poll() is not None:
                    self.fail(f"Temporary UI exited: {child.stderr.read().decode(errors='replace')}")
                time.sleep(.05)
            else:
                self.fail(f"Temporary legacy UI did not start: {last_error}")
            self.launcher.write_state("ui-process.json", {"pid": child.pid, "command": command})
            script.write_bytes(new_source)
            with mock.patch.object(launch.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, '{"owners":[]}', "")):
                with self.assertRaises(RuntimeError):
                    launch.stop_owned_ui(health["pid"], command, None, self.launcher.port, launcher_pid=child.pid)
            self.assertIsNone(child.poll(), "An unverified listener must remain running")
            metadata = launch.windows_ui_metadata(health["pid"], self.launcher.port)
            self.assertTrue(launch.matching_ui_process(metadata, health["pid"], command, child.pid),
                            json.dumps({"metadata": metadata, "expected": command, "base": sys._base_executable}))
            self.launcher.ensure_ui()
            new_health = self.launcher.existing_ui()
            self.assertNotEqual(new_health["pid"], child.pid)
            self.assertTrue(self.launcher.current_ui(new_health))
            self.assertEqual(child.wait(timeout=5), 0)
            self.assertEqual(Path(new_health["data_dir"]), self.launcher.state_dir)
            saved = launch.read_json(self.launcher.state_dir / "ui-process.json")
            self.assertEqual(saved["pid"], new_health["pid"])
            self.assertTrue(saved["started"])
            self.launcher.ensure_ui()
            self.assertEqual(self.launcher.existing_ui()["pid"], new_health["pid"])
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            child.stderr.close()
            if new_health is not None:
                saved = launch.read_json(self.launcher.state_dir / "ui-process.json")
                launch.stop_owned_ui(saved["pid"], saved["command"], saved["started"], self.launcher.port)
                self.launcher.ui_process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
