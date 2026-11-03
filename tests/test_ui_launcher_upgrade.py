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
