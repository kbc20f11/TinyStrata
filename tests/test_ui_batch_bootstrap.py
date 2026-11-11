"""Provided Windows BAT/PowerShell entrypoint, without a UI/model service.

Private Python fixtures record arguments and exit. PATH is isolated so success
cannot be hidden by a globally installed Python. No persistent settings change.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "nt", "Windows provided BAT/PowerShell bootstrap")
class UiBatchBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_temp = tempfile.TemporaryDirectory(prefix="tinystrata-ui-batch-seed-")
        cls.seed = Path(cls.seed_temp.name) / "seed"
        venv.EnvBuilder(with_pip=False).create(cls.seed)

    @classmethod
    def tearDownClass(cls):
        cls.seed_temp.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-ui-batch-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "日本語のアプリ 𠮷🌱"
        (self.source / "ui").mkdir(parents=True)
        self.custom_work = self.base / "別の保存先 データ𠮷🌱"
        self.default_work = self.source.with_name(self.source.name + "-work")
        self.marker = self.base / "child-marker.json"
        for src, dst in ((ROOT / "run-ui.bat", self.source / "run-ui.bat"),
                         (ROOT / "ui/start.ps1", self.source / "ui/start.ps1")):
            shutil.copy2(src, dst)
        (self.source / "ui/launch.py").write_text(
            "import json,os,sys\nfrom pathlib import Path\n"
            "sys.stdout.reconfigure(encoding='utf-8')\nsys.stderr.reconfigure(encoding='utf-8')\n"
            "Path(os.environ['BOOTSTRAP_MARKER']).write_text(json.dumps({'args':sys.argv[1:],'python':sys.executable,'cwd':os.getcwd()},ensure_ascii=False),encoding='utf-8')\n"
            "print('起動記録: 日本語𠮷🌱',flush=True)\n"
            "print('診断記録: 日本語𠮷🌱',file=sys.stderr,flush=True)\n"
            "raise SystemExit(int(os.environ.get('BOOTSTRAP_EXIT','0')))\n",
            encoding="utf-8", newline="\n")
        self.powershell = Path(os.environ["WINDIR"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        self.env = dict(os.environ, BOOTSTRAP_MARKER=str(self.marker),
                        PATH=str(self.powershell.parent) + os.pathsep + str(Path(os.environ["WINDIR"]) / "System32"))

    def interpreter(self, directory):
        scripts = directory / "Scripts"
        scripts.mkdir(parents=True)
        shutil.copy2(self.seed / "Scripts/python.exe", scripts / "python.exe")
        shutil.copy2(self.seed / "pyvenv.cfg", directory / "pyvenv.cfg")
        return scripts / "python.exe"

    def pointer(self, value):
        path = self.default_work / "runtime/ui/location.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def invoke(self, arguments=None, *, env=None):
        arguments = ["--configure-only"] if arguments is None else arguments
        command = '"' + os.environ["COMSPEC"] + '" /d /s /c "' + subprocess.list2cmdline([str(self.source / "run-ui.bat"), *arguments]) + '"'
        # Failed BAT runs pause for a real double-click user; one newline avoids
        # any wait in this owned, hidden test process.
        return subprocess.run(command, cwd=self.base, env=env or self.env, input=b"\n",
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              creationflags=subprocess.CREATE_NO_WINDOW, timeout=25)

    def successful(self, expected_python, arguments=None):
        result = self.invoke(arguments)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertTrue(self.marker.is_file(), result.stdout.decode("utf-8", "replace"))
        marker = json.loads(self.marker.read_text(encoding="utf-8"))
        self.assertEqual(Path(marker["python"]).resolve(), expected_python.resolve())
        self.assertEqual(Path(marker["cwd"]), self.source)
        self.assertEqual(marker["args"], ["--configure-only"] if arguments is None else arguments)
        self.assertIn("起動記録: 日本語𠮷🌱", result.stdout.decode("utf-8"))
        self.assertIn("診断記録: 日本語𠮷🌱", result.stderr.decode("utf-8"))
        return marker

    def test_utf8_custom_pointer_keeps_unicode_source_and_private_python(self):
        python = self.interpreter(self.custom_work / "runtime/.venv")
        self.pointer({"source_dir": str(self.source), "work_dir": str(self.custom_work)})
        self.successful(python)

    def test_empty_quotes_and_trailing_backslashes_reach_python_exactly(self):
        python = self.interpreter(self.custom_work / "runtime/.venv")
        self.pointer({"source_dir": str(self.source), "work_dir": str(self.custom_work)})
        slash = chr(92)
        arguments = ["--configure-only", "", "space text", "C:" + slash + "tail" + slash,
                     "C:" + slash + "space tail" + slash * 2, 'quote"inside',
                     "back" + slash * 3 + '"quote', "日本語 𠮷🌱"]
        self.successful(python, arguments)

    def test_missing_pointer_uses_sibling_private_python(self):
        python = self.interpreter(self.default_work / "runtime/.venv")
        self.successful(python)

    def test_foreign_pointer_does_not_change_selected_source(self):
        python = self.interpreter(self.default_work / "runtime/.venv")
        self.pointer({"source_dir": str(self.base / "別アプリ"), "work_dir": str(self.custom_work)})
        self.successful(python)

    def test_malformed_pointer_retains_sibling_private_python(self):
        python = self.interpreter(self.default_work / "runtime/.venv")
        self.pointer("{broken JSON")
        self.successful(python)

    def test_source_venv_is_second_fallback(self):
        python = self.interpreter(self.source / ".venv")
        self.successful(python)

    def test_path_python_is_last_fallback(self):
        python = self.seed / "Scripts/python.exe"
        self.env["PATH"] = str(python.parent) + os.pathsep + self.env["PATH"]
        self.successful(python)

    def test_missing_python_reports_error_and_nonzero_exit(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.marker.exists())
        self.assertIn("Python was not found", result.stderr.decode("utf-8"))

    def test_child_exit_code_is_not_hidden_by_pause(self):
        self.interpreter(self.custom_work / "runtime/.venv")
        self.pointer({"source_dir": str(self.source), "work_dir": str(self.custom_work)})
        result = self.invoke(env=dict(self.env, BOOTSTRAP_EXIT="13"))
        self.assertEqual(result.returncode, 13)
        self.assertTrue(self.marker.exists())

    def test_missing_launcher_keeps_unicode_diagnostic_and_nonzero_exit(self):
        self.interpreter(self.default_work / "runtime/.venv")
        launcher = self.source / "ui/launch.py"
        launcher.unlink()
        result = self.invoke()
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.marker.exists())
        self.assertIn(str(launcher), result.stderr.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
