"""Unicode Python entrypoints and real child handoff; CPU only, no services.

Native/model behavior is verified separately with actual models. These cases
cover the Python side without changing the process-wide ANSI code page.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from serve.stdio import configure_utf8_stdio
from serve import config, server
from ui.stdio import configure_utf8_stdio as configure_ui_utf8_stdio

HELPERS = (configure_utf8_stdio, configure_ui_utf8_stdio)


class Utf8Diagnostics(unittest.TestCase):
    def test_valid_unicode_is_kept_and_other_stream_options_are_preserved(self):
        for helper in HELPERS:
            with self.subTest(helper=helper.__module__):
                raw = io.BytesIO()
                stream = io.TextIOWrapper(raw, encoding="cp932", errors="strict", line_buffering=True, write_through=True)
                try:
                    helper([stream])
                    print("設定とモデル: 日本語 𠮷🌱.gguf / 読めない名前 �", file=stream)
                    stream.flush()
                    self.assertEqual(raw.getvalue().decode("utf-8"), "設定とモデル: 日本語 𠮷🌱.gguf / 読めない名前 �" + os.linesep)
                    self.assertTrue(stream.line_buffering)
                    self.assertTrue(stream.write_through)
                finally:
                    stream.detach()

    def test_diagnostic_lone_surrogate_is_visible_instead_of_raising(self):
        for helper in HELPERS:
            with self.subTest(helper=helper.__module__):
                raw = io.BytesIO()
                stream = io.TextIOWrapper(raw, encoding="cp932")
                try:
                    helper([stream])
                    stream.write("未対応の名前\udcff🌱")
                    stream.flush()
                    self.assertEqual(raw.getvalue().decode("utf-8"), "未対応の名前\\udcff🌱")
                finally:
                    stream.detach()

    def test_none_memory_and_closed_or_disconnected_streams_do_not_block_startup(self):
        memory = io.StringIO()
        closed = io.TextIOWrapper(io.BytesIO(), encoding="cp932")
        closed.close()

        class Disconnected:
            def reconfigure(self, **_kwargs):
                raise BrokenPipeError("diagnostic consumer exited")

        for helper in HELPERS:
            helper([None, memory, closed, Disconnected()])
        memory.write("日本語🌱")
        self.assertEqual(memory.getvalue(), "日本語🌱")


class UnicodeChildEntrypoints(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-unicode-python-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name) / "日本語の作業 𠮷🌱"
        self.base.mkdir()

    def child(self, code, *arguments):
        env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="cp932", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-X", "utf8=0", "-B", "-c", code,
                                 str(ROOT), *map(str, arguments)], cwd=self.base, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        return result.stdout.decode("utf-8")

    def test_importing_modules_does_not_mutate_host_streams(self):
        output = self.child(
            "import importlib,json,sys\n"
            "sys.path.insert(0,sys.argv[1])\n"
            "before=[sys.stdout.encoding,sys.stderr.encoding,sys.stdout.errors,sys.stderr.errors]\n"
            "for name in ('run','serve.server','ui.launch','ui.server','serve.stdio','ui.stdio'): importlib.import_module(name)\n"
            "after=[sys.stdout.encoding,sys.stderr.encoding,sys.stdout.errors,sys.stderr.errors]\n"
            "print(json.dumps({'before':before,'after':after}))\n")
        data = json.loads(output)
        self.assertEqual(data["before"], data["after"])
        self.assertEqual(data["before"][0], "cp932")

    def test_four_entrypoints_configure_only_their_diagnostics_before_argument_processing(self):
        code = (
            "import importlib,json,sys\nfrom unittest import mock\n"
            "sys.path.insert(0,sys.argv[1])\nmodule=importlib.import_module(sys.argv[2])\n"
            "class StopBeforeStartup(Exception): pass\n"
            "with mock.patch.object(module.argparse,'ArgumentParser',side_effect=StopBeforeStartup):\n"
            " try: module.main()\n except StopBeforeStartup: pass\n"
            "print(json.dumps({'stdout':sys.stdout.encoding,'stderr':sys.stderr.encoding,"
            "'errors':sys.stdout.errors,'path':sys.argv[3]},ensure_ascii=False))\n")
        path = self.base / "モデル model🌱.gguf"
        for name in ("run", "serve.server", "ui.launch", "ui.server"):
            with self.subTest(entrypoint=name):
                result = json.loads(self.child(code, name, path))
                self.assertEqual(result, {"stdout": "utf-8", "stderr": "utf-8", "errors": "backslashreplace", "path": str(path)})

    def test_run_configure_only_prints_unicode_config_in_legacy_environment(self):
        path = self.base / "設定 config🌱.json"
        output = self.child(
            "import pathlib,sys\nfrom unittest import mock\nsys.path.insert(0,sys.argv[1])\nimport run\n"
            "path=pathlib.Path(sys.argv[2]);sys.argv=['run.py','--configure-only']\n"
            "with mock.patch('serve.install_guard.assert_installation_ready'), "
            "mock.patch.object(run.importlib.util,'find_spec',return_value=object()), "
            "mock.patch.object(run,'prepare',return_value=({'port':18080},path)):\n"
            " raise SystemExit(run.main())\n", path)
        self.assertEqual(output.strip(), "TinyStrata config: " + str(path))

    def test_provided_launcher_configure_only_saves_unicode_source_work_and_pointer(self):
        source = self.base / "アプリ source🌱"
        (source / "ui/web").mkdir(parents=True)
        (source / "ui/server.py").write_text("# configure-only fixture\n", encoding="utf-8")
        (source / "ui/web/index.html").write_text("<!doctype html>", encoding="utf-8")
        work = self.base / "データ work🌱"
        env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="cp932", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-X", "utf8=0", "-B", str(ROOT / "ui/launch.py"),
                                 "--source-dir", str(source), "--work-dir", str(work), "--configure-only"],
                                cwd=self.base, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        settings = work / "runtime/ui/settings.json"
        self.assertIn(str(settings), result.stdout.decode("utf-8"))
        value = json.loads(settings.read_text(encoding="utf-8"))
        self.assertEqual(Path(value["source_dir"]), source)
        self.assertEqual(Path(value["work_dir"]), work)
        pointer = source.with_name(source.name + "-work") / "runtime/ui/location.json"
        self.assertEqual(json.loads(pointer.read_text(encoding="utf-8")),
                         {"source_dir": str(source), "work_dir": str(work)})

    def test_ui_addon_does_not_require_the_new_core_stdio_module(self):
        # The optional UI installer adds ui/** to an older API installation;
        # it deliberately does not refresh that API's serve directory.
        source = self.base / "旧APIと追加UI source🌱"
        (source / "ui/web").mkdir(parents=True)
        (source / "serve").mkdir()
        for name in ("launch.py", "stdio.py"):
            shutil.copy2(ROOT / "ui" / name, source / "ui" / name)
        shutil.copy2(ROOT / "serve/install_guard.py", source / "serve/install_guard.py")
        (source / "ui/server.py").write_text("# configure-only fixture\n", encoding="utf-8")
        (source / "ui/web/index.html").write_text("<!doctype html>", encoding="utf-8")
        work = self.base / "追加UI work🌱"
        env = dict(os.environ, PYTHONUTF8="0", PYTHONIOENCODING="cp932", PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([sys.executable, "-X", "utf8=0", "-B", str(source / "ui/launch.py"),
                                 "--source-dir", str(source), "--work-dir", str(work), "--configure-only"],
                                cwd=self.base, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        settings = work / "runtime/ui/settings.json"
        self.assertIn(str(settings), result.stdout.decode("utf-8"))
        self.assertFalse((source / "serve/stdio.py").exists())
        self.assertEqual(json.loads(settings.read_text(encoding="utf-8"))["source_dir"], str(source))

    def test_config_and_actual_worker_handoff_preserve_unicode_argv_cwd_environment(self):
        path = self.base / "設定 config🌱.json"
        arguments = [str(self.base / "日本語 model🌱.gguf"), str(self.base / "画像 projector🌱.gguf"),
                     str(self.base / "思考 draft🌱.gguf"), str(self.base / "tokenizer 𠮷")]
        cfg = {"exe": sys.executable, "args": arguments, "tokenizer": arguments[-1], "log": str(self.base / "記録 log🌱.txt")}
        config.save(path, cfg)
        self.assertEqual(config.load(path), cfg)
        code = "import json,os,sys; print(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd(),'env':os.environ['UNICODE_TEST_PATH']},ensure_ascii=True))"
        env = dict(os.environ, UNICODE_TEST_PATH=str(path))
        process = server.popen("Unicode Python test worker", [sys.executable, "-B", "-c", code, *arguments],
                               cwd=self.base, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8")
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(json.loads(stdout), {"args": arguments, "cwd": str(self.base), "env": str(path)})


if __name__ == "__main__":
    unittest.main()
