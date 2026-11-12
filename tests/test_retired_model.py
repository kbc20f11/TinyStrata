"""Retirement boundaries and existing-user migration; no inference or GPU calls."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import run
from serve import backend, model_download, model_setup, tinyconfig
from serve.model_profiles import ModelProfiles
from serve.model_switch import ModelControlError
from ui import launch
import test_model_library as library_fixture

ALIASES = ("gpt-oss", "gptoss", "gpt-oss-20b")


class RetirementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tinystrata-retirement-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_registry_and_all_model_control_entrypoints_reject_retired_aliases(self):
        fixture = library_fixture.create_fixture(self.root)
        self.addCleanup(fixture.control.close)
        self.assertEqual(len(fixture.control.catalog()), 5)
        for alias in ALIASES:
            with self.subTest(alias=alias):
                for action in (fixture.profiles.get, fixture.control.start, fixture.control.delete,
                               lambda name: fixture.control.download(name, start=True)):
                    with self.assertRaisesRegex(ValueError, "no longer supported"):
                        action(alias)
                with self.assertRaises(ModelControlError) as raised:
                    fixture.control.check_model({"model": alias})
                self.assertEqual(raised.exception.status, 404)
                self.assertIn("no longer supported", str(raised.exception))
        self.assertFalse(fixture.control._downloads)

    def test_retired_architecture_or_disguised_model_rejects_before_backend_selection(self):
        for cfg in ({"arch": "gpt-oss", "model_name": "custom-api-name"},
                    {"arch": "qwen35moe", "model_name": "gptoss"}):
            path = self.root / "config.json"
            original = json.dumps({**cfg, "args": ["--model", "old.gguf"]}).encode()
            path.write_bytes(original)
            args = argparse.Namespace(model=None, model_family=None, config=path, configure_only=True)
            with patch("serve.backend.hardware_backend", side_effect=AssertionError("No device probe")), \
                 patch("serve.model_setup.resolve_executable", side_effect=AssertionError("No native lookup")):
                with self.assertRaisesRegex(ValueError, "no longer supported"):
                    run.prepare(args)
            self.assertEqual(path.read_bytes(), original)
            for action in (tinyconfig.validate, tinyconfig.migrate_settings, backend.configure_backend,
                           model_setup.configure_download):
                with self.assertRaisesRegex(ValueError, "no longer supported"):
                    action(cfg, path) if action is model_setup.configure_download else action(cfg)

    def test_manifest_and_local_gguf_cannot_restore_retired_support(self):
        for arch, name in (("gpt-oss", None), ("qwen35moe", "gpt-oss-20b")):
            for action in (model_download.manifest_for_arch, model_download.projector_for_arch):
                with self.assertRaisesRegex(ValueError, "no longer supported"):
                    action(arch, name)
        with patch("serve.gguf_reader.GGUFFile", return_value=argparse.Namespace(metadata={"general.architecture": "gpt-oss"})):
            with self.assertRaisesRegex(ValueError, "no longer supported"):
                model_setup.model_architecture(self.root / "old-model.gguf")

    def test_saved_ui_selection_uses_downloaded_supported_model_and_preserves_data(self):
        work = self.root / "work"
        runtime = work / "runtime"
        configs = runtime / ".tinystrata"
        ui = runtime / "ui"
        configs.mkdir(parents=True)
        ui.mkdir()
        weights = configs / "gpt-oss-20b-MXFP4.gguf"
        weights.write_bytes(b"existing retired model weights")
        old = {"arch": "gpt-oss", "model_name": "gpt-oss-20b", "args": ["--model", str(weights)]}
        for name in ("config.json", "config-gpt-oss.json"):
            (configs / name).write_text(json.dumps(old), encoding="utf-8")
        gemma = configs / "gemma.gguf"
        gemma.write_bytes(b"existing supported weights")
        (configs / "config-gemma4.json").write_text(json.dumps({"arch": "gemma4", "model_name": "gemma-4-26b-a4b-it",
            "args": ["--model", str(gemma)]}), encoding="utf-8")
        # A stale/missing Qwen download is not selected over an installed Gemma.
        (configs / "config-qwen36.json").write_text(json.dumps({"arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b",
            "args": ["--model", str(configs / "not-downloaded.gguf")]}), encoding="utf-8")
        (ui / "settings.json").write_text(json.dumps({"model_family": "gptoss"}), encoding="utf-8")
        history = ui / "chat-history.sqlite3"
        with closing(sqlite3.connect(history)) as connection:
            connection.execute("CREATE TABLE history(model TEXT, content TEXT)")
            connection.execute("INSERT INTO history VALUES (?, ?)", ("gpt-oss-20b", "retained conversation"))
            connection.commit()
        preserved = {p: p.read_bytes() for p in (*configs.iterdir(), history)}
        args = argparse.Namespace(api_port=18080, port=3000, model_family=None)
        with patch("subprocess.Popen", side_effect=AssertionError("Migration must not launch/download")):
            launcher = launch.Launcher(args, ROOT, work)
        self.assertEqual(launcher.family, "gemma4")
        self.assertEqual(launch.profile_path(runtime, "qwen36").name, "config-qwen36.json")
        self.assertEqual({p: p.read_bytes() for p in preserved}, preserved)
        self.assertEqual(json.loads((ui / "settings.json").read_text())["model_family"], "gptoss")

    def test_explicit_retired_ui_selection_is_rejected(self):
        for alias in ALIASES:
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, "no longer supported"):
                launch.Launcher(argparse.Namespace(api_port=18080, port=3000, model_family=alias), ROOT, self.root / alias)


class RetirementHttpTests(unittest.TestCase):
    """Use the actual HTTP handler and no-inference fixture, never real services."""
    setUp = library_fixture.LibraryHTTPTests.setUp
    start_http = library_fixture.LibraryHTTPTests.start_http
    http = library_fixture.LibraryHTTPTests.http
    def test_retired_generation_is_rejected_before_load_across_api_dialects(self):
        self.start_http()
        for alias in ALIASES:
            for path, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": "test"}]}),
                               ("/v1/responses", {"input": "test"}),
                               ("/v1/messages", {"messages": [{"role": "user", "content": "test"}], "max_tokens": 16})):
                with self.subTest(alias=alias, path=path), patch.object(self.f.svc, "load", side_effect=AssertionError("No load before rejection")):
                    status, result = self.http(path=path, body={"model": alias, **body})
                self.assertEqual(status, 404, result)
                self.assertIn("no longer supported", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
