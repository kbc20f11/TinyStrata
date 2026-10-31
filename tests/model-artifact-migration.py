"""CPU-only managed-model migration regressions.

Run from any directory: python tests/model-artifact-migration.py
All model files are tiny temporary fixtures; no download, engine or GPU starts.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run as launcher
from serve import backend, model_download as downloads, model_profiles, model_setup as setup


class ManagedCloverMigration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-artifact-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "model-files"
        self.directory.mkdir()
        self.path = self.root / "config-clover.json"
        self.old_bytes = b"old pinned model"
        self.old = downloads._Manifest("fixture/old", "old-revision", "Clover-fixture-Q4_0.gguf",
            len(self.old_bytes), hashlib.sha256(self.old_bytes).hexdigest(), "https://example.invalid/old")
        self.new = downloads._Manifest("fixture/new", "new-revision", "Clover-fixture-IQ3_XXS.gguf",
            3, hashlib.sha256(b"new").hexdigest(), "https://example.invalid/new")
        self.old_path = self.directory / self.old.filename
        self.new_path = self.directory / self.new.filename
        self.cfg = {"arch":"clover_moe", "cwd":str(self.root), "backend":"cuda", "vision_mode":"no",
                    "args":["--model", str(self.old_path), "--clover-mode", "causal", "--clover-draft", "0"],
                    "tokenizer":str(self.root / "old-tokenizer")}
        for name, value in (("_LEGACY_CLOVER_MODEL", self.old), ("FIXED_CLOVER_MODEL", self.new)):
            replacement = patch.object(downloads, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        setup._artifact_digest.cache_clear()
        self.addCleanup(setup._artifact_digest.cache_clear)

    def migrate(self, cfg=None):
        return setup.configure_download(copy.deepcopy(self.cfg if cfg is None else cfg), self.path)

    def write_old(self):
        self.old_path.write_bytes(self.old_bytes)

    def profile(self):
        self.path.write_text(json.dumps(self.cfg), encoding="utf-8")
        qwen = {"arch":"qwen35moe", "cwd":str(self.root), "backend":"cuda", "vision_mode":"no",
                "args":["--model", str(self.root / "qwen.gguf")]}
        with patch.object(setup, "resolve_executable", return_value=None), \
             patch.object(setup, "shared_library_dirs", side_effect=lambda cfg:cfg), \
             patch.object(backend, "configure_backend", side_effect=lambda cfg, *a, **kw:cfg):
            return model_profiles.ModelProfiles(self.root / "config.json", qwen)

    def test_fresh_profile_uses_new_managed_artifact(self):
        cfg = {"arch":"clover_moe", "args":[]}
        result = self.migrate(cfg)
        self.assertEqual(setup.model_path(result), self.new_path)
        self.assertEqual(result["model_source"], "managed")
        self.assertEqual(cfg["args"], [])

    def test_verified_old_managed_default_moves_without_deletion(self):
        self.write_old()
        result = self.migrate()
        self.assertEqual(setup.model_path(result), self.new_path)
        self.assertEqual(result["model_source"], "managed")
        self.assertTrue(result["model_bootstrap"])
        self.assertNotIn("tokenizer", result)
        self.assertEqual(self.old_path.read_bytes(), self.old_bytes)
        self.assertEqual(setup.model_path(self.cfg), self.old_path)
        self.assertIn("tokenizer", self.cfg)

    def test_missing_old_target_moves_and_preserves_partial(self):
        partial = self.old_path.with_name(self.old_path.name + ".part")
        partial.write_bytes(b"partial")
        result = self.migrate()
        self.assertEqual(setup.model_path(result), self.new_path)
        self.assertEqual(partial.read_bytes(), b"partial")

    def test_existing_new_target_preferred_over_old(self):
        self.write_old()
        self.new_path.write_bytes(b"new")
        result = self.migrate()
        self.assertEqual(setup.model_path(result), self.new_path)
        self.assertNotIn("model_bootstrap", result)

    def test_explicit_local_at_old_managed_path_is_preserved(self):
        self.write_old()
        cfg = {**self.cfg, "model_source":"local"}
        with patch.object(setup, "_artifact_digest", side_effect=AssertionError("local files must not be hashed")):
            result = self.migrate(cfg)
        self.assertEqual(setup.model_path(result), self.old_path)
        self.assertEqual(result["model_source"], "local")

    def test_same_filename_outside_managed_directory_is_preserved(self):
        external = self.root / "custom" / self.old.filename
        external.parent.mkdir()
        external.write_bytes(self.old_bytes)
        cfg = copy.deepcopy(self.cfg)
        cfg["args"][1] = str(external)
        with patch.object(setup, "_artifact_digest", side_effect=AssertionError("custom files must not be hashed")):
            self.assertEqual(setup.model_path(self.migrate(cfg)), external)

    def test_custom_filename_inside_managed_directory_is_preserved(self):
        custom = self.directory / "my-custom-clover.gguf"
        custom.write_bytes(self.old_bytes)
        cfg = copy.deepcopy(self.cfg)
        cfg["args"][1] = str(custom)
        self.assertEqual(setup.model_path(self.migrate(cfg)), custom)

    def test_wrong_size_at_old_path_is_preserved(self):
        self.old_path.write_bytes(b"different")
        with patch.object(setup, "_artifact_digest", side_effect=AssertionError("wrong-size files must not be hashed")):
            self.assertEqual(setup.model_path(self.migrate()), self.old_path)

    def test_wrong_hash_at_old_path_is_preserved(self):
        self.old_path.write_bytes(b"x" * len(self.old_bytes))
        self.assertEqual(setup.model_path(self.migrate()), self.old_path)

    def test_unreadable_old_file_is_preserved(self):
        self.write_old()
        with patch.object(setup, "_artifact_digest", side_effect=PermissionError("fixture")):
            self.assertEqual(setup.model_path(self.migrate()), self.old_path)

    def test_directory_at_old_path_is_preserved(self):
        self.old_path.mkdir()
        self.assertEqual(setup.model_path(self.migrate()), self.old_path)

    def test_file_modified_during_verification_is_not_adopted(self):
        self.write_old()
        stamp = self.old_path.stat().st_mtime_ns
        def replaced(*args):
            os.utime(self.old_path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
            return self.old.sha256
        with patch.object(setup, "_artifact_digest", side_effect=replaced):
            self.assertEqual(setup.model_path(self.migrate()), self.old_path)

    def test_relative_managed_path_resolves_against_cwd(self):
        self.write_old()
        cfg = copy.deepcopy(self.cfg)
        cfg["args"][1] = str(Path("model-files") / self.old.filename)
        self.assertEqual(setup.model_path(self.migrate(cfg)), self.new_path)

    def test_repeat_migration_is_idempotent(self):
        self.write_old()
        first = self.migrate()
        self.assertEqual(self.migrate(first), first)

    def test_other_families_are_not_migrated(self):
        self.write_old()
        for arch in ("qwen35moe", "lfm2moe", "gemma4"):
            with self.subTest(arch=arch):
                result = self.migrate({**self.cfg, "arch":arch})
                self.assertEqual(setup.model_path(result), self.old_path)
                self.assertNotIn("model_source", result)

    def test_profile_does_not_reselect_existing_old_default(self):
        self.write_old()
        profiles = self.profile()
        clover = profiles.get("clover")
        self.assertEqual(setup.model_path(clover.config), self.new_path)
        catalog = next(row for row in profiles.list() if row["arch"] == "clover_moe")
        self.assertFalse(catalog["installed"])
        self.assertEqual(Path(catalog["path"]), self.new_path)

    def test_profile_uses_new_default_when_both_exist(self):
        self.write_old()
        self.new_path.write_bytes(b"new")
        profiles = self.profile()
        self.assertEqual(setup.model_path(profiles.get("clover").config), self.new_path)
        self.assertTrue(next(row for row in profiles.list() if row["arch"] == "clover_moe")["installed"])

    def test_profile_preserves_explicit_local_selection(self):
        self.write_old()
        self.cfg["model_source"] = "local"
        profiles = self.profile()
        self.assertEqual(setup.model_path(profiles.get("clover").config), self.old_path)

    def test_missing_local_selection_does_not_fall_back_to_new_default(self):
        self.new_path.write_bytes(b"new")
        self.cfg["model_source"] = "local"
        profiles = self.profile()
        self.assertEqual(setup.model_path(profiles.get("clover").config), self.old_path)
        self.assertTrue(profiles.get("clover").config["model_bootstrap"])
        self.assertFalse(next(row for row in profiles.list() if row["arch"] == "clover_moe")["installed"])

    def test_missing_custom_path_does_not_fall_back_to_new_default(self):
        self.new_path.write_bytes(b"new")
        custom = self.root / "my-custom-clover.gguf"
        self.cfg["args"][1] = str(custom)
        profiles = self.profile()
        self.assertEqual(setup.model_path(profiles.get("clover").config), custom)
        self.assertFalse(next(row for row in profiles.list() if row["arch"] == "clover_moe")["installed"])

    def test_launcher_explicit_model_always_wins(self):
        self.write_old()
        self.path.write_text(json.dumps(self.cfg), encoding="utf-8")
        args = SimpleNamespace(model=self.old_path, model_family="clover", config=self.path, backend=None,
            exe=None, kv_type=None, lib_dir=None, port=None, host=None, api_key=None, gpu=None)
        with patch.object(setup, "model_architecture", return_value="clover_moe"), \
             patch.object(setup, "resolve_executable", return_value=None), \
             patch.object(setup, "shared_library_dirs", side_effect=lambda cfg:cfg), \
             patch.object(setup, "prepare_installed_model", side_effect=lambda cfg, *a, **kw:cfg), \
             patch.object(backend, "configure_backend", side_effect=lambda cfg, *a, **kw:cfg), \
             patch.object(backend, "executable_backend", return_value=None), \
             patch.object(backend, "hardware_backend", side_effect=AssertionError("no GPU probing")):
            result, _ = launcher.prepare(args)
        self.assertEqual(setup.model_path(result), self.old_path)
        self.assertEqual(result["model_source"], "local")


if __name__ == "__main__":
    unittest.main(verbosity=2)
