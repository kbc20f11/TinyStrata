"""Same-architecture models retain distinct state and pinned download bundles."""
from copy import deepcopy
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import run as launcher
from serve import backend, config, model_download, model_setup, tinyconfig
from serve.model_profiles import ModelProfile, ModelProfiles
from serve.model_switch import ModelControlError, ModelSwitcher
from serve.server import configure_model_downloads, default_model_name, sampling_defaults_from_config


QWEN = "qwen3.6-35b-a3b"
TIEL = "tiel-coder-35b-a3b"


class SameArchitectureProfiles(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-profile-identity-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cfg = {"arch": "qwen35moe", "model_name": QWEN, "backend": "cuda",
                    "args": ["--max-context", "auto", "--kv-type", "auto", "--chunk", "auto", "--mtp", "3"],
                    "cwd": str(self.root), "private_setting": "keep Qwen settings"}
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(self.cfg, indent=3) + "\n", encoding="utf-8")
        self.original = self.path.read_bytes()
        for target in (patch.object(model_setup, "resolve_executable", return_value=None),
                       patch.object(model_setup, "shared_library_dirs", side_effect=lambda cfg: cfg)):
            target.start()
            self.addCleanup(target.stop)

    def profiles(self, cfg=None, path=None):
        return ModelProfiles(path or self.path, self.cfg if cfg is None else cfg)

    def test_aliases_select_a_distinct_model_with_shared_architecture(self):
        for name in (TIEL, "tiel", " TIEL-CODER "):
            self.assertEqual(ModelProfiles.resolve(name), TIEL)
        self.assertNotEqual(ModelProfiles.resolve("qwen35moe"), TIEL)
        profiles = self.profiles()
        self.assertEqual(profiles.current_id, QWEN)
        self.assertEqual(profiles.get("tiel").arch, profiles.get("qwen36").arch)
        self.assertNotEqual(profiles.get("tiel").id, profiles.get("qwen36").id)

    def test_config_artifact_tokenizer_and_shared_settings_are_separate(self):
        profiles = self.profiles()
        qwen, tiel = profiles.get(QWEN), profiles.get(TIEL)
        self.assertEqual(tiel.path, self.root / "config-tiel-coder.json")
        self.assertNotEqual(qwen.path, tiel.path)
        self.assertNotEqual(qwen.shared_path, tiel.shared_path)
        self.assertEqual(model_setup.model_path(qwen.config).name, model_download.FIXED_MODEL.filename)
        self.assertEqual(model_setup.model_path(tiel.config).name, model_download.FIXED_TIEL_MODEL.filename)
        self.assertNotEqual(model_setup.model_path(qwen.config), model_setup.model_path(tiel.config))
        self.assertNotIn("tokenizer", tiel.config)
        self.assertEqual(config.value_of(qwen.config, "mtp"), 3)
        self.assertEqual(config.value_of(tiel.config, "mtp"), 3)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse(tiel.path.exists(), "Catalog construction must not write an inactive profile")

    def test_tiel_restart_does_not_become_qwen(self):
        cfg = {**deepcopy(self.cfg), "model_name": TIEL, "private_setting": "keep Tiel settings"}
        path = self.root / "config-tiel-coder.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        profiles = self.profiles(cfg, path)
        self.assertEqual(profiles.current_id, TIEL)
        self.assertEqual(profiles.get(TIEL).path, path)
        self.assertEqual(profiles.get(TIEL).config["private_setting"], "keep Tiel settings")
        self.assertEqual(profiles.get(QWEN).path, self.path)
        self.assertEqual(profiles.get(QWEN).config["private_setting"], "keep Qwen settings")

    def test_legacy_and_custom_api_name_still_select_qwen(self):
        for name in (None, "my-custom-qwen-api"):
            cfg = deepcopy(self.cfg)
            if name is None:
                cfg.pop("model_name")
            else:
                cfg["model_name"] = name
            profiles = self.profiles(cfg)
            self.assertEqual(profiles.current_id, QWEN)
            if name is not None:
                self.assertEqual(profiles.get(QWEN).config["model_name"], name)

    def test_known_alias_normalizes_on_tiel_restart(self):
        cfg = {**deepcopy(self.cfg), "model_name": "tiel-coder"}
        profiles = self.profiles(cfg, self.root / "custom-tiel.json")
        self.assertEqual(profiles.current_id, TIEL)
        self.assertEqual(profiles.get(TIEL).config["model_name"], TIEL)
        self.assertEqual(model_setup.model_path(profiles.get(TIEL).config).name,
                         model_download.FIXED_TIEL_MODEL.filename)
        self.assertEqual(profiles.get(TIEL).config["sampling"]["temperature"], 1.0)

    def test_known_model_name_with_wrong_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "another model architecture"):
            self.profiles({**self.cfg, "model_name": "gemma-4-26b-a4b-it"})

    def test_same_architecture_wrong_identity_cannot_update_or_prepare(self):
        profiles = self.profiles()
        tiel = profiles.get(TIEL)
        wrong = {**tiel.config, "model_name": QWEN}
        with self.assertRaisesRegex(ValueError, "another model"):
            profiles.update(TIEL, wrong, save=True)
        self.assertFalse(tiel.path.exists())
        tiel.path.write_text(json.dumps(wrong), encoding="utf-8")
        saved = tiel.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "another model"):
            profiles.prepare(TIEL)
        self.assertEqual(tiel.path.read_bytes(), saved)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_occupied_same_architecture_primary_uses_distinct_fallback(self):
        occupied = self.root / "config-tiel-coder.json"
        occupied.write_text(json.dumps(self.cfg), encoding="utf-8")
        before = occupied.read_bytes()
        profiles = self.profiles()
        self.assertEqual(profiles.get(TIEL).path, self.root / "config-tiel-coder-profile.json")
        self.assertEqual(occupied.read_bytes(), before)
        occupied.with_name("config-tiel-coder-profile.json").write_text(json.dumps(self.cfg), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No separate configuration"):
            self.profiles()

    def test_reusing_a_primary_filename_never_aliases_current_config(self):
        cfg = {**self.cfg, "model_name": TIEL}
        profiles = self.profiles(cfg)
        self.assertEqual(profiles.get(TIEL).path, self.path)
        self.assertEqual(profiles.get(QWEN).path, self.root / "config-qwen36.json")
        self.assertEqual(len({p.path for p in profiles._profiles.values()}), 5)

    def test_catalog_availability_checks_only_the_matching_artifact(self):
        directory = self.root / "model-files"
        directory.mkdir()
        (directory / model_download.FIXED_TIEL_MODEL.filename).write_bytes(b"fixture")
        rows = {row["id"]: row for row in self.profiles().list()}
        self.assertEqual(len(rows), 5)
        self.assertTrue(rows[TIEL]["installed"])
        self.assertFalse(rows[QWEN]["installed"])
        self.assertEqual(rows[TIEL]["context"]["model_limit"], 262144)
        self.assertTrue(rows[TIEL]["vision"])

    def test_downloader_uses_model_specific_weight_and_projector(self):
        profiles = self.profiles()
        switcher = ModelSwitcher.__new__(ModelSwitcher)
        switcher.profiles, switcher._downloads = profiles, {}
        with patch.object(model_download, "FixedModelBundleDownloader") as bundle:
            tiel_download = switcher._downloader(TIEL)
            call = bundle.call_args.kwargs
            self.assertEqual(call["_manifests"], (model_download.FIXED_TIEL_MODEL, model_download.FIXED_TIEL_MMPROJ))
            self.assertNotIn(model_download.FIXED_MODEL, call["_manifests"])
            self.assertIs(switcher._downloader(TIEL), tiel_download)
            self.assertEqual(bundle.call_count, 1)
            switcher._downloader(QWEN)
            self.assertEqual(bundle.call_args.kwargs["_manifests"],
                             (model_download.FIXED_MODEL, model_download.FIXED_MMPROJ))

    def test_same_architecture_request_still_requires_explicit_switch(self):
        switcher = ModelSwitcher.__new__(ModelSwitcher)
        switcher._lock, switcher.profiles = threading.RLock(), self.profiles()
        switcher.active_id = QWEN
        switcher.svc = SimpleNamespace(model_names=lambda: [QWEN])
        with self.assertRaises(ModelControlError) as raised:
            switcher.check_model({"model": "tiel"})
        self.assertEqual(raised.exception.status, 409)
        switcher.active_id = TIEL
        switcher.check_model({"model": TIEL})

    def test_initial_api_downloader_uses_tiel_model_and_own_projector(self):
        profile = self.profiles().get(TIEL)
        svc = SimpleNamespace(downloads=None)
        with patch.object(model_download, "FixedModelBundleDownloader") as bundle:
            manager = configure_model_downloads(svc, profile.config, profile.path)
            self.assertIs(manager, svc.downloads)
            self.assertIs(manager, bundle.return_value)
            self.assertEqual(bundle.call_args.kwargs["_manifests"],
                             (model_download.FIXED_TIEL_MODEL, model_download.FIXED_TIEL_MMPROJ))
        with patch.object(model_download, "FixedModelBundleDownloader") as bundle:
            qwen = self.profiles().get(QWEN)
            configure_model_downloads(svc, qwen.config, qwen.path)
            self.assertEqual(bundle.call_args.kwargs["_manifests"],
                             (model_download.FIXED_MODEL, model_download.FIXED_MMPROJ))

    def test_tiel_sampler_defaults_are_shared_by_migration_and_api(self):
        original = {"arch": "qwen35moe", "model_name": TIEL}
        migrated, _ = tinyconfig.migrate_generation_settings(original)
        expected = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0,
                    "frequency_penalty": 0.0, "repetition_penalty": 1.0, "min_p": 0.0}
        self.assertEqual(migrated["sampling"], expected)
        self.assertEqual(sampling_defaults_from_config(original), expected)
        self.assertEqual(sampling_defaults_from_config(migrated), expected)
        self.assertNotIn("sampling", original)

    def test_handwritten_tiel_aliases_keep_artifact_projector_api_and_sampler_identity(self):
        for name in ("tiel", " TIEL-CODER ", TIEL):
            with self.subTest(model_name=name):
                cfg = {"arch": "qwen35moe", "model_name": name}
                self.assertEqual(model_download.manifest_for_arch("qwen35moe", name), model_download.FIXED_TIEL_MODEL)
                self.assertEqual(model_download.projector_for_arch("qwen35moe", name), model_download.FIXED_TIEL_MMPROJ)
                self.assertEqual(default_model_name(cfg), TIEL)
                migrated, _ = tinyconfig.migrate_generation_settings(cfg)
                self.assertEqual(migrated["sampling"], tinyconfig.TIEL_SAMPLING)
                self.assertEqual(sampling_defaults_from_config(cfg), tinyconfig.TIEL_SAMPLING)
                greedy, _ = tinyconfig.migrate_generation_settings({**cfg, "sampling": {"temperature": 0}})
                self.assertEqual(greedy["sampling"]["temperature"], 0)

    def test_saved_tiel_greedy_choice_is_not_migrated_as_legacy_qwen(self):
        original = {"arch": "qwen35moe", "model_name": TIEL, "sampling": {"temperature": 0}}
        migrated, _ = tinyconfig.migrate_generation_settings(original)
        self.assertEqual(migrated["sampling"]["temperature"], 0)
        self.assertEqual(sampling_defaults_from_config(original)["temperature"], 0)
        self.assertEqual(sampling_defaults_from_config(migrated)["temperature"], 0)
        self.assertEqual(migrated["sampling"]["presence_penalty"], 0)
        legacy_qwen, _ = tinyconfig.migrate_generation_settings({"arch": "qwen35moe", "sampling": {"temperature": 0}})
        self.assertEqual(legacy_qwen["sampling"]["temperature"], tinyconfig.DEFAULT_SAMPLING["temperature"])

    def launch(self, family="tiel", path=None):
        args = SimpleNamespace(model_family=family, config=path, model=None, exe=None, backend="cuda",
                               kv_type=None, lib_dir=None, port=None, host=None, api_key=None, gpu=None)
        with patch.object(launcher, "RUNTIME", self.root / "runtime"), \
             patch.object(launcher, "ROOT", self.root), \
             patch.object(backend, "configure_backend", side_effect=lambda cfg, *a, **kw: {**cfg, "backend": "cuda"}), \
             patch.object(backend, "executable_backend", return_value=None), \
             contextlib.redirect_stdout(io.StringIO()):
            return launcher.prepare(args)

    def test_launcher_tiel_family_saves_separate_config_log_port(self):
        cfg, path = self.launch()
        self.assertEqual(path, self.root / "runtime/.tinystrata/config-tiel-coder.json")
        self.assertTrue(path.is_file())
        self.assertEqual(cfg["model_name"], TIEL)
        self.assertEqual(cfg["arch"], "qwen35moe")
        self.assertEqual(cfg["port"], 18084)
        self.assertEqual(Path(cfg["log"]).name, "engine-tiel-coder.log")
        self.assertEqual(config.value_of(cfg, "mtp"), 3)
        self.assertEqual(model_setup.model_path(cfg).name, model_download.FIXED_TIEL_MODEL.filename)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_launcher_rejects_existing_other_model_even_with_shared_arch(self):
        with self.assertRaisesRegex(ValueError, "another model"):
            self.launch(path=self.path)
        self.assertEqual(self.path.read_bytes(), self.original)
        _, tiel_path = self.launch()
        before = tiel_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "another model"):
            self.launch("qwen36", tiel_path)
        self.assertEqual(tiel_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
