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

    def test_qwen_official_thinking_sampler_defaults_reach_profiles_and_api(self):
        original = {"arch": "qwen35moe", "model_name": QWEN}
        expected = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 1.5, "repetition_penalty": 1.0}
        migrated, _ = tinyconfig.migrate_generation_settings(original)
        self.assertEqual(migrated["sampling"], expected)
        self.assertTrue(migrated["enable_thinking"])
        self.assertEqual(sampling_defaults_from_config(original), expected)
        self.assertEqual(sampling_defaults_from_config(migrated), expected)
        self.assertEqual(self.profiles().get(QWEN).config["sampling"], expected)
        self.assertNotIn("sampling", original)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_saved_qwen_sampling_choices_are_not_reclassified_as_defaults(self):
        # Old stock 0.7 and a deliberate choice of the same numbers cannot be
        # distinguished in a saved profile. Preserve both rather than guessing.
        old_stock = {**tinyconfig.DEFAULT_SAMPLING, "temperature": 0.7}
        choices = (old_stock, {"temperature": 0},
                   {"temperature": 0.6, "top_p": 0.8, "presence_penalty": 0.2},
                   {"top_k": 50, "repeat_penalty": 1.1})
        for choice in choices:
            with self.subTest(sampling=choice):
                cfg = {"arch": "qwen35moe", "model_name": QWEN,
                       "generation_policy_version": tinyconfig.GENERATION_POLICY_VERSION,
                       "sampling": deepcopy(choice), "enable_thinking": False,
                       "max_tokens": 2048, "args": ["--kv-type", "fp32"]}
                migrated, _ = tinyconfig.migrate_generation_settings(cfg)
                for key, value in choice.items():
                    self.assertEqual(migrated["sampling"][key], value)
                effective = sampling_defaults_from_config(migrated)
                for key, value in choice.items():
                    self.assertEqual(effective["repetition_penalty" if key == "repeat_penalty" else key], value)
                self.assertFalse(migrated["enable_thinking"])
                self.assertEqual(migrated["max_tokens"], 2048)
                self.assertEqual(migrated["args"], cfg["args"])
                again, changed = tinyconfig.migrate_generation_settings(migrated)
                self.assertFalse(changed)
                self.assertEqual(again, migrated)

    def test_qwen_default_change_preserves_other_model_sampler_contracts(self):
        for arch, model, expected in (
                ("gemma4", "gemma-4-26b-a4b-it", tinyconfig.GEMMA4_SAMPLING),
                ("lfm2moe", "lfm2.5-24b-a2b", tinyconfig.LFM2_SAMPLING),
                ("clover_moe", "clover-35b-a3b-preview", tinyconfig.CLOVER_SAMPLING),
                ("qwen35moe", TIEL, tinyconfig.TIEL_SAMPLING)):
            with self.subTest(model=model):
                cfg = {"arch": arch, "model_name": model}
                migrated, _ = tinyconfig.migrate_generation_settings(cfg)
                self.assertEqual(migrated["sampling"], expected)
                self.assertEqual(sampling_defaults_from_config(cfg), expected)

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


class ManagedNativeGenerations(unittest.TestCase):
    """Real saved profiles and generation folders; native programs never run."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-native-profiles-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.work = self.root / "work"
        self.settings = self.work / "runtime/.tinystrata"
        self.settings.mkdir(parents=True)
        self.old = self.work / "native/cuda-old"
        self.new = self.work / "native/cuda-current"
        self.engines(self.old)
        self.engines(self.new)
        self.model = self.root / "selected-model.gguf"
        self.model.write_bytes(b"model-path-only fixture")
        self.current = {"arch": "qwen35moe", "model_name": QWEN, "backend": "cuda", "gpu": 0,
                        "exe": str(self.new / "tinystrata.exe"), "cwd": str(self.root),
                        "lib_dirs": [str(self.new), "current-sdk"],
                        "args": ["--max-context", "auto", "--kv-type", "auto", "--mtp", "3"]}
        self.saved = {"arch": "gemma4", "model_name": "gemma-4-26b-a4b-it", "backend": "cuda", "gpu": 2,
                      "exe": str(self.old / "tinystrata-gemma4.exe"), "cwd": str(self.root),
                      "lib_dirs": [str(self.old), "old-sdk"], "tokenizer": str(self.root / "chosen-tokenizer"),
                      "args": ["--model", str(self.model), "--max-context", "65536", "--kv-type", "fp16",
                               "--chunk", "768", "--mtp", "1", "--expert-slots", "42"],
                      "memory_policy_version": tinyconfig.MEMORY_POLICY_VERSION,
                      "chunk_policy_version": tinyconfig.CHUNK_POLICY_VERSION,
                      "sampling": {**tinyconfig.GEMMA4_SAMPLING, "temperature": .63, "top_k": 41, "seed": 777},
                      "enable_thinking": False, "max_tokens": 1234, "private_setting": "preserve",
                      "vision_exe": str(self.old / "tinystrata-vision.exe"),
                      "vision": {"exe": str(self.old / "tinystrata-vision.exe"), "model": str(self.model),
                                 "mmproj": str(self.root / "chosen-projector.gguf"), "gpu": True, "max_tokens": 768}}
        self.path = self.settings / "config.json"
        self.gemma = self.settings / "config-gemma4.json"

    @staticmethod
    def engines(directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("tinystrata.exe", "tinystrata-gemma4.exe", "tinystrata-lfm2.exe", "tinystrata-vision.exe"):
            (directory / name).write_bytes(b"CPU-only executable path fixture")

    def profiles(self, path=None):
        path = path or self.path
        path.write_text(json.dumps(self.current), encoding="utf-8")
        selected = path.parent / "config-gemma4.json"
        selected.write_text(json.dumps(self.saved, indent=2), encoding="utf-8")
        before = {file: file.read_bytes() for file in path.parent.iterdir() if file.is_file()}
        with patch.object(model_setup, "resolve_executable", return_value=Path(self.current["exe"])), \
             patch.object(model_setup, "shared_library_dirs", side_effect=lambda value: value), \
             patch("subprocess.Popen", side_effect=AssertionError("Profile reads must not start a native process")):
            profiles = ModelProfiles(path, self.current)
        self.assertEqual({file: file.read_bytes() for file in before}, before, "Catalog reads must not rewrite profiles")
        self.assertEqual({file for file in path.parent.iterdir() if file.is_file()}, set(before))
        return profiles

    def test_inactive_cuda_uses_updated_generation_without_changing_model_preferences(self):
        profiles = self.profiles()
        value = profiles.get("gemma4").config
        self.assertEqual(value["exe"], str(self.new / "tinystrata-gemma4.exe"))
        self.assertEqual(value["vision_exe"], str(self.new / "tinystrata-vision.exe"))
        self.assertEqual(value["vision"], {**self.saved["vision"], "exe": str(self.new / "tinystrata-vision.exe")})
        self.assertEqual(value["lib_dirs"], self.current["lib_dirs"])
        self.assertEqual(value["gpu"], self.current["gpu"])
        for field in ("args", "tokenizer", "sampling", "enable_thinking", "max_tokens", "private_setting"):
            self.assertEqual(value[field], self.saved[field], field)
        before = self.path.read_bytes()
        profiles.update("gemma4", value, save=True)
        self.assertEqual(config.load(self.gemma)["exe"], str(self.new / "tinystrata-gemma4.exe"))
        self.assertEqual(self.path.read_bytes(), before, "Saving the selected inactive profile must not rewrite the active one")

    def test_managed_hip_keeps_shared_generation_and_runtime_alignment(self):
        self.current.update(backend="hip", hip_ordinal=1)
        self.saved.update(backend="hip", hip_ordinal=0)
        self.saved["args"][self.saved["args"].index("--kv-type") + 1] = "int8"
        value = self.profiles().get("gemma4").config
        self.assertEqual(value["exe"], str(self.new / "tinystrata-gemma4.exe"))
        self.assertEqual(value["lib_dirs"], self.current["lib_dirs"])
        self.assertEqual(value["hip_ordinal"], 1)
        self.assertEqual(config.value_of(value, "kv_type"), "int8")

    def test_existing_backend_change_migration_still_selects_the_current_sibling(self):
        self.current.update(backend="hip", hip_ordinal=1)
        value = self.profiles().get("gemma4").config
        self.assertEqual(value["backend"], "hip")
        self.assertEqual(value["exe"], str(self.new / "tinystrata-gemma4.exe"))
        self.assertEqual(config.value_of(value, "kv_type"), "auto")
        self.assertEqual(value["sampling"], self.saved["sampling"])

    def test_custom_or_renamed_executables_are_not_replaced_on_the_same_backend(self):
        for executable in (self.root / "custom/tinystrata-gemma4.exe", self.old / "my-gemma.exe",
                           self.work / "custom/tinystrata-gemma4.exe",
                           self.root / "work-other/native/old/tinystrata-gemma4.exe",
                           self.work / "build/custom/tinystrata-gemma4.exe"):
            with self.subTest(executable=executable):
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.write_bytes(b"custom")
                self.saved["exe"] = str(executable)
                value = self.profiles().get("gemma4").config
                self.assertEqual(value["exe"], str(executable))
                self.assertEqual(value["lib_dirs"], self.saved["lib_dirs"])
                self.assertEqual(value["vision_exe"], self.saved["vision_exe"])

    def test_custom_vision_is_preserved_when_managed_language_engine_updates(self):
        custom = self.root / "custom/encoder.exe"
        for nested in (False, True):
            with self.subTest(nested=nested):
                self.saved["vision"]["exe"] = str(custom)
                if nested:
                    self.saved.pop("vision_exe", None)
                else:
                    self.saved["vision_exe"] = str(custom)
                value = self.profiles().get("gemma4").config
                self.assertEqual(value["exe"], str(self.new / "tinystrata-gemma4.exe"))
                self.assertEqual(value.get("vision_exe"), self.saved.get("vision_exe"))
                self.assertEqual(value["vision"]["exe"], str(custom))

    def test_custom_active_engine_cannot_retarget_inactive_managed_profiles(self):
        custom = self.root / "custom"
        self.engines(custom)
        self.current["exe"] = str(custom / "tinystrata.exe")
        value = self.profiles().get("gemma4").config
        self.assertEqual(value["exe"], self.saved["exe"])
        self.assertEqual(value["lib_dirs"], self.saved["lib_dirs"])

    def test_recognized_legacy_builds_and_missing_old_generations_can_update(self):
        for directory in (self.work / "build/Release", self.work / "build-cuda-132"):
            with self.subTest(directory=directory):
                self.saved["exe"] = str(directory / "tinystrata-gemma4.exe")
                value = self.profiles().get("gemma4").config
                self.assertEqual(value["exe"], str(self.new / "tinystrata-gemma4.exe"))
        legacy = self.work / "build-cuda/Release"
        self.engines(legacy)
        self.current["exe"] = str(legacy / "tinystrata.exe")
        self.saved["exe"] = str(self.old / "tinystrata-gemma4.exe")
        self.assertEqual(self.profiles().get("gemma4").config["exe"], str(legacy / "tinystrata-gemma4.exe"))

    def test_missing_sibling_or_nonmanaged_config_does_not_partially_retarget(self):
        (self.new / "tinystrata-gemma4.exe").unlink()
        value = self.profiles().get("gemma4").config
        self.assertEqual(value["exe"], self.saved["exe"])
        self.assertEqual(value["lib_dirs"], self.saved["lib_dirs"])
        self.engines(self.new)
        outside = self.root / "custom-config"
        outside.mkdir()
        value = self.profiles(outside / "config.json").get("gemma4").config
        self.assertEqual(value["exe"], self.saved["exe"])


if __name__ == "__main__":
    unittest.main()
