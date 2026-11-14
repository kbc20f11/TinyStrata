"""Intel launch policy and state preservation; no model/GPU process runs."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import run as launcher
from serve import backend, config, intel_backend, model_setup, tinyconfig


class IntelLaunch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tinystrata-intel-launch-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "runtime"
        self.primary = self.runtime / ".tinystrata/config.json"
        self.primary.parent.mkdir(parents=True)
        self.original = {"backend": "cuda", "arch": "qwen35moe", "api_key": "fixture-private-key",
                         "args": ["--kv-type", "fp8"], "sampling": {"temperature": .7}}
        self.primary.write_text(json.dumps(self.original), encoding="utf-8")
        self.original_bytes = self.primary.read_bytes()
        for target in (patch.object(launcher, "RUNTIME", self.runtime),
                       patch.object(launcher, "ROOT", self.root),
                       patch.object(model_setup, "resolve_executable", return_value=None)):
            target.start()
            self.addCleanup(target.stop)

    def prepare(self, **overrides):
        values = dict(model_family="qwen36", config=None, model=None, exe=None, backend="sycl",
                      intel_profile="b390-qwen36", kv_type=None, lib_dir=None, port=None, host=None,
                      api_key=None, gpu=None, configure_only=True)
        values.update(overrides)
        with contextlib.redirect_stdout(io.StringIO()):
            return launcher.prepare(SimpleNamespace(**values))

    def test_separate_config_port_log_and_physical_pool_policy(self):
        cfg, path = self.prepare()
        self.assertEqual(path, self.primary.with_name("config-sycl.json"))
        self.assertEqual(cfg["port"], 18085)
        self.assertEqual(Path(cfg["log"]).name, "engine-intel.log")
        self.assertEqual(cfg["vision_mode"], "no")
        self.assertEqual(config.value_of(cfg, "kv_type"), "auto")
        self.assertEqual(intel_backend.context_tokens(cfg), 8192)
        self.assertEqual(self.primary.read_bytes(), self.original_bytes)
        self.assertTrue(cfg["model_bootstrap"])

    def test_explicit_mtp_zero_survives_profile_defaults_and_restart(self):
        cfg, path = self.prepare(mtp=0)
        native = intel_backend.engine_args(cfg, allow_uninstalled=True)
        self.assertEqual(config.value_of(cfg, "mtp"), 0)
        self.assertNotIn("--mtp-stochastic", native)
        self.assertNotIn("--draft-vocab", native)
        restarted, _ = self.prepare(config=path, backend="auto", intel_profile=None)
        self.assertEqual(config.value_of(restarted, "mtp"), 0)

    def test_explicit_other_config_cannot_overwrite_cuda_or_keys(self):
        with self.assertRaisesRegex(ValueError, "separate Intel"):
            self.prepare(config=self.primary)
        self.assertEqual(self.primary.read_bytes(), self.original_bytes)

    def test_other_family_and_same_architecture_tiel_reject_before_save(self):
        for family in ("tiel", "gemma4", "lfm2", "clover"):
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, "Qwen3.6"):
                self.prepare(model_family=family)
            self.assertFalse(self.primary.with_name("config-sycl.json").exists())
        self.assertEqual(self.primary.read_bytes(), self.original_bytes)

    def test_unsupported_context_or_images_do_not_publish(self):
        for options in ({"max_context": 32769}, {"vision": "gpu"}, {"expert_vram": 4}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.prepare(**options)
            self.assertFalse(self.primary.with_name("config-sycl.json").exists())

    def test_saved_generation_auth_and_arbitrary_user_state_survive(self):
        cfg, path = self.prepare()
        cfg.update(api_key="private-fixture-key", sampling={"temperature": .7, "top_k": 9},
                   enable_thinking=False, max_tokens=777, private_state={"history": ["kept"]})
        config.save(path, cfg)
        wanted = deepcopy(cfg)
        actual, _ = self.prepare(config=path, backend="auto", intel_profile=None)
        for key in ("api_key", "enable_thinking", "max_tokens", "private_state"):
            self.assertEqual(actual[key], wanted[key])
        for key, value in wanted["sampling"].items():
            self.assertEqual(actual["sampling"][key], value)
        self.assertEqual(self.primary.read_bytes(), self.original_bytes)

    def test_migration_does_not_erase_unsupported_intel_options(self):
        cfg = {"backend": "sycl", "arch": "qwen35moe", "args": ["--expert-vram", "4"]}
        migrated, _ = tinyconfig.migrate_settings(cfg)
        self.assertEqual(migrated["args"], cfg["args"])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            tinyconfig.validate(migrated, allow_uninstalled=True)

    def test_actual_gguf_identity_rejects_tiel_before_tokenizer_or_config_writes(self):
        model = self.root / "aliased-qwen.gguf"
        model.write_bytes(b"fixture; not a real model")
        cfg = {"backend": "sycl", "arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b",
               "vision_mode": "no", "args": ["--model", str(model)]}
        with patch.object(model_setup, "model_architecture", return_value="qwen35moe"), \
             patch("serve.gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata={"general.name": "Tiel Coder"})), \
             patch.object(model_setup, "_extract_tokenizer") as extract:
            with self.assertRaisesRegex(ValueError, "not a validated Intel model"):
                model_setup.prepare_installed_model(cfg, self.root / "new.json", model)
            extract.assert_not_called()
        self.assertFalse((self.root / "new.json").exists())

    def test_native_metadata_recognizes_sycl_without_hardware_detection(self):
        exe = self.root / "native.exe"
        data = bytearray(256)
        data[:2] = b"MZ"
        data[60:64] = (128).to_bytes(4, "little")
        data[128:132] = b"PE\0\0"
        exe.write_bytes(data)
        result = SimpleNamespace(returncode=0, stdout=json.dumps({"backend": "sycl", "experimental": True}))
        with patch.object(backend.subprocess, "run", return_value=result) as metadata:
            self.assertEqual(backend.executable_backend(exe), "sycl")
        self.assertEqual(metadata.call_args.args[0], [str(exe), "--capabilities"])


if __name__ == "__main__":
    unittest.main()
