"""CPU-only API/launcher contract tests for Gemma/LFM conversation parking settings."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import backend, config, model_setup, tinyconfig

spec = importlib.util.spec_from_file_location("parking_launcher_test", ROOT / "run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

BOUNDS = {"conversation_cache_mib": (0, 65536), "conversation_cache_disk_mib": (0, 65536),
          "conversation_cache_slots": (1, 16), "conversation_cache_checkpoints": (1, 8),
          "conversation_cache_min_tokens": (0, 262144)}


def profile(arch):
    return {"arch": arch, "backend": "cuda", "exe": "engine.exe", "vision_mode": "no",
            "args": ["--model", "fixture.gguf", "--kv-type", "int8", "--chunk", "auto"]}


class ParkingSettings(unittest.TestCase):
    def test_api_boundaries_round_trip_to_native_flags(self):
        for arch in ("gemma4", "lfm2moe"):
            for key, bounds in BOUNDS.items():
                for value in bounds:
                    with self.subTest(arch=arch, key=key, value=value):
                        original = profile(arch)
                        before = copy.deepcopy(original)
                        cfg, changed = config.apply(original, {key: value})
                        self.assertEqual(changed, [key])
                        self.assertEqual(config.value_of(cfg, key), value)
                        self.assertIn("--" + key.replace("_", "-"), cfg["args"])
                        tinyconfig.validate(cfg, allow_uninstalled=True)
                        self.assertEqual(original, before)

    def test_numeric_boundaries_and_invalid_json_types_are_rejected(self):
        for arch in ("gemma4", "lfm2moe"):
            for key, (low, high) in BOUNDS.items():
                for value in (low - 1, high + 1, 10**1000, -(10**1000), 1.5, True, False, "1", "auto", [], {}, float("inf"), float("nan")):
                    with self.subTest(arch=arch, key=key, value=value):
                        with self.assertRaises(ValueError):
                            config.apply(profile(arch), {key: value})

    def test_wrong_architecture_is_rejected_at_both_api_and_raw_args(self):
        for arch in ("qwen35moe", "clover_moe"):
            for key, (low, _) in BOUNDS.items():
                with self.subTest(arch=arch, key=key):
                    with self.assertRaisesRegex(ValueError, "only by Gemma"):
                        config.apply(profile(arch), {key: low})
                    cfg = profile(arch)
                    cfg["args"] += [config.ENGINE_OPTIONS[key][0], str(low)]
                    with self.assertRaisesRegex(ValueError, "only by Gemma"):
                        tinyconfig.validate(cfg, allow_uninstalled=True)

    def test_reset_removes_explicit_flag_for_supported_architectures(self):
        for arch in ("gemma4", "lfm2moe"):
            for key, (low, _) in BOUNDS.items():
                cfg, _ = config.apply(profile(arch), {key: low})
                cfg, _ = config.apply(cfg, {key: None})
                self.assertIsNone(config.value_of(cfg, key))
                self.assertNotIn(config.ENGINE_OPTIONS[key][0], cfg["args"])

    def test_migration_keeps_all_explicit_parking_budgets(self):
        values = {"conversation_cache_mib": 512, "conversation_cache_disk_mib": 256,
                  "conversation_cache_slots": 4, "conversation_cache_checkpoints": 3,
                  "conversation_cache_min_tokens": 1024}
        for arch in ("gemma4", "lfm2moe"):
            cfg, _ = config.apply(profile(arch), values)
            before = copy.deepcopy(cfg)
            after, _ = tinyconfig.migrate_settings(cfg)
            self.assertEqual({key: config.value_of(after, key) for key in BOUNDS}, values)
            self.assertEqual(cfg, before)
            tinyconfig.validate(after, allow_uninstalled=True)

    def test_omitted_settings_do_not_inject_native_defaults(self):
        for arch in ("qwen35moe", "gemma4", "lfm2moe", "clover_moe"):
            cfg, _ = tinyconfig.migrate_settings(profile(arch))
            self.assertFalse(any(arg.startswith("--conversation-cache-") for arg in cfg["args"]))

    def test_raw_args_reject_duplicate_incomplete_and_non_numeric_values(self):
        for key, (low, _) in BOUNDS.items():
            flag = config.ENGINE_OPTIONS[key][0]
            for tail in ([flag], [flag, "bad"], [flag, str(low), flag, str(low)], [flag, "1.5"]):
                cfg = profile("lfm2moe"); cfg["args"] += tail
                with self.subTest(key=key, tail=tail), self.assertRaises(ValueError):
                    tinyconfig.validate(cfg, allow_uninstalled=True)

    def test_launcher_accepts_cli_values_without_starting_a_service(self):
        seen = []
        def prepare(args):
            seen.append(args)
            return {}, Path("unused.json")
        argv = ["run.py", "--configure-only", "--model-family", "lfm2"]
        for key, (low, _) in BOUNDS.items():
            argv += [config.ENGINE_OPTIONS[key][0], str(low)]
        with patch.object(sys, "argv", argv), patch.object(launcher, "prepare", side_effect=prepare), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(), 0)
        self.assertEqual({key: getattr(seen[0], key) for key in BOUNDS}, {key: b[0] for key, b in BOUNDS.items()})

    def test_launcher_prepare_forwards_all_values(self):
        class Args(SimpleNamespace):
            def __getattr__(self, key):
                return None
        values = {key: b[0] for key, b in BOUNDS.items()}
        for arch, family in (("gemma4", "gemma4"), ("lfm2moe", "lfm2")):
            with tempfile.TemporaryDirectory(prefix="tinystrata-parking-settings-") as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(profile(arch)), encoding="utf-8")
                args = Args(model_family=family, config=path, backend="cuda", **values)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(backend, "configure_backend", side_effect=lambda cfg,*a,**kw:cfg))
                    stack.enter_context(patch.object(backend, "executable_backend", return_value="cuda"))
                    stack.enter_context(patch.object(model_setup, "configure_download", side_effect=lambda cfg,*a,**kw:cfg))
                    stack.enter_context(patch.object(model_setup, "configure_vision", side_effect=lambda cfg,*a,**kw:cfg))
                    stack.enter_context(patch.object(model_setup, "shared_library_dirs", side_effect=lambda cfg:cfg))
                    stack.enter_context(patch.object(model_setup, "save_settings"))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    cfg, _ = launcher.prepare(args)
                self.assertEqual({key: config.value_of(cfg, key) for key in BOUNDS}, values)


if __name__ == "__main__":
    unittest.main()
