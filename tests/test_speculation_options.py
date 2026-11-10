"""Target-verified speculation request admission; no GPU or network required."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import backend, config, tinyconfig
from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, check_request_sampling,
                          clean_shared_defaults, engine_args, sampling_defaults_from_config)


def profile(arch="qwen35moe"):
    args = ["--model", "model.gguf", "--mtp", "0" if arch == "gpt-oss" else "3"]
    if arch == "gemma4":
        args += ["--draft-model", "assistant.gguf"]
    return {"arch": arch, "exe": "fixture.exe", "vision_mode": "no",
            "args": args, "sampling": {"temperature": 0.0}}


def service(arch="qwen35moe", *, lookup=0, auto=0, temperature=0.0):
    tok = ByteTokenizer()
    engine = MockEngine(tok, "unused")
    engine._tinystrata_arch = arch
    engine.info = {"mtp_auto": auto, "lookup_draft": lookup, "mtp": 0 if arch == "gpt-oss" else 3}
    return Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                   model_name=tinyconfig.MODEL_NAMES[arch], generation_defaults={"arch": arch},
                   sampling_defaults={"temperature": temperature})


def eagle_profile(depth="1", acceptance=None, model="eagle model.gguf"):
    cfg = profile("gpt-oss")
    cfg["args"] += ["--eagle", str(depth)]
    if model is not None:
        cfg["args"] += ["--eagle-model", model]
    if acceptance is not None:
        cfg["args"] += ["--eagle-accept", acceptance]
    return cfg


class SpeculationOptions(unittest.TestCase):
    def test_default_is_unchanged(self):
        self.assertEqual(config.speculation_request_options({}), {})
        self.assertEqual(StrataEngine.sampling_keys({}), "")
        for arch in ("qwen35moe", "gemma4", "gpt-oss"):
            self.assertEqual(service(arch).speculation_request_settings(), {"mtp_auto": 0, "lookup_draft": 0})

    def test_request_values_are_strict(self):
        for key, limit in (("mtp_auto", 1), ("lookup_draft", 7)):
            for value in range(limit + 1):
                self.assertEqual(config.speculation_request_options({key: value}), {key: value})
            for value in (-1, limit + 1, True, "1", 1.0, [], float("nan"), float("inf"), 10**1000):
                with self.subTest(key=key, value=repr(value)), self.assertRaises(ValueError):
                    check_request_sampling({key: value})
        self.assertEqual(config.speculation_request_options({"mtp_auto": None, "lookup_draft": None}), {})

    def test_launch_roundtrip_does_not_mutate_profile(self):
        for arch in ("qwen35moe", "gemma4"):
            original = profile(arch)
            before = deepcopy(original)
            cfg, _ = config.apply(original, {"mtp_auto": 1, "lookup_draft": 7})
            self.assertEqual(config.value_of(cfg, "mtp_auto"), 1)
            self.assertEqual(config.value_of(cfg, "lookup_draft"), 7)
            tinyconfig.validate(cfg, allow_uninstalled=True)
            self.assertEqual(original, before)

    def test_other_models_reject_controls(self):
        for arch in ("clover_moe", "lfm2moe"):
            for key in config.SPECULATION_REQUEST_KEYS:
                with self.assertRaisesRegex(ValueError, "Qwen35MoE or Gemma"):
                    config.apply(profile(arch), {key: 1})
                with self.assertRaisesRegex(ValueError, "Qwen35MoE or Gemma"):
                    service(arch).speculation_request_settings({key: 0})

    def test_gpt_oss_lookup_roundtrips_without_enabling_mtp(self):
        original = profile("gpt-oss")
        before = deepcopy(original)
        cfg, _ = config.apply(original, {"mtp_auto": 0, "lookup_draft": 7})
        self.assertEqual(config.value_of(cfg, "lookup_draft"), 7)
        self.assertEqual(config.value_of(cfg, "mtp"), 0)
        tinyconfig.validate(cfg, allow_uninstalled=True)
        self.assertEqual(original, before)
        self.assertEqual(service("gpt-oss", lookup=7, temperature=1.).speculation_request_settings(),
                         {"mtp_auto": 0, "lookup_draft": 7})
        for value in (1,):
            with self.assertRaisesRegex(ValueError, "no MTP head"):
                config.apply(original, {"mtp_auto": value})
            with self.assertRaisesRegex(ValueError, "no MTP head"):
                service("gpt-oss").speculation_request_settings({"mtp_auto": value})

    def test_gpt_oss_legacy_engine_identity_still_allows_lookup(self):
        svc = service("gpt-oss", lookup=7, temperature=1.)
        del svc.engine._tinystrata_arch
        self.assertEqual(svc.speculation_request_settings()["lookup_draft"], 7)

    def test_gpt_oss_fp16_kv_matches_the_native_engine(self):
        cfg = {**profile("gpt-oss"), "backend": "hip"}
        cfg, _ = config.apply(cfg, {"kv_type": "fp16", "lookup_draft": 3})
        tinyconfig.validate(cfg, allow_uninstalled=True)
        self.assertEqual(config.value_of(backend.configure_backend(cfg, "hip"), "kv_type"), "fp16")
        self.assertEqual(config.value_of(backend.configure_backend(cfg, "hip", explicit_kv="fp16"), "kv_type"), "fp16")
        with self.assertRaises(ValueError):
            backend.configure_backend(cfg, "hip", explicit_kv="fp8-mma")

    def test_gpt_oss_preserves_full_vocabulary_sampling(self):
        self.assertEqual(sampling_defaults_from_config({"arch": "gpt-oss"})["top_k"], 0)
        for model in ("gpt-oss-20b", "gpt-oss", "gptoss"):
            self.assertEqual(clean_shared_defaults({"top_k": 0}, model), {"top_k": 0})
        self.assertEqual(StrataEngine.sampling_keys({"temperature": 1., "top_k": 0}, arch="gpt-oss"),
                         " temperature=1.0 top_k=0")
        self.assertEqual(StrataEngine.sampling_keys({"top_k": 64}, arch="gpt-oss"), " top_k=64")
        for arch in ("qwen35moe", "clover_moe", "lfm2moe", "gemma4", None):
            self.assertEqual(StrataEngine.sampling_keys({"top_k": 0}, arch=arch), " top_k=64")
        with self.assertRaises(ValueError):
            clean_shared_defaults({"top_k": 0}, "qwen3.6-35b-a3b")
        with self.assertRaises(SystemExit):
            sampling_defaults_from_config({"arch": "qwen35moe", "sampling": {"top_k": 0}})

    def test_forwarding_and_explicit_off(self):
        keys = StrataEngine.sampling_keys({"mtp_auto": 1, "lookup_draft": 7})
        self.assertIn(" mtp_auto=1", keys)
        self.assertIn(" lookup_draft=7", keys)
        svc = service(lookup=7, auto=1, temperature=.7)
        self.assertEqual(svc.speculation_request_settings({"lookup_draft": 0}), {"mtp_auto": 1, "lookup_draft": 0})

    def test_both_controls_preserve_valid_sampling_temperatures(self):
        for arch in ("qwen35moe", "gemma4", "gpt-oss"):
            if arch != "gpt-oss":
                self.assertEqual(service(arch, auto=1, temperature=1.).speculation_request_settings()["mtp_auto"], 1)
            for temperature in (0, .7, 1.0):
                self.assertEqual(service(arch, lookup=7, temperature=temperature).speculation_request_settings()["lookup_draft"], 7)
            for temperature in (-1, True, "0", float("nan"), float("inf")):
                with self.subTest(arch=arch, temperature=temperature), self.assertRaises(ValueError):
                    service(arch, lookup=7, temperature=temperature).speculation_request_settings()
            self.assertEqual(service(arch, lookup=7).speculation_request_settings({"lookup_draft": None})["lookup_draft"], 7)

    def test_profile_keeps_its_real_temperature_default(self):
        for arch in ("qwen35moe", "gemma4", "gpt-oss"):
            cfg, _ = config.apply(profile(arch), {"lookup_draft": 3})
            cfg.pop("sampling")
            tinyconfig.validate(cfg, allow_uninstalled=True)
            self.assertNotIn("sampling", cfg)

    def test_eagle_raw_profile_is_optional_and_preserves_native_rejection_default(self):
        self.assertFalse(any(key.startswith("eagle") for key in config.ENGINE_OPTIONS))
        cfg, _ = tinyconfig.migrate_settings(profile("gpt-oss"))
        self.assertFalse(any(arg.startswith("--eagle") for arg in cfg["args"]))
        tinyconfig.validate(cfg, allow_uninstalled=True)
        for depth in ("1", "2"):
            for acceptance in (None, "target", "rejection"):
                with self.subTest(depth=depth, acceptance=acceptance):
                    cfg = eagle_profile(depth, acceptance)
                    before = deepcopy(cfg)
                    tinyconfig.validate(cfg, allow_uninstalled=True)
                    self.assertEqual(engine_args(cfg), cfg["args"])
                    self.assertEqual(cfg, before)
                    if acceptance is None:
                        self.assertNotIn("--eagle-accept", cfg["args"], "native rejection default must stay omitted")
        off = eagle_profile("0", model=None)
        tinyconfig.validate(off, allow_uninstalled=True)
        off, _ = config.apply(off, {"lookup_draft": 3})
        tinyconfig.validate(off, allow_uninstalled=True)

    def test_eagle_raw_flags_are_gpt_oss_only(self):
        for arch in ("qwen35moe", "gemma4", "clover_moe", "lfm2moe"):
            for flags in (["--eagle", "0"], ["--eagle", "1", "--eagle-model", "draft.gguf"],
                          ["--eagle-accept", "rejection"], ["--eagle-model", "draft.gguf"]):
                with self.subTest(arch=arch, flags=flags), self.assertRaisesRegex(ValueError, "require gpt-oss"):
                    cfg = profile(arch)
                    cfg["args"] += flags
                    tinyconfig.validate(cfg, allow_uninstalled=True)

    def test_eagle_values_and_dependencies_are_strict(self):
        for depth in ("-1", "3", "1.0", "1e0", "true", "nan", "inf", "", " 1", "01"):
            with self.subTest(depth=depth), self.assertRaisesRegex(ValueError, "whole number"):
                tinyconfig.validate(eagle_profile(depth), allow_uninstalled=True)
        for acceptance in ("exact", "approximate", "", "TARGET"):
            with self.subTest(acceptance=acceptance), self.assertRaisesRegex(ValueError, "target or rejection"):
                tinyconfig.validate(eagle_profile(acceptance=acceptance), allow_uninstalled=True)
        for flags in (["--eagle", "1"], ["--eagle", "2", "--eagle-model", ""],
                      ["--eagle-model", "draft.gguf"], ["--eagle-accept", "target"],
                      ["--eagle-accept", "rejection"], ["--eagle", "0", "--eagle-model", "draft.gguf"],
                      ["--eagle", "0", "--eagle-accept", "target"]):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "EAGLE3|require --eagle"):
                cfg = profile("gpt-oss")
                cfg["args"] += flags
                tinyconfig.validate(cfg, allow_uninstalled=True)
        for flags in (["--eagle"], ["--eagle-model"], ["--eagle-accept"],
                      ["--eagle", "1", "--eagle-model", "--no-graph"],
                      ["--eagle", "1", "--eagle", "2"]):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "incomplete|Duplicate"):
                cfg = profile("gpt-oss")
                cfg["args"] += flags
                tinyconfig.validate(cfg, allow_uninstalled=True)

    def test_eagle_rejects_other_drafters_and_multiple_gpus(self):
        for key in ("mtp", "mtp_auto", "lookup_draft"):
            cfg, _ = config.apply(eagle_profile(), {key: 1}) if key != "mtp_auto" else (eagle_profile(), [])
            if key == "mtp_auto":
                cfg["args"] += ["--mtp-auto", "1"]
            with self.subTest(key=key), self.assertRaises(ValueError):
                tinyconfig.validate(cfg, allow_uninstalled=True)
        for gpu in ([0, 1], (0, 1), "0,1"):
            with self.subTest(gpu=gpu), self.assertRaisesRegex(ValueError, "one GPU"):
                tinyconfig.validate({**eagle_profile(), "gpu": gpu}, allow_uninstalled=True)
        for gpu in (None, 0, 1, "0", "1", [0], (1,)):
            tinyconfig.validate({**eagle_profile(), "gpu": gpu}, allow_uninstalled=True)
        for key, value in (("parallel", 2), ("replicas", 2), ("layer_split", "auto")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                tinyconfig.validate({**eagle_profile(), key: value}, allow_uninstalled=True)

    def test_eagle_migration_backend_and_engine_command_preserve_raw_flags(self):
        for acceptance in (None, "target", "rejection"):
            original = {**eagle_profile("2", acceptance), "backend": "cuda"}
            before = deepcopy(original)
            migrated, _ = tinyconfig.migrate_settings(original)
            for requested in ("cuda", "hip", "cuda"):
                migrated = backend.configure_backend(migrated, requested)
                tinyconfig.validate(migrated, allow_uninstalled=True)
                for flag in ("--eagle", "--eagle-model", "--eagle-accept"):
                    if flag in before["args"]:
                        self.assertEqual(migrated["args"][migrated["args"].index(flag) + 1],
                                         before["args"][before["args"].index(flag) + 1])
                    else:
                        self.assertNotIn(flag, migrated["args"])
                self.assertEqual(engine_args(migrated), migrated["args"])
                self.assertEqual(config.value_of(migrated, "mtp"), 0)
            self.assertEqual(original, before)

    def test_eagle_files_checked_only_when_installation_is_required(self):
        with tempfile.TemporaryDirectory(prefix="tinystrata-eagle-profile-") as directory:
            root = Path(directory)
            for name in ("fixture.exe", "model.gguf", "eagle model.gguf"):
                (root / name).touch()
            for model in ("eagle model.gguf", str(root / "eagle model.gguf")):
                cfg = {**eagle_profile(model=model), "cwd": str(root)}
                tinyconfig.validate(cfg)
                tinyconfig.check_files(cfg)
            for model in ("missing.gguf", str(root)):
                cfg = {**eagle_profile(model=model), "cwd": str(root)}
                tinyconfig.validate(cfg, allow_uninstalled=True)
                tinyconfig.check_files(cfg, allow_uninstalled=True)
                tinyconfig.validate(cfg)
                with self.subTest(check_files=model), self.assertRaisesRegex(ValueError, "EAGLE3 draft model not found"):
                    tinyconfig.check_files(cfg)

    def test_eagle_request_admission_checks_info_and_unloaded_spawn(self):
        for identity in ("info", "spawn", "stale_info_and_spawn"):
            for legacy in (False, True):
                with self.subTest(identity=identity, legacy=legacy):
                    svc = service("gpt-oss", temperature=1.)
                    if legacy:
                        del svc.engine._tinystrata_arch
                    if identity == "info":
                        svc.engine.info["eagle"] = 1
                    else:
                        svc.engine.info = {} if identity == "spawn" else {"eagle": 0}
                        svc.engine.spawn = ("fixture.exe", eagle_profile()["args"])
                    self.assertEqual(svc.speculation_request_settings(), {"mtp_auto": 0, "lookup_draft": 0})
                    self.assertEqual(svc.speculation_request_settings({"mtp_auto": 0, "lookup_draft": 0}),
                                     {"mtp_auto": 0, "lookup_draft": 0})
                    for key in ("mtp_auto", "lookup_draft"):
                        with self.subTest(key=key), self.assertRaisesRegex(ValueError, "EAGLE3"):
                            svc.speculation_request_settings({key: 1})
        svc = service("gpt-oss", lookup=3, temperature=1.)
        svc.engine.info["eagle"] = "2"
        with self.assertRaisesRegex(ValueError, "EAGLE3"):
            svc.speculation_request_settings()
        self.assertEqual(svc.speculation_request_settings({"lookup_draft": 0}), {"mtp_auto": 0, "lookup_draft": 0})
        svc.engine.info["eagle"] = 0
        svc.engine.spawn = ("fixture.exe", ["--eagle", "0"])
        self.assertEqual(svc.speculation_request_settings({"lookup_draft": 7})["lookup_draft"], 7)


if __name__ == "__main__":
    unittest.main()
