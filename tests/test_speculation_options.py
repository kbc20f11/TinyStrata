"""Target-verified speculation request admission; no GPU or network required."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import config, tinyconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, check_request_sampling


def profile(arch="qwen35moe"):
    args = ["--model", "model.gguf", "--mtp", "3"]
    if arch == "gemma4":
        args += ["--draft-model", "assistant.gguf"]
    return {"arch": arch, "exe": "fixture.exe", "vision_mode": "no",
            "args": args, "sampling": {"temperature": 0.0}}


def service(arch="qwen35moe", *, lookup=0, auto=0, temperature=0.0):
    tok = ByteTokenizer()
    engine = MockEngine(tok, "unused")
    engine._tinystrata_arch = arch
    engine.info = {"mtp_auto": auto, "lookup_draft": lookup, "mtp": 3}
    return Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                   model_name=tinyconfig.MODEL_NAMES[arch], generation_defaults={"arch": arch},
                   sampling_defaults={"temperature": temperature})


class SpeculationOptions(unittest.TestCase):
    def test_default_is_unchanged(self):
        self.assertEqual(config.speculation_request_options({}), {})
        self.assertEqual(StrataEngine.sampling_keys({}), "")
        for arch in ("qwen35moe", "gemma4"):
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


    def test_forwarding_and_explicit_off(self):
        keys = StrataEngine.sampling_keys({"mtp_auto": 1, "lookup_draft": 7})
        self.assertIn(" mtp_auto=1", keys)
        self.assertIn(" lookup_draft=7", keys)
        svc = service(lookup=7, auto=1, temperature=.7)
        self.assertEqual(svc.speculation_request_settings({"lookup_draft": 0}), {"mtp_auto": 1, "lookup_draft": 0})

    def test_both_controls_preserve_valid_sampling_temperatures(self):
        for arch in ("qwen35moe", "gemma4"):
            self.assertEqual(service(arch, auto=1, temperature=1.).speculation_request_settings()["mtp_auto"], 1)
            for temperature in (0, .7, 1.0):
                self.assertEqual(service(arch, lookup=7, temperature=temperature).speculation_request_settings()["lookup_draft"], 7)
            for temperature in (-1, True, "0", float("nan"), float("inf")):
                with self.subTest(arch=arch, temperature=temperature), self.assertRaises(ValueError):
                    service(arch, lookup=7, temperature=temperature).speculation_request_settings()
            self.assertEqual(service(arch, lookup=7).speculation_request_settings({"lookup_draft": None})["lookup_draft"], 7)

    def test_profile_keeps_its_real_temperature_default(self):
        for arch in ("qwen35moe", "gemma4"):
            cfg, _ = config.apply(profile(arch), {"lookup_draft": 3})
            cfg.pop("sampling")
            tinyconfig.validate(cfg, allow_uninstalled=True)
            self.assertNotIn("sampling", cfg)


if __name__ == "__main__":
    unittest.main()
