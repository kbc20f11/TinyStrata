"""CPU-only lookup admission/forwarding; never starts native or HTTP processes."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import config, tinyconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, check_request_sampling


def profile():
    return {"arch": "clover_moe", "exe": "fixture.exe", "vision_mode": "no",
            "args": ["--model", "fixture.gguf", "--kv-type", "int8",
                     "--clover-mode", "causal", "--clover-draft", "0"],
            "sampling": deepcopy(tinyconfig.CLOVER_SAMPLING)}


def service(*, lookup=0, vision=None, arch="clover_moe"):
    tok = ByteTokenizer()
    engine = MockEngine(tok, "unused")
    engine.info = {"clover_mode": "causal", "clover_draft": 0, "clover_lookup": lookup,
                   "clover_accept": "exact", "vision": int(vision is not None)}
    return Service(engine, tok, ChatTemplate(ROOT / "serve" / "chat_template.jinja"),
                   model_name="clover-35b-a3b-preview" if arch == "clover_moe" else "qwen3.6-35b-a3b",
                   vision=vision, generation_defaults={"arch": arch},
                   sampling_defaults=deepcopy(tinyconfig.CLOVER_SAMPLING))


class LookupOptions(unittest.TestCase):
    def test_request_type_and_bounds_are_strict(self):
        for value in range(8):
            self.assertEqual(config.clover_request_options({"clover_lookup": value}), {"clover_lookup": value})
        for value in (True, False, "3", 3.0, 0.0, -1, 8, float("inf"), float("nan"), [], 10 ** 1000):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    check_request_sampling({"clover_lookup": value})
        self.assertEqual(config.clover_request_options({"clover_lookup": None}), {})

    def test_config_roundtrip_and_no_implicit_default(self):
        original = profile()
        before = deepcopy(original)
        for value in (0, 1, 7):
            cfg, _ = config.apply(original, {"clover_lookup": value})
            self.assertEqual(config.value_of(cfg, "clover_lookup"), value)
            tinyconfig.validate(cfg, allow_uninstalled=True)
        self.assertEqual(original, before)
        self.assertNotIn("--clover-lookup", original["args"])
        self.assertEqual(StrataEngine.sampling_keys({}), "")

    def test_config_rejects_wrong_family_and_fractional_arg(self):
        with self.assertRaisesRegex(ValueError, "Clover"):
            config.apply({**profile(), "arch": "qwen35moe"}, {"clover_lookup": 1})
        cfg = profile()
        cfg["args"] += ["--clover-lookup", "3.0"]
        with self.assertRaisesRegex(ValueError, "whole number"):
            tinyconfig.validate(cfg, allow_uninstalled=True)

    def test_config_rejects_self_spec_diffusion_and_sampled_temperature(self):
        for options in ({"clover_mode": "self-spec", "clover_draft": 3}, {"clover_mode": "diffusion"}):
            cfg, _ = config.apply(profile(), {"clover_lookup": 3, **options})
            with self.assertRaisesRegex(ValueError, "causal"):
                tinyconfig.validate(cfg, allow_uninstalled=True)
        for value in (.7, -1, True, "0", float("nan"), float("inf")):
            cfg, _ = config.apply(profile(), {"clover_lookup": 3})
            cfg["sampling"]["temperature"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                tinyconfig.validate(cfg, allow_uninstalled=True)

    def test_default_off_and_explicit_lookup_keep_causal_mode(self):
        svc = service()
        self.assertEqual(svc.clover_request_settings()["clover_lookup"], 0)
        self.assertEqual(svc.clover_request_settings({"clover_lookup": 7})["clover_mode"], "causal")
        self.assertEqual(svc.clover_request_settings({"clover_lookup": 7})["clover_draft"], 0)

    def test_inherits_resident_flag_and_null_keeps_it(self):
        svc = service(lookup=3)
        self.assertEqual(svc.clover_request_settings()["clover_lookup"], 3)
        self.assertEqual(svc.clover_request_settings({"clover_lookup": None})["clover_lookup"], 3)
        self.assertEqual(svc.clover_request_settings({"clover_lookup": 0})["clover_lookup"], 0)

    def test_lookup_allows_resident_images(self):
        self.assertEqual(service(vision=object()).clover_request_settings({"clover_lookup": 3})["clover_lookup"], 3)

    def test_request_rejects_incompatible_mode_or_draft(self):
        for req in ({"clover_mode": "self-spec"}, {"clover_draft": 3}, {"clover_mode": "diffusion"}):
            with self.subTest(req=req), self.assertRaisesRegex(ValueError, "causal"):
                service(lookup=3).clover_request_settings(req)
        with self.assertRaisesRegex(ValueError, "causal"):
            service().clover_request_settings({"clover_lookup": 3, "clover_draft": 1})

    def test_request_temperature_resolves_request_shared_and_config(self):
        for value in (.7, -1, True, "0", float("nan"), float("inf")):
            for origin in ("request", "shared", "config"):
                svc = service(lookup=3)
                req = {"temperature": value} if origin == "request" else {}
                if origin == "shared":
                    svc.shared["temperature"] = value
                elif origin == "config":
                    svc.sampling_defaults["temperature"] = value
                with self.subTest(value=value, origin=origin), self.assertRaisesRegex(ValueError, "temperature 0"):
                    svc.clover_request_settings(req)
        self.assertEqual(service(lookup=3).clover_request_settings({"temperature": -0.0})["clover_lookup"], 3)

    def test_explicit_off_allows_other_modes_and_sampling(self):
        svc = service(lookup=3)
        self.assertEqual(svc.clover_request_settings({"clover_lookup": 0, "temperature": .7})["clover_lookup"], 0)
        self.assertEqual(svc.clover_request_settings({"clover_lookup": 0, "clover_mode": "self-spec"})["clover_draft"], 3)

    def test_non_clover_rejects_even_explicit_zero(self):
        for value in (0, 3):
            with self.assertRaisesRegex(ValueError, "Clover"):
                service(arch="qwen35moe").clover_request_settings({"clover_lookup": value})

    def test_forwarding_preserves_explicit_zero_and_maximum(self):
        for value in (0, 7):
            self.assertEqual(StrataEngine.sampling_keys({"clover_lookup": value}), f" clover_lookup={value}")


if __name__ == "__main__":
    unittest.main()
