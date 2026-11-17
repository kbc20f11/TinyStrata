"""CPU-only metadata contracts. Native subprocess calls are always mocked."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from serve import intel_capabilities as intel
from serve.intel_models import MODEL_CONTRACTS, MODEL_PROFILES, GEMMA_MODEL, LFM_MODEL, CLOVER_MODEL


def legacy_report():
    return {"backend": "sycl", "experimental": True, "model_families": [],
            "kv_types": ["fp32", "int8"], "images": False, "mtp": True,
            "experimental_mtp_max_depth": 7, "serve": True}


def report(*, mkl=True, xmx=True):
    return {**legacy_report(), "capabilities_schema": 1,
            "experimental_models": [intel.QWEN_MODEL],
            "experimental_profiles": {intel.B390_PROFILE: 1} if mkl and xmx else {},
            "build": {"mkl": mkl, "int8_xmx": xmx, "aot_target": "intel_gpu_ptl_h"},
            "weight_types": {"scalar": [0, 1, 8, 10, 11, 14, 18, 21, 22, 23, 30],
                             "dp4a": [8, 10, 11, 14, 18, 21, 22, 23],
                             "int8_xmx": [14, 18, 21, 22] if xmx else []}}


class CompatibilityPolicy(unittest.TestCase):
    def test_fp8_requires_qwen_only_explicit_codec_metadata_and_never_enables_old_runtimes(self):
        from serve.intel_models import EXPERIMENTAL_KV_MODELS,FP8_KV_FORMAT,TIEL_MODEL
        current=report()
        for native in (current,legacy_report()):
            with self.assertRaisesRegex(intel.IntelCapabilityError,"requested model-specific KV"):
                intel.validate_capabilities(native,requested_kv="fp8",allow_legacy=True)
        current.update(kv_types=["fp32","int8","fp8"],experimental_kv_models=deepcopy(EXPERIMENTAL_KV_MODELS),
                       fp8_kv_format=FP8_KV_FORMAT)
        intel.validate_capabilities(current,requested_kv="fp8")
        current.update(experimental_models=[intel.QWEN_MODEL,TIEL_MODEL,GEMMA_MODEL,LFM_MODEL,CLOVER_MODEL],
                       experimental_model_contracts=deepcopy(MODEL_CONTRACTS))
        for model in (TIEL_MODEL,GEMMA_MODEL,LFM_MODEL,CLOVER_MODEL):
            with self.subTest(model=model),self.assertRaisesRegex(intel.IntelCapabilityError,"requested model-specific KV"):
                intel.validate_capabilities(current,model=model,requested_kv="fp8")
        for field,bad in (("experimental_kv_models",{}),("experimental_kv_models",{"fp8":[intel.QWEN_MODEL,TIEL_MODEL]}),
                          ("fp8_kv_format","e5m2"),("fp8_kv_format",None)):
            changed=deepcopy(current);changed[field]=bad
            with self.subTest(field=field,bad=bad),self.assertRaisesRegex(intel.IntelCapabilityError,"codec contract"):
                intel.validate_capabilities(changed)
        changed=report();changed["experimental_kv_models"]=deepcopy(EXPERIMENTAL_KV_MODELS)
        with self.assertRaisesRegex(intel.IntelCapabilityError,"Partial experimental FP8"):
            intel.validate_capabilities(changed)

    def test_images_require_explicit_protocol_and_model_contract(self):
        from serve.intel_models import VISION_MODELS, VISION_MODEL_CONTRACTS
        for model in VISION_MODELS:
            native = report()
            native.update(images=True, image_protocol="geni_sve1", vision_contract_version=1,
                          experimental_models=[model], experimental_profiles={MODEL_PROFILES[model]: 1})
            if model in VISION_MODEL_CONTRACTS:
                native.update(mtp=False, experimental_mtp_max_depth=0, kv_types=["int8"],
                              experimental_model_contracts={model: deepcopy(VISION_MODEL_CONTRACTS[model])})
            intel.validate_capabilities(native, model=model, profile=MODEL_PROFILES[model], require_images=True)
            self.assertTrue(intel.supports_images(native, model))
            for field, bad in (("image_protocol", "guess"), ("vision_contract_version", True),
                               ("vision_contract_version", 2)):
                changed = deepcopy(native); changed[field] = bad
                with self.subTest(model=model, field=field), self.assertRaises(intel.IntelCapabilityError):
                    intel.validate_capabilities(changed, model=model, require_images=True)
        for old in (report(), legacy_report()):
            intel.validate_capabilities(old, allow_legacy=True)
            with self.assertRaisesRegex(intel.IntelCapabilityError, "lacks image"):
                intel.validate_capabilities(old, allow_legacy=True, require_images=True)

    def test_listing_uses_published_proof_without_process_or_model_reads(self):
        cfg = {"backend": "sycl", "arch": "qwen35moe", "model_name": intel.QWEN_MODEL,
               "exe": os.path.abspath("native.exe")}
        with patch("subprocess.run", side_effect=AssertionError("Listing must remain cheap")):
            self.assertFalse(intel.configured_images(cfg))
            cfg["intel_vision_contract"] = {"version": 1, "protocol": "geni_sve1", "model": intel.QWEN_MODEL,
                                            "sha256": "a" * 64, "executable": cfg["exe"]}
            self.assertTrue(intel.configured_images(cfg))
            changed = deepcopy(cfg); changed["exe"] = os.path.abspath("other-native.exe")
            self.assertFalse(intel.configured_images(changed))
            changed = deepcopy(cfg); changed["intel_vision_contract"]["version"] = True
            self.assertFalse(intel.configured_images(changed))

    def test_memory_aware_context_requires_actual_native_support(self):
        modern = report()
        modern.update(context_auto=True, max_context=262144)
        before = deepcopy(modern)
        with patch("subprocess.run", side_effect=AssertionError("Pure contract only")):
            for requested in ("auto", 8192, 32768, 262144):
                intel.validate_capabilities(modern, profile=intel.B390_PROFILE, requested_context=requested)
        self.assertEqual(modern, before)
        for old in (report(), legacy_report()):
            for requested in (8192, 32768):
                intel.validate_capabilities(old, allow_legacy=True, requested_context=requested)
            with self.assertRaisesRegex(intel.IntelCapabilityError, "update its model runtime"):
                intel.validate_capabilities(old, allow_legacy=True, requested_context="auto")
            for requested in (32769, 65536):
                with self.assertRaisesRegex(intel.IntelCapabilityError, "not reduced"):
                    intel.validate_capabilities(old, allow_legacy=True, requested_context=requested)

    def test_context_ceiling_is_native_capability_not_allocated_capacity(self):
        current = report()
        current.update(context_auto=True, max_context=65536)
        intel.validate_capabilities(current, requested_context="auto")
        intel.validate_capabilities(current, requested_context=65536)
        for requested in (65537, 262144, 0, True, "8192"):
            with self.subTest(requested=requested), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(current, requested_context=requested)
        lfm = report(xmx=False)
        lfm.update(experimental_models=[LFM_MODEL], experimental_profiles={MODEL_PROFILES[LFM_MODEL]: 1},
                   mtp=False, experimental_mtp_max_depth=0, kv_types=["int8"],
                   experimental_model_contracts={LFM_MODEL: deepcopy(MODEL_CONTRACTS[LFM_MODEL])},
                   context_auto=True, max_context=131072)
        intel.validate_capabilities(lfm, model=LFM_MODEL, requested_context=131072)
        with self.assertRaises(intel.IntelCapabilityError):
            intel.validate_capabilities(lfm, model=LFM_MODEL, requested_context=131073)
        lfm["max_context"] = 262144
        intel.validate_capabilities(lfm, model=LFM_MODEL, requested_context="auto")
        with self.assertRaisesRegex(intel.IntelCapabilityError, "not reduced"):
            intel.validate_capabilities(lfm, model=LFM_MODEL, requested_context=131073)

    def test_mixed_clover_context_contract_does_not_inherit_qwen_ceiling(self):
        current = report()
        current.update(context_auto=True, max_context=262144)
        current["experimental_models"].append(CLOVER_MODEL)
        current["experimental_profiles"][MODEL_PROFILES[CLOVER_MODEL]] = 1
        current["experimental_model_contracts"] = {CLOVER_MODEL: deepcopy(MODEL_CONTRACTS[CLOVER_MODEL])}
        intel.validate_capabilities(current, requested_context=262144)
        for requested in ("auto", 131072):
            intel.validate_capabilities(current, model=CLOVER_MODEL, requested_context=requested)
        with self.assertRaisesRegex(intel.IntelCapabilityError, "not reduced"):
            intel.validate_capabilities(current, model=CLOVER_MODEL, requested_context=131073)
        current["experimental_model_contracts"][CLOVER_MODEL]["context_max"] = 262144
        with self.assertRaisesRegex(intel.IntelCapabilityError, "model-specific"):
            intel.validate_capabilities(current, model=CLOVER_MODEL, requested_context="auto")

    def test_partial_or_malformed_native_context_metadata_fails_closed(self):
        for fields in ({"context_auto": True}, {"max_context": 262144},
                       {"context_auto": 1, "max_context": 262144},
                       {"context_auto": "true", "max_context": 262144},
                       {"context_auto": True, "max_context": True},
                       {"context_auto": True, "max_context": 262145}):
            with self.subTest(fields=fields), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities({**report(), **fields}, requested_context="auto")

    def test_new_target_only_reports_require_explicit_per_model_contracts(self):
        for model in (GEMMA_MODEL, LFM_MODEL, CLOVER_MODEL):
            current = report(xmx=False)
            current.update(experimental_models=[model], experimental_profiles={MODEL_PROFILES[model]: 1},
                           mtp=False, experimental_mtp_max_depth=0, kv_types=["int8"],
                           experimental_model_contracts={model: deepcopy(MODEL_CONTRACTS[model])})
            for profile in ("generic", MODEL_PROFILES[model]):
                intel.validate_capabilities(current, model=model, profile=profile)
            before = deepcopy(current)
            del current["experimental_model_contracts"]
            with self.assertRaisesRegex(intel.IntelCapabilityError, "explicit target-only contract"):
                intel.validate_capabilities(current, model=model)
            current = before
            current["experimental_model_contracts"][model]["mtp"] = True
            with self.assertRaisesRegex(intel.IntelCapabilityError, "model-specific"):
                intel.validate_capabilities(current, model=model)

    def test_mixed_clover_global_mtp_cannot_override_causal_contract(self):
        current = report()
        current["experimental_models"].append(CLOVER_MODEL)
        current["experimental_profiles"][MODEL_PROFILES[CLOVER_MODEL]] = 1
        current["experimental_model_contracts"] = {CLOVER_MODEL: deepcopy(MODEL_CONTRACTS[CLOVER_MODEL])}
        intel.validate_capabilities(current, model=CLOVER_MODEL, profile=MODEL_PROFILES[CLOVER_MODEL])
        for key, value in (("mode", "diffusion"), ("images", True), ("experimental_mtp_max_depth", 3)):
            changed = deepcopy(current)
            changed["experimental_model_contracts"][CLOVER_MODEL][key] = value
            with self.subTest(field=key), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(changed, model=CLOVER_MODEL)

    def test_generic_tiel_needs_declared_model_and_scalar_type18_without_xmx(self):
        current = report(mkl=False, xmx=False)
        current["build"]["aot_target"] = ""
        current["experimental_models"].append(intel.TIEL_MODEL)
        intel.validate_capabilities(current, model=intel.TIEL_MODEL, profile="generic")
        current["weight_types"]["scalar"].remove(18)
        current["weight_types"]["dp4a"].remove(18)
        with self.assertRaisesRegex(intel.IntelCapabilityError, "type18"):
            intel.validate_capabilities(current, model=intel.TIEL_MODEL, profile="generic")

    def test_pure_checks_never_probe_or_modify_the_input(self):
        original = report()
        before = deepcopy(original)
        with patch("subprocess.run", side_effect=AssertionError("No process")):
            intel.validate_capabilities(original, profile=intel.B390_PROFILE,
                                        required_types={"scalar": [0, 14], "int8_xmx": {21, 22}})
        self.assertEqual(original, before)

    def test_legacy_requires_explicit_existing_qwen_compatibility(self):
        old = legacy_report()
        with self.assertRaisesRegex(intel.IntelCapabilityError, "Legacy"):
            intel.validate_capabilities(old)
        for profile in ("generic", intel.B390_PROFILE):
            intel.validate_capabilities(old, allow_legacy=True, profile=profile)
        for options in ({"model": "tiel-coder-35b-a3b"}, {"profile": "b390-tiel"},
                        {"required_types": {"scalar": [18]}}):
            with self.subTest(options=options), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(old, allow_legacy=True, **options)

    def test_missing_or_partial_schema_does_not_downgrade(self):
        current = report()
        del current["capabilities_schema"]
        with self.assertRaisesRegex(intel.IntelCapabilityError, "missing capabilities_schema"):
            intel.validate_capabilities(current, allow_legacy=True)
        for version in (0, 2, True, "1", None):
            current["capabilities_schema"] = version
            with self.subTest(version=version), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(current, allow_legacy=True)

    def test_build_without_either_compiled_feature_cannot_claim_b390(self):
        for mkl, xmx in ((False, False), (False, True), (True, False)):
            current = report(mkl=mkl, xmx=xmx)
            intel.validate_capabilities(current)
            with self.assertRaisesRegex(intel.IntelCapabilityError, "profile"):
                intel.validate_capabilities(current, profile=intel.B390_PROFILE)
            current["experimental_profiles"] = {intel.B390_PROFILE: 1}
            with self.assertRaisesRegex(intel.IntelCapabilityError, "MKL and INT8 XMX"):
                intel.validate_capabilities(current)

    def test_no_tiel_or_formal_family_claim_is_inferred_from_architecture(self):
        with self.assertRaisesRegex(intel.IntelCapabilityError, "does not declare experimental model"):
            intel.validate_capabilities(report(), model="tiel-coder-35b-a3b")
        current = report()
        current["model_families"] = ["qwen35moe"]
        with self.assertRaisesRegex(intel.IntelCapabilityError, "formal-adoption"):
            intel.validate_capabilities(current)

    def test_required_quantization_is_checked_per_path(self):
        current = report()
        intel.validate_capabilities(current, required_types={path: [18] for path in ("scalar", "dp4a", "int8_xmx")})
        current["weight_types"]["dp4a"].remove(18)
        current["weight_types"]["int8_xmx"].remove(18)
        intel.validate_capabilities(current, required_types={"scalar": [18]})
        for path in ("dp4a", "int8_xmx"):
            with self.subTest(path=path), self.assertRaisesRegex(intel.IntelCapabilityError, "lacks GGUF types"):
                intel.validate_capabilities(current, required_types={path: [18]})

    def test_invalid_build_and_quantization_shapes_are_rejected(self):
        cases = [("build", {"mkl": 1, "int8_xmx": True, "aot_target": ""}),
                 ("build", {"mkl": True, "int8_xmx": "true", "aot_target": ""}),
                 ("build", {"mkl": True, "int8_xmx": True, "aot_target": "other_gpu"}),
                 ("experimental_models", [intel.QWEN_MODEL, intel.QWEN_MODEL]),
                 ("experimental_profiles", {intel.B390_PROFILE: True})]
        for key, value in cases:
            current = report()
            current[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(current)
        for value in ([14, 14], [21, 14], [True], [None], [65536], "14"):
            current = report()
            current["weight_types"]["int8_xmx"] = value
            with self.subTest(value=value), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(current)
        current = report(xmx=False)
        current["weight_types"]["int8_xmx"] = [14]
        with self.assertRaisesRegex(intel.IntelCapabilityError, "inconsistent"):
            intel.validate_capabilities(current)

    def test_invalid_required_types_and_profile_model_pairs_are_rejected(self):
        for required in ([], {"unknown": [14]}, {"scalar": "14"}, {"scalar": [True]}, {"scalar": [-1]}):
            with self.subTest(required=required), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(report(), required_types=required)
        current = report()
        current["experimental_models"].append("tiel-coder-35b-a3b")
        with self.assertRaisesRegex(intel.IntelCapabilityError, "specific to qwen3.6"):
            intel.validate_capabilities(current, model="tiel-coder-35b-a3b", profile=intel.B390_PROFILE)

    def test_tiel_requires_new_explicit_model_profile_and_type18_in_every_path(self):
        current = report()
        current["experimental_models"].append(intel.TIEL_MODEL)
        current["experimental_profiles"][intel.TIEL_PROFILE] = 1
        intel.validate_capabilities(current, model=intel.TIEL_MODEL, profile=intel.TIEL_PROFILE)
        for path in ("scalar", "dp4a", "int8_xmx"):
            changed = deepcopy(current)
            changed["weight_types"][path].remove(18)
            with self.subTest(path=path), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(changed, model=intel.TIEL_MODEL, profile=intel.TIEL_PROFILE)
        for old in (legacy_report(), report()):
            with self.subTest(old=old), self.assertRaises(intel.IntelCapabilityError):
                intel.validate_capabilities(old, model=intel.TIEL_MODEL, profile=intel.TIEL_PROFILE, allow_legacy=True)
        with self.assertRaises(intel.IntelCapabilityError):
            intel.validate_capabilities(current, model=intel.QWEN_MODEL, profile=intel.TIEL_PROFILE)


class MetadataProbe(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.exe = self.root / "tinystrata-intel.exe"
        self.exe.write_bytes(b"CPU metadata fixture")
        self.cfg = {"backend": "sycl", "exe": str(self.exe), "lib_dirs": [str(self.root)]}
        self.inherited = {"SystemRoot": str(self.root / "Windows"),
                          "PATH": "C:\\Program Files (x86)\\Intel\\oneAPI\\compiler\\bin;C:\\unrelated"}
        intel._probe_cached.cache_clear()
        self.addCleanup(intel._probe_cached.cache_clear)

    def completed(self, value=None, returncode=0):
        return subprocess.CompletedProcess([str(self.exe), "--capabilities"], returncode,
                                           json.dumps(report() if value is None else value), "")

    def probe(self, **options):
        return intel.probe_capabilities(self.cfg, inherited=self.inherited, **options)

    def test_cached_probe_is_sha_bound_and_returns_unshared_reports(self):
        with patch.object(intel.subprocess, "run", return_value=self.completed()) as run:
            first = self.probe()
            first.report["build"]["mkl"] = False
            second = self.probe()
        self.assertEqual(run.call_count, 1)
        self.assertTrue(second.report["build"]["mkl"])
        self.assertEqual(second.sha256, hashlib.sha256(self.exe.read_bytes()).hexdigest())
        self.assertFalse(second.legacy)

    def test_metadata_replacement_cannot_reuse_cached_report(self):
        with patch.object(intel.subprocess, "run", return_value=self.completed()) as run:
            before = self.probe()
            info = self.exe.stat()
            data = self.exe.read_bytes()
            self.exe.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            os.utime(self.exe, ns=(info.st_atime_ns, info.st_mtime_ns))
            after = self.probe()
        self.assertEqual(run.call_count, 2)
        self.assertNotEqual(before.sha256, after.sha256)

    def test_child_command_is_model_free_and_path_is_sdk_free(self):
        original, inherited = deepcopy(self.cfg), deepcopy(self.inherited)
        with patch.object(intel.subprocess, "run", return_value=self.completed()) as run:
            self.probe(timeout=4)
        args, options = run.call_args
        self.assertEqual(args[0], [str(self.exe), "--capabilities"])
        self.assertEqual(options["cwd"], str(self.root))
        self.assertEqual(options["timeout"], 4)
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        environment = options["env"]
        self.assertEqual(environment["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
        self.assertEqual(environment["UR_LOADER_PRELOAD_FILTER"], "1")
        self.assertEqual(environment["PATH"].split(";"),
                         [str(self.root), str(self.root / "Windows" / "System32"), str(self.root / "Windows")])
        self.assertNotIn("oneapi", environment["PATH"].lower())
        self.assertEqual(self.cfg, original)
        self.assertEqual(self.inherited, inherited)

    def test_case_insensitive_controls_are_retained_without_duplicate_path_keys(self):
        self.inherited.update(Path="C:\\SDK2", oneapi_device_selector="level_zero:1",
                              ur_loader_preload_filter="0")
        self.cfg["env"] = {"ONEAPI_DEVICE_SELECTOR": "level_zero:2", "Path": "C:\\SDK3"}
        with patch.object(intel.subprocess, "run", return_value=self.completed()) as run:
            self.probe()
        env = {key.upper(): value for key, value in run.call_args.kwargs["env"].items()}
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:2")
        self.assertEqual(env["UR_LOADER_PRELOAD_FILTER"], "0")
        self.assertEqual(sum(key.upper() == "PATH" for key in run.call_args.kwargs["env"]), 1)
        self.assertNotIn("SDK", env["PATH"])

    def test_changed_selector_invalidates_process_result_cache(self):
        with patch.object(intel.subprocess, "run", return_value=self.completed()) as run:
            self.probe()
            self.cfg["env"] = {"ONEAPI_DEVICE_SELECTOR": "level_zero:1"}
            self.probe()
        self.assertEqual(run.call_count, 2)

    def test_expected_hash_is_checked_before_execution(self):
        expected = hashlib.sha256(self.exe.read_bytes()).hexdigest()
        with patch.object(intel.subprocess, "run", return_value=self.completed()) as run:
            self.probe(expected_sha256=expected.upper())
            with self.assertRaisesRegex(intel.IntelCapabilityError, "does not match"):
                self.probe(expected_sha256="0" * 64)
        self.assertEqual(run.call_count, 1)

    def test_executable_changes_during_probe_are_rejected(self):
        def changed(*args, **kwargs):
            self.exe.write_bytes(b"replacement fixture")
            return self.completed()
        with patch.object(intel.subprocess, "run", side_effect=changed):
            with self.assertRaisesRegex(intel.IntelCapabilityError, "changed during"):
                self.probe()

    def test_timeout_nonzero_and_malformed_json_do_not_downgrade(self):
        results = [subprocess.TimeoutExpired("fixture", 1), self.completed(returncode=1),
                   subprocess.CompletedProcess([], 0, "{", ""),
                   subprocess.CompletedProcess([], 0, '{"backend":"sycl","backend":"cuda"}', ""),
                   subprocess.CompletedProcess([], 0, '{"backend":NaN}', ""),
                   subprocess.CompletedProcess([], 0, "x" * 65537, "")]
        for result in results:
            intel._probe_cached.cache_clear()
            behavior = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
            with self.subTest(result=result), patch.object(intel.subprocess, "run", **behavior):
                with self.assertRaises(intel.IntelCapabilityError):
                    self.probe()

    def test_legacy_probe_reports_unknown_build_until_policy_explicitly_allows_it(self):
        with patch.object(intel.subprocess, "run", return_value=self.completed(legacy_report())):
            observed = self.probe()
        self.assertTrue(observed.legacy)
        with self.assertRaises(intel.IntelCapabilityError):
            intel.validate_capabilities(observed.report)
        intel.validate_capabilities(observed.report, allow_legacy=True, profile=intel.B390_PROFILE)

    def test_invalid_probe_inputs_never_execute(self):
        with patch.object(intel.subprocess, "run", side_effect=AssertionError("No process")):
            for options in ({"timeout": 0}, {"timeout": True}, {"timeout": float("inf")},
                            {"timeout": 61}, {"expected_sha256": "bad"}):
                with self.subTest(options=options), self.assertRaises(intel.IntelCapabilityError):
                    self.probe(**options)
            self.cfg["backend"] = "cuda"
            with self.assertRaises(intel.IntelCapabilityError):
                self.probe()


if __name__ == "__main__":
    unittest.main()
