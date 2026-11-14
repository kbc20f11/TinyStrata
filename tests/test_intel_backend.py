"""Pure CPU policy checks; no native/device availability probe is performed."""
from copy import deepcopy
import os
import unittest
from unittest.mock import patch

from serve import intel_backend as intel
from serve import config


def configuration(*extra, **fields):
    return {"arch": "qwen35moe", "model_name": intel.QWEN_MODEL, "backend": "sycl",
            "exe": "runtime/native/sycl-test/tinystrata-intel.exe", "vision_mode": "no",
            "args": ["--model", "models/qwen.gguf", *extra], **fields}


def value(args, flag):
    return args[args.index(flag) + 1]


class IntelBackendPolicy(unittest.TestCase):
    def test_policy_helpers_do_not_launch_processes_or_load_runtime_dlls(self):
        with patch("subprocess.Popen", side_effect=AssertionError("No native process allowed")), \
             patch("ctypes.WinDLL", side_effect=AssertionError("No runtime DLL allowed"), create=True):
            result = intel.configure_backend(configuration())
            self.assertEqual(value(intel.engine_args(result), "--kv-type"), "int8")

    def test_explicit_pinned_identity_is_required_even_for_same_architecture(self):
        for fields in ({"arch": "gemma4"}, {"arch": "lfm2moe"}, {"arch": "clover_moe"},
                       {"model_name": "qwen3.8-flash"}, {"model_name": 7}):
            with self.subTest(fields=fields):
                cfg = configuration(**fields)
                self.assertFalse(intel.supports_model(cfg))
                with self.assertRaisesRegex(ValueError, "identities|models only"):
                    intel.configure_backend(cfg)
        self.assertTrue(intel.supports_model({"arch": "qwen35moe"}))

    def test_tiel_requires_its_own_profile_and_derives_only_its_native_admission(self):
        cfg = configuration(model_name=intel.TIEL_MODEL, intel_profile=intel.TIEL_PROFILE,
                            sampling={"presence_penalty": 0.0}, reasoning_effort="medium")
        self.assertTrue(intel.supports_model(cfg))
        self.assertEqual(intel.canonical_model(cfg), intel.TIEL_MODEL)
        self.assertEqual(intel.canonical_model(configuration()), intel.QWEN_MODEL)
        original = deepcopy(cfg)
        prepared = intel.configure_backend(cfg)
        native = intel.engine_args(prepared)
        self.assertIn("--experimental-tiel", native)
        self.assertNotIn("--experimental-tiel", prepared["args"])
        self.assertNotIn("--experimental-tiel", intel.engine_args(intel.configure_backend(configuration(intel_profile=intel.B390_PROFILE))))
        self.assertEqual((value(native, "--max-context"), value(native, "--chunk"), value(native, "--mtp")), ("8192", "1024", "3"))
        self.assertEqual(prepared["sampling"], cfg["sampling"])
        self.assertEqual(prepared["reasoning_effort"], "medium")
        self.assertEqual(cfg, original)
        for wrong in (configuration(model_name=intel.TIEL_MODEL),
                      configuration(model_name=intel.TIEL_MODEL, intel_profile=intel.B390_PROFILE),
                      configuration(intel_profile=intel.TIEL_PROFILE),
                      configuration("--experimental-tiel", intel_profile=intel.B390_PROFILE)):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                intel.configure_backend(wrong)

    def test_configure_deepcopy_preserves_sampling_thinking_auth_history(self):
        original = configuration(sampling={"temperature": .42, "presence_penalty": .7},
                                 enable_thinking=False, max_tokens=7000, api_key=["private-key"],
                                 history={"messages": [{"role": "user", "content": "saved"}]},
                                 aliases=["custom-api-alias"], custom={"nested": [1, 2]})
        before = deepcopy(original)
        result = intel.configure_backend(original, profile=intel.B390_PROFILE)
        for key in ("sampling", "enable_thinking", "max_tokens", "api_key", "history", "aliases", "custom"):
            self.assertEqual(result[key], original[key])
        result["sampling"]["temperature"] = 0
        result["api_key"].append("other")
        result["history"]["messages"].clear()
        result["args"].append("--no-graph")
        self.assertEqual(original, before)

    def test_generic_defaults_do_not_enable_b390_optimizations(self):
        result = intel.configure_backend(configuration())
        command = intel.engine_args(result)
        self.assertEqual(value(command, "--max-context"), "4096")
        self.assertEqual(value(command, "--chunk"), "128")
        self.assertEqual(value(command, "--mtp"), "0")
        self.assertEqual(value(command, "--kv-type"), "int8")
        self.assertNotIn("--decode-dp4a", command)
        self.assertNotIn("--draft-vocab", command)

    def test_b390_requested_profile_fills_validated_engine_options_only(self):
        result = intel.configure_backend(configuration(), profile=intel.B390_PROFILE)
        command = intel.engine_args(result)
        self.assertEqual(value(command, "--max-context"), "8192")
        self.assertEqual(value(command, "--chunk"), "1024")
        self.assertEqual(value(command, "--mtp"), "3")
        self.assertEqual(value(command, "--draft-vocab"), "2048")
        for flag in ("--prefill-mkl", "--decode-dp4a", "--verify-xmx", "--prefill-int8-xmx", "--vocab-xmx",
                     "--expert-xmx-layout", "--device-state", "--dense-xmx-layout", "--dense-split-k", "--mtp-stochastic"):
            self.assertEqual(command.count(flag), 1)
        self.assertEqual(value(result["args"], "--max-context"), "auto")
        self.assertEqual(value(result["args"], "--kv-type"), "auto")
        self.assertEqual(value(result["args"], "--chunk"), "auto")
        for flag in ("--decode-dp4a", "--mtp-stochastic", "--draft-vocab"):
            self.assertNotIn(flag, result["args"])

    def test_profile_defaults_follow_later_mtp_settings_updates_without_persisting_flags(self):
        initial = intel.configure_backend(configuration(sampling={"temperature": .8}, enable_thinking=True,
                                                      api_key=["saved"], history={"turns": [1]}), profile=intel.B390_PROFILE)
        initial_before = deepcopy(initial)
        self.assertEqual(value(initial["args"], "--mtp"), "3")
        off, changed = config.apply(initial, {"mtp": 0})
        self.assertEqual(changed, ["mtp"])
        off_before = deepcopy(off)
        intel.validate_config(off)
        off_command = intel.engine_args(off)
        self.assertEqual(value(off_command, "--mtp"), "0")
        self.assertNotIn("--mtp-stochastic", off_command)
        self.assertNotIn("--draft-vocab", off_command)
        self.assertIn("--decode-dp4a", off_command)
        self.assertEqual(off, off_before)
        on, _ = config.apply(off, {"mtp": 3})
        on_before = deepcopy(on)
        on_command = intel.engine_args(on)
        self.assertEqual(value(on_command, "--mtp"), "3")
        self.assertIn("--mtp-stochastic", on_command)
        self.assertEqual(value(on_command, "--draft-vocab"), "2048")
        for settings in (initial, off, on):
            for flag in ("--decode-dp4a", "--mtp-stochastic", "--draft-vocab"):
                self.assertNotIn(flag, settings["args"])
            for key in ("sampling", "enable_thinking", "api_key", "history"):
                self.assertEqual(settings[key], initial_before[key])
        self.assertEqual(initial, initial_before)
        self.assertEqual(on, on_before)

    def test_explicit_flags_and_draft_zero_remain_user_settings(self):
        initial = intel.configure_backend(configuration("--mtp", "3", "--draft-vocab", "0", "--prefill-mkl", "--no-graph"),
                                          profile=intel.B390_PROFILE)
        before = deepcopy(initial)
        self.assertIn("--prefill-mkl", initial["args"])
        self.assertIn("--no-graph", initial["args"])
        for depth in (0, 3):
            with self.subTest(depth=depth):
                changed, _ = config.apply(initial, {"mtp": depth})
                command = intel.engine_args(changed)
                self.assertEqual(value(command, "--draft-vocab"), "0")
                self.assertIn("--prefill-mkl", changed["args"])
                self.assertIn("--no-graph", changed["args"])
        self.assertEqual(initial, before)

    def test_explicit_stochastic_or_positive_draft_cannot_survive_mtp_zero(self):
        for explicit in (("--mtp-stochastic",), ("--draft-vocab", "2048")):
            with self.subTest(explicit=explicit):
                initial = intel.configure_backend(configuration("--mtp", "3", *explicit), profile=intel.B390_PROFILE)
                off, _ = config.apply(initial, {"mtp": 0})
                before = deepcopy(off)
                with self.assertRaises(ValueError):
                    intel.validate_config(off)
                with self.assertRaises(ValueError):
                    intel.engine_args(off)
                self.assertEqual(off, before)

    def test_explicit_mtp_zero_and_draft_zero_are_preserved(self):
        result = intel.configure_backend(configuration("--mtp", "0"), profile=intel.B390_PROFILE)
        command = intel.engine_args(result)
        self.assertEqual(value(command, "--mtp"), "0")
        self.assertNotIn("--mtp-stochastic", command)
        self.assertNotIn("--draft-vocab", command)
        result = intel.configure_backend(configuration("--mtp", "3", "--draft-vocab", "0"), profile=intel.B390_PROFILE)
        self.assertEqual(value(intel.engine_args(result), "--draft-vocab"), "0")

    def test_b390_auto1024_preserves_both_explicit512_profiles(self):
        for model, profile in intel.MODEL_PROFILES.items():
            fields = {"model_name": model, "intel_profile": profile}
            auto = intel.configure_backend(configuration(**fields))
            self.assertEqual(value(auto["args"], "--chunk"), "auto")
            self.assertEqual(value(intel.engine_args(auto), "--chunk"), "1024")
            manual = configuration("--chunk", "512", sampling={"temperature": .37}, **fields)
            before = deepcopy(manual)
            configured = intel.configure_backend(manual)
            self.assertEqual(value(intel.engine_args(configured), "--chunk"), "512")
            self.assertEqual(value(configured["args"], "--chunk"), "512")
            self.assertEqual(configured["sampling"], {"temperature": .37})
            self.assertEqual(manual, before)

    def test_explicit_context_chunk_kv_and_device_are_not_retuned(self):
        result = intel.configure_backend(configuration("--max-context", "16384", "--chunk", "64", "--kv-type", "fp32", "--mtp", "0",
                                                      gpu=2), profile=intel.B390_PROFILE)
        command = intel.engine_args(result)
        for flag, expected in (("--max-context", "16384"), ("--chunk", "64"), ("--kv-type", "fp32"), ("--device", "2")):
            self.assertEqual(value(command, flag), expected)
        result = intel.configure_backend(configuration("--kv-type", "fp32"), explicit_kv="auto")
        self.assertEqual(value(result["args"], "--kv-type"), "auto")
        self.assertEqual(value(intel.engine_args(result), "--kv-type"), "int8")

    def test_device_index_matches_native_cli_ceiling_without_probing_availability(self):
        for cfg in (configuration(gpu=1024), configuration("--device", "1024")):
            with self.subTest(cfg=cfg), patch("subprocess.run", side_effect=AssertionError("No device probe")):
                self.assertEqual(value(intel.engine_args(cfg), "--device"), "1024")
        for index in (1025, 65535):
            for cfg in (configuration(gpu=index), configuration("--device", str(index))):
                with self.subTest(index=index, args=cfg["args"]), self.assertRaisesRegex(ValueError, "device index.*1024"):
                    intel.configure_backend(cfg)

    def test_context_is_bounded_without_hardware_or_vram_probe(self):
        for limit in ("16", "32768"):
            self.assertEqual(intel.context_tokens(configuration("--max-context", limit)), int(limit))
        for limit in ("15", "32769", "262144", "-1", "4.5", "inf"):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "Intel context"):
                intel.configure_backend(configuration("--max-context", limit))
        with patch("subprocess.run", side_effect=AssertionError("No VRAM/device inference")):
            self.assertEqual(intel.context_tokens(configuration()), 4096)
            self.assertEqual(intel.context_tokens(configuration(intel_profile=intel.B390_PROFILE)), 8192)

    def test_unsupported_images_placement_and_state_are_rejected(self):
        for fields in ({"vision": {"exe": "encoder.exe"}}, {"images": {"enabled": True}}, {"vision_mode": "gpu"},
                       {"vision_mode": "cpu"}, {"layer_split": "auto"}, {"vram_elastic": True},
                       {"slot_save_path": "snapshots"}, {"parallel": 2}, {"replicas": 2},
                       {"expert_gpu_percent": 0}, {"expert_gpu_count": 8}, {"conversation_cache_mib": 0}, {"gpu": [0, 1]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                intel.configure_backend(configuration(**fields))

    def test_discrete_memory_tail_role_and_expert_learning_controls_are_rejected(self):
        for fields in ({"min_free_vram_mib": 768}, {"min_free_vram_mib": -1},
                       {"effort_position": "end"}, {"expert_profile_save_every": 1}, {"hip_ordinal": 0}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                intel.configure_backend(configuration(**fields))
        cfg = configuration(min_free_vram_mib=0, effort_position="start", expert_profile_save_every=0)
        intel.validate_config(cfg)

    def test_unsupported_native_flags_duplicate_and_incomplete_values_are_rejected(self):
        for arguments in (("--vision",), ("--kv-grow",), ("--expert-vram", "auto"),
                          ("--lookup-draft", "3"), ("--prefill-npu",), ("--prefill-xmx",),
                          ("--chunk",), ("--mtp", "0", "--mtp", "3"), ("--device", "-1")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                intel.configure_backend(configuration(*arguments))

    def test_unsupported_kv_and_unprofiled_optimizations_are_rejected(self):
        for kv in ("fp8", "fp8-mma", "fp16", "other"):
            with self.subTest(kv=kv), self.assertRaisesRegex(ValueError, "Intel KV"):
                intel.configure_backend(configuration("--kv-type", kv))
        with self.assertRaisesRegex(ValueError, "explicitly requested"):
            intel.configure_backend(configuration("--decode-dp4a"))
        with self.assertRaises(ValueError):
            intel.configure_backend(configuration(), profile="xe2-guessed-from-name")

    def test_invalid_draft_and_mtp_dependencies_are_rejected(self):
        for arguments in (("--mtp", "0", "--mtp-stochastic"), ("--mtp", "8"), ("--chunk", "1", "--mtp", "3"),
                          ("--draft-vocab", "63"), ("--draft-vocab", "4097"), ("--draft-vocab", "-1"),
                          ("--mtp", "0", "--draft-vocab", "64"), ("--mtp", "0", "--verify-graph")):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                intel.configure_backend(configuration(*arguments), profile=intel.B390_PROFILE)

    def test_backend_change_does_not_reuse_cuda_hip_ordinals_or_mutate_source(self):
        source = configuration(backend="hip", gpu=2, hip_ordinal=1)
        before = deepcopy(source)
        result = intel.configure_backend(source)
        self.assertEqual(result["backend"], "sycl")
        self.assertNotIn("gpu", result)
        self.assertNotIn("hip_ordinal", result)
        self.assertEqual(value(intel.engine_args(result), "--device"), "0")
        self.assertEqual(source, before)

    def test_bootstrap_validation_is_pure_and_does_not_require_installed_files(self):
        cfg = {"backend": "sycl", "arch": "qwen35moe", "model_name": intel.QWEN_MODEL, "args": []}
        result = intel.configure_backend(cfg, profile=intel.B390_PROFILE)
        intel.validate_config(result, allow_uninstalled=True)
        with self.assertRaises(ValueError):
            intel.validate_config(result)
        with self.assertRaises(ValueError):
            intel.engine_args(result)

    def test_child_environment_defaults_and_paths_leave_parent_untouched(self):
        inherited = {"Path": "system-runtime", "AUTH_SECRET": "secret"}
        cfg = configuration(lib_dirs=["stage", "stage"], env={"CUSTOM": "1"})
        before = deepcopy(cfg)
        result = intel.runtime_environment(cfg, inherited)
        self.assertEqual(result["Path"], "stage;system-runtime")
        self.assertNotIn("PATH", result)
        self.assertEqual(result["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
        self.assertEqual(result["UR_LOADER_PRELOAD_FILTER"], "1")
        self.assertEqual(inherited, {"Path": "system-runtime", "AUTH_SECRET": "secret"})
        self.assertEqual(cfg, before)

    def test_child_explicit_device_controls_win_case_insensitively(self):
        inherited = {"Path": "system", "oneapi_device_selector": "level_zero:gpu:1", "UR_LOADER_PRELOAD_FILTER": "0"}
        result = intel.runtime_environment(configuration(env={"ONEAPI_DEVICE_SELECTOR": "level_zero:gpu:2"}), inherited)
        self.assertEqual(result["oneapi_device_selector"], "level_zero:gpu:2")
        self.assertNotIn("ONEAPI_DEVICE_SELECTOR", result)
        self.assertEqual(result["UR_LOADER_PRELOAD_FILTER"], "0")
        self.assertEqual(inherited["oneapi_device_selector"], "level_zero:gpu:1")
        with patch.dict(os.environ, {"ONEAPI_DEVICE_SELECTOR": "explicit", "UR_LOADER_PRELOAD_FILTER": "0"}):
            before = dict(os.environ)
            child = intel.runtime_environment(configuration())
            self.assertEqual(child["ONEAPI_DEVICE_SELECTOR"], "explicit")
            self.assertEqual(dict(os.environ), before)


if __name__ == "__main__":
    unittest.main()
