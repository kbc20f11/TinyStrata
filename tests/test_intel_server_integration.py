"""CPU-only normal-server/profile integration; never start a native engine."""
from copy import deepcopy
from dataclasses import replace
from contextlib import redirect_stderr
import http.client
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from serve import intel_backend, intel_capabilities, model_setup, server
from serve.model_profiles import ModelProfiles
from serve.model_switch import ModelBundle, ModelControlError, ModelSwitcher
from serve.model_download import FIXED_MODEL, FIXED_TIEL_MODEL, manifest_for_arch


QWEN = "qwen3.6-35b-a3b"
TIEL = "tiel-coder-35b-a3b"


def capability_report(*, tiel=False):
    result = {"backend": "sycl", "experimental": True, "model_families": [],
            "kv_types": ["fp32", "int8"], "images": False, "mtp": True,
            "experimental_mtp_max_depth": 7, "serve": True, "capabilities_schema": 1,
            "context_auto": True, "max_context": 262144,
            "experimental_models": [QWEN], "experimental_profiles": {"b390-qwen36": 1},
            "build": {"mkl": True, "int8_xmx": True, "aot_target": "intel_gpu_ptl_h"},
            "weight_types": {"scalar": [0, 1, 8, 10, 11, 14, 18, 21, 22, 23, 30],
                             "dp4a": [8, 10, 11, 14, 18, 21, 22, 23], "int8_xmx": [14, 18, 21, 22]}}
    if tiel:
        result["experimental_models"].append(TIEL)
        result["experimental_profiles"]["b390-tiel-coder"] = 1
    return result


class IntelServerIntegration(unittest.TestCase):
    def target_configuration(self, model):
        profile = intel_backend.MODEL_PROFILES[model]
        cfg = intel_backend.configure_backend({"backend": "sycl", "arch": intel_backend.MODEL_ARCHES[model],
            "model_name": model, "intel_profile": profile, "cwd": str(self.root), "exe": str(self.exe),
            "lib_dirs": [str(self.runtime)], "args": ["--model", str(self.root / "absent-pinned.gguf")],
            "vision_mode": "no", "api_key": "private-fixture-key"})
        from serve.intel_models import MODEL_CONTRACTS
        self.capabilities = capability_report()
        self.capabilities.update(experimental_models=[model], experimental_profiles={profile: 1}, mtp=False,
            max_context=intel_backend.MODEL_CONTEXT_LIMITS[model],
            experimental_mtp_max_depth=0, kv_types=["int8"], experimental_model_contracts={model: deepcopy(MODEL_CONTRACTS[model])})
        self.capabilities["build"]["int8_xmx"] = False
        self.capabilities["weight_types"]["int8_xmx"] = []
        return cfg

    def test_new_family_bootstrap_downloads_only_its_pinned_target(self):
        from serve.model_download import FixedModelDownloader
        for model in intel_backend.TARGET_ONLY_MODELS:
            cfg = self.target_configuration(model)
            svc = self.service(cfg)
            path = self.root / (model + ".json")
            with self.subTest(model=model):
                server.configure_model_downloads(svc, cfg, path)
                self.assertIs(type(svc.downloads), FixedModelDownloader)
                self.assertEqual(svc.downloads.manifest, manifest_for_arch(cfg["arch"], model))
                self.assertNotIn("draft_model", cfg)
                self.assertNotIn("vision", cfg)
                svc.downloads.close()

    def test_missing_new_family_runtime_rejects_switch_without_unloading(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        with patch("serve.model_switch.threading.Thread") as thread:
            for model in intel_backend.TARGET_ONLY_MODELS:
                with self.subTest(model=model), self.assertRaisesRegex(ValueError, "dedicated Intel runtime"):
                    control.start(model)
            thread.assert_not_called()
        svc.engine.unload.assert_not_called()
        svc.engine.close.assert_not_called()

    def test_cross_family_current_id_change_cannot_reassign_initial_runtime(self):
        profiles = ModelProfiles(self.path, self.cfg)
        gemma = self.target_configuration(intel_backend.GEMMA_MODEL)
        dedicated = self.root / "gemma-runtime" / "tinystrata-intel.exe"
        dedicated.parent.mkdir()
        dedicated.write_bytes(b"dedicated Gemma fixture; never execute")
        gemma.update(exe=str(dedicated), lib_dirs=[str(dedicated.parent)])
        profiles.current_id = intel_backend.GEMMA_MODEL  # Same transition performed after READY.
        updated = profiles.update(intel_backend.GEMMA_MODEL, gemma)
        self.assertEqual(updated.config["exe"], str(dedicated))
        self.assertEqual(updated.config["lib_dirs"], [str(dedicated.parent)])
        # Qwen/Tiel keep their intentional common-engine generation behavior.
        tiel = profiles.get(TIEL).config
        self.assertEqual(tiel["exe"], str(self.exe))

    def test_intel_new_family_features_and_request_controls_do_not_inherit_cuda(self):
        for model in intel_backend.TARGET_ONLY_MODELS:
            cfg = self.target_configuration(model)
            svc = self.service(cfg)
            with self.subTest(model=model):
                features = server.intel_feature_fields(svc)["features"]
                for key in ("images", "assistant", "mtp", "lookup", "clover_draft"):
                    self.assertIs(features[key], False)
                self.assertEqual(features["kv_types"], ["int8"])
                self.assertTrue(features["target_only"])
                svc.engine.unload.assert_not_called()
                for req in ({"lookup_draft": 4}, {"mtp_auto": 1}, {"clover_mode": "diffusion"},
                            {"clover_mode": "self-spec", "clover_draft": 3}, {"clover_lookup": 4}):
                    with self.assertRaises(ValueError):
                        svc.clover_request_settings(req)
                if model == intel_backend.CLOVER_MODEL:
                    svc.engine.info.update(clover_mode="self-spec", clover_draft=3, clover_lookup=16)
                    self.assertEqual(svc.clover_request_settings({}), {"clover_mode": "causal", "clover_draft": 0})
                    self.assertIsNone(svc.clover_preview_block({}))
                else:
                    self.assertIsNone(svc.clover_request_settings({}))
                svc.engine.unload.assert_not_called()
                svc.engine.close.assert_not_called()

    def test_lfm_shared_thinking_off_is_rejected_before_persistence(self):
        svc = self.service(self.target_configuration(intel_backend.LFM_MODEL))
        shared = self.root / "shared-lfm.json"
        shared.write_text('{"temperature": 0.1}', encoding="utf-8")
        svc.shared, svc.shared_path = {"temperature": .1}, str(shared)
        for disabled in ({"enable_thinking": False}, {"reasoning_effort": "none"}):
            with self.assertRaisesRegex(ValueError, "always uses thinking"):
                svc.set_shared(disabled)
            self.assertEqual(svc.shared, {"temperature": .1})
            self.assertEqual(shared.read_text(encoding="utf-8"), '{"temperature": 0.1}')

    def test_intel_inventory_requires_only_pinned_target_not_cuda_auxiliaries(self):
        from serve.model_library import artifacts, inventory
        directory = self.root / "model-files"
        directory.mkdir()
        for model, arch in intel_backend.MODEL_ARCHES.items():
            artifact = manifest_for_arch(arch, model)
            target = directory / artifact.filename
            target.write_bytes(b"CPU fixture; never open as GGUF")
            profile = SimpleNamespace(id=model, arch=arch, path=self.root / (model + ".json"),
                config={"backend": "sycl", "vision_mode": "no", "args": ["--model", str(target)]})
            with self.subTest(model=model):
                self.assertEqual(artifacts(profile), (artifact,))
                plan = inventory(profile, (profile,))
                self.assertTrue(plan.management["complete"])
                self.assertEqual(plan.management["download_bytes"], artifact.size)
                self.assertEqual([entry.path for entry in plan.files], [target])

    def test_new_family_health_props_and_catalog_expose_runtime_feature_limits(self):
        svc = self.service(self.target_configuration(intel_backend.CLOVER_MODEL))
        svc.template.source, svc.template.caps = "CPU fixture template", {}
        svc.models = ModelSwitcher(svc, self.target_configuration(intel_backend.CLOVER_MODEL),
                                   self.root / "config-clover-sycl.json", server)
        with patch.object(svc, "start_telemetry"):
            httpd = server.serve(svc, "127.0.0.1", 0)
        try:
            for route in ("/health", "/props", "/v1/models"):
                client = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
                try:
                    client.request("GET", route)
                    response = client.getresponse()
                    body = json.loads(response.read())
                finally:
                    client.close()
                self.assertEqual(response.status, 200)
                if route == "/v1/models":
                    for row in body["data"]:
                        self.assertEqual(row["architecture"]["input_modalities"], ["text"])
                        self.assertFalse(row["features"]["assistant"])
                        if row["id"] in intel_backend.TARGET_ONLY_MODELS:
                            self.assertFalse(row["features"]["mtp"])
                    body = next(row for row in body["data"] if row["id"] == intel_backend.CLOVER_MODEL)
                self.assertFalse(body["features"]["images"])
                self.assertFalse(body["features"]["mtp"])
                self.assertFalse(body["features"]["lookup"])
                self.assertEqual(body["features"]["clover_modes"], ["causal"])
        finally:
            svc.stopping.set()
            httpd.shutdown()
            httpd.server_close()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="intel-server-cpu-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        self.exe = self.runtime / "tinystrata-intel.exe"
        self.exe.write_bytes(b"CPU fixture; never execute")
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"CPU fixture; no model loaded")
        self.path = self.root / "config-sycl.json"
        self.cfg = intel_backend.configure_backend({
            "backend": "sycl", "intel_profile": "b390-qwen36", "arch": "qwen35moe", "model_name": QWEN,
            "exe": str(self.exe), "cwd": str(self.root), "lib_dirs": [str(self.runtime)], "gpu": 0,
            "args": ["--model", str(self.model), "--max-context", "auto", "--kv-type", "auto",
                     "--chunk", "auto", "--mtp", "3"],
            "sampling": {"temperature": .7, "presence_penalty": 1.2}, "enable_thinking": False,
            "max_tokens": 4096, "api_key": "fixture-private-key", "aliases": ["personal-qwen"],
            "history_preference": {"retain": True}, "log": str(self.root / "engine-intel.log"),
            "download_dir": str(self.root / "model-files"),
        })
        self.path.write_text(json.dumps(self.cfg), encoding="utf-8")
        self.capabilities = capability_report()
        probe = patch.object(intel_capabilities, "probe_capabilities",
            side_effect=lambda cfg: intel_capabilities.IntelCapabilities(
                cfg["exe"], "0" * 64, (), deepcopy(self.capabilities)))
        self.probe = probe.start()
        self.addCleanup(probe.stop)

    def service(self, cfg=None):
        engine = server.MockEngine(server.ByteTokenizer(), "", max_context=8192)
        engine.info = {"backend": "sycl", "memory_model": "uma"}
        engine.alive = Mock(return_value=True)
        engine.unload, engine.close, engine.vram = Mock(), Mock(), Mock()
        return server.Service(engine, server.ByteTokenizer(), SimpleNamespace(call_format="function"),
                              model_name=server.default_model_name(cfg or self.cfg),
                              generation_defaults=cfg or self.cfg,
                              sampling_defaults=(cfg or self.cfg).get("sampling"))

    def test_native_args_resolve_auto_without_cuda_learning_or_duplicate_serve(self):
        original = deepcopy(self.cfg)
        with patch.object(server, "learned_profile_args", side_effect=AssertionError("CUDA learning")), \
             patch.object(server, "ordered_gpus", side_effect=AssertionError("CUDA ordinal")):
            args = server.engine_args(self.cfg)
        self.assertNotIn("--serve", args)
        self.assertEqual(args[args.index("--max-context") + 1], "auto")
        self.assertEqual(args[args.index("--kv-type") + 1], "int8")
        self.assertEqual(args[args.index("--draft-vocab") + 1], "2048")
        self.assertNotIn("--expert-profile", args)
        self.assertEqual(self.cfg, original)

    def test_auto_context_is_unknown_until_ready_and_uses_actual_capacity_after_reload(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        svc.encode_prompt = Mock(return_value=[1] * 40000)
        svc.prompt_turn_at = Mock(return_value=None)
        svc.engine.known_ctx = 8192
        svc.engine.info.update(max_context=8192, kv_used_tokens=123)
        svc.engine.alive.return_value = False
        self.assertEqual(control.context_info(QWEN), {"allocated_tokens": None, "model_limit": 262144, "requested": "auto"})
        self.assertEqual(svc.reported_ctx(), 0)
        self.assertIsNone(svc.metrics()["engine"]["kv_used_tokens"])
        counted = svc.chat_token_count({"model": QWEN, "messages": [{"role": "user", "content": "data"}]})
        self.assertEqual(counted["max_context"], 0)
        self.assertTrue(counted["context_pending"])
        svc.engine.unload.assert_not_called()
        with self.assertRaises(server.EngineStarting):
            svc.prepare([{"role": "user", "content": "data"}], None, {}, 1)
        svc.engine.alive.return_value = True
        for actual in (262144, 65536):
            svc.engine.max_context = actual  # Real READY is the sole allocation authority.
            self.assertEqual(control.context_info(QWEN)["allocated_tokens"], actual)
            self.assertEqual(svc.metrics()["engine"]["max_context"], actual)
            self.assertEqual(svc.chat_token_count({"model": QWEN, "messages": []})["max_context"], actual)
            ids, _, output = svc.prepare([{"role": "user", "content": "data"}], None, {}, 100)
            self.assertEqual((len(ids), output), (40000, 100))
        svc.engine.max_context = 32768
        with self.assertRaisesRegex(ValueError, "never truncated"):
            svc.prepare([{"role": "user", "content": "data"}], None, {}, 100)
        svc.engine.alive.return_value = False
        self.assertIsNone(control.context_info(QWEN)["allocated_tokens"])
        self.assertEqual(svc.metrics()["engine"]["max_context"], 0)

    def test_old_auto_runtime_rejects_before_config_save_worker_or_unload(self):
        self.capabilities.pop("context_auto")
        self.capabilities.pop("max_context")
        before = self.path.read_bytes()
        svc = self.service({**self.cfg, "args": ["--model", str(self.model), "--max-context", "32768"]})
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        with patch("serve.model_switch.threading.Thread") as thread, patch.object(model_setup, "save_settings") as save:
            with self.assertRaisesRegex(intel_capabilities.IntelCapabilityError, "update its model runtime"):
                control._validate_intel_runtime(QWEN)
            thread.assert_not_called()
            save.assert_not_called()
        svc.engine.unload.assert_not_called()
        svc.engine.close.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_old_runtime_rejects_fp8_before_config_save_worker_or_unload(self):
        svc = self.service()
        cfg = deepcopy(self.cfg)
        cfg["args"][cfg["args"].index("--kv-type")+1] = "fp8"
        self.path.write_text(json.dumps(cfg),encoding="utf-8")
        before = self.path.read_bytes()
        control = ModelSwitcher(svc,cfg,self.path,server)
        with patch("serve.model_switch.threading.Thread") as thread,patch.object(model_setup,"save_settings") as save:
            with self.assertRaisesRegex(intel_capabilities.IntelCapabilityError,"requested model-specific KV"):
                control._validate_intel_runtime(QWEN)
            thread.assert_not_called();save.assert_not_called()
        svc.engine.unload.assert_not_called();svc.engine.close.assert_not_called()
        self.assertEqual(self.path.read_bytes(),before)

    def test_child_environment_uses_only_intel_runtime_policy(self):
        with patch.dict(os.environ, {"PATH": "parent-path"}, clear=True), \
             patch.object(server, "ordered_gpus", side_effect=AssertionError("CUDA ordinal")), \
             patch.object(server, "hip_visible", side_effect=AssertionError("HIP ordinal")):
            env = server.child_env(self.cfg)
            self.assertEqual(os.environ["PATH"], "parent-path")
        self.assertEqual(env["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
        self.assertEqual(env["UR_LOADER_PRELOAD_FILTER"], "1")
        self.assertTrue(env["PATH"].startswith(str(self.runtime)))
        for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HIP_VISIBLE_DEVICES",
                    "TINYSTRATA_Q36_EXPERT_PROFILE_DIR", "TINYSTRATA_Q36_EXPERT_TOPIC"):
            self.assertNotIn(key, env)

    def test_device_option_uses_level_zero_index_without_cuda_ordinal_mapping(self):
        cfg = deepcopy(self.cfg)
        cfg.pop("gpu")
        cfg["args"] += ["--device", "2"]
        with patch.object(server, "ordered_gpus", side_effect=AssertionError("CUDA ordinal")):
            self.assertEqual(server.gpu_list(cfg), [2])
            profiles = ModelProfiles(self.path, cfg)
        self.assertEqual(profiles.get(QWEN).config["gpu"], 2)
        self.assertNotIn("gpu", cfg)

    def test_catalog_reads_only_intel_config_and_preserves_runtime_and_user_fields(self):
        others = {self.root / name: b"malformed unrelated config, must not be read"
                  for name in ("config.json", "config-qwen36.json", "config-tiel-coder.json", "config-gemma4.json")}
        for path, data in others.items():
            path.write_bytes(data)
        with patch("serve.model_profiles.config.load", side_effect=AssertionError("Other config read")):
            profiles = ModelProfiles(self.path, self.cfg)
        self.assertEqual({row["id"] for row in profiles.list()}, set(intel_backend.MODEL_PROFILES))
        for model in intel_backend.TARGET_ONLY_MODELS:
            self.assertNotIn("exe", profiles.get(model).config)
            self.assertNotIn("lib_dirs", profiles.get(model).config)
        profile = profiles.get("qwen36")
        self.assertEqual(profile.path, self.path)
        for key in ("exe", "lib_dirs", "intel_profile", "sampling", "enable_thinking", "api_key", "aliases", "history_preference"):
            self.assertEqual(profile.config[key], self.cfg[key])
        self.assertFalse(profiles.list()[0]["vision"])
        self.assertEqual(profiles.context_info(QWEN), {"allocated_tokens": None, "model_limit": 262144, "requested": "auto"})
        for path, data in others.items():
            self.assertEqual(path.read_bytes(), data)

    def test_generic_runtime_keeps_generic_policy_when_selecting_tiel(self):
        cfg = {**self.cfg, "intel_profile": "generic"}
        profiles = ModelProfiles(self.path, cfg)
        tiel = profiles.get(TIEL).config
        self.assertEqual(tiel["intel_profile"], "generic")
        self.assertEqual(tiel["args"][tiel["args"].index("--mtp") + 1], "0")
        native = intel_backend.engine_args(tiel, allow_uninstalled=True)
        self.assertIn("--experimental-tiel", native)
        self.assertNotIn("--verify-xmx", native)

    def test_tiel_and_other_models_reject_before_job_start_or_unload(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        # A different family's Qwen executable is not a new-model runtime.
        for model in intel_backend.TARGET_ONLY_MODELS:
            profile = control.profiles.get(model)
            cfg = {**profile.config, "exe": str(self.exe), "lib_dirs": [str(self.runtime)]}
            profile.path.write_text(json.dumps(cfg), encoding="utf-8")
        with patch("serve.model_switch.threading.Thread") as thread, patch.object(control, "_downloader") as downloader:
            for model in (TIEL, "gemma4", "lfm2", "clover"):
                with self.subTest(model=model), self.assertRaises(ValueError):
                    control.start(model)
                with self.subTest(download=model), self.assertRaises(ValueError):
                    control.download(model, start=True)
            thread.assert_not_called()
            downloader.assert_not_called()
        svc.engine.unload.assert_not_called()
        svc.engine.close.assert_not_called()
        self.assertTrue(svc.loaded())

    def test_wrong_current_model_and_unsupported_saved_settings_reject_before_prepare(self):
        with self.assertRaisesRegex(ValueError, "match its pinned model"):
            ModelProfiles(self.path, {**self.cfg, "model_name": TIEL})
        profiles = ModelProfiles(self.path, self.cfg)
        for change in ({"model_name": TIEL}, {"vision_mode": "gpu"}, {"images": {"exe": "image.exe"}},
                       {"parallel": 2}, {"effort_position": "end"}, {"min_free_vram_mib": 128}):
            invalid = {**self.cfg, **change}
            self.path.write_text(json.dumps(invalid), encoding="utf-8")
            before = self.path.read_bytes()
            with self.subTest(change=change), patch.object(model_setup, "prepare_installed_model") as prepare:
                with self.assertRaises(ValueError):
                    profiles.prepare(QWEN)
                prepare.assert_not_called()
            self.assertEqual(self.path.read_bytes(), before)

    def test_service_initializes_identity_preserving_generation_policy(self):
        svc = self.service()
        self.assertEqual(server.intel_health_fields(svc),
                         {"backend": "sycl", "intel_profile": "b390-qwen36", "experimental": True})
        self.assertEqual(svc.sampling_defaults, self.cfg["sampling"])
        self.assertFalse(svc.default_enable_thinking)
        self.assertEqual(svc.default_max_tokens, 4096)
        self.assertNotIn("api_key", server.intel_health_fields(svc))
        for backend in (None, "cuda", "hip"):
            self.assertEqual(server.intel_health_fields(SimpleNamespace(backend=backend)), {})
        with self.assertRaises(ValueError):
            self.service({**self.cfg, "vision": {"exe": "unsupported.exe"}})

    def test_uma_never_queries_nvml_or_mutates_vram_or_expert_placement(self):
        svc = self.service()
        with patch("serve.telemetry._Nvml", side_effect=AssertionError("NVML must not be loaded")):
            self.assertIsNone(svc.free_vram_mib())
            svc.check_load_policy()
            svc.min_free_vram_mib = 1
            with self.assertRaises(ValueError):
                svc.check_load_policy()
        with self.assertRaises(ValueError):
            svc.vram(0)
        svc.engine.vram.assert_not_called()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        with patch("serve.model_switch.expert_geometry", side_effect=AssertionError("discrete placement")):
            for operation in (control.expert_placement, lambda: control.start_expert_placement(QWEN, 50),
                              lambda: control.start_expert_count(QWEN, 10)):
                with self.assertRaises(ModelControlError):
                    operation()
        svc.engine.unload.assert_not_called()

    def test_same_model_reload_cannot_change_backend_before_old_model_unload(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        with patch("serve.tinyconfig.validate"), patch("serve.tinyconfig.check_files"), \
             patch.object(server, "StrataEngine") as native:
            with self.assertRaisesRegex(ValueError, "sycl runtime"):
                control._prepare(QWEN, {**self.cfg, "backend": "cuda"})
            native.assert_not_called()
        svc.engine.unload.assert_not_called()
        svc.engine.close.assert_not_called()

    def test_incompatible_profile_or_quantization_keeps_model_and_config_before_prepare(self):
        invalid = []
        for feature in ("mkl", "int8_xmx"):
            changed = capability_report()
            changed["build"][feature] = False
            changed["experimental_profiles"] = {}
            if feature == "int8_xmx":
                changed["weight_types"]["int8_xmx"] = []
            invalid.append(changed)
        for path, missing in (("scalar", 30), ("dp4a", 23), ("int8_xmx", 22)):
            changed = capability_report()
            changed["weight_types"][path].remove(missing)
            invalid.append(changed)
        for changed in invalid:
            self.capabilities = capability_report()
            svc = self.service()
            control = ModelSwitcher(svc, self.cfg, self.path, server)
            engine, before = svc.engine, self.path.read_bytes()
            self.capabilities = changed
            with self.subTest(report=changed), patch.object(control.profiles, "prepare") as prepare, \
                 patch.object(server, "StrataEngine") as native:
                with self.assertRaises(intel_capabilities.IntelCapabilityError):
                    control._prepare(QWEN)
                prepare.assert_not_called()
                native.assert_not_called()
            self.assertIs(svc.engine, engine)
            self.assertTrue(svc.loaded())
            engine.unload.assert_not_called()
            engine.close.assert_not_called()
            self.assertEqual(self.path.read_bytes(), before)

    def test_switch_pipeline_preserves_current_loaded_bundle_on_capability_rejection(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        before = self.path.read_bytes()
        engine = svc.engine
        self.capabilities["weight_types"]["int8_xmx"].remove(21)
        with patch.object(control.profiles, "prepare") as prepare:
            control._switch(QWEN)
            prepare.assert_not_called()
        self.assertEqual(control.status()["state"], "error")
        self.assertIs(control.status()["rolled_back"], True)
        self.assertIs(svc.engine, engine)
        self.assertTrue(svc.loaded())
        engine.unload.assert_not_called()
        engine.close.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_capability_rejection_precedes_switch_worker_and_download_start(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        before = self.path.read_bytes()
        self.capabilities["experimental_profiles"] = {}
        with patch("serve.model_switch.threading.Thread") as thread, \
             patch.object(control, "_downloader") as downloader:
            with self.assertRaises(intel_capabilities.IntelCapabilityError):
                control.download(QWEN, start=True)
            svc.engine.alive.return_value = False
            with self.assertRaises(intel_capabilities.IntelCapabilityError):
                control.start(QWEN)
            thread.assert_not_called()
            downloader.assert_not_called()
        self.assertFalse(control.status()["switching"])
        self.assertEqual(control.status()["state"], "idle")
        self.assertEqual(control._downloads, {})
        svc.engine.unload.assert_not_called()
        svc.engine.close.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_bootstrap_and_activation_check_capabilities_before_manager_or_config_save(self):
        svc = self.service()
        before = self.path.read_bytes()
        self.capabilities["experimental_profiles"] = {}
        with patch("serve.model_download.FixedModelDownloader") as download:
            with self.assertRaises(intel_capabilities.IntelCapabilityError):
                server.configure_model_downloads(svc, self.cfg, self.path)
            download.assert_not_called()
        self.capabilities = capability_report()
        svc.engine = server.UnavailableEngine()
        with patch("serve.model_download.FixedModelDownloader", return_value=SimpleNamespace(closed=False)) as download:
            server.configure_model_downloads(svc, self.cfg, self.path)
            activate = download.call_args.kwargs["on_ready"]
        self.capabilities["weight_types"]["scalar"].remove(30)
        with patch.object(model_setup, "prepare_installed_model") as prepare, \
             patch.object(model_setup, "save_settings") as save, patch.object(server, "StrataEngine") as native:
            with self.assertRaises(intel_capabilities.IntelCapabilityError):
                activate(self.model)
            prepare.assert_not_called()
            save.assert_not_called()
            native.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_server_main_rejects_before_config_migration_save_or_native_start(self):
        before = self.path.read_bytes()
        self.capabilities["experimental_profiles"] = {}
        with patch("sys.argv", ["serve.server", "--engine", "strata", "--config", str(self.path)]), \
             patch.object(model_setup, "save_settings") as save, patch.object(server, "StrataEngine") as native, \
             patch.object(server, "Server") as listener, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                server.main()
            self.assertEqual(stopped.exception.code, 2)
            save.assert_not_called()
            native.assert_not_called()
            listener.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_legacy_qwen_compatibility_is_explicit_and_other_backends_do_not_probe(self):
        self.capabilities = {key: value for key, value in capability_report().items()
                             if key not in ("capabilities_schema", "experimental_models", "experimental_profiles",
                                            "build", "weight_types", "context_auto", "max_context")}
        self.cfg["args"][self.cfg["args"].index("--max-context") + 1] = "32768"
        self.path.write_text(json.dumps(self.cfg), encoding="utf-8")
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        control._validate_intel_runtime(QWEN)
        self.assertEqual(svc.intel_profile, "b390-qwen36")
        with patch.object(model_setup, "validate_intel_executable", side_effect=AssertionError("Intel probe")):
            for backend in ("cuda", "hip"):
                self.service({**self.cfg, "backend": backend})
            ordinary = ModelSwitcher.__new__(ModelSwitcher)
            ordinary._intel = False
            ordinary._validate_intel_runtime(QWEN)

    def test_normal_bootstrap_download_is_text_only(self):
        svc = self.service()
        manager = SimpleNamespace(status=Mock(return_value={"state": "idle"}))
        with patch("serve.model_download.FixedModelDownloader", return_value=manager) as text, \
             patch("serve.model_download.FixedModelBundleDownloader", side_effect=AssertionError("image download")):
            self.assertIs(server.configure_model_downloads(svc, self.cfg, self.path), manager)
            self.assertEqual(text.call_args.kwargs["_manifest"], FIXED_MODEL)
            control = ModelSwitcher(svc, self.cfg, self.path, server)
            self.assertIs(control._downloader(QWEN), manager)

    def test_initial_qwen_and_tiel_download_select_activate_and_cache_own_artifact(self):
        from serve import model_download as downloads
        self.capabilities = capability_report(tiel=True)
        resolve = downloads.manifest_for_arch
        class FixtureResponse(io.BytesIO):
            status = 200
            def __init__(self, payload):
                super().__init__(payload)
                self.headers = {'Content-Length': str(len(payload))}
        class PreparedBoundary(RuntimeError):
            pass
        for model in (QWEN, TIEL):
            for exists in (False, True):
                with self.subTest(model=model, existing=exists):
                    root = self.root / f'{model}-{exists}'
                    root.mkdir()
                    artifact = resolve('qwen35moe', model)
                    payload = model.encode('ascii')
                    local = root / artifact.filename
                    if exists:
                        local.write_bytes(payload)
                    cfg = deepcopy(self.cfg)
                    cfg.update(model_name=model, intel_profile=intel_backend.MODEL_PROFILES[model],
                               download_dir=str(root / 'managed'), model_bootstrap=True)
                    cfg['args'][cfg['args'].index('--model') + 1] = str(local)
                    path = root / ('config-sycl.json' if model == QWEN else 'config-tiel-coder-sycl.json')
                    path.write_text(json.dumps(cfg), encoding='utf-8')
                    before = path.read_bytes()
                    svc = self.service(cfg)
                    svc.engine = server.UnavailableEngine()
                    def tiny_manifest(arch, selected):
                        real = resolve(arch, selected)
                        content = selected.encode('ascii')
                        return replace(real, size=len(content), sha256=hashlib.sha256(content).hexdigest())
                    prepared = []
                    def prepare(latest, config_path, selected_path, **kwargs):
                        prepared.append((latest['model_name'], selected_path))
                        raise PreparedBoundary('intentional GGUF/native boundary')
                    with patch.object(downloads, 'manifest_for_arch', side_effect=tiny_manifest), \
                         patch.object(downloads.urllib.request, 'urlopen', return_value=FixtureResponse(payload)) as http, \
                         patch.object(model_setup, 'prepare_installed_model', side_effect=prepare), \
                         patch('serve.tinyconfig.validate'), \
                         patch.object(server, 'StrataEngine', side_effect=AssertionError('No native')) as native:
                        manager = server.configure_model_downloads(svc, cfg, path)
                        control = ModelSwitcher(svc, cfg, path, server)
                        svc.models = control
                        self.assertIs(control._downloader(model), manager)
                        self.assertEqual(manager.manifest.filename, artifact.filename)
                        manager._run()  # synchronous real tiny SHA/transfer/activation path
                        self.assertEqual(prepared, [(model, local if exists else manager.target)])
                        self.assertEqual(http.call_count, 0 if exists else 1)
                        self.assertIn('intentional GGUF/native boundary', manager.status()['error'])
                        native.assert_not_called()
                        manager.close()
                    self.assertEqual(path.read_bytes(), before)
                    if exists:
                        self.assertEqual(local.read_bytes(), payload)

    def test_tiel_profile_uses_own_files_sampler_effort_and_runtime_contract(self):
        self.capabilities = capability_report(tiel=True)
        tiel = self.root / FIXED_TIEL_MODEL.filename
        tiel.write_bytes(b'CPU-only pinned-basename fixture')
        svc = self.service()
        svc.api_key = 'existing-auth-must-stay'
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        profile = control.profiles.get(TIEL)
        self.assertEqual(profile.path.name, 'config-tiel-coder-sycl.json')
        self.assertEqual(profile.config['intel_profile'], 'b390-tiel-coder')
        self.assertEqual(profile.config['sampling']['presence_penalty'], 0.0)
        self.assertEqual(profile.config['reasoning_effort'], 'medium')
        self.assertEqual(model_setup.model_path(profile.config), tiel)
        self.assertEqual(profile.config['exe'], self.cfg['exe'])
        cfg = {**profile.config, 'tokenizer': str(self.root / 'tiel-tokenizer')}
        native = server.MockEngine(server.ByteTokenizer(), '', max_context=8192)
        native.info = {'backend': 'sycl', 'memory_model': 'uma'}
        with patch.object(control.profiles, 'prepare', return_value=cfg), \
             patch.object(server, 'model_tokenizer', return_value=server.ByteTokenizer()), \
             patch.object(server, 'ChatTemplate', return_value=SimpleNamespace(call_format='function')), \
             patch.object(server, 'effort_end_args', return_value=None), \
             patch.object(server, 'StrataEngine', return_value=native):
            bundle = control._prepare(TIEL)
        self.assertEqual(bundle.fresh.config_effort, 'medium')
        self.assertEqual(bundle.fresh.sampling_defaults['presence_penalty'], 0.0)
        control._publish(bundle)
        self.assertEqual(server.intel_health_fields(svc)['intel_profile'], 'b390-tiel-coder')
        self.assertEqual(svc.model, TIEL)
        self.assertEqual(svc.api_key, 'existing-auth-must-stay')
        manager = SimpleNamespace(status=Mock(return_value={'state': 'idle'}))
        with patch('serve.model_download.FixedModelDownloader', return_value=manager) as text, \
             patch('serve.model_download.FixedModelBundleDownloader', side_effect=AssertionError('No images')):
            self.assertIs(control._downloader(TIEL), manager)
        self.assertEqual(text.call_args.kwargs['_manifest'], FIXED_TIEL_MODEL)

    def test_tiel_old_runtime_or_missing_type18_rejects_before_job_prepare_or_unload(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
        before = self.path.read_bytes()
        reports = [capability_report()]
        for path in ('scalar', 'dp4a', 'int8_xmx'):
            candidate = capability_report(tiel=True)
            candidate['weight_types'][path].remove(18)
            reports.append(candidate)
        for report in reports:
            self.capabilities = report
            with self.subTest(report=report), patch.object(control.profiles, 'prepare') as prepare, \
                 patch('serve.model_switch.threading.Thread') as worker, patch.object(control, '_downloader') as downloader:
                with self.assertRaises(intel_capabilities.IntelCapabilityError):
                    control.start(TIEL)
                with self.assertRaises(intel_capabilities.IntelCapabilityError):
                    control.download(TIEL, start=True)
                with self.assertRaises(intel_capabilities.IntelCapabilityError):
                    control._prepare(TIEL)
                prepare.assert_not_called()
                worker.assert_not_called()
                downloader.assert_not_called()
            self.assertTrue(svc.loaded())
            svc.engine.unload.assert_not_called()
            svc.engine.close.assert_not_called()
            self.assertEqual(self.path.read_bytes(), before)

    def test_tiel_header_and_basename_are_checked_before_tokenizer_or_config_write(self):
        self.capabilities = capability_report(tiel=True)
        cfg = intel_backend.configure_backend({**self.cfg, 'model_name': TIEL, 'intel_profile': 'b390-tiel-coder'})
        good = self.root / FIXED_TIEL_MODEL.filename
        wrong = self.root / 'another-model.gguf'
        good.write_bytes(b'CPU-only header fixture')
        wrong.write_bytes(b'CPU-only header fixture')
        path = self.root / 'config-tiel-coder-sycl.json'
        before = self.path.read_bytes()
        for model, general_name in ((good, 'Qwen3.6-35B'), (wrong, 'Ornith-1.5-35B')):
            with self.subTest(model=model, name=general_name), \
                 patch.object(model_setup, 'model_architecture', return_value='qwen35moe'), \
                 patch('serve.gguf_reader.GGUFFile', return_value=SimpleNamespace(metadata={'general.name': general_name})), \
                 patch.object(model_setup, '_extract_tokenizer') as extract, patch.object(model_setup, 'save_settings') as save:
                with self.assertRaisesRegex(ValueError, 'identity'):
                    model_setup.prepare_installed_model(cfg, path, model)
                extract.assert_not_called()
                save.assert_not_called()
        def fake_extract(model, cache):
            tokenizer = cache / 'tokenizer'
            tokenizer.mkdir(parents=True)
            for name in model_setup.TOKENIZER_FILES:
                (tokenizer / name).write_text('{}', encoding='utf-8')
        with patch.object(model_setup, 'model_architecture', return_value='qwen35moe'), \
             patch('serve.gguf_reader.GGUFFile', return_value=SimpleNamespace(metadata={'general.name': 'Ornith-1.5-35B'})), \
             patch.object(model_setup, '_extract_tokenizer', side_effect=fake_extract):
            prepared = model_setup.prepare_installed_model(cfg, path, good)
        self.assertEqual(Path(prepared['log']).name, 'engine-intel-tiel.log')
        self.assertEqual(prepared['model_name'], TIEL)
        self.assertTrue(path.is_file())
        self.assertEqual(self.path.read_bytes(), before)

    def test_health_http_exposes_only_nonsecret_intel_identity(self):
        svc = self.service()
        with patch.object(svc, "start_telemetry"):
            httpd = server.serve(svc, "127.0.0.1", 0)
        try:
            client = http.client.HTTPConnection("127.0.0.1", httpd.server_port, timeout=3)
            try:
                client.request("GET", "/health")
                response = client.getresponse()
                body = json.loads(response.read())
            finally:
                client.close()
            self.assertEqual(response.status, 200)
            self.assertEqual(body["backend"], "sycl")
            self.assertEqual(body["intel_profile"], "b390-qwen36")
            self.assertIs(body["experimental"], True)
            self.assertNotIn(self.cfg["api_key"], json.dumps(body))
            self.assertFalse(body["images"])
            self.assertFalse(body["features"]["images"])
            self.assertFalse(body["features"]["assistant"])
        finally:
            svc.stopping.set()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
