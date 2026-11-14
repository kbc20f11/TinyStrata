"""CPU-only normal-server/profile integration; never start a native engine."""
from copy import deepcopy
import http.client
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from serve import intel_backend, model_setup, server
from serve.model_profiles import ModelProfiles
from serve.model_switch import ModelControlError, ModelSwitcher
from serve.model_download import FIXED_MODEL


QWEN = "qwen3.6-35b-a3b"
TIEL = "tiel-coder-35b-a3b"


class IntelServerIntegration(unittest.TestCase):
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

    def service(self, cfg=None):
        engine = server.MockEngine(server.ByteTokenizer(), "", max_context=8192)
        engine.info = {"backend": "sycl", "memory_model": "uma"}
        engine.alive = Mock(return_value=True)
        engine.unload, engine.close, engine.vram = Mock(), Mock(), Mock()
        return server.Service(engine, server.ByteTokenizer(), SimpleNamespace(call_format="function"),
                              generation_defaults=cfg or self.cfg,
                              sampling_defaults=(cfg or self.cfg).get("sampling"))

    def test_native_args_resolve_auto_without_cuda_learning_or_duplicate_serve(self):
        original = deepcopy(self.cfg)
        with patch.object(server, "learned_profile_args", side_effect=AssertionError("CUDA learning")), \
             patch.object(server, "ordered_gpus", side_effect=AssertionError("CUDA ordinal")):
            args = server.engine_args(self.cfg)
        self.assertNotIn("--serve", args)
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(args[args.index("--kv-type") + 1], "int8")
        self.assertEqual(args[args.index("--draft-vocab") + 1], "2048")
        self.assertNotIn("--expert-profile", args)
        self.assertEqual(self.cfg, original)

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
        self.assertEqual([row["id"] for row in profiles.list()], [QWEN])
        profile = profiles.get("qwen36")
        self.assertEqual(profile.path, self.path)
        for key in ("exe", "lib_dirs", "intel_profile", "sampling", "enable_thinking", "api_key", "aliases", "history_preference"):
            self.assertEqual(profile.config[key], self.cfg[key])
        self.assertFalse(profiles.list()[0]["vision"])
        self.assertEqual(profiles.context_info(QWEN), {"allocated_tokens": None, "model_limit": 32768, "requested": "auto"})
        for path, data in others.items():
            self.assertEqual(path.read_bytes(), data)

    def test_tiel_and_other_models_reject_before_job_start_or_unload(self):
        svc = self.service()
        control = ModelSwitcher(svc, self.cfg, self.path, server)
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
        with self.assertRaisesRegex(ValueError, "Qwen3.6"):
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

    def test_normal_bootstrap_download_is_text_only(self):
        svc = self.service()
        manager = SimpleNamespace(status=Mock(return_value={"state": "idle"}))
        with patch("serve.model_download.FixedModelDownloader", return_value=manager) as text, \
             patch("serve.model_download.FixedModelBundleDownloader", side_effect=AssertionError("image download")):
            self.assertIs(server.configure_model_downloads(svc, self.cfg, self.path), manager)
            self.assertEqual(text.call_args.kwargs["_manifest"], FIXED_MODEL)
            control = ModelSwitcher(svc, self.cfg, self.path, server)
            self.assertIs(control._downloader(QWEN), manager)

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
        finally:
            svc.stopping.set()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
