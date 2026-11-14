"""CPU-only normal-server/profile integration; never start a native engine."""
from copy import deepcopy
from contextlib import redirect_stderr
import http.client
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
from serve.model_download import FIXED_MODEL, FIXED_TIEL_MODEL


QWEN = "qwen3.6-35b-a3b"
TIEL = "tiel-coder-35b-a3b"


def capability_report(*, tiel=False):
    result = {"backend": "sycl", "experimental": True, "model_families": [],
            "kv_types": ["fp32", "int8"], "images": False, "mtp": True,
            "experimental_mtp_max_depth": 7, "serve": True, "capabilities_schema": 1,
            "experimental_models": [QWEN], "experimental_profiles": {"b390-qwen36": 1},
            "build": {"mkl": True, "int8_xmx": True, "aot_target": "intel_gpu_ptl_h"},
            "weight_types": {"scalar": [0, 1, 8, 10, 11, 14, 18, 21, 22, 23, 30],
                             "dp4a": [8, 10, 11, 14, 18, 21, 22, 23], "int8_xmx": [14, 18, 21, 22]}}
    if tiel:
        result["experimental_models"].append(TIEL)
        result["experimental_profiles"]["b390-tiel-coder"] = 1
    return result


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
        self.assertEqual([row["id"] for row in profiles.list()], [QWEN, TIEL])
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
                                            "build", "weight_types")}
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
        finally:
            svc.stopping.set()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
