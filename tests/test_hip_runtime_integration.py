"""Update/runtime integration regressions; no native processes or GPU calls."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run as launcher
from serve import backend, hip_runtime, install_guard, model_setup, server
from serve.hip_runtime import HipRuntime, RUNTIME_ENV


class AutomaticDeviceRefresh(unittest.TestCase):
    def setUp(self):
        self.mask = patch.dict(os.environ, {"HIP_VISIBLE_DEVICES":"", "ROCR_VISIBLE_DEVICES":""})
        self.mask.start()
        self.addCleanup(self.mask.stop)
        self.platform = patch.object(launcher.os, "name", "nt")
        self.platform.start()
        self.addCleanup(self.platform.stop)
        self.cfg = {"backend":"hip", "exe":"selected.exe", "hip_ordinal":0, "lib_dirs":["sdk"]}

    def test_driver_reorder_refreshes_auto_without_mutating_input(self):
        with patch.object(backend,"hip_ordinal",return_value=1) as probe:
            selected = launcher.refresh_hip_device(self.cfg)
        self.assertEqual(selected["hip_ordinal"],1)
        self.assertEqual(self.cfg["hip_ordinal"],0)
        probe.assert_called_once_with(["sdk"],executable="selected.exe")

    def test_explicit_gpu_zero_does_not_autoselect(self):
        with patch.object(backend,"hip_ordinal") as probe:
            selected = launcher.refresh_hip_device({**self.cfg,"gpu":0})
        probe.assert_not_called()
        self.assertEqual(selected["gpu"],0)

    def test_mask_does_not_keep_old_unmasked_ordinal(self):
        with patch.dict(os.environ,{"HIP_VISIBLE_DEVICES":"2"}), patch.object(backend,"hip_ordinal") as probe:
            selected = launcher.refresh_hip_device(self.cfg)
        self.assertNotIn("hip_ordinal",selected)
        probe.assert_not_called()

    def test_profile_mask_does_not_keep_old_unmasked_ordinal(self):
        with patch.object(backend,"hip_ordinal") as probe:
            selected = launcher.refresh_hip_device({**self.cfg,"env":{"HIP_VISIBLE_DEVICES":"1"}})
        self.assertNotIn("hip_ordinal",selected)
        probe.assert_not_called()

    def test_failed_probe_never_assumes_device_zero(self):
        with patch.object(backend,"hip_ordinal",return_value=None):
            with self.assertRaisesRegex(ValueError,"device 0 was not assumed"):
                launcher.refresh_hip_device(self.cfg)


class RuntimeWiring(unittest.TestCase):
    def test_child_env_aligns_runtime_before_environment_guard(self):
        cfg={"backend":"hip","exe":"engine.exe","hip_ordinal":1}
        def align(value,env):
            self.assertEqual(value,cfg)
            return {**env,RUNTIME_ENV:"selected.dll"}
        with patch.object(hip_runtime,"runtime_environment",side_effect=align), patch.object(server,"hip_env_guard") as guard:
            value=server.child_env(cfg)
        self.assertEqual(value[RUNTIME_ENV],"selected.dll")
        self.assertEqual(guard.call_args.args[0][RUNTIME_ENV],"selected.dll")

    def test_saved_other_family_uses_managed_sdk_before_old_dirs(self):
        cfg={"backend":"hip","arch":"lfm2moe","exe":"engine.exe","lib_dirs":["old-sdk"]}
        with patch.object(model_setup.os,"name","nt"), patch.object(hip_runtime,"resolve_hip_runtime",return_value=HipRuntime("new.dll",("new-sdk",),True)):
            value=model_setup.shared_library_dirs(cfg)
        self.assertEqual(value["lib_dirs"],["new-sdk","old-sdk"])
        self.assertEqual(cfg["lib_dirs"],["old-sdk"])

    def test_vision_resolves_its_own_executable_and_mask(self):
        cfg={"backend":"hip","exe":"main.exe","vision":{"exe":"vision.exe","cuda_device":2}}
        with patch.object(hip_runtime,"resolve_hip_runtime",return_value=None), patch.object(hip_runtime,"runtime_environment",side_effect=lambda c,e:{**e,"selected_exe":c["exe"]}):
            value=server.vision_env(cfg,{"HIP_VISIBLE_DEVICES":"1"})
        self.assertEqual(value["selected_exe"],"vision.exe")
        self.assertEqual(value["HIP_VISIBLE_DEVICES"],"2")

    def test_gpu_encoder_runtime_mismatch_stops_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            left,right=Path(directory)/"main.dll",Path(directory)/"vision.dll"
            left.write_bytes(b"core-A");right.write_bytes(b"core-B")
            cfg={"backend":"hip","exe":"main.exe","vision":{"exe":"vision.exe","gpu":True}}
            with patch.object(hip_runtime,"resolve_hip_runtime",side_effect=[HipRuntime(str(left),(),False),HipRuntime(str(right),(),False)]):
                with self.assertRaisesRegex(ValueError,"different HIP runtimes"):
                    server.vision_env(cfg,{})

    def test_same_core_but_different_math_asset_directories_stops_gpu_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory) / "main.dll", Path(directory) / "vision.dll"
            left.write_bytes(b"same core"); right.write_bytes(b"same core")
            cfg = {"backend": "hip", "exe": "main.exe", "vision": {"exe": "vision.exe", "gpu": True}}
            selected = [HipRuntime(str(left), ("SDK-A",), False), HipRuntime(str(right), ("SDK-B",), False)]
            with patch.object(hip_runtime, "resolve_hip_runtime", side_effect=selected):
                with self.assertRaisesRegex(ValueError, "SDK dependencies"):
                    server.vision_env(cfg, {})

    def test_invalid_settings_do_not_publish_intermediate_runtime_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg_path = Path(directory) / "config.json"
            original = {"backend": "hip", "exe": "old.exe", "model_bootstrap": True}
            cfg_path.write_text(json.dumps(original), encoding="utf-8")
            replacement = {**original, "exe": "new-generation.exe"}
            with patch("sys.argv", ["serve.server", "--engine", "tinystrata", "--config", str(cfg_path)]), \
                    patch.object(hip_runtime, "migrate_installed_runtime", return_value=replacement), \
                    patch("serve.tinyconfig.migrate_settings", return_value=(replacement, True)), \
                    patch("serve.tinyconfig.validate", side_effect=ValueError("invalid saved settings")), \
                    patch.object(model_setup, "save_settings") as save, \
                    patch.object(server.config, "save") as raw_save, \
                    patch.object(server, "Server") as listener, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    server.main()
            save.assert_not_called(); raw_save.assert_not_called(); listener.assert_not_called()
            self.assertEqual(json.loads(cfg_path.read_text(encoding="utf-8")), original)

    def test_direct_server_gpu_override_discards_saved_ordinal(self):
        tree=ast.parse(Path(server.__file__).read_text(encoding="utf-8"))
        main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="main")
        block=next(node for node in main.body if isinstance(node,ast.If) and ast.unparse(node.test)=="a.gpu is not None")
        namespace={"a":SimpleNamespace(gpu="1"),"cfg":{"hip_ordinal":0}}
        exec(compile(ast.Module(body=[block],type_ignores=[]),"server-gpu-override","exec"),namespace)
        self.assertEqual(server.hip_visible(namespace["cfg"]),[1])

    def test_api_update_guard_runs_before_configuration_or_listener(self):
        with patch("sys.argv",["serve.server","--engine","mock","--config","missing.json"]), patch.object(install_guard,"assert_installation_ready",side_effect=ValueError("update active")), patch.object(server,"Server") as listener, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                server.main()
        self.assertEqual(error.exception.code,2)
        listener.assert_not_called()

    def test_launcher_update_guard_runs_before_dependency_repair(self):
        with patch("sys.argv",["run.py"]), patch.object(install_guard,"assert_installation_ready",side_effect=ValueError("update active")), patch.object(launcher.importlib.util,"find_spec") as dependency, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                launcher.main()
        self.assertEqual(error.exception.code,2)
        dependency.assert_not_called()


class EngineQuitBudget(unittest.TestCase):
    def make_engine(self, facts, timeouts=0):
        class Process:
            def __init__(self):
                self.stdin, self.stdout = io.StringIO(), io.StringIO()
                self.waits, self.actions = [], []
                self.returncode = None
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                self.waits.append(timeout)
                if len(self.waits) <= timeouts:
                    raise subprocess.TimeoutExpired("mock-native", timeout)
                self.returncode = 0
                return 0
            def terminate(self):
                self.actions.append("terminate")
            def kill(self):
                self.actions.append("kill")
        process = Process()
        engine = object.__new__(server.StrataEngine)
        engine.proc, engine.info, engine.pump, engine.log = process, facts, None, None
        engine.batch = False
        return engine, process

    def test_only_verified_small_hip_gets_longer_graceful_quit(self):
        cases = [(None, 20), ({}, 20),
                 ({"hip_vmm_verified": 0, "hip_vmm_quantum_kib": 64}, 20),
                 ({"hip_vmm_verified": 1, "hip_vmm_quantum_kib": 4096}, 20),
                 ({"hip_vmm_verified": 1, "hip_vmm_quantum_kib": 0}, 20),
                 ({"hip_vmm_verified": 1, "hip_vmm_quantum_kib": "64"}, 20),
                 ({"hip_vmm_verified": 1, "hip_vmm_quantum_kib": 64}, 60),
                 ({"hip_vmm_verified": 1, "hip_vmm_quantum_kib": 1024}, 60)]
        for facts, expected in cases:
            with self.subTest(facts=facts):
                engine, process = self.make_engine(facts)
                engine.close()
                self.assertEqual(process.waits, [expected])
                self.assertEqual(process.actions, [])
                self.assertIsNone(engine.proc)

    def test_small_hip_timeout_keeps_bounded_terminate_and_kill_fallbacks(self):
        engine, process = self.make_engine({"hip_vmm_verified": 1, "hip_vmm_quantum_kib": 64}, timeouts=2)
        engine.close()
        self.assertEqual(process.waits, [60, 20, 20])
        self.assertEqual(process.actions, ["terminate", "kill"])
        self.assertIsNone(engine.proc)

    def test_still_stuck_after_all_budgets_is_reported_not_hidden(self):
        engine, process = self.make_engine({"hip_vmm_verified": 1, "hip_vmm_quantum_kib": 64}, timeouts=3)
        with self.assertRaises(server.EngineStuck):
            engine.close()
        self.assertEqual(process.waits, [60, 20, 20])
        self.assertIs(engine.proc, process)


if __name__=="__main__":
    unittest.main()
