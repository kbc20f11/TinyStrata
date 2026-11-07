"""CPU-only benchmark profile migration tests; no services, files or GPU work."""
import copy
import importlib.util
import json
import io
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import config, tinyconfig

spec = importlib.util.spec_from_file_location("benchmark_profile_test_module", ROOT / "benchmark/run.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
CASES = json.loads((ROOT / "benchmark/cases.json").read_text(encoding="utf-8"))


def original(model):
    cfg = {"arch": bench.MODELS[model][1], "exe": "engine.exe", "tokenizer": "tokenizer",
           "args": ["--model", "model.gguf", "--max-context", "32768", "--chunk", "512"],
           "vision": {"exe": "vision.exe", "mmproj": "projector.gguf", "model": "model.gguf"}}
    if model == "gemma":
        cfg["args"] += ["--draft-model", "draft.gguf"]
    return cfg


def controlled(model):
    return bench.controlled_config(original(model), model, ROOT / "unused-test-output", 18001, CASES)


class BenchmarkProfileMigration(unittest.TestCase):
    def test_saved_profile_preserves_native_sampler_vision_and_source(self):
        before = original('clover')
        before.update(api_key='private-production-key', backend='hip', enable_thinking=False,
                      sampling={'temperature': .37, 'top_p': .91, 'presence_penalty': .2, 'penalty_last_n': 128},
                      env={'STRATA_IQ_MT_MIN': '2', 'STRATA_API_KEY': 'private-key'}, vision_mode='gpu')
        before['args'] += ['--kv-type', 'fp8-mma', '--mtp', '0', '--vision', '--cpu-threads', '7']
        snapshot = copy.deepcopy(before)
        cfg = bench.saved_config(before, 'clover', ROOT / 'unused-private-output', 18019, CASES)
        self.assertEqual(cfg['args'], before['args'])
        self.assertEqual(cfg['sampling'], before['sampling'])
        self.assertEqual(cfg['vision'], before['vision'])
        self.assertFalse(cfg['enable_thinking'])
        self.assertEqual(cfg['env']['STRATA_IQ_MT_MIN'], '2')
        self.assertNotIn('api_key', cfg)
        self.assertNotIn('STRATA_API_KEY', cfg['env'])
        self.assertEqual(cfg['port'], 18019)
        cfg['vision']['gpu'] = False
        cfg['sampling']['temperature'] = 9
        self.assertEqual(before, snapshot)

    def test_saved_profile_registry_and_generation_counts_are_separate(self):
        self.assertEqual(bench.benchmark_models('all'), list(bench.MODELS))
        self.assertEqual(bench.benchmark_models('all', True), [*bench.MODELS, 'tiel'])
        with self.assertRaises(ValueError): bench.benchmark_models('tiel')
        self.assertEqual(bench.generation_passes(False), (1, 2))
        self.assertEqual(bench.generation_passes(True), (1,))

    def test_saved_sampler_overrides_clover_protocol_without_replacing_workload_controls(self):
        sampling = {'temperature': .37, 'top_k': 17, 'penalty_last_n': 128,
                    'seed': 9, 'max_tokens': 99999, 'strata_checkpoint': True}
        request = bench.request_body('clover', 'question', 32, CASES, thinking=False, sampling=sampling)
        self.assertEqual((request['temperature'], request['top_k'], request['penalty_last_n']), (.37, 17, 128))
        self.assertEqual((request['seed'], request['max_tokens'], request['strata_checkpoint']), (1234, 32, False))
        self.assertFalse(request['enable_thinking'])
        self.assertEqual(sampling['seed'], 9)
        self.assertEqual(bench.request_body('tiel', 'q', 32, CASES)['model'], 'tiel-coder-35b-a3b')

    def test_service_requests_graceful_unload_before_force_cleanup(self):
        events = []
        class Process:
            ended = False
            def poll(self): return 0 if self.ended else None
            def kill(self): events.append('kill_api'); self.ended = True
            def wait(self, timeout): return 0
        service = bench.Service.__new__(bench.Service)
        service.process = Process()
        service.remember = Mock()
        service.stop, service.watcher = Mock(), Mock()
        service.owned, service.lock = {}, threading.Lock()
        service.stdout, service.stderr = io.StringIO(), io.StringIO()
        def json_call(path, body=None, timeout=None):
            events.append(path)
            if path == '/unload': return {'status': 'unloaded'}
            return {'loaded': events.count('/health') == 1}
        service.json = json_call
        with patch.object(bench.psutil, 'wait_procs', return_value=([], [])):
            service.close()
        self.assertLess(events.index('/unload'), events.index('kill_api'))
        self.assertTrue(service.cleanup['graceful_unloaded'])
        self.assertFalse(service.cleanup['forced_fallback'])

    def test_benchmark_requests_are_one_shot_without_parking(self):
        cases = json.loads((ROOT / 'benchmark/cases.json').read_text(encoding='utf-8'))
        for model in bench.MODELS:
            with self.subTest(model=model):
                body = bench.request_body(model, 'fresh input', 128, cases)
                self.assertIs(body['strata_checkpoint'], False)
                self.assertTrue(body['stream'])

    def test_explicit_expert_vram_survives_api_startup_migration(self):
        for model in bench.MODELS:
            with self.subTest(model=model):
                cfg = controlled(model)
                cfg["args"] += ["--expert-vram", "6"]
                after, _ = tinyconfig.migrate_settings(cfg)
                self.assertEqual(config.value_of(after, "expert_vram"), 6)

    def test_explicit_expert_slots_survive_api_startup_migration(self):
        cfg = controlled("qwen")
        cfg["args"] += ["--expert-slots", "64"]
        after, _ = tinyconfig.migrate_settings(cfg)
        at = after["args"].index("--expert-slots")
        self.assertEqual(after["args"][at + 1], "64")

    def test_manual_chunk512_is_not_migrated_to_auto(self):
        for model in bench.MODELS:
            with self.subTest(model=model):
                cfg = controlled(model)
                # A caller may already have protected its fixed Expert budget.
                cfg["memory_policy_version"] = tinyconfig.MEMORY_POLICY_VERSION
                cfg["args"][cfg["args"].index("--chunk") + 1] = "512"
                after, _ = tinyconfig.migrate_settings(cfg)
                self.assertEqual(config.value_of(after, "chunk"), 512)

    def test_markers_follow_the_current_policy_versions(self):
        with patch.object(tinyconfig, "MEMORY_POLICY_VERSION", 17), \
             patch.object(tinyconfig, "CHUNK_POLICY_VERSION", 18), \
             patch.object(tinyconfig, "GENERATION_POLICY_VERSION", 19):
            cfg = controlled("qwen")
            self.assertEqual((cfg.get("memory_policy_version"), cfg.get("chunk_policy_version"),
                              cfg.get("generation_policy_version")), (17, 18, 19))

    def test_sampling_and_automatic_defaults_are_retained_without_mutating_source(self):
        before = original("qwen")
        snapshot = copy.deepcopy(before)
        cfg = bench.controlled_config(before, "qwen", ROOT / "unused-test-output", 18001, CASES)
        cfg["sampling"].update(temperature=0.37, presence_penalty=0.12)
        after, _ = tinyconfig.migrate_settings(cfg)
        self.assertEqual(after["sampling"], cfg["sampling"])
        self.assertEqual(config.value_of(after, "chunk"), "auto")
        self.assertIsNone(config.value_of(after, "expert_vram"))
        self.assertEqual(before, snapshot)


if __name__ == "__main__":
    unittest.main()
