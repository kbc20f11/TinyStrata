"""CPU-only benchmark profile migration tests; no services, files or GPU work."""
import copy
import importlib.util
import json
import io
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
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
    cfg = {"arch": bench.model_definition(model)[1], "exe": "engine.exe", "tokenizer": "tokenizer",
           "args": ["--model", "model.gguf", "--max-context", "32768", "--chunk", "512"],
           "vision": {"exe": "vision.exe", "mmproj": "projector.gguf", "model": "model.gguf"}}
    if model == "gemma":
        cfg["args"] += ["--draft-model", "draft.gguf"]
    return cfg


def controlled(model):
    return bench.controlled_config(original(model), model, ROOT / "unused-test-output", 18001, CASES)


class BenchmarkProfileMigration(unittest.TestCase):
    def test_saved_profile_starts_unactivated_models_without_changing_source(self):
        for model in bench.benchmark_models('all', saved_profile=True):
            with self.subTest(model=model):
                before = original(model)
                before.update(model_bootstrap=True, lazy_load=True,
                              sampling={'temperature': .37, 'top_k': 17})
                snapshot = copy.deepcopy(before)
                cfg = bench.saved_config(before, model, ROOT / 'unused-private-output', 18019, CASES)
                self.assertNotIn('model_bootstrap', cfg)
                self.assertNotIn('lazy_load', cfg)
                for key in ('args', 'sampling', 'vision'):
                    self.assertEqual(cfg[key], snapshot[key])
                self.assertEqual(before, snapshot)

    def test_standard_profile_already_starts_unactivated_models(self):
        for model in bench.MODELS:
            with self.subTest(model=model):
                before = original(model)
                before.update(model_bootstrap=True, lazy_load=True)
                snapshot = copy.deepcopy(before)
                cfg = bench.controlled_config(before, model, ROOT / 'unused-private-output', 18019, CASES)
                self.assertNotIn('model_bootstrap', cfg)
                self.assertNotIn('lazy_load', cfg)
                self.assertEqual(before, snapshot)

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
        with self.assertRaisesRegex(ValueError, 'Unsupported benchmark model'):
            bench.benchmark_models('gptoss')
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


class IntelTextBenchmarkVariant(unittest.TestCase):
    def original(self):
        from serve.intel_backend import configure_backend
        return configure_backend({'backend': 'sycl', 'intel_profile': 'b390-qwen36',
            'arch': 'qwen35moe', 'model_name': 'qwen3.6-35b-a3b', 'gpu': 0,
            'exe': 'runtime/tinystrata-intel.exe', 'tokenizer': 'model/tokenizer', 'lib_dirs': ['runtime'],
            'cwd': str(ROOT), 'args': ['--model', 'models/qwen.gguf', '--max-context', 'auto', '--chunk', 'auto',
                                     '--kv-type', 'auto', '--mtp', '3'],
            'sampling': {'temperature': .37, 'presence_penalty': .12}, 'api_key': 'production-key',
            'lazy_load': True, 'model_bootstrap': True, 'history': {'saved': True}})

    def configured(self, original=None):
        return bench.intel_text_config(self.original() if original is None else original,
                                       'qwen', ROOT / 'unused-private-output', 18019, CASES)

    def engine(self):
        return {'backend': 'sycl', 'model': 'qwen3.6-35b-a3b', 'memory_model': 'uma', 'kv': 'int8',
                'mtp': 3, 'max_context': 32768, 'vision': 0, 'device_vendor': 32902, 'device_integrated': 1,
                'expert_cache_bytes': None, 'expert_cache_mib': None, 'kv_grow': None}

    def health(self):
        return {'backend': 'sycl', 'intel_profile': 'b390-qwen36', 'experimental': True,
                'model': 'qwen3.6-35b-a3b', 'images': False}

    def test_explicit_variant_is_separate_and_only_admitted_models_are_allowed(self):
        self.assertEqual(bench.MODELS['qwen'][0], 'config.json')
        self.assertEqual(bench.INTEL_TEXT_CONFIGS['qwen'], 'config-sycl.json')
        self.assertEqual(bench.benchmark_models('qwen', intel_text=True), ['qwen'])
        self.assertEqual(bench.benchmark_models('tiel', intel_text=True), ['tiel'])
        for model in ('all', 'gemma', 'lfm', 'clover'):
            with self.subTest(model=model), self.assertRaises(ValueError):
                bench.benchmark_models(model, intel_text=True)
        with patch('serve.intel_backend.supports_model', return_value=False), self.assertRaises(ValueError):
            bench.benchmark_models('tiel', intel_text=True)

    def test_tiel_variant_uses_model_specific_profile_and_sampling(self):
        from serve.intel_backend import configure_backend, engine_args
        original = self.original()
        original.update(model_name='tiel-coder-35b-a3b', intel_profile='b390-tiel-coder')
        original = configure_backend(original)
        cfg = bench.intel_text_config(original, 'tiel', ROOT / 'unused', 18019, CASES)
        self.assertEqual(cfg['sampling'], tinyconfig.TIEL_SAMPLING)
        self.assertEqual(cfg['sampling']['presence_penalty'], 0.0)
        self.assertIn('--experimental-tiel', engine_args(cfg))
        self.assertEqual(bench.INTEL_TEXT_CONFIGS['tiel'], 'config-tiel-coder-sycl.json')
        for planner in (bench.controlled_config, bench.saved_config):
            with self.assertRaisesRegex(ValueError, '--intel-text'):
                planner(self.original(), 'qwen', ROOT / 'unused', 18019, CASES)

    def test_fixed_context_retains_profile_and_canonical_workload_without_source_mutation(self):
        from serve.intel_backend import engine_args
        before = self.original()
        snapshot = copy.deepcopy(before)
        with patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('No process')):
            cfg = self.configured(before)
        actual = engine_args(cfg)
        for flag, value in (('--max-context', '32768'), ('--chunk', '1024'), ('--kv-type', 'int8'),
                            ('--mtp', '3'), ('--draft-vocab', '2048')):
            self.assertEqual(actual[actual.index(flag) + 1], value)
        self.assertEqual(config.value_of(cfg, 'chunk'), 'auto')
        self.assertNotIn('--kv-grow', actual)
        self.assertNotIn('--vision', actual)
        self.assertNotIn('vision', cfg)
        self.assertEqual(cfg['vision_mode'], 'no')
        self.assertEqual(cfg['sampling'], CASES['sampling']['qwen'])
        self.assertEqual(cfg['intel_profile'], before['intel_profile'])
        for key in ('api_key', 'model_bootstrap', 'lazy_load', 'history'):
            self.assertNotIn(key, cfg)
        self.assertEqual(before, snapshot)
        self.assertEqual(bench.generation_passes(False), (1, 2))
        body = bench.request_body('qwen', CASES['generation'][0]['text'], CASES['generation_max_tokens'], CASES)
        self.assertEqual((body['seed'], body['max_tokens'], body['temperature']), (1234, 128, .7))
        self.assertEqual(body['presence_penalty'], 1.5)
        self.assertFalse(body['strata_checkpoint'])
        self.assertTrue(body['enable_thinking'])

    def test_incompatible_saved_flags_and_backends_are_rejected(self):
        changes = [{'backend': 'cuda'}, {'backend': 'hip'}, {'intel_profile': 'generic'},
                   {'vision_mode': 'gpu'}, {'env': {'STRATA_TUNING': '1'}}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.configured({**self.original(), **change})
        for setting, value in (('mtp', 0), ('kv_type', 'fp32'), ('chunk', 128), ('chunk', 512)):
            modified, _ = config.apply(self.original(), {setting: value})
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.configured(modified)
        for flag in ('--profile', '--no-graph', '--verify-graph'):
            modified = self.original()
            modified['args'].append(flag)
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.configured(modified)
        modified = self.original()
        modified['args'] += ['--draft-vocab', '4096']
        with self.assertRaisesRegex(ValueError, 'draft-vocab'):
            self.configured(modified)

    def test_conflicting_cli_options_fail_before_output_or_service_creation(self):
        conflicts = [('--saved-profile',), ('--include-64k',), ('--clover-vision',),
                     ('--clover-mode', 'self-spec'), ('--thinking', 'off'), ('--kv-type', 'int8'),
                     ('--chunk', '512'), ('--cpu-workers', '2'), ('--engine-env', 'STRATA_X=1')]
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'must-not-exist'
            for extra in conflicts:
                command = ['benchmark/run.py', '--intel-text', '--model', 'qwen', '--output', str(output), *extra]
                with self.subTest(extra=extra), patch.object(sys, 'argv', command), \
                     patch.object(bench, 'Service', side_effect=AssertionError('No service')), \
                     patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as stop:
                    bench.main()
                self.assertEqual(stop.exception.code, 2)
                self.assertFalse(output.exists())

    def test_actual_api_profile_fixed_kv_and_uma_metadata_are_required(self):
        cfg, info, health = self.configured(), self.engine(), self.health()
        bench.validate_intel_text_engine(info, health, cfg)
        for change in ({'backend': 'cuda'}, {'intel_profile': 'generic'}, {'experimental': False}, {'images': True}):
            with self.subTest(health=change), self.assertRaises(ValueError):
                bench.validate_intel_text_engine(info, {**health, **change}, cfg)
        for change in ({'backend': 'cuda'}, {'mtp': 0}, {'kv': 'fp32'}, {'max_context': 8192},
                       {'kv_grow': 1}, {'memory_model': 'discrete'}, {'expert_cache_bytes': 0},
                       {'device_integrated': 0}):
            with self.subTest(engine=change), self.assertRaises(ValueError):
                bench.validate_intel_text_engine({**info, **change}, health, cfg)

    def test_uma_cache_stays_unavailable_and_rate_pooling_is_unchanged(self):
        info = self.engine()
        self.assertIsNone(bench.cache_bytes(info, intel_text=True))
        with self.assertRaises(ValueError):
            bench.cache_bytes(info)
        runs = [{'complete': True, 'startup_engine': info, 'after_16k_engine': info, 'records': [
            {'category': 'generation', 'native': {'engine_generated': 128, 'decode_ms': 4000}},
            {'category': 'input_4k', 'native': {'prompt_read': 4096, 'prompt_ms': 1000}},
            {'category': 'input_16k', 'native': {'prompt_read': 16384, 'prompt_ms': 4000}}]}]
        result = bench.aggregate_model(runs, intel_text=True)
        self.assertEqual(result['generation']['tok_s'], 32)
        self.assertEqual(result['input_16k']['tok_s'], 4096)
        self.assertFalse(result['expert_cache']['applicable'])
        self.assertEqual(result['expert_cache']['startup_engine'], [None])
        report = {'protocol': 'tinystrata-bench-v1-intel-text', 'standard': False, 'intel_text': True,
                  'complete': True, 'runs': 4, 'models': {'qwen': {'intel_text': True, 'runs': runs}}}
        text = bench.render_report(report)
        self.assertIn('N/A (UMA)', text)
        self.assertIn('Image is N/A', text)
        self.assertIn('32.00', text)

    def test_current_capability_proof_and_stage_hash_are_recorded_without_model_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / '.tinystrata-intel-runtime.json'
            manifest.write_text('{"fixture":true}', encoding='utf-8')
            cfg = self.configured()
            cfg.update(exe=str(root / 'tinystrata-intel.exe'), lib_dirs=[str(root)])
            proof = SimpleNamespace(executable=cfg['exe'], sha256='a' * 64, report={'capabilities_schema': 1})
            with patch('serve.model_setup.validate_intel_executable', return_value=proof) as gate:
                record = bench.intel_text_build(cfg)
            gate.assert_called_once_with(cfg, allow_legacy=False)
            self.assertEqual(record['executable_sha256'], 'a' * 64)
            self.assertEqual(record['stage_manifests'][0]['sha256'], bench.sha256(manifest))
            manifest.unlink()
            with patch('serve.model_setup.validate_intel_executable', return_value=proof), self.assertRaises(ValueError):
                bench.intel_text_build(cfg)

    def test_native_identity_excludes_capability_probes_and_refuses_changed_arguments(self):
        from serve.intel_backend import engine_args
        cfg = self.configured()
        service = bench.Service.__new__(bench.Service)
        service.cfg, service.lock, service.owned = cfg, threading.Lock(), {10: 1.0, 11: 2.0}
        service.remember = Mock()
        probe = Mock()
        probe.create_time.return_value = 1.0
        probe.is_running.return_value = True
        probe.exe.return_value = cfg['exe']
        probe.cmdline.return_value = [cfg['exe'], '--capabilities']
        resident = Mock()
        resident.create_time.return_value = 2.0
        resident.is_running.return_value = True
        resident.exe.return_value = cfg['exe']
        resident.cmdline.return_value = [cfg['exe'], '--serve', *engine_args(cfg)]
        with patch.object(bench.psutil, 'Process', side_effect=lambda pid: {10: probe, 11: resident}[pid]):
            identity = service.intel_native_identity()
            self.assertEqual(identity['pid'], 11)
            resident.cmdline.return_value[-1] = '999'
            with self.assertRaisesRegex(ValueError, 'arguments differ'):
                service.intel_native_identity()

    def test_prepare_only_writes_four_separate_nonstandard_plans_without_starting_services(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = self.original()
            native, model = root / 'tinystrata-intel.exe', root / 'model.gguf'
            native.write_bytes(b'CPU-only fixture')
            model.write_bytes(b'CPU-only model placeholder')
            cfg.update(exe=str(native), cwd=str(root), lib_dirs=[str(root)])
            cfg['args'][cfg['args'].index('--model') + 1] = str(model)
            profile = root / 'config-sycl.json'
            profile.write_text(json.dumps(cfg), encoding='utf-8')
            before = profile.read_bytes()
            output = root / 'plans'
            argv = ['benchmark/run.py', '--intel-text', '--model', 'qwen', '--prepare-only',
                    '--config-dir', str(root), '--output', str(output)]
            proof = {'executable_sha256': 'f' * 64, 'stage_manifests': [{'sha256': 'a' * 64}]}
            with patch.object(sys, 'argv', argv), patch.object(bench, 'intel_text_build', return_value=proof), \
                 patch.object(bench.subprocess, 'check_output', return_value='fixture-head\n'), \
                 patch.object(bench, 'Service', side_effect=AssertionError('No service/native/model allowed')):
                bench.main()
            report = json.loads((output / 'result.json').read_text(encoding='utf-8'))
            self.assertEqual(report['protocol'], 'tinystrata-bench-v1-intel-text')
            self.assertFalse(report['standard'])
            self.assertTrue(report['prepared_only'])
            self.assertTrue(report['configs_unchanged'])
            self.assertEqual((report['runs'], report['generation_passes'], report['seed']), (4, 2, 1234))
            self.assertFalse(report['protocol_differences']['expert_placement']['applicable'])
            self.assertEqual(len(list(output.glob('qwen-*/config.json'))), 4)
            self.assertFalse(report['models']['qwen']['image_supported'])
            self.assertEqual(report['models']['qwen']['unsupported_cases'][0]['status'], 'unsupported')
            self.assertEqual(profile.read_bytes(), before)

    def test_tiel_main_prepare_uses_its_sampler_before_any_service_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = self.original()
            native, model = root / 'tinystrata-intel.exe', root / 'Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf'
            native.write_bytes(b'CPU-only fixture')
            model.write_bytes(b'CPU-only model placeholder')
            cfg.update(exe=str(native), cwd=str(root), lib_dirs=[str(root)],
                       model_name='tiel-coder-35b-a3b', intel_profile='b390-tiel-coder')
            cfg['args'][cfg['args'].index('--model') + 1] = str(model)
            profile = root / 'config-tiel-coder-sycl.json'
            profile.write_text(json.dumps(cfg), encoding='utf-8')
            output = root / 'plans'
            argv = ['benchmark/run.py', '--intel-text', '--model', 'tiel', '--prepare-only',
                    '--config-dir', str(root), '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(bench, 'intel_text_build', return_value={'fixture': True}), \
                 patch.object(bench.subprocess, 'check_output', return_value='fixture-head\n'), \
                 patch.object(bench, 'Service', side_effect=AssertionError('No service/native/model allowed')):
                bench.main()
            report = json.loads((output / 'result.json').read_text(encoding='utf-8'))
            self.assertFalse(report['standard'])
            self.assertEqual(report['models']['tiel']['sampling'], tinyconfig.TIEL_SAMPLING)
            self.assertEqual(len(list(output.glob('tiel-*/config.json'))), 4)


if __name__ == "__main__":
    unittest.main()
