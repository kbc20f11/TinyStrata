"""Offline worker-probe admission and ownership tests; no API/GPU/process starts."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('worker_probe_test_module', ROOT / 'benchmark/probe.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def row(rate, category='generation', pp=1000):
    n = 128 if category == 'generation' else 32
    count = int(category.split('_')[1]) if category.startswith('input_') else 32
    return {'category': category, 'native': {'engine_generated': n, 'decode_ms': n * 1000 / rate,
            'prompt_read': count, 'prompt_ms': count * 1000 / pp}}


def run(rate, role='baseline', workers=0, order='', post=None, pp=1000):
    categories = ('generation', 'input_4096', 'input_16384') if order else ('generation',)
    return {'role': role, 'workers': workers, 'order': order,
            'records': [row(post if category != 'generation' and post is not None else rate,
                            category, pp) for category in categories]}


def confirmation(a=100, b=120, *, post=None, pp=1000, reverse_b=None):
    return [run(a if arm == 'A' else reverse_b if order == 'BAAB' and reverse_b else b,
                'baseline' if arm == 'A' else 'candidate', 0 if arm == 'A' else 7,
                order, None if arm == 'A' else post, 1000 if arm == 'A' else pp)
            for order in ('ABBA', 'BAAB') for arm in order]


class WorkerAdmission(unittest.TestCase):
    def test_source_provenance_hashes_the_actual_tool_and_labels_checkout_head(self):
        expected = hashlib.sha256((ROOT / 'benchmark/probe.py').read_bytes()).hexdigest()
        for status, dirty in (('', False), ('?? benchmark/probe.py\n', True), (OSError('git unavailable'), None)):
            with self.subTest(dirty=dirty), \
                 patch.object(P.subprocess, 'check_output', side_effect=['test-head', status]):
                provenance = P.source_provenance()
            self.assertEqual(provenance['tool_sha256'], expected)
            self.assertEqual(provenance['commit'], 'test-head')
            self.assertEqual(provenance['commit_kind'], 'source_checkout_HEAD_not_native_build')
            self.assertIs(provenance['source_worktree_dirty'], dirty)

    def test_candidates_cover_half_and_powers_without_machine_constant(self):
        self.assertEqual(P.candidate_counts(15), [1, 2, 4, 7, 8, 15])
        self.assertEqual(P.candidate_counts(9), [1, 2, 4, 8, 9])
        self.assertEqual(P.candidate_counts(1), [1])
        self.assertEqual(P.candidate_counts(15, [4, 7]), [4, 7, 15])
        for count in (0, -1, True):
            with self.assertRaises(ValueError): P.candidate_counts(count)
        with self.assertRaises(ValueError): P.candidate_counts(15, [16])

    def test_selects_fewest_within_two_percent_of_best(self):
        rows = [run(100), run(100), run(119, 'candidate', 7), run(120, 'candidate', 8)]
        self.assertEqual(P.pick_screen(rows, 15)['candidate'], 7)
        rows[2] = run(116, 'candidate', 7)
        self.assertEqual(P.pick_screen(rows, 15)['candidate'], 8)

    def test_near_best_still_must_exceed_baseline_noise(self):
        rows = [run(100), run(100), run(102, 'candidate', 7), run(104, 'candidate', 8)]
        self.assertEqual(P.pick_screen(rows, 15)['candidate'], 8)
        rows = [run(90), run(110), run(115, 'candidate', 7)]
        self.assertIsNone(P.pick_screen(rows, 15)['candidate'])
        self.assertIsNone(P.pick_screen([run(100), run(103, 'candidate', 7)], 15)['candidate'])

    def test_confirmation_requires_both_orders_and_long_workloads(self):
        self.assertTrue(P.confirm_result(confirmation())['admitted'])
        self.assertFalse(P.confirm_result(confirmation(reverse_b=95))['admitted'])
        self.assertFalse(P.confirm_result(confirmation(post=85))['admitted'])
        self.assertFalse(P.confirm_result(confirmation(pp=800))['admitted'])
        self.assertFalse(P.confirm_result(confirmation(b=102))['admitted'])

    def test_geometry_drops_only_worker_metadata_and_accepts_cpu_only_tier(self):
        info = {'expert_total': 8, 'expert_slots': 4, 'expert_cache_bytes': 1024,
                'expert_arena_bytes': 2048, 'max_context': 262144, 'chunk': 2048,
                'cpu_workers': 7, 'cpu_threads': 8, 'q36_cpu_compact': 1}
        other = {**info, 'cpu_workers': 15, 'cpu_threads': 16}
        self.assertEqual(P.geometry(info), P.geometry(other))
        self.assertNotEqual(P.geometry(info), P.geometry({**other, 'expert_cache_bytes': 1025}))
        self.assertNotEqual(P.geometry(info), P.geometry({**other, 'q36_cpu_compact': 0}))
        P.geometry({**info, 'expert_slots': 0, 'expert_cache_bytes': 0, 'expert_arena_bytes': 0})
        with self.assertRaises(ValueError): P.geometry({**info, 'expert_arena_bytes': -1})

    def test_response_identity_includes_reasoning_and_draft_counts(self):
        first = {'response': {'content': 'answer', 'reasoning': 'reason', 'elapsed_s': .1},
                 'native': {**row(100)['native'], 'drafts_offered': 3, 'drafts_accepted': 2}}
        second = copy.deepcopy(first)
        second['response']['elapsed_s'] = 100
        self.assertEqual(P.response_signature(first), P.response_signature(second))
        second['response']['reasoning'] = 'different'
        self.assertNotEqual(P.response_signature(first), P.response_signature(second))
        second = copy.deepcopy(first); second['native']['drafts_offered'] = 4
        self.assertNotEqual(P.response_signature(first), P.response_signature(second))

    def test_profile_fields_and_layout_are_preserved_without_credentials(self):
        original = {'arch': 'clover_moe', 'exe': 'engine.exe', 'tokenizer': 'tokenizer', 'cwd': str(ROOT),
            'args': ['--model', 'model.gguf', '--chunk', '2048', '--kv-type', 'fp8-mma', '--no-graph',
                     '--cpu-threads', '99'],
            'env': {'STRATA_EXP_MODE': '8', 'STRATA_IQ_MT_MIN': '1'}, 'api_key': 'DO_NOT_COPY'}
        before = copy.deepcopy(original)
        args = SimpleNamespace(model='clover', expert_gib=None, kv_type=None, context=None,
                               chunk=None, engine_env=[])
        with patch.object(P.B, 'free_port', return_value=18001):
            cfg = P.configuration(original, args, ROOT / 'unused', 7)
        self.assertEqual(original, before)
        self.assertEqual(cfg['env'], original['env'])
        self.assertIn('--no-graph', cfg['args'])
        self.assertEqual(cfg['args'][cfg['args'].index('--chunk') + 1], '2048')
        self.assertEqual(cfg['args'][cfg['args'].index('--cpu-threads') + 1], '7')
        self.assertNotIn('DO_NOT_COPY', json.dumps(cfg))
        with self.assertRaises(ValueError): P.preserved_environment({'TINYSTRATA_API_KEY': 'secret'}, {})

    def test_actual_server_config_validation_accepts_canonical_workers_and_integer_chunk(self):
        original = {'arch': 'clover_moe', 'exe': 'engine.exe', 'tokenizer': 'tokenizer', 'cwd': str(ROOT),
            'args': ['--model', 'model.gguf', '--max-context', '262144', '--chunk', '512',
                     '--kv-type', 'fp8-mma', '--pool-workers', '15', '--clover-mode', 'causal', '--clover-draft', '0'],
            'env': {'STRATA_EXP_MODE': '8', 'STRATA_IQ_MT_MIN': '1'}}
        args = SimpleNamespace(model='clover', expert_gib=4.0, kv_type=None, context=None,
                               chunk=P.B.chunk_option('2048'), engine_env=[])
        with patch.object(P.B, 'free_port', return_value=18001):
            cfg = P.configuration(original, args, ROOT / 'unused', 7)
        migrated, _ = P.B.tinyconfig.migrate_settings(cfg)
        P.B.tinyconfig.validate(migrated)
        self.assertNotIn('--pool-workers', migrated['args'])
        self.assertEqual(migrated['args'][migrated['args'].index('--cpu-threads') + 1], '7')
        self.assertEqual(P.B.tinyconfig.config.value_of(migrated, 'chunk'), 2048)
        self.assertEqual(P.B.tinyconfig.config.value_of(migrated, 'expert_vram'), 4.0)
        self.assertEqual(migrated['env'], original['env'])


class FakeService:
    instances = []
    failure = None
    all_gpu = False
    rates = {1: 30, 2: 60, 4: 90, 7: 119, 8: 120, 15: 100}

    def __init__(self, cfg, folder, timeout):
        self.cfg = cfg; self.closed = False; self.generated = 0
        args = cfg['args']
        self.workers = int(args[args.index('--cpu-threads') + 1]) if '--cpu-threads' in args else 15
        self.instances.append(self)

    def ready(self, timeout):
        if self.failure == 'ready': raise RuntimeError('simulated startup failure')
        return None

    def json(self, path, body=None):
        assert path == '/metrics'
        payload = 1025 if self.failure == 'capacity' and self.workers == 4 else 1024
        engine = {'model': 'clover-35b-a3b-preview', 'version': 'test', 'kv': 'fp8-mma',
            'mtp': 0, 'clover_mode': 'causal', 'clover_draft': 0, 'vision': 0, 'images': False,
            'max_context': 262144, 'chunk': 2048, 'expert_slots': 8 if self.all_gpu else 4,
            'expert_total': 8, 'expert_cache_bytes': payload, 'expert_arena_bytes': 2048,
            'cpu_workers': self.workers, 'cpu_threads': self.workers + 1}
        return {'engine': engine}

    def generate(self, body):
        self.generated += 1
        text = body['messages'][0]['content']
        target = int(text.split('_')[1]) if text.startswith('long_') else 32
        limit = body['max_tokens']
        response = text + ('changed' if self.failure == 'hash' and self.workers == 4 else '')
        return {'response': {'content': response, 'reasoning': 'same', 'elapsed_s': .1},
            'native': {'engine_generated': limit, 'output_tokens': limit,
                'prompt_total': target, 'prompt_read': target, 'reused': 0,
                'prompt_ms': target, 'decode_ms': limit * 1000 / self.rates[self.workers],
                'drafts_accepted': 0, 'drafts_offered': 0, 'finish': 'length'}}

    def close(self):
        self.closed = True
        if self.failure == 'close': raise RuntimeError('simulated cleanup failure')


class OfflineController(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.config = self.base / 'profiles'; self.config.mkdir()
        self.exe = self.base / 'engine.exe'; self.exe.write_bytes(b'MZ-test')
        self.model = self.base / 'model.gguf'; self.model.write_bytes(b'test-model')
        profile = {'exe': str(self.exe), 'tokenizer': str(self.base / 'tokenizer'), 'cwd': str(ROOT),
            'arch': 'clover_moe', 'model_name': 'clover-35b-a3b-preview',
            'args': ['--model', str(self.model), '--chunk', '2048', '--kv-type', 'fp8-mma',
                     '--max-context', '262144', '--clover-mode', 'causal', '--clover-draft', '0'],
            'env': {'STRATA_EXP_MODE': '8', 'STRATA_IQ_MT_MIN': '1'}, 'api_key': 'PRIVATE_KEY'}
        for name in ('config-clover.json', 'config.json', 'config-gemma4.json', 'config-lfm2.json'):
            (self.config / name).write_text(json.dumps(profile), encoding='utf-8')
        self.initial = {p.name: p.read_bytes() for p in self.config.iterdir()}
        FakeService.instances = []; FakeService.failure = None; FakeService.all_gpu = False

    def invoke(self, *extra, service_class=FakeService):
        output = self.base / ('result-' + str(len(list(self.base.glob('result-*')))))
        argv = ['probe.py', '--model', 'clover', '--config-dir', str(self.config),
                '--output', str(output), '--workers', '4,7,8', *extra]
        with patch.object(sys, 'argv', argv), patch.object(P, 'OwnedService', service_class), \
             patch.object(P.B, 'free_port', return_value=18001), \
             patch.object(P.B, 'assets', return_value=[self.exe, self.model]), \
             patch.object(P.subprocess, 'check_output', side_effect=lambda command, **kwargs:
                          'test-commit' if 'rev-parse' in command else '?? benchmark/probe.py\n'), \
             patch.object(P.B, 'exact_length_text', side_effect=lambda count, paragraph, question, target: ('long_' + str(target), target)), \
             patch('builtins.print'):
            P.main()
        return json.loads((output / 'result.json').read_text()), output

    def assert_profiles_and_owned_children(self):
        self.assertEqual(self.initial, {p.name: p.read_bytes() for p in self.config.iterdir()})
        self.assertTrue(all(instance.closed for instance in FakeService.instances))

    def test_mirrored_screen_and_full_confirm_recommend_without_writing_settings(self):
        result, output = self.invoke()
        self.assertTrue(result['complete'])
        self.assertEqual(result['recommendation']['workers'], 7)
        self.assertTrue(result['confirmation']['admitted'])
        self.assertEqual(len([r for r in result['runs'] if r['phase'] == 'confirm']), 8)
        self.assertTrue(all(len(r['records']) == 5 for r in result['runs'] if r['phase'] == 'confirm'))
        self.assertFalse(result['settings_written'])
        for run in result['runs']:
            self.assertEqual(run['config']['env']['STRATA_EXP_MODE'], '8')
            self.assertNotIn('PRIVATE_KEY', json.dumps(run['config']))
        self.assert_profiles_and_owned_children()

    def test_prepare_only_never_constructs_a_service(self):
        result, output = self.invoke('--prepare-only', '--default-workers', '15')
        self.assertTrue(result['complete']); self.assertIsNone(result['recommendation'])
        self.assertEqual(FakeService.instances, [])
        self.assertEqual(result['tool_sha256'], hashlib.sha256((ROOT / 'benchmark/probe.py').read_bytes()).hexdigest())
        self.assertTrue(result['source_worktree_dirty'])
        self.assertEqual(result['commit_kind'], 'source_checkout_HEAD_not_native_build')
        self.assert_profiles_and_owned_children()

    def test_screen_only_never_recommends_even_when_screen_is_faster(self):
        result, output = self.invoke('--screen-only')
        self.assertEqual(result['screen']['candidate'], 7)
        self.assertIsNone(result['recommendation'])
        self.assert_profiles_and_owned_children()

    def test_all_gpu_is_not_a_cpu_worker_speed_claim(self):
        FakeService.all_gpu = True
        result, output = self.invoke()
        self.assertIn('not_applicable', result)
        self.assertIsNone(result['recommendation'])
        self.assertEqual(FakeService.instances[0].generated, 0)
        self.assert_profiles_and_owned_children()

    def test_failures_close_only_owned_services_and_preserve_profiles(self):
        for failure in ('ready', 'hash', 'capacity', 'close'):
            with self.subTest(failure=failure):
                FakeService.instances = []; FakeService.failure = failure
                with self.assertRaises((RuntimeError, ValueError)):
                    self.invoke()
                self.assert_profiles_and_owned_children()
                latest = sorted(self.base.glob('result-*'), key=lambda p: int(p.name.split('-')[1]))[-1]
                result = json.loads((latest / 'result.json').read_text())
                self.assertFalse(result['complete']); self.assertIsNone(result['recommendation'])
                self.assertTrue(result['profiles_unchanged'])

    def test_partial_constructor_failure_reaps_owned_handle_without_starting_a_process(self):
        class Child:
            pid = 424242
            returncode = None
            killed = False
            def poll(child): return child.returncode
            def kill(child): child.killed = True; child.returncode = -9
            def wait(child, timeout): return child.returncode
        child = Child()
        streams = []
        class UnstartedWatcher:
            ident = None
            def join(watcher, timeout=None): raise AssertionError('Cannot join an unstarted watcher')
        def partial(service, cfg, folder, timeout):
            service.cfg = cfg; service.owned = {}; service.lock = threading.Lock()
            service.stop = threading.Event(); service.process = child
            service.stdout = io.StringIO(); service.stderr = io.StringIO()
            streams.extend((service.stdout, service.stderr))
            service.watcher = UnstartedWatcher()
            raise RuntimeError('simulated failure after Popen before watcher.start')
        class Identity:
            pid = child.pid
            def create_time(identity): return 1234.0
            def children(identity, recursive=False): return []
            def kill(identity): child.kill()
        identity = Identity()
        with patch.object(P.B.Service, '__init__', partial), \
             patch.object(P.B.psutil, 'Process', side_effect=lambda pid: identity if pid == child.pid else self.fail('Unrelated PID lookup')), \
             patch.object(P.B.psutil, 'wait_procs', return_value=([], [])):
            with self.assertRaisesRegex(RuntimeError, 'after Popen'):
                self.invoke(service_class=P.OwnedService)
        self.assertTrue(child.killed)
        self.assertTrue(all(stream.closed for stream in streams))
        self.assert_profiles_and_owned_children()

    def test_constructor_cleanup_failure_still_kills_exact_popen_handle(self):
        class Child:
            pid = 434343
            returncode = None
            killed = False
            def poll(child): return child.returncode
            def kill(child): child.killed = True; child.returncode = -9
            def wait(child, timeout): return child.returncode
        child = Child(); streams = []
        def partial(service, cfg, folder, timeout):
            service.process = child; service.owned = {}; service.stop = threading.Event()
            service.stdout = io.StringIO(); service.stderr = io.StringIO()
            streams.extend((service.stdout, service.stderr))
            raise RuntimeError('constructor failed')
        with patch.object(P.B.Service, '__init__', partial), \
             patch.object(P.B.Service, 'close', side_effect=RuntimeError('normal cleanup failed')):
            with self.assertRaisesRegex(RuntimeError, 'constructor failed'):
                self.invoke(service_class=P.OwnedService)
        self.assertTrue(child.killed)
        self.assertTrue(all(stream.closed for stream in streams))
        self.assert_profiles_and_owned_children()


if __name__ == '__main__':
    unittest.main()
