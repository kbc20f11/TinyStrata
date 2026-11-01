"""CPU-only benchmark extension regressions: python tests/benchmark-64k.py.

Uses tiny temporary assets and a fake API service; never starts an engine,
queries a GPU, opens a socket, downloads weights, or sends API requests.
"""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('tinystrata_benchmark', ROOT / 'benchmark/run.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
from serve import config, tinyconfig


class FakeService:
    instances = []
    capacity = 131072

    def __init__(self, cfg, folder, timeout):
        self.cfg, self.requests, self.closed = cfg, [], False
        self.largest = 0
        self.instances.append(self)

    def ready(self, timeout):
        return None

    def engine(self):
        args = self.cfg['args']
        def option(name, fallback):
            return args[args.index(name) + 1] if name in args else fallback
        mode = option('--clover-mode', 'causal')
        committed = max(16384, ((self.largest + 7 + 8191) // 8192) * 8192)
        return dict(kv=option('--kv-type', 'int8'), kv_grow=1, mtp=0,
                    max_context=self.capacity, clover_mode=mode,
                    clover_draft=int(option('--clover-draft', '0')),
                    clover_accept=option('--clover-accept', 'exact'),
                    clover_accept_ratio=float(option('--clover-accept-ratio', '.9')),
                    clover_accept_delta=float(option('--clover-accept-delta', '.6')),
                    clover_last_mode=mode, clover_last_accept=option('--clover-accept', 'exact'),
                    clover_last_accept_ratio=float(option('--clover-accept-ratio', '.9')),
                    clover_last_accept_delta=float(option('--clover-accept-delta', '.6')),
                    expert_cache_bytes=(10 - self.largest // 16384) * bench.GIB,
                    kv_committed_rows=committed, kv_mapped_bytes=committed * 128)

    def vision_identity(self):
        return None

    def json(self, path, body):
        if path != '/v1/messages/count_tokens':
            raise AssertionError('Unexpected fake API endpoint')
        text = body['messages'][0]['content']
        return {'input_tokens': int(text.split(':')[1])}

    def generate(self, body):
        self.requests.append(copy.deepcopy(body))
        text = body['messages'][0]['content']
        length = int(text.split(':')[1]) if text.startswith('BENCH-LENGTH:') else 20
        self.largest = max(self.largest, length)
        native = dict(engine_generated=7 if length > 20 else 11, prompt_read=length,
                      prompt_total=length, reused=0, prompt_ms=100, decode_ms=50,
                      finish='stop', drafts_offered=0, drafts_accepted=0)
        return dict(native=native, engine_after=self.engine(),
                    response=dict(complete=True, content='answer', reasoning='',
                                  first_output_s=.1, elapsed_s=.2))

    def close(self):
        self.closed = True


class Benchmark64K(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tinystrata-bench64-test-')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.config = self.folder / 'configs'
        self.config.mkdir()
        for name in ('engine.exe', 'model.gguf'):
            (self.folder / name).write_bytes(b'test asset')
        self.original = dict(exe=str(self.folder / 'engine.exe'), arch='clover_moe',
                             args=['--model', str(self.folder / 'model.gguf')])
        self.saved = self.config / 'config-clover.json'
        self.saved.write_text(json.dumps(self.original), encoding='utf-8')
        self.cases = json.loads((ROOT / 'benchmark/cases.json').read_text(encoding='utf-8'))
        FakeService.instances, FakeService.capacity = [], 131072
        self.sequence = 0

    def invoke(self, *flags):
        self.sequence += 1
        output = self.folder / f'result-{self.sequence}'
        arguments = ['benchmark/run.py', '--model', 'clover', '--config-dir', str(self.config),
                     '--output', str(output), *flags]
        def counted_document(count, paragraph, question, length):
            text = f'BENCH-LENGTH:{length}'
            self.assertEqual(count(text), length)
            return text, length
        with patch.object(sys, 'argv', arguments), patch.object(bench, 'Service', FakeService), \
                patch.object(bench, 'free_port', return_value=12345), \
                patch.object(bench, 'exact_length_text', side_effect=counted_document), \
                patch.object(tinyconfig, 'validate'), \
                patch.object(bench.subprocess, 'check_output', return_value='a' * 40), \
                patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('No process may start')), \
                patch.object(bench.psutil, 'virtual_memory', return_value=SimpleNamespace(total=bench.GIB)), \
                contextlib.redirect_stdout(io.StringIO()):
            bench.main()
        return json.loads((output / 'result.json').read_text(encoding='utf-8'))

    def test_01_clover_only(self):
        for model in ('qwen', 'gemma', 'lfm', 'all'):
            with self.subTest(model=model), patch.object(sys, 'argv', ['run.py', '--model', model, '--include-64k']), \
                    patch.object(bench, 'Service') as service, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.main()
                self.assertEqual(error.exception.code, 2)
                service.assert_not_called()

    def test_02_default_prepare_keeps_v1_metadata(self):
        report = self.invoke('--prepare-only')
        self.assertEqual(report['protocol'], 'tinystrata-bench-v1')
        self.assertTrue(report['standard'])
        self.assertNotIn('include_64k', report)
        self.assertNotIn('include_64k', report['clover_profile'])
        self.assertTrue(report['prepared_only'])
        self.assertEqual(FakeService.instances, [])

    def test_03_extension_prepare_is_labelled_without_api(self):
        report = self.invoke('--prepare-only', '--include-64k')
        self.assertEqual(report['protocol'], 'tinystrata-bench-v1-clover-causal-64k')
        self.assertFalse(report['standard'])
        self.assertTrue(report['include_64k'])
        self.assertTrue(report['clover_profile']['include_64k'])
        self.assertEqual(FakeService.instances, [])

    def test_04_default_request_sequence_and_body(self):
        report = self.invoke('--runs', '1')
        run = report['models']['clover']['runs'][0]
        self.assertEqual([r['category'] for r in run['records']], ['generation'] * 6 + ['input_4k', 'input_16k'])
        self.assertNotIn('after_64k_engine', run)
        self.assertNotIn('input_64k', report['models']['clover']['aggregate'])
        requests = FakeService.instances[0].requests
        self.assertEqual(len(requests), 9)  # Existing warm-up + eight measurements.
        expected = dict(model='clover-35b-a3b-preview', messages=[dict(role='user', content=self.cases['generation'][0]['text'])],
                        max_tokens=128, stream=True, enable_thinking=True, seed=1234,
                        frequency_penalty=0.0, temperature=0.0, top_p=1.0, top_k=64,
                        min_p=0.0, presence_penalty=.9, repetition_penalty=1.05)
        self.assertEqual(requests[1], expected)
        self.assertEqual(json.loads(self.saved.read_text(encoding='utf-8')), self.original)

    def test_05_extension_order_and_actual_snapshots(self):
        report = self.invoke('--runs', '1', '--include-64k', '--clover-mode', 'self-spec', '--clover-draft', '3')
        self.assertEqual(report['protocol'], 'tinystrata-bench-v1-clover-self-spec-64k')
        run = report['models']['clover']['runs'][0]
        self.assertEqual([r['category'] for r in run['records']], ['generation'] * 6 + ['input_4k', 'input_16k', 'input_64k'])
        self.assertEqual(run['records'][-1]['name'], 'input_64k')
        native = run['records'][-1]['native']
        self.assertEqual((native['prompt_read'], native['reused']), (65536, 0))
        self.assertEqual(FakeService.instances[0].requests[-1]['max_tokens'], 32)
        self.assertEqual(run['after_16k_engine']['kv_committed_rows'], 24576)
        self.assertEqual(run['after_64k_engine']['kv_committed_rows'], 73728)
        self.assertEqual(run['after_16k_engine']['expert_cache_bytes'], 9 * bench.GIB)
        self.assertEqual(run['after_64k_engine']['expert_cache_bytes'], 6 * bench.GIB)
        self.assertEqual(run['after_64k_engine']['kv_mapped_bytes'], 73728 * 128)
        self.assertTrue(FakeService.instances[0].closed)

    def test_06_template_inclusive_length_and_fresh_native_counts(self):
        text, count = bench.exact_length_text(lambda s: len(s) + 13,
            self.cases['long_paragraph'], self.cases['long_question'], 65536)
        self.assertEqual((len(text) + 13, count), (65536, 65536))
        self.assertTrue(text.endswith(self.cases['long_question']))
        native = dict(engine_generated=7, prompt_read=65536, prompt_total=65536,
                      reused=0, prompt_ms=100, decode_ms=50)
        bench.validate_native(native, 65536)
        with self.assertRaises(ValueError):
            bench.validate_native({**native, 'prompt_read': 65535, 'prompt_total': 65535}, 65536)
        with self.assertRaises(ValueError):
            bench.validate_native({**native, 'reused': 1}, 65536)

    def test_07_capacity_gate_precedes_all_requests(self):
        bench.validate_64k_context({'max_context': 65568}, 32)
        for value in (65567, 65536, None, '131072', True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '65,568'):
                bench.validate_64k_context({'max_context': value}, 32)
        FakeService.capacity = 65567
        with self.assertRaisesRegex(ValueError, '65,568'):
            self.invoke('--runs', '1', '--include-64k')
        self.assertEqual(FakeService.instances[0].requests, [])
        self.assertTrue(FakeService.instances[0].closed)

    def test_08_aggregate_uses_actual_output_and_conditional_cache(self):
        runs = []
        for generated, prompt_ms, decode_ms in ((7, 1000, 20), (9, 3000, 30)):
            runs.append(dict(complete=True, records=[dict(category='input_64k', native=dict(
                engine_generated=generated, prompt_read=65536, prompt_ms=prompt_ms, decode_ms=decode_ms))],
                startup_engine={'expert_cache_bytes': 10 * bench.GIB},
                after_16k_engine={'expert_cache_bytes': 9 * bench.GIB},
                after_64k_engine={'expert_cache_bytes': 6 * bench.GIB}))
        aggregate = bench.aggregate_model(runs)
        self.assertEqual(aggregate['input_64k']['tok_s'], 32768)
        self.assertEqual(aggregate['decode_after_64k']['tokens'], 16)
        self.assertEqual(aggregate['decode_after_64k']['tok_s'], 320)
        self.assertEqual(aggregate['expert_cache']['after_16k_engine'], [9 * bench.GIB] * 2)
        self.assertEqual(aggregate['expert_cache']['after_64k_engine'], [6 * bench.GIB] * 2)
        empty = bench.aggregate_model([])
        self.assertNotIn('input_64k', empty)
        self.assertNotIn('decode_after_64k', empty)
        self.assertEqual(set(empty['expert_cache']), {'startup_engine', 'after_16k_engine'})

    def test_09_renderer_only_extends_flagged_reports(self):
        original = self.invoke('--runs', '1')
        plain = bench.render_report(original)
        self.assertNotIn('64K', plain)
        self.assertIn('Expert cache GiB: startup → after16K', plain)
        extended = self.invoke('--runs', '1', '--include-64k')
        text = bench.render_report(extended)
        self.assertIn('64K input tok/s', text)
        self.assertIn('Generation after64K tok/s', text)
        self.assertIn('startup → after16K → after64K', text)
        self.assertIn('10.00 → 9.00 → 6.00', text)

    def test_10_saved_fast_app_profile_does_not_change_standard_benchmark(self):
        # App defaults and the fixed public comparison protocol have different
        # jobs. An installed fast profile must not leak its mode/CPU settings.
        old_app = copy.deepcopy(self.original)
        fast_app = {**copy.deepcopy(self.original), 'model_source': 'managed',
                    'env': {'STRATA_IQ_MT_MIN': '1'},
                    'sampling': dict(tinyconfig.CLOVER_SAMPLING),
                    'enable_thinking': True, 'vision_mode': 'no',
                    'clover_decode_policy_version': 1}
        fast_app['args'] += ['--max-context', 'auto', '--chunk', 'auto', '--kv-type', 'int8',
                             '--clover-mode', 'self-spec', '--clover-draft', '3', '--clover-accept', 'exact']
        before = copy.deepcopy(fast_app)
        expected = bench.controlled_config(old_app, 'clover', self.folder, 12345, self.cases)
        controlled = bench.controlled_config(fast_app, 'clover', self.folder, 12345, self.cases)
        self.assertEqual(controlled, expected)
        self.assertEqual(fast_app, before)
        # The real server's settings migration also must retain the explicit
        # benchmark controls when app defaults change in tinyconfig.
        migrated, _ = tinyconfig.migrate_settings(controlled)
        self.assertEqual(config.clover_mode(migrated), 'causal')
        self.assertEqual(config.value_of(migrated, 'clover_draft'), 0)
        self.assertEqual(config.value_of(migrated, 'kv_type'), 'int8')
        self.assertEqual(migrated['env'], {})
        self.original = fast_app
        self.saved.write_text(json.dumps(fast_app), encoding='utf-8')
        report = self.invoke('--runs', '4')
        self.assertTrue(report['standard'])
        launched = FakeService.instances[0].cfg
        self.assertEqual(config.clover_mode(launched), 'causal')
        self.assertEqual(config.value_of(launched, 'clover_draft'), 0)
        self.assertEqual(config.value_of(launched, 'kv_type'), 'int8')
        self.assertEqual(launched['env'], {})
        self.assertEqual(json.loads(self.saved.read_text(encoding='utf-8')), before)

    def test_11_self_spec_omitted_draft_is_three_and_explicit_value_survives(self):
        controlled = bench.controlled_config(self.original, 'clover', self.folder, 12345, self.cases,
                                            clover_mode='self-spec')
        self.assertEqual(config.value_of(controlled, 'clover_draft'), 3)
        self.assertEqual(config.value_of(controlled, 'clover_accept'), 'exact')
        explicit = bench.controlled_config(self.original, 'clover', self.folder, 12345, self.cases,
                                          clover_mode='self-spec', clover_draft=7)
        self.assertEqual(config.value_of(explicit, 'clover_draft'), 7)
        report = self.invoke('--prepare-only', '--clover-mode', 'self-spec')
        self.assertEqual(report['clover_draft'], 3)
        self.assertEqual(report['clover_profile']['draft'], 3)
        self.assertFalse(report['standard'])
        self.assertEqual(FakeService.instances, [])


if __name__ == '__main__':
    unittest.main()
