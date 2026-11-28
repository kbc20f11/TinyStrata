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
    reported_kv = {}

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
        kv = option('--kv-type', 'int8')
        if kv == 'auto':
            kv = 'int8' if self.cfg['arch'] == 'gemma4' else 'fp8-mma'
        kv = self.reported_kv.get(self.cfg['model_name'], kv)
        return dict(kv=kv, kv_grow=1, mtp=int(option('--mtp', '0')),
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
        length = int(text.split(':')[1]) if isinstance(text, str) and text.startswith('BENCH-LENGTH:') else 20
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
        FakeService.instances, FakeService.capacity, FakeService.reported_kv = [], 131072, {}
        self.sequence = 0

    def invoke(self, *flags, model='clover'):
        self.sequence += 1
        output = self.folder / f'result-{self.sequence}'
        arguments = ['benchmark/run.py', '--model', model, '--config-dir', str(self.config),
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
        report = self.invoke('--prepare-only', '--protocol', 'v1')
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
        report = self.invoke('--runs', '1', '--protocol', 'v1')
        run = report['models']['clover']['runs'][0]
        self.assertEqual([r['category'] for r in run['records']], ['generation'] * 6 + ['input_4k', 'input_16k'])
        self.assertNotIn('after_64k_engine', run)
        self.assertNotIn('input_64k', report['models']['clover']['aggregate'])
        requests = FakeService.instances[0].requests
        self.assertEqual(len(requests), 9)  # Existing warm-up + eight measurements.
        expected = dict(model='clover-35b-a3b-preview', messages=[dict(role='user', content=self.cases['generation'][0]['text'])],
                        max_tokens=128, stream=True, enable_thinking=True, strata_checkpoint=False, seed=1234,
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
        original = self.invoke('--runs', '1', '--protocol', 'v1')
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
        report = self.invoke('--runs', '4', '--protocol', 'v1')
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

    def vision_app(self):
        projector = self.folder / 'projector.gguf'
        projector.write_bytes(b'fixture projector')
        app = copy.deepcopy(self.original)
        app['vision_mode'] = 'gpu'
        app['vision'] = dict(exe=str(self.folder / 'engine.exe'),
                             model=str(self.folder / 'model.gguf'), mmproj=str(projector))
        return app

    def test_12_vision_app_does_not_change_fixed_causal_baseline(self):
        app = self.vision_app()
        before = copy.deepcopy(app)
        controlled = bench.controlled_config(app, 'clover', self.folder, 12345, self.cases)
        self.assertNotIn('vision', controlled)
        self.assertEqual(controlled['vision_mode'], 'no')
        self.assertNotIn('--vision', controlled['args'])
        self.assertEqual(app, before)

    def test_v3_defaults_include_tiel_and_causal_clover_vision(self):
        self.saved.write_text(json.dumps(self.vision_app()), encoding='utf-8')
        report = self.invoke('--prepare-only')
        self.assertEqual(report['protocol'], 'tinystrata-bench-v3')
        self.assertTrue(report['standard'])
        self.assertEqual(report['clover_mode'], 'causal')
        self.assertTrue(report['models']['clover']['image_supported'])
        self.assertEqual(bench.benchmark_models('all', v2=True), [*bench.MODELS, 'tiel'])
        self.assertEqual(bench.benchmark_models('tiel', v2=True), ['tiel'])

    def write_discrete_profiles(self):
        snapshots = {}
        for model in bench.benchmark_models('all', v2=True):
            filename, arch, name = bench.model_definition(model)
            app = self.vision_app()
            app.update(arch=arch, model_name=name, backend='hip', sampling={'temperature': .37, 'top_p': .61})
            if model == 'clover':
                app['env'] = {'STRATA_IQ_MT_MIN': '5'}
            app['args'] += ['--kv-type', 'fp8-mma' if model == 'clover' else 'int8',
                            '--chunk', '512', '--expert-vram', '6']
            if model == 'gemma':
                app['args'] += ['--draft-model', str(self.folder / 'model.gguf')]
            path = self.config / filename
            path.write_text(json.dumps(app), encoding='utf-8')
            snapshots[path] = path.read_bytes()
        return snapshots

    def test_v3_records_managed_kv_requests_and_formats_selected_by_the_engine(self):
        snapshots = self.write_discrete_profiles()
        requested = {'qwen': 'auto', 'gemma': 'auto', 'lfm': 'auto', 'clover': 'int8', 'tiel': 'auto'}
        observed = {'qwen': 'fp8-mma', 'gemma': 'int8', 'lfm': 'fp8-mma', 'clover': 'int8', 'tiel': 'fp8-mma'}

        plan = self.invoke('--prepare-only', model='all')
        self.assertEqual(plan['protocol'], 'tinystrata-bench-v3')
        self.assertEqual(plan['kv_policy'], 'application-default')
        self.assertTrue(plan['standard'])
        self.assertEqual(FakeService.instances, [])
        self.assertEqual({key: model['requested_kv'] for key, model in plan['models'].items()}, requested)

        with patch.dict('os.environ', {'STRATA_IQ_MT_MIN': '7'}):
            report = self.invoke('--protocol', 'v3', '--runs', '1', model='all')
        self.assertEqual(report['protocol'], plan['protocol'])
        self.assertEqual(report['kv_policy'], 'application-default')
        self.assertTrue(report['complete'])
        for (key, model), service in zip(report['models'].items(), FakeService.instances, strict=True):
            with self.subTest(model=key):
                self.assertEqual(config.value_of(service.cfg, 'kv_type'), requested[key])
                self.assertEqual(config.value_of(service.cfg, 'chunk'), 'auto')
                self.assertIsNone(config.value_of(service.cfg, 'expert_vram'))
                self.assertNotIn('--kv-grow', service.cfg['args'])
                self.assertEqual(model['requested_kv'], requested[key])
                self.assertEqual(model['observed_kv'], [observed[key]])
                self.assertEqual(model['runs'][0]['startup_engine']['kv'], observed[key])
                self.assertTrue(all('enable_thinking' not in body for body in service.requests))
                if model['image_supported']:
                    image = service.requests[-1]
                    self.assertEqual(image['messages'][0]['content'][0]['type'], 'image')
                    self.assertNotIn('max_tokens', image)
                if key == 'clover':
                    self.assertEqual(service.cfg['env'], {'STRATA_IQ_MT_MIN': '1'})
                if key == 'qwen':
                    self.assertEqual(model['sampling']['temperature'], 1.0)
                    self.assertEqual(service.requests[1]['temperature'], 1.0)
                    self.assertEqual(service.requests[1]['top_p'], .95)
                    self.assertIs(model['reasoning_default'], True)
                    self.assertEqual(service.requests[1]['max_tokens'], self.cases['generation_max_tokens'])
                if key == 'tiel':
                    self.assertEqual(model['reasoning_default'], 'medium')
        markdown = bench.render_report(report).lower()
        self.assertRegex(markdown, r'\| model \|[^\n]*\bkv\b[^\n]*\|')
        self.assertIn('| fp8-mma |', markdown)
        self.assertIn('| int8 |', markdown)
        self.assertEqual({path: path.read_bytes() for path in snapshots}, snapshots)

    def test_v3_uses_updated_defaults_from_the_managed_profile_factory(self):
        from serve import model_profiles

        self.write_discrete_profiles()
        factory = model_profiles.fresh_profile_defaults

        def changed_defaults(name, arch):
            return config.apply(factory(name, arch), {'kv_type': 'fp8', 'mtp': 1, 'chunk': 1024})[0]

        with patch.object(model_profiles, 'fresh_profile_defaults', side_effect=changed_defaults):
            report = self.invoke('--runs', '1', model='qwen')
        model = report['models']['qwen']
        self.assertEqual(model['requested_kv'], 'fp8')
        self.assertEqual(model['observed_kv'], ['fp8'])
        self.assertEqual(model['runs'][0]['startup_engine']['mtp'], 1)
        self.assertEqual(config.value_of(FakeService.instances[0].cfg, 'chunk'), 1024)

    def test_v2_and_v1_keep_fixed_int8_and_their_original_workloads(self):
        self.write_discrete_profiles()
        fixed = self.invoke('--protocol', 'v2', '--prepare-only', model='all')
        self.assertEqual(fixed['protocol'], 'tinystrata-bench-v2')
        self.assertEqual(fixed['kv_policy'], 'fixed-int8')
        self.assertTrue(fixed['standard'])
        self.assertEqual(list(fixed['models']), bench.benchmark_models('all', v2=True))
        self.assertTrue(fixed['models']['clover']['image_supported'])
        self.assertEqual(fixed['models']['qwen']['sampling']['temperature'], .7)
        self.assertTrue(all(model['requested_kv'] == 'int8' for model in fixed['models'].values()))
        clover = self.folder / f'result-{self.sequence}' / 'clover-1' / 'config.json'
        self.assertEqual(json.loads(clover.read_text(encoding='utf-8'))['env'], {})

        legacy = self.invoke('--protocol', 'v1', '--prepare-only', model='all')
        self.assertEqual(legacy['protocol'], 'tinystrata-bench-v1')
        self.assertEqual(legacy['kv_policy'], 'fixed-int8')
        self.assertTrue(legacy['standard'])
        self.assertEqual(list(legacy['models']), bench.benchmark_models('all'))
        self.assertFalse(legacy['models']['clover']['image_supported'])
        self.assertEqual(legacy['models']['qwen']['sampling']['temperature'], .7)
        self.assertTrue(all(model['requested_kv'] == 'int8' for model in legacy['models'].values()))

        self.invoke('--protocol', 'v2', '--runs', '1', model='qwen')
        qwen = FakeService.instances[-1]
        self.assertIn('--kv-grow', qwen.cfg['args'])
        self.assertIs(qwen.requests[1]['enable_thinking'], True)
        self.assertIs(qwen.requests[-1]['enable_thinking'], False)
        self.assertEqual(qwen.requests[-1]['max_tokens'], self.cases['image_max_tokens'])

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as blocked:
            self.invoke('--protocol', 'v2', '--kv-type', 'int8', '--prepare-only', model='inkling')
        self.assertEqual(blocked.exception.code, 2)

    def test_inkling_extension_preserves_automatic_kv_on_v2_and_v3(self):
        name, arch, model = bench.model_definition('inkling')
        inkling = dict(self.original, arch=arch, model_name=model, backend='hip',
                       inkling_vision=str(self.folder / 'model.gguf'))
        (self.config / name).write_text(json.dumps(inkling), encoding='utf-8')
        for revision, flags in (('v3', ()), ('v2', ('--protocol', 'v2'))):
            with self.subTest(revision=revision):
                plan = self.invoke('--prepare-only', *flags, model='inkling')
                self.assertEqual(plan['protocol'], f'tinystrata-bench-{revision}-inkling')
                self.assertEqual(plan['kv_policy'], 'application-default' if revision == 'v3' else 'engine-auto')
                self.assertFalse(plan['standard'])
                self.assertTrue(plan['models']['inkling']['image_supported'])
                self.assertEqual(plan['models']['inkling']['requested_kv'], 'auto')

    def test_reported_auto_kv_must_be_resolved_and_fixed_int8_must_match(self):
        self.write_discrete_profiles()
        name = bench.model_definition('qwen')[2]
        for reported, requested, flags in (('auto', 'auto', ()),
                                           ('fp8-mma', 'int8', ('--protocol', 'v2'))):
            with self.subTest(reported=reported):
                FakeService.reported_kv[name] = reported
                with self.assertRaisesRegex(ValueError, f'requested KV {requested}'):
                    self.invoke('--runs', '1', *flags, model='qwen')
                self.assertEqual(FakeService.instances[-1].requests, [])
                self.assertTrue(FakeService.instances[-1].closed)

        FakeService.reported_kv[name] = 'int8'
        fallback = self.invoke('--runs', '1', model='qwen')['models']['qwen']
        self.assertEqual(fallback['requested_kv'], 'auto')
        self.assertEqual(fallback['observed_kv'], ['int8'])

    def test_13_explicit_vision_d3_variant_records_resident_encoder_and_assets(self):
        app = self.vision_app()
        controlled = bench.controlled_config(app, 'clover', self.folder, 12345, self.cases,
                                            clover_mode='self-spec', clover_vision=True)
        self.assertEqual(config.value_of(controlled, 'clover_draft'), 3)
        self.assertEqual(controlled['vision_mode'], 'gpu')
        self.assertIn('--vision', controlled['args'])
        self.assertNotIn('--mtp', controlled['args'])
        self.assertTrue(controlled['vision']['gpu'])
        self.assertEqual(controlled['vision']['min_tokens'], 1024)
        self.assertEqual(controlled['vision']['max_tokens'], 4096)
        self.assertIn(Path(app['vision']['mmproj']), bench.assets(controlled))
        self.saved.write_text(json.dumps(app), encoding='utf-8')
        report = self.invoke('--prepare-only', '--clover-mode', 'self-spec', '--clover-vision')
        self.assertEqual(report['protocol'], 'tinystrata-bench-v1-clover-self-spec-vision')
        self.assertTrue(report['clover_profile']['vision'])
        self.assertTrue(report['models']['clover']['image_supported'])
        self.assertFalse(report['standard'])
        self.assertEqual(FakeService.instances, [])
        self.assertEqual(json.loads(self.saved.read_text(encoding='utf-8')), app)

    def test_14_vision_does_not_enable_unverified_decoders(self):
        app = self.vision_app()
        for options in ({'clover_mode': 'diffusion'},
                        {'clover_mode': 'self-spec', 'clover_accept': 'threshold'}):
            with self.assertRaisesRegex(ValueError, 'Vision requires'):
                bench.controlled_config(app, 'clover', self.folder, 12345, self.cases,
                                        clover_vision=True, **options)


if __name__ == '__main__':
    unittest.main()
