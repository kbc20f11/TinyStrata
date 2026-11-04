"""Expert placement uses a guarded model-reload transaction; no GPU/model files needed."""
from copy import deepcopy
import http.client
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import config
from serve.frontend import ChatTemplate
from serve.model_profiles import ModelProfile
from serve.model_switch import ModelBundle, ModelControlError, ModelSwitcher
from serve.server import ByteTokenizer, MockEngine, Server, Service, make_handler
from serve.tinyconfig import MEMORY_POLICY_VERSION, automatic_experts

MODEL = 'qwen3.6-35b-a3b'
OTHER = 'gemma-4-26b-a4b-it'
GEOMETRY = {'total_expert_bytes': 2 ** 30 + 1234, 'expert_total': 100, 'experts_per_layer': 10}


class Engine(MockEngine):
    def __init__(self, tok, slots=70, loaded=True):
        super().__init__(tok, 'never generated', max_context=8192)
        self.info = {'expert_slots': slots, 'expert_total': 100}
        self.loaded = loaded
        self.restarts = 0
        self.unloads = 0
    def alive(self):
        return self.loaded
    def unload(self):
        self.unloads += 1
        self.loaded = False
    def close(self):
        self.loaded = False
    def restart(self, tries=1):
        self.restarts += 1
        self.loaded = True
    def generate(self, *args, **kwargs):
        raise AssertionError('Generation is forbidden in the placement fixture')


class Download:
    state = 'idle'
    def status(self):
        return {'state': self.state}
    def close(self):
        pass


class Profiles:
    current_id = MODEL
    def __init__(self, root):
        self.rows = {}
        self.fail_save = False
        for name, arch, filename in ((MODEL, 'qwen35moe', 'config.json'), (OTHER, 'gemma4', 'config-gemma4.json')):
            cfg = {'arch': arch, 'model_name': name, 'args': ['--model', 'fixture.gguf', '--max-context', '8192'],
                   'memory_policy_version': MEMORY_POLICY_VERSION, 'private_setting': 'preserve me'}
            path = root / filename
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=3) + '\n', encoding='utf-8')
            self.rows[name] = ModelProfile(name, arch, path, cfg)
    def get(self, name):
        return self.rows[name]
    def resolve(self, name):
        if name not in self.rows:
            raise ValueError('unknown fixture model')
        return name
    def update(self, name, cfg, save=False):
        old = self.get(name)
        if save:
            if self.fail_save:
                raise OSError('fixture disk failure')
            old.path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        self.rows[name] = ModelProfile(name, old.arch, old.path, deepcopy(cfg))
        return self.rows[name]
    def context_info(self, model):
        return {'native_tokens': 8192, 'allocated_tokens': None, 'max_tokens': 8192}


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tinystrata-placement-test-')
        self.addCleanup(self.temp.cleanup)
        self.profiles = Profiles(Path(self.temp.name))
        self.original = self.profiles.get(MODEL).path.read_bytes()
        self.other_original = self.profiles.get(OTHER).path.read_bytes()
        tok = ByteTokenizer()
        self.svc = Service(Engine(tok), tok, ChatTemplate(ROOT / 'serve' / 'chat_template.jinja'))
        self.svc.downloads = Download()
        api = SimpleNamespace(responses_api=SimpleNamespace(_kept_prompt_tools=(None, None)))
        with mock.patch('serve.model_switch.ModelProfiles', return_value=self.profiles):
            self.control = ModelSwitcher(self.svc, self.profiles.get(MODEL).config,
                                         self.profiles.get(MODEL).path, api)
        self.svc.models = self.control
        self.addCleanup(self.control.close)
        self.geometry = mock.patch('serve.model_switch.expert_geometry', return_value=GEOMETRY)
        self.geometry.start(); self.addCleanup(self.geometry.stop)
        self.prepare = mock.patch.object(self.control, '_prepare', side_effect=self.prepared)
        self.prepare.start(); self.addCleanup(self.prepare.stop)
        self.activate = mock.patch.object(self.control, '_activate', side_effect=lambda bundle: bundle.engine.restart(tries=1))
        self.activate.start(); self.addCleanup(self.activate.stop)
        self.httpd = Server(('127.0.0.1', 0), make_handler(self.svc))
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.thread.start(); self.addCleanup(self.stop_http)

    def prepared(self, model, configuration):
        self.assertEqual(model, MODEL)
        percent = configuration.get('expert_gpu_percent')
        desired = configuration.get('expert_gpu_count', percent)
        slots = min(70, desired) if desired is not None else 70
        fresh = Service(Engine(self.svc.tok, slots, loaded=False), self.svc.tok, self.svc.template,
                        model_name=MODEL, generation_defaults=configuration)
        if 'kv_grow' in self.svc.engine.info:
            fresh.engine.info['kv_grow'] = self.svc.engine.info['kv_grow']
        fresh.engine.info.update(getattr(self, 'candidate_native_info', {}))
        return ModelBundle(MODEL, deepcopy(configuration), self.profiles.get(MODEL).path, fresh)

    def stop_http(self):
        self.httpd.shutdown(); self.thread.join(3); self.httpd.server_close()

    def wait(self):
        worker = self.control._worker
        if worker is not None:
            worker.join(3)
            self.assertFalse(worker.is_alive())
        return self.control.expert_placement()

    def http(self, method='GET', req=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=3)
        try:
            connection.request(method, '/v1/model/expert-placement', json.dumps(req) if req is not None else None,
                               {'Content-Type': 'application/json', 'Connection': 'close', **(headers or {})})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def apply(self, percent):
        code, value = self.http('POST', {'model': MODEL, 'gpu_percent': percent})
        self.assertIn(code, (200, 202), value)
        return self.wait()

    def apply_count(self, count):
        code, value = self.http('POST', {'model': MODEL, 'gpu_experts': count})
        self.assertIn(code, (200, 202), value)
        return self.wait()

    def test_auto_and_requested_percent_are_distinct_from_observed_residency(self):
        code, value = self.http()
        self.assertEqual(code, 200, value)
        placement = value['placement']
        self.assertTrue(placement['auto'])
        self.assertIsNone(placement['gpu_percent'])
        self.assertEqual(placement['actual_percent'], 70)
        self.assertEqual(placement['expert_total'], 100)
        self.assertTrue(placement['supported'])
        self.assertIsNone(placement['requested_expert_bytes'])

    def test_zero_saves_cpu_experts_only_for_active_profile(self):
        old = self.svc.engine
        value = self.apply(0)
        self.assertEqual(value['operation']['state'], 'ready')
        self.assertEqual(value['placement']['gpu_percent'], 0)
        self.assertEqual(value['placement']['actual_percent'], 0)
        saved = json.loads(self.profiles.get(MODEL).path.read_text(encoding='utf-8'))
        self.assertEqual(config.value_of(saved, 'expert_vram'), 0)
        self.assertEqual(saved['expert_gpu_percent'], 0)
        self.assertEqual(saved['private_setting'], 'preserve me')
        self.assertEqual(self.profiles.get(OTHER).path.read_bytes(), self.other_original)
        self.assertEqual(old.unloads, 1)
        self.assertIsNot(self.svc.engine, old)

    def test_full_request_uses_uncapped_native_placement_and_reports_memory_limited_actual(self):
        original_args = list(self.profiles.get(MODEL).config['args'])
        value = self.apply(100)['placement']
        self.assertEqual(value['requested_expert_bytes'], GEOMETRY['total_expert_bytes'])
        self.assertIsNone(value['expert_vram_gib'])
        self.assertEqual(value['gpu_percent'], 100)
        self.assertEqual(value['actual_percent'], 70)
        self.assertFalse(value['auto'])
        saved = self.profiles.get(MODEL).config
        self.assertEqual(saved['expert_gpu_percent'], 100)
        self.assertIsNone(config.value_of(saved, 'expert_vram'))
        self.assertEqual(saved['args'], original_args)

    def test_intermediate_budget_and_auto_restore(self):
        value = self.apply(30)['placement']
        expected = round(GEOMETRY['total_expert_bytes'] * .30)
        self.assertEqual(value['requested_expert_bytes'], expected)
        self.assertEqual(value['expert_vram_gib'], expected / 2 ** 30)
        value = self.apply(None)['placement']
        self.assertTrue(value['auto'])
        saved = self.profiles.get(MODEL).config
        self.assertNotIn('expert_gpu_percent', saved)
        self.assertIsNone(config.value_of(saved, 'expert_vram'))

    def test_legacy_manual_budget_is_not_mislabeled_automatic_or_as_percentage(self):
        cfg, _ = config.apply(self.profiles.get(MODEL).config, {'expert_vram': .5})
        self.profiles.update(MODEL, cfg)
        value = self.control.expert_placement()['placement']
        self.assertTrue(value['legacy_budget'])
        self.assertFalse(value['auto'])
        self.assertIsNone(value['gpu_percent'])
        self.assertEqual(value['expert_vram_gib'], .5)

    def test_direct_gib_override_clears_stale_percentage_descriptor(self):
        cfg = {**self.profiles.get(MODEL).config, 'expert_gpu_percent': 50}
        updated, _ = config.apply(cfg, {'expert_vram': .5})
        self.assertNotIn('expert_gpu_percent', updated)
        same = {**updated, 'expert_gpu_percent': 50}
        unchanged, changed = config.apply(same, {'expert_vram': .5})
        self.assertEqual(unchanged, same)
        self.assertEqual(changed, [])

    def test_direct_gib_override_clears_only_percentage_owned_fixed_slots(self):
        cfg = deepcopy(self.profiles.get(MODEL).config)
        cfg['args'] += ['--expert-slots', '5']
        owned = {**cfg, 'expert_gpu_percent': 50}
        updated, _ = config.apply(owned, {'expert_vram': 8})
        self.assertNotIn('expert_gpu_percent', updated)
        self.assertNotIn('--expert-slots', updated['args'])
        legacy, _ = config.apply(cfg, {'expert_vram': 8})
        self.assertIn('--expert-slots', legacy['args'])
        reset = automatic_experts(owned)
        self.assertNotIn('expert_gpu_percent', reset)
        self.assertNotIn('--expert-slots', reset['args'])
        counted = {**cfg, 'expert_gpu_count': 50}
        updated, _ = config.apply(counted, {'expert_vram': 8})
        self.assertNotIn('expert_gpu_count', updated)
        self.assertNotIn('--expert-slots', updated['args'])
        reset = automatic_experts(counted)
        self.assertNotIn('expert_gpu_count', reset)
        self.assertNotIn('--expert-slots', reset['args'])

    def test_verified_qwen_vmm_uses_requested_expert_count_without_staging_budget_tax(self):
        self.svc.engine.info['kv_grow'] = 1
        value = self.apply(50)['placement']
        saved = self.profiles.get(MODEL).config
        self.assertIsNone(config.value_of(saved, 'expert_vram'))
        self.assertEqual(saved['args'][saved['args'].index('--expert-slots') + 1], '5')
        self.assertEqual(value['gpu_percent'], 50)
        self.assertEqual(value['actual_percent'], 50)
        self.assertEqual(value['expert_slots_per_layer'], 5)
        self.assertFalse(value['legacy_budget'])
        self.assertEqual(self.control._prepare.call_count, 1)
        self.apply(0)
        self.assertEqual(config.value_of(self.profiles.get(MODEL).config, 'expert_vram'), 0)
        self.assertNotIn('--expert-slots', self.profiles.get(MODEL).config['args'])
        self.apply(100)
        self.assertIsNone(config.value_of(self.profiles.get(MODEL).config, 'expert_vram'))
        self.assertNotIn('--expert-slots', self.profiles.get(MODEL).config['args'])

    def test_vmm_disable_flags_and_environment_keep_safe_gib_fallback(self):
        self.svc.engine.info['kv_grow'] = 1
        cfg = deepcopy(self.profiles.get(MODEL).config)
        for changes in ({'args': cfg['args'] + ['--kv-whole']},
                        {'args': cfg['args'] + ['--no-kv-grow']}, {'env': {'STRATA_KV_GROW': '0'}},
                        {'env': {'STRATA_KV_GROW': 'invalid'}}):
            with self.subTest(changes=changes):
                new, _ = self.control._placement_configuration({**cfg, **changes}, 50, GEOMETRY, self.svc.engine.info)
                self.assertNotIn('--expert-slots', new['args'])
                self.assertGreater(config.value_of(new, 'expert_vram'), 0)
        for arch in ('gemma4', 'lfm2moe'):
            new, _ = self.control._placement_configuration({**cfg, 'arch': arch}, 50, GEOMETRY, {'kv_grow': 1})
            self.assertNotIn('--expert-slots', new['args'])
        new, _ = self.control._placement_configuration(
            {**cfg, 'args': cfg['args'] + ['--kv-grow'], 'env': {'STRATA_KV_GROW': '0'}},
            50, GEOMETRY, {'kv_grow': 1})
        self.assertIn('--expert-slots', new['args'])

    def test_cold_candidate_can_adopt_verified_vmm_once_before_publication(self):
        self.candidate_native_info = {'kv_grow': 1}
        value = self.apply(50)
        saved = self.profiles.get(MODEL).config
        self.assertIsNone(config.value_of(saved, 'expert_vram'))
        self.assertIn('--expert-slots', saved['args'])
        self.assertEqual(value['operation']['state'], 'ready')
        self.assertEqual(self.control._prepare.call_count, 2)

    def test_count_api_maps_5120_to_128_per_layer_and_preserves_exact_requested_count(self):
        sizes = {'total_expert_bytes': 13246627840, 'expert_total': 10240, 'experts_per_layer': 256}
        self.svc.engine.info.update(kv_grow=1, expert_total=10240, expert_slots=10240)
        self.candidate_native_info = {'kv_grow': 1, 'expert_total': 10240, 'expert_slots': 5120}
        with mock.patch('serve.model_switch.expert_geometry', return_value=sizes):
            placement = self.apply_count(5120)['placement']
            self.assertEqual(placement['gpu_experts'], 5120)
            self.assertEqual(placement['expert_slots'], 5120)
            self.assertEqual(placement['count_step'], 40)
            self.assertEqual(placement['expert_total'], 10240)
            self.assertEqual(placement['actual_percent'], 50)
            self.assertIsNone(placement['gpu_percent'])
            self.assertEqual(placement['requested_percent'], 50)
            self.assertFalse(placement['auto'])
            self.assertFalse(placement['legacy_budget'])
            saved = self.profiles.get(MODEL).config
            self.assertEqual(saved['expert_gpu_count'], 5120)
            self.assertNotIn('expert_gpu_percent', saved)
            self.assertEqual(saved['args'][saved['args'].index('--expert-slots') + 1], '128')
            self.assertIsNone(config.value_of(saved, 'expert_vram'))
            self.assertEqual(self.control._prepare.call_count, 1)
            self.svc.engine.loaded = False
            self.assertEqual(self.control.expert_placement()['placement']['count_step'], 40)

    def test_count_zero_full_and_auto_have_distinct_persisted_descriptors(self):
        self.svc.engine.info['kv_grow'] = 1
        value = self.apply_count(0)['placement']
        self.assertEqual(value['gpu_experts'], 0)
        self.assertFalse(value['auto'])
        self.assertEqual(config.value_of(self.profiles.get(MODEL).config, 'expert_vram'), 0)
        value = self.apply_count(100)['placement']
        self.assertEqual(value['gpu_experts'], 100)
        self.assertFalse(value['auto'])
        self.assertIsNone(config.value_of(self.profiles.get(MODEL).config, 'expert_vram'))
        self.assertNotIn('--expert-slots', self.profiles.get(MODEL).config['args'])
        value = self.apply_count(None)['placement']
        self.assertIsNone(value['gpu_experts'])
        self.assertTrue(value['auto'])
        self.assertNotIn('expert_gpu_count', self.profiles.get(MODEL).config)
        self.assertNotIn('expert_gpu_percent', self.profiles.get(MODEL).config)

    def test_count_schema_and_stable_granularity_reject_unrepresentable_values(self):
        for count in (True, -1, 101, 55, 50.0, '50'):
            code, value = self.http('POST', {'model': MODEL, 'gpu_experts': count})
            self.assertEqual(code, 400, value)
        code, value = self.http('POST', {'model': MODEL, 'gpu_experts': 50, 'gpu_percent': 50})
        self.assertEqual(code, 400, value)
        sizes = {'total_expert_bytes': 13246627840, 'expert_total': 10240, 'experts_per_layer': 256}
        with mock.patch('serve.model_switch.expert_geometry', return_value=sizes):
            code, value = self.http('POST', {'model': MODEL, 'gpu_experts': 5111})
            self.assertEqual(code, 400, value)
            self.assertEqual(self.control.expert_placement()['placement']['count_step'], 40)

    def test_old_percentage_metadata_is_reported_as_representable_requested_count(self):
        cfg = {**self.profiles.get(MODEL).config, 'expert_gpu_percent': 10}
        self.profiles.update(MODEL, cfg)
        sizes = {'total_expert_bytes': 13246627840, 'expert_total': 10240, 'experts_per_layer': 256}
        with mock.patch('serve.model_switch.expert_geometry', return_value=sizes):
            value = self.control.expert_placement()['placement']
            self.assertEqual(value['gpu_experts'], 1000)
            self.assertEqual(value['gpu_percent'], 10)
            self.assertEqual(value['count_step'], 40)
            self.assertFalse(value['auto'])

    def test_percentage_and_auto_clear_conflicting_legacy_fixed_slots(self):
        cfg = deepcopy(self.profiles.get(MODEL).config)
        cfg['args'] += ['--expert-slots', '32']
        self.profiles.update(MODEL, cfg, save=True)
        before = self.control.expert_placement()['placement']
        self.assertTrue(before['legacy_budget'])
        self.assertFalse(before['auto'])
        self.assertEqual(before['legacy_expert_slots'], 32)
        self.apply(0)
        saved = self.profiles.get(MODEL).config
        self.assertNotIn('--expert-slots', saved['args'])
        self.assertEqual(config.value_of(saved, 'expert_vram'), 0)
        self.apply(None)
        self.assertTrue(self.control.expert_placement()['placement']['auto'])

    def test_unchanged_auto_is_a_noop(self):
        value = self.apply(None)
        self.assertFalse(value['operation']['switching'])
        self.assertIsNone(self.control._worker)
        self.assertEqual(self.profiles.get(MODEL).path.read_bytes(), self.original)

    def test_invalid_percent_wrong_model_and_unknown_fields_are_rejected(self):
        for percent in (True, -10, 1, 25, 20.0, '30', 110):
            with self.subTest(percent=percent):
                code, value = self.http('POST', {'model': MODEL, 'gpu_percent': percent})
                self.assertEqual(code, 400, value)
        code, value = self.http('POST', {'model': OTHER, 'gpu_percent': 30})
        self.assertEqual(code, 409, value)
        code, value = self.http('POST', {'model': MODEL, 'gpu_percent': 30, 'unload': True})
        self.assertEqual(code, 400, value)
        self.assertEqual(self.profiles.get(MODEL).path.read_bytes(), self.original)

    def test_generation_model_operations_downloads_and_non_http_fifo_work_refuse_change(self):
        self.control._requests = 1
        with self.assertRaises(ModelControlError):
            self.control.start_expert_placement(MODEL, 30)
        self.control._requests = 0
        self.control._busy = True
        code, value = self.http()
        self.assertEqual(code, 200, value)  # Polling is allowed while a reload is running.
        self.assertIsNone(value['placement']['actual_percent'])
        with self.assertRaises(ModelControlError):
            self.control.start_expert_placement(MODEL, 30)
        self.control._busy = False
        self.svc.downloads.state = 'downloading'
        with self.assertRaises(ModelControlError):
            self.control.start_expert_placement(MODEL, 30)
        self.svc.downloads.state = 'idle'
        self.svc.fifo.acquire()
        try:
            with self.assertRaises(ModelControlError):
                self.control.start_expert_placement(MODEL, 30)
        finally:
            self.svc.fifo.release()

    def test_failed_activation_restores_native_model_and_keeps_exact_config(self):
        old = self.svc.engine
        self.activate.stop()
        with mock.patch.object(self.control, '_activate', side_effect=ValueError('fixture activation failure')):
            value = self.apply(40)
        self.assertEqual(value['operation']['state'], 'error')
        self.assertTrue(value['operation']['rolled_back'])
        self.assertEqual(self.profiles.get(MODEL).path.read_bytes(), self.original)
        self.assertNotIn('expert_gpu_percent', self.profiles.get(MODEL).config)
        self.assertIs(self.svc.engine, old)
        self.assertTrue(old.alive())
        self.assertEqual(old.restarts, 1)
        self.assertIsNone(value['placement']['pending'])

    def test_failure_after_save_rolls_back_persisted_config_and_partial_publication(self):
        old = self.svc.engine
        publish = self.control._publish
        def broken_publish(bundle):
            publish(bundle)
            raise ValueError('fixture publish failure')
        with mock.patch.object(self.control, '_publish', side_effect=broken_publish):
            value = self.apply(40)
        self.assertEqual(value['operation']['state'], 'error')
        self.assertTrue(value['operation']['rolled_back'])
        self.assertEqual(self.profiles.get(MODEL).path.read_bytes(), self.original)
        self.assertNotIn('expert_gpu_percent', self.profiles.get(MODEL).config)
        self.assertIs(self.svc.engine, old)
        self.assertTrue(old.alive())

    def test_disk_save_failure_rolls_back_without_reporting_ready(self):
        old = self.svc.engine
        self.profiles.fail_save = True
        value = self.apply(40)
        self.assertEqual(value['operation']['state'], 'error')
        self.assertTrue(value['operation']['rolled_back'])
        self.assertEqual(self.profiles.get(MODEL).path.read_bytes(), self.original)
        self.assertIs(self.svc.engine, old)
        self.assertTrue(old.alive())

    def test_external_config_edit_during_reload_is_preserved_and_refuses_commit(self):
        path = self.profiles.get(MODEL).path
        newer = self.original + b'\n'
        self.activate.stop()
        def activate_and_edit(bundle):
            bundle.engine.restart(tries=1)
            path.write_bytes(newer)
        with mock.patch.object(self.control, '_activate', side_effect=activate_and_edit):
            value = self.apply(40)
        self.assertEqual(value['operation']['state'], 'error')
        self.assertTrue(value['operation']['rolled_back'])
        self.assertEqual(path.read_bytes(), newer)
        self.assertNotIn('expert_gpu_percent', self.profiles.get(MODEL).config)

    def test_unloaded_model_reports_no_stale_residency_and_can_be_configured(self):
        self.svc.engine.loaded = False
        value = self.control.expert_placement()['placement']
        self.assertIsNone(value['actual_percent'])
        self.assertIsNone(value['expert_slots'])
        self.assertEqual(value['expert_total'], 100)
        value = self.apply(20)
        self.assertTrue(self.svc.loaded())
        self.assertEqual(value['placement']['gpu_percent'], 20)

    def test_missing_geometry_is_unavailable_and_never_unloads_model(self):
        old = self.svc.engine
        with mock.patch('serve.model_switch.expert_geometry', side_effect=ValueError('unsupported expert encoding')):
            code, value = self.http()
            self.assertEqual(code, 200, value)
            self.assertFalse(value['placement']['supported'])
            code, value = self.http('POST', {'model': MODEL, 'gpu_percent': 30})
            self.assertEqual(code, 400, value)
        self.assertEqual(old.unloads, 0)

    def test_authentication_and_same_origin_apply(self):
        self.svc.api_key = 'fixture-secret'
        code, _ = self.http()
        self.assertEqual(code, 401)
        code, value = self.http(headers={'Authorization': 'Bearer fixture-secret'})
        self.assertEqual(code, 200, value)
        code, value = self.http('POST', {'model': MODEL, 'gpu_percent': 30},
            {'Authorization': 'Bearer fixture-secret', 'Origin': 'https://untrusted.example'})
        self.assertEqual(code, 403, value)

    def test_regular_model_switch_still_uses_original_transaction_without_placement(self):
        self.control._downloads[OTHER] = Download()
        profile = self.profiles.get(OTHER)
        fresh = Service(Engine(self.svc.tok, loaded=False), self.svc.tok, self.svc.template,
                        model_name=OTHER, generation_defaults=profile.config)
        bundle = ModelBundle(OTHER, deepcopy(profile.config), profile.path, fresh)
        self.prepare.stop()
        with mock.patch.object(self.control, '_prepare', return_value=bundle) as prepare:
            self.control.start(OTHER)
            value = self.wait()
        prepare.assert_called_once_with(OTHER)
        self.assertEqual(value['operation']['state'], 'ready')
        self.assertEqual(self.control.active_id, OTHER)
        self.assertEqual(self.profiles.get(MODEL).path.read_bytes(), self.original)
        self.assertEqual(self.profiles.get(OTHER).path.read_bytes(), self.other_original)


if __name__ == '__main__':
    unittest.main()
