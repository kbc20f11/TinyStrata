"""Retired generator settings cannot block Intel profiles or reactivate HTTP generation.

No native model/device is started. Profile checks use absent model files; the
HTTP case runs the real server main with its explicit mock engine.
"""
from copy import deepcopy
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import intel_backend
from serve.intel_models import MODEL_CONFIGS
from serve.model_profiles import ModelProfiles


QWEN = intel_backend.QWEN_MODEL
TIEL = intel_backend.TIEL_MODEL


class GeneratorProfileMigration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.active_path = self.directory / MODEL_CONFIGS[QWEN]

    def configuration(self, model):
        return intel_backend.configure_backend({
            'backend': 'sycl', 'intel_profile': 'generic', 'arch': 'qwen35moe',
            'model_name': model, 'exe': str(self.directory / 'absent-intel.exe'),
            'cwd': str(self.directory), 'args': ['--model', str(self.directory / (model + '.gguf'))],
            'vision_mode': 'no', 'sampling': {'temperature': .7},
            'api_key': 'private-fixture', 'private_state': {'images': ['keep-user-data']},
        })

    def test_active_constructor_discards_legacy_generator_without_mutating_caller(self):
        for legacy in ({'exe': 'never-run-generator.exe'}, 'malformed', True, None):
            with self.subTest(legacy=legacy):
                cfg = self.configuration(QWEN)
                cfg['images'] = legacy
                before = deepcopy(cfg)
                profiles = ModelProfiles(self.active_path, cfg)
                normalized = profiles.get(QWEN).config
                self.assertNotIn('images', normalized)
                self.assertEqual(cfg, before)
                self.assertEqual(normalized['private_state'], before['private_state'])
                self.assertEqual(normalized['sampling'], before['sampling'])
                self.assertEqual(normalized['api_key'], before['api_key'])

    def test_inactive_profile_can_be_listed_and_prepared_for_switch_without_rewriting_it(self):
        active = self.configuration(QWEN)
        inactive = self.configuration(TIEL)
        inactive['images'] = {'exe': 'never-run-generator.exe', 'out_dir': 'keep-assets'}
        path = self.directory / MODEL_CONFIGS[TIEL]
        path.write_text(json.dumps(inactive), encoding='utf-8')
        saved = path.read_bytes()
        profiles = ModelProfiles(self.active_path, active)
        self.assertIn(TIEL, {row['id'] for row in profiles.list()})
        self.assertNotIn('images', profiles.get(TIEL).config)
        # prepare() rereads the on-disk profile. It must reach the genuine
        # missing-model refusal rather than reject the retired generator.
        with self.assertRaisesRegex(ValueError, 'is not installed'):
            profiles.prepare(TIEL)
        self.assertEqual(path.read_bytes(), saved)
        self.assertEqual(profiles.get(TIEL).config['private_state'], inactive['private_state'])

    def test_refreshing_a_saved_profile_drops_only_the_generator_setting(self):
        profiles = ModelProfiles(self.active_path, self.configuration(QWEN))
        stale = self.configuration(TIEL)
        stale['images'] = 'malformed'
        before = deepcopy(stale)
        updated = profiles.update(TIEL, stale)
        self.assertNotIn('images', updated.config)
        self.assertEqual(stale, before)
        self.assertEqual(updated.config['private_state'], before['private_state'])
        self.assertEqual(updated.config['vision_mode'], before['vision_mode'])


class RetiredGeneratorHttp(unittest.TestCase):
    def test_real_mock_server_main_migrates_stale_config_and_exposes_no_generator(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config = directory / 'mock-config.json'
            output = directory / 'server.log'
            retained = directory / 'retained-output.png'
            retained.write_bytes(b'user-owned-output-must-stay')
            original = {'images': {'exe': str(directory / 'never-run.exe'), 'out_dir': str(directory)},
                        'vision': {'exe': 'vision.exe', 'model': 'model.gguf', 'mmproj': 'mmproj.gguf'},
                        'private_state': {'images': ['user-content']}}
            config.write_text(json.dumps(original), encoding='utf-8')
            with socket.socket() as reserve:
                reserve.bind(('127.0.0.1', 0))
                port = reserve.getsockname()[1]
            # Suppress only hardware telemetry. Main, config persistence,
            # HTTP routing and the explicit mock inference path run normally.
            bootstrap = ('from serve import server\n'
                         'server.Service.start_telemetry = lambda self: None\n'
                         'raise SystemExit(server.main())\n')
            with output.open('wb') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-X', 'utf8', '-B', '-c', bootstrap,
                                         '--engine', 'mock', '--config', str(config), '--port', str(port)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    def request(method, route, body=None):
                        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                        try:
                            connection.request(method, route, json.dumps(body) if body is not None else None,
                                               {'Content-Type': 'application/json', 'Connection': 'close'})
                            response = connection.getresponse()
                            return response.status, json.loads(response.read())
                        finally:
                            connection.close()
                    deadline = time.monotonic() + 12
                    while True:
                        if proc.poll() is not None:
                            self.fail('Mock API ended during startup: ' + output.read_text(encoding='utf-8'))
                        try:
                            status, health = request('GET', '/health')
                            if status == 200: break
                        except OSError:
                            pass
                        if time.monotonic() >= deadline:
                            self.fail('Mock API readiness deadline: ' + output.read_text(encoding='utf-8'))
                        time.sleep(.05)
                    self.assertIn('images', health)       # image-understanding capability remains
                    self.assertNotIn('image_generation', health)
                    for route in ('/v1/images/generations', '/v1/images/edits'):
                        status, _ = request('POST', route, {'prompt': 'test'})
                        self.assertEqual(status, 404)
                    self.assertEqual(request('GET', '/v1/images/files/' + 'a'*32 + '.png')[0], 404)
                    status, discovered = request('GET', '/v1/models')
                    self.assertEqual(status, 200)
                    self.assertTrue(discovered['data'])
                    self.assertTrue(all(row['architecture']['output_modalities'] == ['text']
                                        for row in discovered['data']))
                    status, state = request('GET', '/v1/status')
                    self.assertEqual(status, 200)
                    self.assertIn('vision', state)
                    self.assertNotIn('image_generation', state)
                    status, answer = request('POST', '/v1/chat/completions', {
                        'messages': [{'role': 'user', 'content': 'test'}], 'max_tokens': 16, 'stream': False})
                    self.assertEqual(status, 200)
                    self.assertTrue(answer['choices'])
                    migrated = json.loads(config.read_text(encoding='utf-8-sig'))
                    self.assertEqual(migrated, {key: value for key, value in original.items() if key != 'images'})
                    self.assertEqual(retained.read_bytes(), b'user-owned-output-must-stay')
                finally:
                    proc.kill()
                    proc.wait(timeout=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
