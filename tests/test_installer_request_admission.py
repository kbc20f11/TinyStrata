"""Installer admission with real HTTP handling; fixture model never infers."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import test_model_library as fixtures
from serve import install_guard
from serve.model_switch import ModelControlError


class InstallerAdmissionTests(unittest.TestCase):
    setUp = fixtures.LibraryHTTPTests.setUp
    start_http = fixtures.LibraryHTTPTests.start_http
    http = fixtures.LibraryHTTPTests.http

    def marker(self):
        work = self.f.root / 'managed-work'
        work.mkdir(exist_ok=True)
        self.f.control._config_path = work / 'runtime/.tinystrata/config.json'
        marker = work / install_guard.GUARD_NAME
        marker.write_text(json.dumps(dict(schema=1, pid=1234, started_unix=1000.0,
                                         token='fixture-private-token')), encoding='utf-8')
        owner = patch.object(install_guard, '_owner_state', return_value='alive')
        owner.start()
        self.addCleanup(owner.stop)
        return marker

    def test_admitted_work_finishes_but_new_leases_and_mutations_are_refused(self):
        with self.f.control.admit():
            self.assertEqual(self.f.control.status()['in_flight'], 1)
            marker = self.marker()
            before = marker.read_bytes()
            with self.assertRaises(ModelControlError) as blocked:
                with self.f.control.admit():
                    self.fail('New work was admitted during an update')
            self.assertEqual(blocked.exception.status, 503)
            self.assertEqual(self.f.control.status()['in_flight'], 1)
            for action in (self.f.control.start, self.f.control.delete,
                           lambda name: self.f.control.download(name, start=True)):
                with self.subTest(action=action), self.assertRaisesRegex(ModelControlError, 'being updated'):
                    action(fixtures.GEMMA)
            self.assertEqual(marker.read_bytes(), before)
        self.assertEqual(self.f.control.status()['in_flight'], 0)
        marker.unlink()
        with self.f.control.admit():
            self.assertEqual(self.f.control.status()['in_flight'], 1)

    def test_http_generation_load_and_mutations_are_blocked_before_native_work(self):
        self.marker()
        self.start_http()
        for path, body in (('/v1/chat/completions', {'messages': [{'role':'user','content':'test'}]}),
                           ('/v1/responses', {'input':'test'}),
                           ('/v1/messages', {'messages': [{'role':'user','content':'test'}], 'max_tokens':8}),
                           ('/load', {}),
                           ('/v1/model/switch', {'model':fixtures.GEMMA}),
                           ('/v1/model/download', {'model':fixtures.GEMMA}),
                           ('/v1/model/delete', {'model':fixtures.GEMMA})):
            with self.subTest(path=path), patch.object(self.f.svc, 'load', side_effect=AssertionError('No load')):
                status, result = self.http(path=path, body=body)
                self.assertEqual(status, 503, result)
                self.assertIn('being updated', json.dumps(result))
        self.assertFalse(self.f.control._downloads)

    def test_http_status_and_graceful_unload_remain_available(self):
        self.marker()
        self.start_http()
        for path in ('/health', '/v1/status', '/metrics', '/v1/model/switch'):
            with self.subTest(path=path):
                status, result = self.http(method='GET', path=path)
                self.assertEqual(status, 200, result)
        for path in ('/unload', '/v1/unload'):
            with self.subTest(path=path), patch.object(self.f.svc, 'unload_waiting', return_value='unloaded'):
                status, result = self.http(path=path, body={})
                self.assertEqual(status, 200, result)
        self.assertEqual(self.f.control.status()['in_flight'], 0)

    def test_download_activity_is_reported_even_for_active_uninstalled_model(self):
        self.assertFalse(self.f.control.status()['downloading'])
        manager = SimpleNamespace(running=lambda: True, status=lambda: {'state':'downloading'}, close=lambda:None)
        self.f.control._downloads[self.f.control.active_id] = manager
        self.marker()
        self.start_http()
        status, result = self.http(method='GET', path='/v1/model/switch')
        self.assertEqual(status, 200, result)
        self.assertTrue(result['downloading'])
        manager.running = lambda: False
        manager.status = lambda: {'state':'verifying'}
        self.assertTrue(self.f.control.status()['downloading'])
        manager.status = lambda: {'state':'error'}
        self.assertFalse(self.f.control.status()['downloading'])


if __name__ == '__main__':
    unittest.main()
