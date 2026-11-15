"""Concurrent file preparation and native switching; isolated files/fake engines only.

Production-model/browser workflow is exercised separately. These bounded event
tests reproduce the admission/publication races without network or GPU access.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import test_model_library as library
import test_model_switch_vision_resilience as switching
from serve import model_setup, server
from serve.model_switch import ModelBundle, ModelControlError


class Download:
    def __init__(self, state='idle', *, running=False):
        self.state, self.running, self.closed, self.starts = state, running, False, 0

    def status(self):
        return {'state': self.state}

    def start(self):
        self.starts += 1
        self.state, self.running = 'downloading', True
        return self.status()

    def cancel(self):
        self.state = 'cancelled'
        return self.status()

    def close(self):
        self.closed = True


class DownloadChatConcurrency(unittest.TestCase):
    def setUp(self):
        self.f = switching.ModelSwitchVisionResilience()
        self.addCleanup(self.f.doCleanups)
        self.f.setUp()

    def bootstrap_callback(self):
        owned = Download()
        cfg = {**self.f.profiles.get(switching.OLD).config, 'vision_mode': 'no'}
        with patch('serve.model_download.FixedModelBundleDownloader', return_value=owned) as construct:
            server.configure_model_downloads(self.f.svc, cfg, self.f.profiles.get(switching.OLD).path)
        self.f.control._downloads[switching.OLD] = owned
        return construct.call_args.kwargs['on_ready'], owned

    def test_unrelated_transfer_allows_generation_and_real_switch_transaction(self):
        manager = Download('verifying', running=True)
        self.f.control._downloads[switching.OLD] = manager
        with self.f.control.admit():
            self.f.assert_text_works('old')
            self.assertEqual(self.f.control.status()['in_flight'], 1)
        result = self.f.switch()
        self.assertEqual(result['model'], switching.NEW)
        self.assertEqual(result['state'], 'ready')
        self.assertTrue(result['downloading'])
        self.assertEqual(manager.state, 'verifying')
        self.f.assert_text_works('new')

    def test_target_download_and_ready_worker_tail_block_switch_before_native_work(self):
        for state, running in (('downloading', False), ('verifying', False), ('loading', False), ('ready', True)):
            with self.subTest(state=state), patch.object(self.f.control, '_prepare') as prepare:
                self.f.control._downloads[switching.NEW] = Download(state, running=running)
                with self.assertRaises(ModelControlError) as caught:
                    self.f.control.start(switching.NEW)
                self.assertEqual(caught.exception.status, 409)
                prepare.assert_not_called()
                self.assertIs(self.f.svc.engine, self.f.old_engine)
                self.assertTrue(self.f.svc.loaded())

    def test_expert_reload_ignores_other_download_but_not_active_target(self):
        profile = self.f.profiles.get(switching.OLD)
        cfg = {**profile.config, 'fixture_placement': 50}
        self.f.control._downloads[switching.NEW] = Download('verifying', running=True)
        with patch('serve.model_switch.expert_geometry', return_value={'total_expert_bytes': 100}), \
             patch.object(self.f.control, '_placement_configuration', return_value=(cfg, {'gpu_percent': 50})), \
             patch('serve.model_switch.threading.Thread') as worker:
            self.f.control.start_expert_placement(switching.OLD, 50)
            worker.return_value.start.assert_called_once()
        self.f.control._busy = False
        self.f.control._worker = None
        self.f.control._downloads[switching.OLD] = Download('ready', running=True)
        with self.assertRaises(ModelControlError):
            self.f.control.start_expert_placement(switching.OLD, 50)

    def test_second_download_cannot_write_a_shared_bundle_asset(self):
        first = Download('downloading', running=True)
        self.f.control._downloads[switching.OLD] = first
        with patch.object(self.f.control, '_downloader') as create:
            with self.assertRaises(ModelControlError) as caught:
                self.f.control.download(switching.NEW, start=True)
            self.assertEqual(caught.exception.status, 409)
            create.assert_not_called()
        self.assertEqual(first.state, 'downloading')
        # Repeated starts/cancellation of the same job remain usable.
        self.f.control.download(switching.OLD, start=True)
        self.f.control.download(switching.OLD, cancel=True)
        self.assertEqual(first.state, 'cancelled')

    def test_cancel_remains_available_while_an_unrelated_native_switch_runs(self):
        manager = Download('downloading', running=True)
        self.f.control._downloads[switching.OLD] = manager
        self.f.control._busy = True
        self.f.control._target = switching.NEW
        result = self.f.control.download(switching.OLD, cancel=True)
        self.assertEqual(result['state'], 'cancelled')
        self.assertTrue(self.f.control._busy)
        self.assertEqual(self.f.control._target, switching.NEW)
        self.f.control._busy = False

    def test_prepared_profile_waits_for_controller_publication_lock(self):
        entered, called, done = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def prepare(*args, **kwargs):
            called.set()
            return {'prepared': True}
        def run():
            entered.set()
            try:
                self.f.control.prepare_download(switching.NEW, Path('verified.gguf'))
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()
        # This is the same lease used for a switch's profile publication.
        with patch.object(self.f.profiles, 'prepare', side_effect=prepare, create=True):
            with self.f.control._lock:
                worker = threading.Thread(target=run)
                worker.start()
                self.assertTrue(entered.wait(1))
                self.assertFalse(called.wait(.05))
            self.assertTrue(done.wait(2))
            worker.join(2)
        self.assertTrue(called.is_set())
        self.assertFalse(errors)

    def test_original_callback_during_pending_switch_never_overwrites_selected_engine(self):
        activate, _ = self.bootstrap_callback()
        entered, release = threading.Event(), threading.Event()
        engine = switching.Engine(self.f.tok, self.f.events, 'new')
        vision = switching.Vision(self.f.events, 'new')
        fresh = server.Service(engine, self.f.tok, self.f.svc.template, model_name=switching.NEW, vision=vision)
        profile = self.f.profiles.get(switching.NEW)
        bundle = ModelBundle(switching.NEW, deepcopy(profile.config), profile.path, fresh)
        def prepare(*args):
            entered.set()
            if not release.wait(2):
                raise AssertionError('fixture switch was not released')
            return bundle
        with patch.object(self.f.control, '_prepare', side_effect=prepare), \
             patch.object(self.f.control, 'prepare_download') as files_only, \
             patch.object(server, 'model_tokenizer', side_effect=AssertionError('No stale activation')), \
             redirect_stdout(StringIO()):
            self.f.control.start(switching.NEW)
            worker = self.f.control._worker
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(self.f.control.active_id, switching.OLD)
                activate(Path('verified-original.gguf'))
                files_only.assert_called_once_with(switching.OLD, Path('verified-original.gguf'), None, None)
                self.assertTrue(self.f.control._busy)
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(self.f.control.active_id, switching.NEW)
        self.assertIs(self.f.svc.engine, engine)
        self.f.assert_text_works('new')

    def test_active_bootstrap_reserves_runtime_before_waiting_for_native_work(self):
        entered, release = threading.Event(), threading.Event()
        errors = []
        armed = False
        def current_path(_):
            if not armed:
                return Path('verified.gguf')
            entered.set()
            if not release.wait(2):
                raise AssertionError('fixture activation was not released')
            return Path('verified.gguf')
        def run():
            try:
                activate(Path('verified.gguf'))
            except BaseException as error:
                errors.append(error)
        self.f.svc.vision = None
        with patch.object(model_setup, 'model_path', side_effect=current_path):
            activate, _ = self.bootstrap_callback()
            armed = True
            worker = threading.Thread(target=run)
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                with self.assertRaises(ModelControlError):
                    self.f.control.start(switching.NEW)
                with self.assertRaises(ModelControlError):
                    with self.f.control.admit():
                        self.fail('bootstrap activation must retain its runtime lease')
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertFalse(self.f.control._busy)
        self.assertIs(self.f.svc.engine, self.f.old_engine)

    def test_inactive_callback_checks_its_owner_not_new_selected_download_manager(self):
        activate, owned = self.bootstrap_callback()
        self.f.control.active_id = switching.NEW
        self.f.svc.downloads = SimpleNamespace(closed=True)
        with patch.object(self.f.control, 'prepare_download') as prepare:
            activate(Path('verified-original.gguf'))
            prepare.assert_called_once()
        self.assertFalse(owned.closed)
        self.assertEqual(self.f.control.active_id, switching.NEW)


class CatalogDownloadAdmission(unittest.TestCase):
    def test_active_generation_blocks_deletion_but_keeps_download_action_available(self):
        with tempfile.TemporaryDirectory() as root, \
             patch.object(model_setup, 'resolve_executable', return_value=None), \
             patch.object(model_setup, 'shared_library_dirs', side_effect=lambda cfg: cfg):
            fixture = library.create_fixture(root)
            try:
                with fixture.control.admit():
                    row = next(row for row in fixture.control.catalog() if row['id'] == library.GEMMA)
                    self.assertTrue(row['management']['can_download'])
                    self.assertFalse(row['management']['can_delete'])
                    fixture.control.download(library.GEMMA, cancel=True)
            finally:
                fixture.control.close()


if __name__ == '__main__':
    unittest.main()
