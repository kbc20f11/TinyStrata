"""Temporary fixed-model files and real loopback HTTP; no native process or GPU.

create_fixture(Path(tempdir)) is also usable by a private browser QA server.
Nothing in this fixture reads the real user's model/configuration directories.
"""
from contextlib import contextmanager
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import model_download as D, model_library as L, model_setup
from serve.frontend import ChatTemplate
from serve.model_profiles import ModelProfiles
from serve.model_switch import ModelControlError, ModelSwitcher
from serve.server import ByteTokenizer, MockEngine, Server, Service, make_handler

QWEN = 'qwen3.6-35b-a3b'
GEMMA = 'gemma-4-26b-a4b-it'
CLOVER = 'clover-35b-a3b-preview'
TIEL = 'tiel-coder-35b-a3b'


class NoInference(MockEngine):
    def generate(self, *args, **kwargs):
        raise AssertionError('Inference is forbidden in the model-library fixture')


def create_fixture(directory, api_key=''):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    models = root / 'model-files'
    models.mkdir(exist_ok=True)
    cfg = dict(arch='qwen35moe', model_name=QWEN, backend='cuda', cwd=str(root),
               args=['--model', str(models / D.FIXED_MODEL.filename), '--max-context', 'auto', '--mtp', '3'],
               private_setting='preserve fixture settings')
    config = root / 'config.json'
    config.write_text(json.dumps(cfg), encoding='utf-8')
    with patch.object(model_setup, 'resolve_executable', return_value=None), \
         patch.object(model_setup, 'shared_library_dirs', side_effect=lambda value: value):
        profiles = ModelProfiles(config, cfg)
    tok = ByteTokenizer()
    svc = Service(NoInference(tok, 'forbidden'), tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
    svc.api_key = api_key
    api = SimpleNamespace(responses_api=SimpleNamespace(_kept_prompt_tools=(None, None)))
    with patch('serve.model_switch.ModelProfiles', return_value=profiles):
        controller = ModelSwitcher(svc, cfg, config, api)
    svc.models = controller
    return SimpleNamespace(root=root, models=models, profiles=profiles, svc=svc, control=controller)


def install_fixture_download(fixture, model, url, content):
    """Owner-supplied tiny HTTP test artifact; never a public/HuggingFace URL."""
    profile = fixture.profiles.get(model)
    artifact = D.manifest_for_arch(profile.arch, profile.id)
    manifest = D._Manifest('fixture', 'fixture', artifact.filename, len(content), hashlib.sha256(content).hexdigest(), url)
    manager = D.FixedModelDownloader(fixture.models, _manifest=manifest)
    fixture.control._downloads[profile.id] = manager
    return manager


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='tinystrata-library-')
        self.addCleanup(self.temp.cleanup)
        self.f = create_fixture(self.temp.name)
        self.addCleanup(self.f.control.close)

    def file(self, artifact=D.FIXED_GEMMA4_MODEL, content=b'model fixture'):
        path = self.f.models / artifact.filename
        path.write_bytes(content)
        return path

    def inventory(self, model=GEMMA):
        return L.inventory(self.f.profiles.get(model), tuple(self.f.profiles._profiles.values()))

    def test_catalog_metadata_is_cheap_and_missing_models_stay_downloadable(self):
        with patch('hashlib.file_digest', side_effect=AssertionError('No artifact hashing in catalog')):
            rows = {row['id']: row for row in self.f.control.catalog()}
        self.assertEqual(len(rows), 5)
        meta = rows[GEMMA]['management']
        self.assertFalse(meta['can_delete']); self.assertTrue(meta['can_download'])
        self.assertEqual(meta['delete_reason'], 'not_downloaded')
        self.assertEqual(meta['download_bytes'], D.FIXED_GEMMA4_MODEL.size + D.FIXED_GEMMA4_MMPROJ.size + D.FIXED_GEMMA4_ASSISTANT.size)
        self.assertEqual(rows[QWEN]['management']['delete_reason'], 'active')

    def test_delete_bundle_partial_and_exact_tokenizer_preserves_unknown_files_and_settings(self):
        model = self.file()
        self.file(D.FIXED_GEMMA4_MMPROJ, b'projector')
        self.file(D.FIXED_GEMMA4_ASSISTANT, b'assistant')
        partial = model.with_name(model.name + '.part'); partial.write_bytes(b'partial')
        info = model.stat()
        identity = f'{model.resolve()}\n{info.st_size}\n{info.st_mtime_ns}'
        cache = self.f.root / 'models' / hashlib.sha256(identity.encode()).hexdigest()[:16] / 'tokenizer'
        cache.mkdir(parents=True)
        for name in (*model_setup.TOKENIZER_FILES, 'merges.json'):
            (cache / name).write_text('fixture', encoding='utf-8')
        unknown = cache / 'personal-note.txt'; unknown.write_text('keep')
        profile = self.f.profiles.get(GEMMA)
        self.f.profiles.update(GEMMA, {**profile.config, 'tokenizer': str(cache)})
        settings = self.f.root / 'config.json'; before = settings.read_bytes()
        history = self.f.root / 'chat-history.sqlite3'; history.write_bytes(b'keep history')
        key = self.f.root / 'search-settings.json'; key.write_text('keep keys')
        plan = self.inventory()
        result = self.f.control.delete(GEMMA)
        self.assertEqual(result['deleted_bytes'], plan.management['reclaimable_bytes'])
        self.assertFalse(model.exists()); self.assertFalse(partial.exists())
        self.assertTrue(unknown.is_file()); self.assertEqual(unknown.read_text(), 'keep')
        self.assertEqual(settings.read_bytes(), before)
        self.assertEqual(history.read_bytes(), b'keep history'); self.assertEqual(key.read_text(), 'keep keys')
        self.assertFalse(next(row for row in result['catalog'] if row['id'] == GEMMA)['installed'])

    def test_partial_only_is_removable_and_redownload_has_fresh_manager(self):
        partial = self.f.models / (D.FIXED_GEMMA4_MODEL.filename + '.part')
        partial.write_bytes(b'partial')
        manager = install_fixture_download(self.f, GEMMA, 'http://127.0.0.1:1/never', b'tiny')
        self.assertEqual(self.f.control.delete(GEMMA)['deleted_bytes'], 7)
        self.assertTrue(manager.closed); self.assertNotIn(GEMMA, self.f.control._downloads)
        self.assertTrue(self.inventory().management['can_download'])

    def test_shared_projector_and_tokenizer_references_are_retained(self):
        clover = self.file(D.FIXED_CLOVER_MODEL, b'clover')
        shared = self.file(D.FIXED_MMPROJ, b'shared projector')
        qwen = self.f.profiles.get(QWEN)
        self.f.profiles.update(QWEN, {**qwen.config, 'vision_mmproj': str(shared)})
        result = self.f.control.delete(CLOVER)
        self.assertFalse(clover.exists()); self.assertEqual(shared.read_bytes(), b'shared projector')
        self.assertEqual(result['retained_shared_bytes'], len(b'shared projector'))

    def test_custom_models_even_at_fixed_filename_are_not_deleted(self):
        model = self.file()
        profile = self.f.profiles.get(GEMMA)
        self.f.profiles.update(GEMMA, {**profile.config, 'model_source': 'local'})
        self.assertEqual(self.inventory().management['delete_reason'], 'custom')
        self.assertIsNone(self.inventory().management['complete'])
        row = next(row for row in self.f.control.catalog() if row['id'] == GEMMA)
        self.assertTrue(row['installed'] and row['management']['complete'] is not False)
        with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
        self.assertTrue(model.exists())
        external = self.f.root / 'my-model.gguf'; external.write_bytes(b'custom')
        cfg = {**profile.config, 'args': ['--model', str(external)]}
        self.f.profiles.update(GEMMA, cfg)
        with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
        self.assertEqual(external.read_bytes(), b'custom')

    def test_hardlink_removal_safety_does_not_change_readiness_or_size(self):
        model = self.file()
        self.file(D.FIXED_GEMMA4_MMPROJ, b'projector')
        self.file(D.FIXED_GEMMA4_ASSISTANT, b'assistant')
        alias = self.f.root / 'benchmark-alias.gguf'; os.link(model, alias)
        meta = self.inventory().management
        self.assertTrue(meta['complete']); self.assertGreater(meta['disk_bytes'], 0)
        self.assertFalse(meta['can_delete']); self.assertEqual(meta['delete_reason'], 'unsafe_path')
        row = next(row for row in self.f.control.catalog() if row['id'] == GEMMA)
        self.assertTrue(row['installed'] and row['management']['complete'] is not False)
        with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
        self.assertEqual(alias.read_bytes(), b'model fixture'); self.assertTrue(model.exists())

    def test_another_profiles_explicit_vision_model_reference_is_shared(self):
        model = self.file()
        qwen = self.f.profiles.get(QWEN)
        self.f.profiles.update(QWEN, {**qwen.config, 'vision': {'model': str(model)}})
        self.assertEqual(self.inventory().management['delete_reason'], 'shared')
        with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
        self.assertTrue(model.exists())

    def test_active_rejected_even_unloaded_and_all_conflicts_leave_files_untouched(self):
        model = self.file()
        for name in (QWEN, 'qwen36'):
            with self.assertRaises(ModelControlError) as raised: self.f.control.delete(name)
            self.assertEqual(raised.exception.status, 409)
        for field, value in (('_requests', 1), ('_busy', True), ('_closed', True)):
            setattr(self.f.control, field, value)
            with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
            setattr(self.f.control, field, False if isinstance(value, bool) else 0)
            self.assertTrue(model.exists())
        manager = install_fixture_download(self.f, GEMMA, 'http://127.0.0.1:1/never', b'tiny')
        manager._state = 'verifying'
        with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
        manager._state = 'idle'
        for lock in (self.f.svc.fifo, self.f.svc.install_lock):
            lock.acquire()
            try:
                with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
            finally: lock.release()
        self.assertTrue(model.exists())

    def test_staging_failure_rolls_back_before_any_unlink(self):
        a = self.file(); b = self.file(D.FIXED_GEMMA4_ASSISTANT)
        original = Path.rename
        def fail(path, target):
            if path == b: raise PermissionError('fixture locked file')
            return original(path, target)
        with patch.object(Path, 'rename', fail), self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
        self.assertEqual(a.read_bytes(), b'model fixture'); self.assertEqual(b.read_bytes(), b'model fixture')
        self.assertFalse(list(self.f.models.glob('.tinystrata-delete-*')))

    def test_changed_file_between_inventory_and_removal_is_rejected(self):
        model = self.file(); plan = self.inventory()
        model.write_bytes(b'replaced content')
        with self.assertRaises(L.UnsafeModelPath): L.delete_files(plan)
        self.assertEqual(model.read_bytes(), b'replaced content')

    def test_partial_unlink_failure_restores_every_remaining_staged_file(self):
        model = self.file(); projector = self.file(D.FIXED_GEMMA4_MMPROJ, b'projector')
        assistant = self.file(D.FIXED_GEMMA4_ASSISTANT, b'assistant')
        manager = install_fixture_download(self.f, GEMMA, 'http://127.0.0.1:1/never', b'tiny')
        manager._state = 'ready'
        original = Path.unlink
        def fail(path, *args, **kwargs):
            if path.name.startswith('.tinystrata-delete-') and path.name.endswith('-1'):
                raise PermissionError('fixture transient unlink failure')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'unlink', fail), self.assertRaises(ModelControlError) as raised:
            self.f.control.delete(GEMMA)
        self.assertEqual(raised.exception.status, 500)
        self.assertIn(f'deleting {len(b"model fixture")} bytes', str(raised.exception))
        self.assertFalse(model.exists())
        self.assertEqual(projector.read_bytes(), b'projector'); self.assertEqual(assistant.read_bytes(), b'assistant')
        self.assertFalse(list(self.f.models.glob('.tinystrata-delete-*')))
        self.assertTrue(manager.closed); self.assertNotIn(GEMMA, self.f.control._downloads)

    def test_rollback_does_not_overwrite_a_concurrently_recreated_file(self):
        self.file(); projector = self.file(D.FIXED_GEMMA4_MMPROJ, b'projector')
        original = Path.unlink
        def fail(path, *args, **kwargs):
            if path.name.startswith('.tinystrata-delete-') and path.name.endswith('-1'):
                projector.write_bytes(b'newer user file')
                raise PermissionError('fixture occupied restoration destination')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'unlink', fail), self.assertRaises(ModelControlError) as raised:
            self.f.control.delete(GEMMA)
        self.assertIn('1 undeleted files remain', str(raised.exception))
        self.assertEqual(projector.read_bytes(), b'newer user file')
        staged = list(self.f.models.glob('.tinystrata-delete-*'))
        self.assertEqual(len(staged), 1); self.assertEqual(staged[0].read_bytes(), b'projector')

    def test_ready_downloader_with_live_worker_blocks_deletion(self):
        model = self.file()
        manager = install_fixture_download(self.f, GEMMA, 'http://127.0.0.1:1/never', b'tiny')
        manager._state = 'ready'
        manager._thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None)
        try:
            self.assertEqual(self.f.control._library_conflict(), 'downloading')
            with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
            self.assertTrue(model.exists())
        finally: manager._thread = None

    def test_symlink_file_and_parent_refuse_deletion(self):
        target = self.f.root / 'unmanaged.gguf'; target.write_bytes(b'keep')
        path = self.f.models / D.FIXED_GEMMA4_MODEL.filename
        try: path.symlink_to(target)
        except OSError: self.skipTest('Host does not permit symlink creation')
        self.assertEqual(self.inventory().management['delete_reason'], 'unsafe_path')
        with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
        self.assertEqual(target.read_bytes(), b'keep')

    @unittest.skipUnless(os.name == 'nt', 'Windows junction containment')
    def test_junction_managed_directory_is_rejected(self):
        outside = self.f.root / 'outside'; outside.mkdir()
        target = outside / D.FIXED_GEMMA4_MODEL.filename; target.write_bytes(b'keep')
        self.f.models.rmdir()
        command = ['cmd', '/c', 'mklink', '/J', str(self.f.models), str(outside)]
        created = subprocess.run(command, capture_output=True)
        if created.returncode: self.skipTest('Junction fixture unavailable')
        try:
            self.assertEqual(self.inventory().management['delete_reason'], 'unsafe_path')
            with self.assertRaises(ModelControlError): self.f.control.delete(GEMMA)
            self.assertEqual(target.read_bytes(), b'keep')
        finally: self.f.models.rmdir()  # Remove the verified junction itself, never recurse.

    def test_admission_cannot_overlap_delete_transaction(self):
        self.file()
        entered, release, admitted = threading.Event(), threading.Event(), threading.Event()
        original = L.delete_files
        def held(plan):
            entered.set(); self.assertTrue(release.wait(3)); return original(plan)
        with patch.object(L, 'delete_files', held):
            deletion = threading.Thread(target=lambda: self.f.control.delete(GEMMA)); deletion.start()
            self.assertTrue(entered.wait(3))
            def request():
                with self.f.control.admit(): admitted.set()
            admission = threading.Thread(target=request); admission.start()
            self.assertFalse(admitted.wait(.05)); release.set()
            deletion.join(3); admission.join(3)
            self.assertFalse(deletion.is_alive()); self.assertTrue(admitted.is_set())


class LibraryHTTPTests(unittest.TestCase):
    # Only these route tests start a small CPU-only loopback HTTP listener.
    setUp = LibraryTests.setUp
    file = LibraryTests.file
    def start_http(self):
        self.httpd = Server(('127.0.0.1', 0), make_handler(self.f.svc))
        self.worker = threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
        self.worker.start()
        self.addCleanup(lambda: (self.httpd.shutdown(), self.worker.join(3), self.httpd.server_close()))

    def http(self, method='POST', body=None, headers=None, path='/v1/model/delete'):
        connection = http.client.HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=3)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None,
                               {'Content-Type': 'application/json', **(headers or {})})
            response = connection.getresponse(); return response.status, json.loads(response.read())
        finally: connection.close()

    def test_route_auth_origin_media_type_and_exact_model_only_body(self):
        model = self.file(); self.f.svc.api_key = 'fixture-key'; self.start_http()
        self.assertEqual(self.http(body={'model': GEMMA})[0], 401)
        auth = {'Authorization': 'Bearer fixture-key'}
        self.assertEqual(self.http(body={'model': GEMMA}, headers={**auth, 'Origin': 'http://foreign.invalid'})[0], 403)
        self.assertEqual(self.http(body={'model': GEMMA}, headers={**auth, 'Content-Type': 'text/plain'})[0], 415)
        for body in ({}, {'model': '../outside'}, {'model': GEMMA, 'path': str(model)}, {'model': False}):
            self.assertEqual(self.http(body=body, headers=auth)[0], 400)
            self.assertTrue(model.is_file())
        status, data = self.http(body={'model': GEMMA}, headers=auth)
        self.assertEqual(status, 200, data); self.assertEqual(data['model_id'], GEMMA)
        self.assertFalse(model.exists()); self.assertEqual(len(data['catalog']), 5)

    def test_control_route_does_not_admit_itself_and_busy_rejects_over_http(self):
        model = self.file(); self.start_http()
        self.f.control._requests = 1
        self.assertEqual(self.http(body={'model': GEMMA})[0], 409)
        self.f.control._requests = 0
        self.assertEqual(self.http(body={'model': QWEN})[0], 409)
        self.assertEqual(self.http(method='GET', path='/v1/models')[0], 200)
        self.assertEqual(self.http(body={'model': GEMMA})[0], 200)
        self.assertFalse(model.exists()); self.assertEqual(self.f.control._requests, 0)


if __name__ == '__main__': unittest.main()
