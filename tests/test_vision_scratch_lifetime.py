"""Resident encoder file lifetimes, including real Windows sharing violations.

The pipe-driven child substitutes image computation; the protocol, filesystem,
Windows handles, retries and cache publication use the production Vision code.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

from serve import server as api


ENCODER = r'''
import ctypes, hashlib, pathlib, sys
mode = sys.argv[1]
handles = []
if sys.platform == 'win32':
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
count = 0
for command in sys.stdin:
    if command.strip() == 'QUIT': break
    _, source, output = command.split()
    source, output = pathlib.Path(source), pathlib.Path(output)
    data = source.read_bytes()
    with open('attempts.txt', 'a', encoding='utf-8') as attempts:
        attempts.write(source.name + '\n')
    if 'held' in mode and sys.platform == 'win32':
        # Permit readers, but deliberately withhold FILE_SHARE_DELETE/WRITE.
        handle = kernel.CreateFileW(str(source.resolve()), 0x80000000, 1,
                                   None, 3, 0x80, None)
        if handle == ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
        handles.append(handle)
    count += 1
    if 'error' in mode and count == 1:
        output.write_bytes(b'incomplete')
        print('ERR unsupported image', flush=True)
    elif mode == 'bad-count':
        output.write_bytes(b'incomplete')
        print('OK not-a-number', flush=True)
    else:
        output.write_bytes(b'SVE-fixture:' + hashlib.sha256(data).digest())
        print('OK 257', flush=True)
if sys.platform == 'win32':
    for handle in handles: kernel.CloseHandle(handle)
'''


class VisionScratchLifetime(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        # stb_image-supported header avoids optional conversion dependencies.
        self.image = b'\x89PNG\r\n\x1a\n' + b'representative-image-data' * 16384

    def worker(self, mode='normal', protocol='enc'):
        proc = subprocess.Popen([sys.executable, '-u', '-c', ENCODER, mode],
                                cwd=self.directory, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', bufsize=1,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        def close():
            if proc.poll() is None:
                try:
                    proc.stdin.write('QUIT\n'); proc.stdin.flush()
                    proc.wait(timeout=3)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    proc.kill(); proc.wait(timeout=3)
            for stream in (proc.stdin, proc.stdout, proc.stderr): stream.close()
        self.addCleanup(close)
        worker = object.__new__(api.Vision)
        worker.dir, worker.proc = self.directory, proc
        worker.input_protocol, worker.stopped = protocol, False
        worker.lock, worker.cache = threading.Lock(), {}
        worker._count_pending = None
        return worker

    def attempts(self):
        return (self.directory / 'attempts.txt').read_text(encoding='utf-8').splitlines()

    def canonical(self, image=None):
        key = hashlib.sha256(self.image if image is None else image).hexdigest()[:32]
        return self.directory / (key + '.sve')

    def check_result(self, worker, image=None):
        image = self.image if image is None else image
        output, rows = worker.encode(image)
        self.assertEqual(output, self.canonical(image))
        self.assertEqual(rows, 257)
        self.assertEqual(output.read_bytes(), b'SVE-fixture:' + hashlib.sha256(image).digest())
        return output, rows

    def test_complete_file_is_cached_for_both_encoder_protocols(self):
        for protocol in ('enc', 'enc-tabs'):
            with self.subTest(protocol=protocol):
                worker = self.worker(protocol=protocol)
                before = len(self.attempts()) if (self.directory / 'attempts.txt').exists() else 0
                expected = self.check_result(worker)
                self.assertEqual(worker.encode(self.image), expected)
                self.assertEqual(len(self.attempts()), before + 1)
                self.assertEqual(list(self.directory.glob('*.part')), [])
                self.assertEqual(list(self.directory.glob('*.img')), [])

    @unittest.skipUnless(os.name == 'nt', 'requires Windows file sharing semantics')
    def test_encoder_retained_input_does_not_fail_success_or_retry(self):
        worker = self.worker('held')
        expected = self.check_result(worker)
        scratch = next(self.directory.glob('*.img'))
        # This is a real sharing violation, not a mocked unlink exception.
        with self.assertRaises(PermissionError): scratch.unlink()
        self.assertEqual(worker.encode(self.image), expected)
        self.assertEqual(len(self.attempts()), 1)
        worker.cache.clear()
        self.assertEqual(self.check_result(worker), expected)
        self.assertEqual(len(set(self.attempts())), 2)
        self.assertEqual(list(self.directory.glob('*.part')), [])

    @unittest.skipUnless(os.name == 'nt', 'requires Windows file sharing semantics')
    def test_failed_encode_keeps_its_error_and_retries_with_a_new_locked_input(self):
        worker = self.worker('held-error', 'enc-tabs')
        with self.assertRaisesRegex(ValueError, 'unsupported image'):
            worker.encode(self.image)
        self.assertEqual(worker.cache, {})
        self.assertFalse(self.canonical().exists())
        self.assertEqual(list(self.directory.glob('*.part')), [])
        self.check_result(worker)
        self.assertEqual(len(set(self.attempts())), 2)

    def test_malformed_success_does_not_publish_partial_output(self):
        worker = self.worker('bad-count')
        with self.assertRaises(ValueError): worker.encode(self.image)
        self.assertEqual(worker.cache, {})
        self.assertFalse(self.canonical().exists())
        self.assertEqual(list(self.directory.glob('*.part')), [])

    def test_failed_publication_retains_the_prior_file_and_can_retry(self):
        worker = self.worker()
        self.canonical().write_bytes(b'prior-complete-result')
        with mock.patch.object(api.os, 'replace', side_effect=PermissionError('output in use')):
            with self.assertRaisesRegex(PermissionError, 'output in use'):
                worker.encode(self.image)
        self.assertEqual(worker.cache, {})
        self.assertEqual(self.canonical().read_bytes(), b'prior-complete-result')
        self.assertEqual(list(self.directory.glob('*.part')), [])
        self.check_result(worker)
        self.assertEqual(len(set(self.attempts())), 2)

    def test_pipe_failure_is_not_replaced_by_cleanup_failure(self):
        worker = self.worker()
        worker.proc.stdin.close()
        original = Path.unlink
        def denied(path, *args, **kwargs):
            if path.suffix in ('.img', '.part'): raise PermissionError('cleanup denied')
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, 'unlink', denied):
            with self.assertRaisesRegex(ValueError, 'closed file'):
                worker.encode(self.image)
        self.assertEqual(worker.cache, {})

    def test_concurrent_requests_share_the_completed_cache_entry(self):
        worker = self.worker()
        results, failures = [], []
        def run():
            try: results.append(worker.encode(self.image))
            except Exception as error: failures.append(error)
        threads = [threading.Thread(target=run) for _ in range(4)]
        for thread in threads: thread.start()
        for thread in threads: thread.join(timeout=5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(failures, [])
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result == results[0] for result in results))
        self.assertEqual(len(self.attempts()), 1)

    def test_cancellation_before_the_next_image_preserves_completed_results(self):
        worker = self.worker()
        cancel = threading.Event()
        encode = worker.encode
        def first(source, keep=()):
            result = encode(source, keep=keep)
            cancel.set()
            return result
        with mock.patch.object(worker, 'encode', first):
            with self.assertRaises(api.ClientGone):
                worker.encode_all([self.image, self.image + b'next'], cancel=cancel)
        self.assertEqual(len(self.attempts()), 1)
        self.assertEqual(len(worker.cache), 1)
        self.check_result(worker)


if __name__ == '__main__':
    unittest.main()
