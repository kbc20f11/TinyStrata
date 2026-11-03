"""Isolated UI-search regressions: no production service, GPU or external HTTP."""
import concurrent.futures
import contextlib
import json
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import web_search as search

ROWS = [{'title': 'Example', 'url': 'https://example.org/page', 'snippet': 'A result.'}]


def service(runner, **kwargs):
    return search.SearchService(runner=runner, intervals={'ddgs': 0, 'brave': 0}, **kwargs)


class SearchTests(unittest.TestCase):
    def test_settings_and_secret_public_shape(self):
        value = {'provider': 'brave', 'max_concurrency': 2, 'brave_api_key': 'private-api-token'}
        self.assertEqual(search.public_settings(value), {'provider': 'brave', 'max_concurrency': 2, 'has_brave_key': True})
        self.assertNotIn('private-api-token', json.dumps(search.public_settings(value)))
        for invalid in [False, 0, 5, 1.5, '2']:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                search.validate_settings({'max_concurrency': invalid})
        for value in [{'provider': 'custom'}, {'url': 'http://localhost/'}, {'brave_api_key': 'bad\r\nHeader:x'}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                search.validate_settings(value)

    def test_query_bounds(self):
        self.assertEqual(search.normalize_query('  a\n b  '), 'a b')
        for value in ['', ' ', False, None, 'x' * 501, 'word ' * 76, 'a\0b']:
            with self.subTest(value=repr(value)), self.assertRaises(search.SearchError) as caught:
                search.normalize_query(value)
            self.assertEqual(caught.exception.code, 'invalid_query')

    def test_safe_links_and_no_result_fetch(self):
        invalid = ['javascript:alert(1)', 'file:///etc/passwd', 'data:text/html,x', 'https://user:secret@example.org/',
                   'http://localhost/', 'http://localhost.localdomain/', 'http://127.0.0.1/', 'http://127.1/',
                   'http://2130706433/', 'http://0x7f000001/', 'http://10.1.2.3/', 'http://169.254.169.254/',
                   'http://100.64.0.1/', 'http://224.0.0.1/', 'http://[::1]/', 'http://[fc00::1]/',
                   'https://example.org\\@127.0.0.1/', 'https://exa\nmple.org/', 'https://example.org:99999/']
        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(search.safe_url(value))
        self.assertEqual(search.safe_url('HTTPS://Example.ORG:443/path#frag'), 'https://example.org/path')
        self.assertEqual(search.safe_url('https://cafe123.de/'), 'https://cafe123.de/')
        self.assertEqual(search.safe_url('https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2F'), 'https://example.org/')
        with patch.object(socket, 'getaddrinfo', side_effect=AssertionError('result URLs must not be fetched')):
            out = search.normalize_results(ROWS * 2 + [{'url': 'http://127.0.0.1/'}])
        self.assertEqual(len(out), 1)

    def test_result_text_bounds_and_script_removal(self):
        rows = [{'url': f'https://example.org/{n}', 'title': '<b>' + 't' * 400 + '</b>',
                 'snippets': ['visible<script>secret()</script>', 's' * 4000]} for n in range(20)]
        out = search.normalize_results(rows)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(len(r['title']) == 300 and len(r['snippet']) <= 2000 for r in out))
        self.assertNotIn('secret', out[0]['snippet'])

    def test_retry_windows_and_http_date(self):
        self.assertEqual(search.retry_delay({'Retry-After': '3', 'X-RateLimit-Limit': '1,1000', 'X-RateLimit-Remaining': '0, 2', 'X-RateLimit-Reset': '1, 999'}), 3)
        self.assertEqual(search.retry_delay({'X-RateLimit-Limit': '1,1000', 'X-RateLimit-Remaining': '1,0', 'X-RateLimit-Reset': '1,3600'}), 3600)
        self.assertEqual(search.retry_delay({'Retry-After': 'Thu, 01 Jan 1970 00:01:00 GMT'}, now=30), 30)
        self.assertIsNone(search.retry_delay({'Retry-After': 'nan', 'X-RateLimit-Reset': 'not-a-number'}))

    def test_unlimited_brave_window_is_not_exhausted_monthly_quota(self):
        headers = {'X-RateLimit-Limit': '50, 0', 'X-RateLimit-Policy': '50;w=1, 0;w=2592000',
                   'X-RateLimit-Remaining': '49, 0', 'X-RateLimit-Reset': '1, 2380757'}
        self.assertIsNone(search.retry_delay(headers))
        self.assertEqual(search.retry_delay({**headers, 'X-RateLimit-Remaining': '0, 0'}), 1)
        self.assertEqual(search.retry_delay({**headers, 'Retry-After': '3'}), 3)
        self.assertEqual(search.retry_delay({**headers, 'X-RateLimit-Limit': '50, 1000'}), 2380757)

    def test_rate_limit_policy_fallback_and_malformed_tuple_alignment(self):
        headers = {'X-RateLimit-Policy': '50;w=1, 0;w=2592000',
                   'X-RateLimit-Remaining': '0, 0', 'X-RateLimit-Reset': '1, 2380757'}
        self.assertEqual(search.retry_delay(headers), 1)
        # Unidentified or malformed limits must not manufacture a month-long
        # local ban. Provider 429 and Retry-After remain independently handled.
        self.assertIsNone(search.retry_delay({k: v for k, v in headers.items() if k != 'X-RateLimit-Policy'}))
        for invalid in ('bad', 'nan', '-1', ''):
            with self.subTest(limit=invalid):
                self.assertEqual(search.retry_delay({**headers, 'X-RateLimit-Limit': '50,' + invalid}), 1)
        self.assertIsNone(search.retry_delay({**headers, 'X-RateLimit-Limit': '50,0', 'X-RateLimit-Reset': 'bad,2380757'}))
        self.assertEqual(search.retry_delay({**headers, 'X-RateLimit-Limit': '50,1000', 'X-RateLimit-Reset': 'bad,2380757'}), 2380757)

    def test_public_resolution_rejects_any_private_answer(self):
        public = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))
        private = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('127.0.0.1', 443))
        with patch.object(socket, 'getaddrinfo', return_value=[public, private]), self.assertRaises(search.SearchError):
            search._public_addresses(search.BRAVE_HOST, 443)

    def test_fixed_provider_connection_pins_ip_and_tls_host(self):
        conn = search._PinnedHTTPS(search.BRAVE_HOST, timeout=1, context=ssl.create_default_context())
        fake_socket = Mock()
        wrapped = object()
        context = Mock()
        context.wrap_socket.return_value = wrapped
        conn._context = context
        address = (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 443))
        with patch.object(search, '_public_addresses', return_value=[address]) as resolver, patch.object(socket, 'socket', return_value=fake_socket):
            conn.connect()
        resolver.assert_called_once_with(search.BRAVE_HOST, 443)
        fake_socket.connect.assert_called_once_with(('8.8.8.8', 443))
        context.wrap_socket.assert_called_once_with(fake_socket, server_hostname=search.BRAVE_HOST)
        self.assertIs(conn.sock, wrapped)

    def test_brave_official_shape_auth_and_no_redirect_follow(self):
        response = Mock(status=200)
        response.getheaders.return_value = [('X-RateLimit-Limit', '1,1000'), ('X-RateLimit-Remaining', '0,100'), ('X-RateLimit-Reset', '1,999')]
        response.read.return_value = json.dumps({'grounding': {'generic': [{'url': 'https://example.org/', 'title': 'Example', 'snippets': ['one', 'two']}]}}).encode()
        conn = Mock()
        conn.getresponse.return_value = response
        with patch.object(search, '_PinnedHTTPS', return_value=conn):
            result = search._brave_search('test query', 'private-test-key')
        call = conn.request.call_args
        self.assertEqual(call.args, ('POST', '/res/v1/llm/context'))
        self.assertEqual(call.kwargs['headers']['X-Subscription-Token'], 'private-test-key')
        self.assertEqual(json.loads(call.kwargs['body'])['maximum_number_of_urls'], 5)
        self.assertEqual(result['results'][0]['snippet'], 'one two')
        self.assertEqual(result['cooldown'], 1)
        response.status = 302
        response.getheaders.return_value = [('Location', 'http://127.0.0.1/private')]
        with patch.object(search, '_PinnedHTTPS', return_value=conn), self.assertRaises(search.SearchError):
            search._brave_search('test query', 'private-test-key')
        self.assertEqual(conn.getresponse.call_count, 2)

    def test_brave_429_monthly_quota_and_secret_safe_errors(self):
        response = Mock(status=429)
        response.getheaders.return_value = [('X-RateLimit-Limit', '1,1000'), ('X-RateLimit-Remaining', '1,0'), ('X-RateLimit-Reset', '1,999')]
        response.read.return_value = b'{"error":{"code":"QUOTA_LIMITED","detail":"private-test-key"}}'
        conn = Mock()
        conn.getresponse.return_value = response
        with patch.object(search, '_PinnedHTTPS', return_value=conn), self.assertRaises(search.SearchError) as caught:
            search._brave_search('test query', 'private-test-key')
        self.assertEqual(caught.exception.code, 'quota_exceeded')
        self.assertEqual(caught.exception.retry_after, 999)
        self.assertNotIn('private-test-key', repr(caught.exception.to_dict()))

    def test_successful_unlimited_plan_allows_subsequent_distinct_search(self):
        response = Mock(status=200)
        response.getheaders.return_value = [('X-RateLimit-Limit', '50, 0'),
                                            ('X-RateLimit-Policy', '50;w=1, 0;w=2592000'),
                                            ('X-RateLimit-Remaining', '49, 0'),
                                            ('X-RateLimit-Reset', '1, 2380757')]
        response.read.return_value = json.dumps({'grounding': {'generic': ROWS}}).encode()
        conn = Mock()
        conn.getresponse.return_value = response
        settings = {'provider': 'brave', 'brave_api_key': 'private-test-key'}
        s = service(lambda query, settings, *_: search._brave_search(query, settings['brave_api_key']))
        try:
            with patch.object(search, '_PinnedHTTPS', return_value=conn):
                first = s.search('first query', settings=settings)
                second = s.search('different query', settings=settings)
            self.assertFalse(first['cached'])
            self.assertFalse(second['cached'])
            self.assertEqual(first['results'], second['results'])
            self.assertEqual(conn.request.call_count, 2)
        finally:
            s.close()

    def test_ddgs_interface_limits_and_missing_dependency(self):
        klass = Mock()
        klass.return_value.__enter__ = Mock(return_value=Mock(text=Mock(return_value=[{'href': 'https://example.org/', 'title': 'x', 'body': 'y'}])))
        klass.return_value.__exit__ = Mock(return_value=False)
        with patch.dict(sys.modules, {'ddgs': types.SimpleNamespace(DDGS=klass)}):
            result = search._ddgs_search('test')
        self.assertEqual(klass.threads, 1)
        klass.assert_called_once_with(timeout=12, verify=True)
        klass.return_value.__enter__.return_value.text.assert_called_once_with('test', safesearch='moderate', max_results=5, backend='auto')
        self.assertEqual(len(result['results']), 1)
        with patch.dict(sys.modules, {'ddgs': None}), self.assertRaises(search.SearchError) as caught:
            search._ddgs_search('test')
        self.assertEqual(caught.exception.code, 'dependency_missing')

    def test_cache_is_bounded_and_results_are_copied(self):
        calls = []
        s = service(lambda *args: calls.append(args[0]) or {'results': ROWS})
        try:
            first = s.search('same')
            first['results'][0]['title'] = 'modified'
            second = s.search('same')
            self.assertTrue(second['cached'])
            self.assertEqual(second['results'][0]['title'], 'Example')
            self.assertEqual(calls, ['same'])
            for n in range(130):
                s.search('query ' + str(n))
            self.assertLessEqual(len(s._cache), 128)
        finally:
            s.close()

    def test_expired_cache_and_missing_key(self):
        runner = Mock(return_value={'results': ROWS})
        s = service(runner, cache_ttl=.001)
        try:
            s.search('same'); time.sleep(.01); s.search('same')
            self.assertEqual(runner.call_count, 2)
            with self.assertRaises(search.SearchError) as caught:
                s.search('private', settings={'provider': 'brave'})
            self.assertEqual(caught.exception.code, 'missing_key')
            self.assertEqual(runner.call_count, 2)
        finally:
            s.close()

    def test_identical_concurrent_queries_coalesce(self):
        release = threading.Event()
        calls = []
        def runner(query, settings, cancel, timeout):
            calls.append(query)
            release.wait(1)
            return {'results': ROWS}
        s = service(runner)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(s.search, 'shared') for _ in range(4)]
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    with s._condition:
                        if s._flights and next(iter(s._flights.values())).waiters == 4:
                            break
                    time.sleep(.002)
                release.set()
                self.assertTrue(all(f.result(timeout=2)['results'] for f in futures))
            self.assertEqual(calls, ['shared'])
        finally:
            release.set(); s.close()

    def test_global_concurrency_queue_limit_and_pacing(self):
        started = threading.Event(); release = threading.Event()
        def runner(*args):
            started.set(); release.wait(1)
            return {'results': ROWS}
        s = service(runner, max_queue=0)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(s.search, 'first')
                self.assertTrue(started.wait(1))
                with self.assertRaises(search.SearchError) as caught:
                    s.search('second')
                self.assertEqual(caught.exception.code, 'busy')
                release.set(); first.result(timeout=2)
        finally:
            release.set(); s.close()
        starts = []
        s = search.SearchService(runner=lambda *args: starts.append(time.monotonic()) or {'results': ROWS}, intervals={'ddgs': .06, 'brave': .06})
        try:
            s.search('one'); s.search('two')
            self.assertGreaterEqual(starts[1] - starts[0], .055)
        finally:
            s.close()

    def test_configured_concurrency_two_is_shared_across_unique_queries(self):
        release = threading.Event(); lock = threading.Lock(); active = peak = 0
        def runner(*args):
            nonlocal active, peak
            with lock:
                active += 1; peak = max(peak, active)
            release.wait(1)
            with lock:
                active -= 1
            return {'results': ROWS}
        s = service(runner)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(s.search, str(n), settings={'max_concurrency': 2}) for n in range(3)]
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    with lock:
                        if active == 2: break
                    time.sleep(.002)
                with lock:
                    self.assertEqual(active, 2)
                release.set()
                for future in futures: future.result(timeout=2)
            self.assertEqual(peak, 2)
        finally:
            release.set(); s.close()

    def test_rate_limit_cooldown_prevents_repeated_provider_calls(self):
        runner = Mock(side_effect=search.SearchError('rate_limited', retry_after=60))
        s = service(runner)
        try:
            for query in ('first', 'second'):
                with self.assertRaises(search.SearchError) as caught:
                    s.search(query)
                self.assertEqual(caught.exception.code, 'rate_limited')
            self.assertEqual(runner.call_count, 1)
        finally:
            s.close()

    def test_brave_short_429_retries_once(self):
        runner = Mock(side_effect=[search.SearchError('rate_limited', retry_after=1), {'results': ROWS}])
        s = service(runner, queue_timeout=2, execution_timeout=3)
        try:
            result = s.search('one', settings={'provider': 'brave', 'brave_api_key': 'test-key'})
            self.assertEqual(len(result['results']), 1)
            self.assertEqual(runner.call_count, 2)
        finally:
            s.close()

    def test_cancellation_only_kills_work_without_other_waiters(self):
        started = threading.Event(); release = threading.Event(); owner_cancel = threading.Event(); saw_cancel = []
        def runner(query, settings, cancel, timeout):
            started.set(); release.wait(1); saw_cancel.append(cancel.is_set())
            return {'results': ROWS}
        s = service(runner)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                owner = pool.submit(s.search, 'same', cancel=owner_cancel)
                self.assertTrue(started.wait(1))
                follower = pool.submit(s.search, 'same')
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline:
                    with s._condition:
                        if next(iter(s._flights.values())).waiters == 2: break
                    time.sleep(.002)
                owner_cancel.set()
                with self.assertRaises(search.SearchError): owner.result(timeout=1)
                release.set()
                self.assertTrue(follower.result(timeout=2)['results'])
            self.assertEqual(saw_cancel, [False])
        finally:
            release.set(); s.close()

    def test_isolated_worker_timeout_cancel_and_no_window_flag(self):
        # Owned, temporary Python fixture; it does no HTTP and loads no model.
        with tempfile.TemporaryDirectory(prefix='tinystrata-search-test-') as folder:
            fixture = Path(folder) / 'worker.py'
            fixture.write_text('import sys,time\nsys.stdin.buffer.read()\ntime.sleep(10)\n', encoding='utf-8')
            processes = []
            real_popen = subprocess.Popen
            def launch(*args, **kwargs):
                self.assertEqual(kwargs['creationflags'], getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.assertNotIn('private-key', ' '.join(args[0]))
                child = real_popen(*args, **kwargs); processes.append(child)
                return child
            cancel = threading.Event()
            timer = threading.Timer(.1, cancel.set)
            timer.start()
            try:
                with patch.object(search, '__file__', str(fixture)), patch.object(subprocess, 'Popen', side_effect=launch):
                    with self.assertRaises(search.SearchError) as caught:
                        search._isolated_search('test', {'provider': 'ddgs', 'max_concurrency': 1, 'brave_api_key': 'private-key'}, cancel, 2)
                self.assertEqual(caught.exception.code, 'cancelled')
                self.assertIsNotNone(processes[0].poll())
                with patch.object(search, '__file__', str(fixture)), patch.object(subprocess, 'Popen', side_effect=launch):
                    with self.assertRaises(search.SearchError) as caught:
                        search._isolated_search('test', search.validate_settings(), threading.Event(), .1)
                self.assertEqual(caught.exception.code, 'timeout')
                self.assertIsNotNone(processes[1].poll())
            finally:
                timer.cancel()
                for child in processes:
                    if child.poll() is None: child.kill(); child.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
