"""Public-page transport/extraction limits, no real provider or model involved."""
from concurrent.futures import ThreadPoolExecutor
import builtins
import io
import json
import runpy
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import web_page as page


class Response:
    def __init__(self, body=b'<p>Public reference</p>', status=200, headers=None):
        self.status = status
        self.headers = {'Content-Type': 'text/html; charset=utf-8', **(headers or {})}
        self.body = io.BytesIO(body)
    def getheader(self, key, default=None): return self.headers.get(key, default)
    def read(self, size): return self.body.read(size)


class PageTests(unittest.TestCase):
    def assert_code(self, code, operation):
        with self.assertRaises(page.PageError) as caught: operation()
        self.assertEqual(caught.exception.code, code)

    def fetch(self, responses, url='https://example.com/reference'):
        requests, connections = [], []
        class Connection:
            def __init__(inner, host, **kwargs):
                inner.host = host
                inner.closed = False
                connections.append(inner)
            def request(inner, method, target, headers):
                requests.append((inner.host, method, target, dict(headers)))
            def getresponse(inner): return responses.pop(0)
            def close(inner): inner.closed = True
        with mock.patch.object(page, '_PublicHTTPS', Connection), mock.patch.object(page, '_PublicHTTP', Connection):
            result = page._fetch(url, 3)
        self.assertTrue(all(conn.closed for conn in connections))
        return result, requests

    def test_invoice_page_preserves_facts_units_table_and_title(self):
        raw = '''<!doctype html><html><head><title>M42 delivery update</title>
        <style>.x {display:none}</style><script>secret()</script></head>
        <body><nav>Login Tracking</nav><main><h1>Approved M42</h1>
        <p>担当:高橋　期限:2026-12-22</p><table><tr><th>数量</th><th>金額</th></tr>
        <tr><td>26台</td><td>136万円</td></tr></table><p>北町倉庫、一括納品、午前指定</p>
        <p hidden>Deadline changed tomorrow</p><p aria-hidden="true">Unapproved</p>
        <div style="display: none">Amount999</div><footer>Cookies</footer></main></body></html>'''.encode()
        value = page.extract_text(raw, 'text/html; charset=utf-8')
        self.assertEqual(value['title'], 'M42 delivery update')
        for fact in ('M42', '高橋', '2026-12-22', '26台', '136万円', '北町倉庫', '一括納品', '午前指定'):
            self.assertIn(fact, value['text'])
        for absent in ('Tracking', 'secret()', 'Unapproved', 'Amount999', 'Cookies', 'changed tomorrow'):
            self.assertNotIn(absent, value['text'])
        self.assertIn('| 26台 | 136万円', value['text'])
        self.assertFalse(value['truncated'])

    def test_plain_text_long_content_flags_truncation(self):
        value = page.extract_text(('資料\n' + '甲' * 30000).encode(), 'text/plain')
        self.assertTrue(value['truncated'])
        self.assertEqual(len(value['text']), 20000)
        self.assertEqual(value['extracted_characters'], 20000)
        self.assertTrue(value['text'].startswith('資料\n甲'))

    def test_japanese_legacy_meta_charset_http_precedence_bom_and_invalid(self):
        body = '<html><head><meta charset="shift_jis"><title>市内の乗換案内</title></head><body><p>武蔵新城駅から南武線で武蔵小杉駅へ、横須賀線に乗り換えて品川駅</p><p>運賃320円、所要26分</p></body></html>'
        expected = '武蔵新城駅から南武線で武蔵小杉駅へ、横須賀線に乗り換えて品川駅'
        value = page.extract_text(body.encode('shift_jis'), 'text/html')
        self.assertIn(expected, value['text'])
        self.assertIn('320円', value['text'])
        self.assertEqual(value['title'], '市内の乗換案内')
        # The real response header overrides a stale charset in the HTML.
        self.assertIn(expected, page.extract_text(body.encode(), 'text/html;charset=utf-8')['text'])
        self.assertEqual(page.extract_text(b'\xef\xbb\xbf' + '駅の案内'.encode(), 'text/plain')['text'], '駅の案内')
        self.assertIn(expected, page.extract_text(body.replace('<meta charset="shift_jis">', '<meta http-equiv="Content-Type" content="text/html; charset=shift_jis">').encode('shift_jis'), 'text/html')['text'])
        self.assert_code('fetch_error', lambda: page.extract_text(b'\xff\xff', 'text/plain'))
        self.assert_code('fetch_error', lambda: page.extract_text(b'text', 'text/plain;charset=not-an-encoding'))

    def test_malformed_html_hidden_tags_and_plain_charset(self):
        value = page.extract_text(b'<title>Title</title><p aria-hidden>Shown</p><div hidden><b>Secret</div><p>Next', 'text/html')
        self.assertIn('Shown', value['text'])
        self.assertIn('Next', value['text'])
        self.assertNotIn('Secret', value['text'])
        self.assertEqual(page.extract_text('Résumé'.encode('latin-1'), 'text/plain;charset=iso-8859-1')['text'], 'Résumé')
        self.assert_code('fetch_error', lambda: page.extract_text(b'<script>document.write("only js")</script>', 'text/html'))

    def test_omitted_head_close_and_hidden_css_whitespace_comments(self):
        for html in (b'<html><head><title>Example</title><body><p>Visible body</p></body></html>',
                     b'<html><head><title>Example</title><p>Visible body</p></html>'):
            value = page.extract_text(html, 'text/html')
            self.assertEqual(value['text'], 'Visible body')
            self.assertEqual(value['title'], 'Example')
        value = page.extract_text(b'<p>Visible</p><p style="display:\nnone">Hidden1</p><p style="display:/**/none">Hidden2</p><p style="visibility:\t hidden">Hidden3</p>', 'text/html')
        self.assertEqual(value['text'], 'Visible')

    def test_non_web_types_are_rejected(self):
        for mime in ('application/pdf', 'image/png', 'application/octet-stream', 'text/javascript', ''):
            self.assert_code('unsupported_type', lambda: page.extract_text(b'not page text', mime))

    def test_url_protocol_credentials_private_aliases_and_ports(self):
        for url in ('file:///etc/passwd', 'http://localhost/', 'http://127.0.0.1/', 'http://10.0.0.1/',
                    'http://169.254.169.254/latest/meta-data/', 'http://[::1]/', 'http://[::ffff:127.0.0.1]/',
                    'http://2130706433/', 'http://127.1/', 'http://0x7f000001/', 'https://user:pass@example.com/',
                    'https://example.com:444/', 'http://example.com:443/', 'https://example.com/\nsecret',
                    'http://127.0.0.1\\@example.com/', 'https://example.local/', 'https://example.test/'):
            self.assert_code('invalid_url', lambda: page.normalize_url(url))
        self.assertEqual(page.normalize_url('https://example.com:443/a#section'), 'https://example.com/a')
        self.assertEqual(page.normalize_url('http://example.com:80/'), 'http://example.com/')

    def test_mixed_dns_is_rejected_before_any_socket_connect(self):
        rows = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 80)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, '', ('192.168.1.1', 80))]
        with mock.patch.object(page.socket, 'getaddrinfo', return_value=rows), mock.patch.object(page.socket, 'socket') as create:
            self.assert_code('blocked_address', lambda: page._PublicHTTP('example.com').connect())
            create.assert_not_called()

    def test_connect_uses_the_exact_validated_address_once(self):
        rows = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('8.8.8.8', 80))]
        with mock.patch.object(page.socket, 'getaddrinfo', return_value=rows) as resolve, mock.patch.object(page.socket, 'socket') as create:
            conn = page._PublicHTTP('example.com', timeout=2)
            conn.connect()
            resolve.assert_called_once()
            create.return_value.connect.assert_called_once_with(('8.8.8.8', 80))
            create.return_value.settimeout.assert_called_with(2)

    def test_https_preserves_hostname_verification(self):
        with mock.patch.object(page._PublicHTTP, 'connect', lambda conn: setattr(conn, 'sock', mock.Mock())), mock.patch.object(page.ssl, 'create_default_context') as context:
            conn = page._PublicHTTPS('example.com', timeout=2)
            conn.connect()
            self.assertEqual(context.return_value.wrap_socket.call_args.kwargs['server_hostname'], 'example.com')
            self.assertEqual(conn.port, 443)

    def test_public_redirects_are_followed_without_cookies_or_auth(self):
        value, requests = self.fetch([Response(status=302, headers={'Location':'https://www.example.com/final'}),
                                     Response('<title>Official</title><p>Confirmed136万円</p>'.encode())])
        self.assertEqual(value['url'], 'https://www.example.com/final')
        self.assertEqual(value['requested_url'], 'https://example.com/reference')
        self.assertIn('Confirmed136万円', value['text'])
        self.assertEqual([x[0] for x in requests], ['example.com', 'www.example.com'])
        for _, method, target, headers in requests:
            self.assertEqual(method, 'GET')
            self.assertEqual(headers['Accept-Encoding'], 'identity')
            self.assertNotIn('Cookie', headers)
            self.assertNotIn('Authorization', headers)
            self.assertNotIn('X-Subscription-Token', headers)
        self.assertFalse(value['javascript_executed'])

    def test_redirect_private_downgrade_loop_and_fourth_hop_are_rejected(self):
        for target in ('http://127.0.0.1/private', 'https://user:secret@example.com/', 'http://example.com/downgrade'):
            self.assert_code('invalid_url', lambda: self.fetch([Response(status=302, headers={'Location':target})]))
        self.assert_code('redirect_limit', lambda: self.fetch([Response(status=302, headers={'Location':'/reference'})]))
        self.assert_code('redirect_limit', lambda: self.fetch([Response(status=302, headers={'Location':f'/hop{i}'}) for i in range(4)]))

    def test_http_errors_oversize_and_compression_no_automatic_retries(self):
        self.assert_code('http_error', lambda: self.fetch([Response(status=429)]))
        self.assert_code('too_large', lambda: self.fetch([Response(headers={'Content-Length':str(page.MAX_BYTES + 1)})]))
        self.assert_code('too_large', lambda: self.fetch([Response(b'x' * (page.MAX_BYTES + 1))]))
        self.assert_code('unsupported_type', lambda: self.fetch([Response(headers={'Content-Encoding':'gzip'})]))
        self.assert_code('unsupported_type', lambda: self.fetch([Response(headers={'Content-Type':'application/pdf'})]))

    def test_service_cache_error_retry_and_web_settings_no_key_forward(self):
        calls = []
        def runner(url, cancel, timeout):
            calls.append(url)
            return {'url':url, 'text':'Approved26台', 'truncated':False}
        service = page.PageService(runner=runner, interval=0)
        settings = {'max_concurrency':2, 'provider':'brave', 'brave_api_key':'MUST-NOT-FORWARD'}
        self.assertFalse(service.read('https://example.com/', settings=settings)['cached'])
        self.assertTrue(service.read('https://example.com/', settings=settings)['cached'])
        self.assertEqual(len(calls), 1)
        service.close()
        self.assert_code('cancelled', lambda: service.read('https://example.com/'))
        failing = mock.Mock(side_effect=[page.PageError('http_error'), {'text':'Retry succeeded'}])
        service = page.PageService(runner=failing, interval=0)
        self.assert_code('http_error', lambda: service.read('https://example.com/'))
        self.assertEqual(service.read('https://example.com/')['text'], 'Retry succeeded')
        self.assertEqual(failing.call_count, 2)
        service.close()

    def test_service_concurrency_spacing_and_cancellation(self):
        lock, release = threading.Lock(), threading.Event()
        current, peak, starts = 0, 0, []
        def runner(url, cancel, timeout):
            nonlocal current, peak
            with lock:
                current += 1; peak = max(peak, current); starts.append(time.monotonic())
            try:
                while not release.wait(.01):
                    if cancel.is_set(): raise page.PageError('cancelled')
                return {'text':'Complete'}
            finally:
                with lock: current -= 1
        service = page.PageService(runner=runner, interval=.06, timeout=1)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(service.read, f'https://example.com/{n}', settings={'max_concurrency':2}) for n in range(3)]
            until = time.monotonic() + .7
            while len(starts) < 2 and time.monotonic() < until: time.sleep(.01)
            self.assertEqual(peak, 2)
            self.assertGreaterEqual(starts[1] - starts[0], .05)
            release.set()
            self.assertTrue(all(f.result()['text'] == 'Complete' for f in futures))
        service.close()

    def test_service_close_cancels_active_and_queue(self):
        started = threading.Event()
        def runner(url, cancel, timeout):
            started.set()
            cancel.wait(2)
            raise page.PageError('cancelled')
        service = page.PageService(runner=runner, interval=0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(service.read, 'https://example.com/a')
            self.assertTrue(started.wait(1))
            b = pool.submit(service.read, 'https://example.com/b')
            service.close()
            for future in (a, b): self.assert_code('cancelled', future.result)

    def test_actual_isolation_kills_hung_dns_or_body_worker(self):
        # A hung child models an OS resolver that ignores socket timeouts. The
        # parent must terminate it rather than leave resolver threads/processes.
        original = subprocess.Popen
        children = []
        def hung(command, **kwargs):
            child = original([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
            children.append(child)
            return child
        started = time.monotonic()
        with mock.patch.object(page, 'start_contained', side_effect=hung):
            self.assert_code('timeout', lambda: page._isolated_read('https://example.com/', threading.Event(), .15))
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(all(child.poll() is not None for child in children))

    def test_launch_prefers_new_containment_and_supports_legacy_core(self):
        launch = mock.Mock(return_value='current-child')
        with mock.patch.object(page.os, 'name', 'nt'), mock.patch.object(page.importlib, 'import_module', return_value=SimpleNamespace(start_contained=launch)), mock.patch.object(page.subprocess, 'Popen') as popen:
            self.assertEqual(page.start_contained(['python', 'worker'], stdin='pipe'), 'current-child')
            launch.assert_called_once_with(['python', 'worker'], stdin='pipe')
            popen.assert_not_called()
        contain = mock.Mock()
        with mock.patch.object(page.os, 'name', 'nt'), mock.patch.object(page.importlib, 'import_module', return_value=SimpleNamespace(contain=contain)), mock.patch.object(page.subprocess, 'Popen') as popen:
            self.assertIs(page.start_contained(['python', 'worker'], stdout='output'), popen.return_value)
            contain.assert_called_once_with(popen.return_value)
            popen.assert_called_once_with(['python', 'worker'], stdout='output')

    def test_missing_containment_fails_only_page_tool_and_legacy_error_cleans_child(self):
        with mock.patch.object(page.os, 'name', 'nt'), mock.patch.object(page.importlib, 'import_module', side_effect=ModuleNotFoundError), mock.patch.object(page.subprocess, 'Popen') as popen:
            self.assert_code('runtime_unavailable', lambda: page.start_contained(['worker']))
            popen.assert_not_called()
        with mock.patch.object(page.os, 'name', 'nt'), mock.patch.object(page.importlib, 'import_module', return_value=SimpleNamespace()), mock.patch.object(page.subprocess, 'Popen') as popen:
            self.assert_code('runtime_unavailable', lambda: page.start_contained(['worker']))
            popen.assert_not_called()
        with mock.patch.object(page.os, 'name', 'nt'), mock.patch.object(page.importlib, 'import_module', return_value=SimpleNamespace(contain=mock.Mock(side_effect=RuntimeError('job error')))), mock.patch.object(page.subprocess, 'Popen') as popen:
            with self.assertRaises(RuntimeError): page.start_contained(['worker'])
            popen.return_value.kill.assert_called_once()
            popen.return_value.wait.assert_called_once_with(timeout=3)
            popen.return_value.stdin.close.assert_called_once()

    def test_ui_module_import_does_not_require_a_core_containment_module(self):
        original_import = builtins.__import__
        attempted = []
        def legacy_import(name, *args, **kwargs):
            if name == 'serve' or name.startswith('serve.'):
                attempted.append(name)
                raise ModuleNotFoundError('No installed API containment module')
            return original_import(name, *args, **kwargs)
        with mock.patch.object(builtins, '__import__', side_effect=legacy_import):
            loaded = runpy.run_path(str(Path(page.__file__)), run_name='ui_compat_fixture')
        self.assertIn('PageService', loaded)
        self.assertEqual(attempted, [])

    def test_non_windows_worker_has_no_windows_core_requirement(self):
        with mock.patch.object(page.os, 'name', 'posix'), mock.patch.object(page.importlib, 'import_module', side_effect=ModuleNotFoundError) as imports, mock.patch.object(page.subprocess, 'Popen') as popen:
            self.assertIs(page.start_contained(['worker']), popen.return_value)
            imports.assert_not_called()

    def test_shared_web_gate_combines_search_and_page_concurrency(self):
        gate = page.WebRequestGate(timeout=1)
        lock, release, entered = threading.Lock(), threading.Event(), threading.Event()
        active, peak = 0, 0
        def call(kind):
            nonlocal active, peak
            with gate.acquire(2, threading.Event()):
                with lock:
                    active += 1; peak = max(active, peak)
                    if active == 2: entered.set()
                release.wait(.7)
                with lock: active -= 1
                return kind
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(call, kind) for kind in ('search', 'page', 'search', 'page')]
            self.assertTrue(entered.wait(.5))
            self.assertEqual(peak, 2)
            release.set()
            self.assertEqual([f.result() for f in futures], ['search', 'page', 'search', 'page'])
        self.assertEqual(gate.active, 0)
        gate.close()
        self.assert_code('cancelled', lambda: gate.acquire().__enter__())

    def test_shared_gate_wait_timeout_and_cancel(self):
        gate = page.WebRequestGate(timeout=.08)
        with gate.acquire():
            self.assert_code('timeout', lambda: gate.acquire().__enter__())
            cancelled = threading.Event(); cancelled.set()
            self.assert_code('cancelled', lambda: gate.acquire(cancel=cancelled).__enter__())
        gate.close()

    def test_shared_gate_wait_consumes_page_deadline_before_network(self):
        gate = page.WebRequestGate(timeout=1)
        runner = mock.Mock(return_value={'text':'Must not fetch after deadline'})
        service = page.PageService(runner=runner, interval=0)
        deadline = time.monotonic() + .05
        def queued_read():
            with gate.acquire():
                return service.read('https://example.com/', deadline=deadline)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with gate.acquire():
                future = pool.submit(queued_read)
                time.sleep(.08)
            self.assert_code('timeout', future.result)
        runner.assert_not_called()
        self.assertEqual(service.active, 0)
        self.assertEqual(gate.active, 0)
        gate.close(); service.close()

    def test_shared_wait_leaves_only_remaining_budget_and_expired_cache_fails(self):
        budgets = []
        service = page.PageService(runner=lambda url,cancel,timeout: budgets.append(timeout) or {'text':'Evidence'}, interval=0)
        gate = page.WebRequestGate(timeout=1)
        deadline = time.monotonic() + .3
        def queued_read():
            with gate.acquire():
                return service.read('https://example.com/', deadline=deadline)
        with ThreadPoolExecutor(max_workers=1) as pool:
            with gate.acquire():
                future = pool.submit(queued_read)
                time.sleep(.08)
            self.assertEqual(future.result()['text'], 'Evidence')
        self.assertEqual(len(budgets), 1)
        self.assertGreater(budgets[0], 0)
        self.assertLess(budgets[0], .25)
        self.assert_code('timeout', lambda: service.read('https://example.com/', deadline=time.monotonic() - .01))
        self.assertEqual(len(budgets), 1)
        gate.close(); service.close()


if __name__ == '__main__': unittest.main()
