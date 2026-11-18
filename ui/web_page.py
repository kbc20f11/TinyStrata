"""Small public-web text reader for UI tools, without browser/runtime dependencies.

Independently implemented; fetched text is data, never model instructions. Network
work is isolated so DNS resolution, slow bodies and cancellation have a hard bound.
"""
from __future__ import annotations

from collections import OrderedDict
import codecs
from contextlib import contextmanager
from html.parser import HTMLParser
import http.client
import importlib
import ipaddress
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit

MAX_BYTES = 512 * 1024
MAX_TEXT = 20000
MAX_WORKER_BYTES = 128 * 1024
TIMEOUT = 10
_MESSAGES = {
    'invalid_url': 'Use a public HTTP or HTTPS URL on its default port, without credentials.',
    'blocked_address': 'This URL resolves to a private or unsupported network address.',
    'unsupported_type': 'Only static HTML and plain-text web pages are supported; JavaScript and document files are not executed.',
    'too_large': 'The page exceeds the512 KiB download limit.',
    'redirect_limit': 'The page redirected too many times.',
    'timeout': 'The page could not be read within10 seconds.',
    'cancelled': 'The page read was cancelled.',
    'busy': 'The page-reading queue is full. Try again shortly.',
    'http_error': 'The website did not return a successful page response.',
    'fetch_error': 'The page could not be read. It may require JavaScript, authentication or allow browser access only.',
    'runtime_unavailable': 'The page reader requires process containment. Update the TinyStrata core and restart the UI.',
}


class PageError(Exception):
    def __init__(self, code):
        self.code = code if code in _MESSAGES else 'fetch_error'
        self.message = _MESSAGES[self.code]
        super().__init__(self.message)

    def to_dict(self):
        return {'code': self.code, 'message': self.message}


def start_contained(command, **kwargs):
    """Keep UI-only updates compatible with the older installed API core.

    New cores assign the job before resuming the worker. Older cores expose
    contain(), which is sufficient here because the page worker never launches
    descendants. A missing core helper disables this tool, not UI startup.
    """
    if os.name != 'nt':
        return subprocess.Popen(command, **kwargs)
    try:
        winjob = importlib.import_module('serve.winjob')
    except ImportError:
        raise PageError('runtime_unavailable') from None
    launch = getattr(winjob, 'start_contained', None)
    if callable(launch):
        return launch(command, **kwargs)
    contain = getattr(winjob, 'contain', None)
    if not callable(contain):
        raise PageError('runtime_unavailable')
    process = subprocess.Popen(command, **kwargs)
    try:
        contain(process)
        return process
    except BaseException:
        try:
            process.kill()
            process.wait(timeout=3)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        raise


def normalize_url(value):
    # safe_url screens confusing parser spellings, credentials, local names,
    # numeric IP aliases and literal private IPs. It does not perform DNS.
    from ui.web_search import safe_url
    normalized = safe_url(value)
    if normalized is None:
        raise PageError('invalid_url')
    parts = urlsplit(normalized)
    if parts.port not in (None, 443 if parts.scheme == 'https' else 80):
        raise PageError('invalid_url')
    return normalized


def _addresses(host, port):
    rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not rows:
        raise PageError('fetch_error')
    for family, kind, proto, canonical, target in rows:
        ip = ipaddress.ip_address(target[0])
        # Reject mixed public/private DNS answers before connecting to ANY.
        if not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise PageError('blocked_address')
    return rows


class _PublicHTTP(http.client.HTTPConnection):
    def connect(self):
        last = None
        for family, kind, proto, canonical, target in _addresses(self.host, self.port):
            sock = socket.socket(family, kind, proto)
            sock.settimeout(self.timeout)
            try:
                sock.connect(target)
                self.sock = sock
                return
            except OSError as exc:
                sock.close()
                last = exc
        raise last or PageError('fetch_error')


class _PublicHTTPS(_PublicHTTP):
    default_port = 443

    def connect(self):
        super().connect()
        try:
            self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=self.host)
        except BaseException:
            self.sock.close()
            raise


class _Text(HTMLParser):
    """Extract static readable body text, preserving paragraph/table boundaries."""
    _skip = {'script', 'style', 'template', 'noscript', 'svg', 'head', 'nav', 'footer', 'aside', 'form'}
    _blocks = {'p', 'div', 'section', 'article', 'main', 'li', 'ul', 'ol', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr', 'table', 'pre', 'blockquote'}
    _void = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.titles, self.stack = [], [], []
        self.in_title = False
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        # HTML permits omission of </head>. Body tags implicitly end it; do not
        # mistake the remainder of a valid page for hidden metadata.
        if tag not in {'head', 'title', 'base', 'basefont', 'bgsound', 'link', 'meta', 'noframes', 'script', 'style', 'template'} and any(name == 'head' for name, _ in self.stack):
            self.handle_endtag('head')
            self.in_title = False
        attrs = dict(attrs)
        if tag == 'title':
            self.in_title = True
        style = re.sub(r'/\*.*?\*/|\s+', '', (attrs.get('style') or '').lower(), flags=re.S)
        hide = tag in self._skip or 'hidden' in attrs or (attrs.get('aria-hidden') or '').strip().lower() == 'true' or 'display:none' in style or 'visibility:hidden' in style
        if tag not in self._void:
            self.stack.append((tag, hide))
            if hide:
                self.hidden += 1
        if not self.hidden:
            if tag in self._blocks:
                self.parts.append('\n')
            elif tag in ('td', 'th'):
                self.parts.append(' | ')

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._void:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == 'title':
            self.in_title = False
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                self.hidden -= sum(hidden for _, hidden in self.stack[index:])
                del self.stack[index:]
                break
        if not self.hidden and tag in self._blocks:
            self.parts.append('\n')

    def handle_data(self, text):
        if self.in_title:
            self.titles.append(text)
        if not self.hidden:
            self.parts.append(text)


def extract_text(raw, content_type):
    mime = content_type.split(';', 1)[0].strip().lower()
    if mime not in ('text/html', 'application/xhtml+xml', 'text/plain'):
        raise PageError('unsupported_type')
    from email.message import Message
    header = Message()
    header['content-type'] = content_type
    charset = header.get_content_charset()
    if not charset:
        if raw.startswith(codecs.BOM_UTF8):
            charset = 'utf-8-sig'
        elif raw.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
            charset = 'utf-32'
        elif raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            charset = 'utf-16'
        elif mime != 'text/plain':
            # Legacy Japanese municipal/rail pages often declare the charset
            # only in HTML. Inspect bounded metadata, never guess encodings.
            for tag in re.findall(rb'<meta\b[^>]{0,1024}>', raw[:4096], flags=re.I):
                match = re.search(rb'charset\s*=\s*[\'\"]?\s*([a-zA-Z0-9._-]{1,50})', tag, flags=re.I)
                if match:
                    charset = match.group(1).decode('ascii')
                    break
    charset = charset or 'utf-8'
    try:
        codecs.lookup(charset)
        text = raw.decode(charset).lstrip('\ufeff')
    except (LookupError, UnicodeError):
        # Replacing a station name/number with mojibake creates misleading
        # evidence. An unreadable page must be an explicit failed tool result.
        raise PageError('fetch_error') from None
    title = ''
    if mime != 'text/plain':
        parser = _Text()
        parser.feed(text)
        title = ' '.join(''.join(parser.titles).split())[:300]
        title = ''.join(c for c in title if ord(c) >= 32 and c not in '\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069')
        text = ''.join(parser.parts)
    # No controls/bidi overrides in the evidence presented to a model or UI.
    text = ''.join(c for c in text if (ord(c) >= 32 and c not in '\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069') or c in '\n\t')
    lines = [' '.join(line.split()) for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line).strip()
    if not text:
        raise PageError('fetch_error')
    return {'title': title, 'text': text[:MAX_TEXT], 'truncated': len(text) > MAX_TEXT,
            'content_type': mime, 'extracted_characters': min(len(text), MAX_TEXT)}


def _fetch(url, timeout):
    original = normalize_url(url)
    current, seen = original, set()
    deadline = time.monotonic() + timeout
    for hop in range(4):
        if current in seen:
            raise PageError('redirect_limit')
        seen.add(current)
        parts = urlsplit(current)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PageError('timeout')
        conn = (_PublicHTTPS if parts.scheme == 'https' else _PublicHTTP)(parts.hostname, timeout=remaining)
        try:
            conn.request('GET', parts.path + ('?' + parts.query if parts.query else ''), headers={
                'Accept': 'text/html,application/xhtml+xml,text/plain',
                'Accept-Encoding': 'identity', 'User-Agent': 'TinyStrata/1', 'Connection': 'close'})
            response = conn.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                target = response.getheader('Location')
                if hop == 3 or not target:
                    raise PageError('redirect_limit')
                current = normalize_url(urljoin(current, target))
                # Prevent secure-page downgrades when following links.
                if parts.scheme == 'https' and urlsplit(current).scheme != 'https':
                    raise PageError('invalid_url')
                continue
            if response.status != 200:
                raise PageError('http_error')
            encodings = response.getheader('Content-Encoding', '').lower().split(',')
            if any(item.strip() not in ('', 'identity') for item in encodings):
                raise PageError('unsupported_type')
            content_type = response.getheader('Content-Type', '')
            if content_type.split(';', 1)[0].strip().lower() not in ('text/html', 'application/xhtml+xml', 'text/plain'):
                raise PageError('unsupported_type')
            try:
                if int(response.getheader('Content-Length', '0')) > MAX_BYTES:
                    raise PageError('too_large')
            except ValueError:
                raise PageError('fetch_error') from None
            chunks, size = [], 0
            while True:
                if time.monotonic() >= deadline:
                    raise PageError('timeout')
                chunk = response.read(min(16384, MAX_BYTES + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > MAX_BYTES:
                    raise PageError('too_large')
            value = extract_text(b''.join(chunks), content_type)
            return {'url': current, 'requested_url': original, **value,
                    'source': 'public_web_page', 'javascript_executed': False}
        finally:
            conn.close()
    raise PageError('redirect_limit')


def _worker_main():
    try:
        value = json.loads(sys.stdin.buffer.read(5000))
        result = {'ok': True, 'page': _fetch(value['url'], min(TIMEOUT, float(value['timeout'])))}
    except PageError as exc:
        result = {'ok': False, 'error': exc.to_dict()}
    except (TimeoutError, socket.timeout):
        result = {'ok': False, 'error': PageError('timeout').to_dict()}
    except Exception:
        result = {'ok': False, 'error': PageError('fetch_error').to_dict()}
    encoded = json.dumps(result, ensure_ascii=False).encode('utf-8')
    if len(encoded) > MAX_WORKER_BYTES:
        encoded = json.dumps({'ok': False, 'error': PageError('fetch_error').to_dict()}).encode()
    sys.stdout.buffer.write(encoded)


def _isolated_read(url, cancel, timeout):
    # -m preserves package imports. cwd is explicit; no arbitrary URL headers,
    # cookies, .netrc or environment-proxy support is present in http.client.
    command = [sys.executable, '-B', '-m', 'ui.web_page', '--page-worker']
    payload = json.dumps({'url': url, 'timeout': timeout}).encode()
    with tempfile.TemporaryFile() as output:
        process = start_contained(command, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.DEVNULL,
                                  cwd=str(Path(__file__).resolve().parents[1]),
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic() + timeout
        try:
            process.stdin.write(payload)
            process.stdin.close()
            while process.poll() is None:
                if cancel.wait(.025):
                    raise PageError('cancelled')
                if time.monotonic() >= deadline:
                    raise PageError('timeout')
                if os.fstat(output.fileno()).st_size > MAX_WORKER_BYTES:
                    raise PageError('fetch_error')
            if cancel.is_set():
                raise PageError('cancelled')
            output.seek(0)
            data = output.read(MAX_WORKER_BYTES + 1)
            if process.returncode != 0 or len(data) > MAX_WORKER_BYTES:
                raise PageError('fetch_error')
            value = json.loads(data)
            if not value.get('ok'):
                raise PageError(value.get('error', {}).get('code'))
            return value['page']
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)


class WebRequestGate:
    """Share the configured network concurrency across UI search/page requests.

    Provider-specific rate intervals and caches remain in their own services.
    Closing this gate wakes queued callers; service close() cancels active work.
    """
    def __init__(self, *, timeout=TIMEOUT):
        self.timeout = timeout
        self.condition = threading.Condition()
        self.active = self.waiting = 0
        self.limit = 1
        self.closed = False

    @contextmanager
    def acquire(self, limit=1, cancel=None):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 4:
            raise ValueError('Web concurrency must be an integer from1 to4')
        deadline = time.monotonic() + self.timeout
        with self.condition:
            self.limit = limit
            if self.waiting >= 4:
                raise PageError('busy')
            self.waiting += 1
            try:
                while True:
                    if self.closed or (cancel is not None and cancel.is_set()):
                        raise PageError('cancelled')
                    if time.monotonic() >= deadline:
                        raise PageError('timeout')
                    if self.active < self.limit:
                        self.active += 1
                        break
                    self.condition.wait(min(.05, max(0, deadline - time.monotonic())))
            finally:
                self.waiting -= 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class PageService:
    """Per-UI bounded queue/cache; provider search quotas stay independent."""
    def __init__(self, *, runner=None, interval=2, timeout=TIMEOUT, cache_ttl=300):
        self.runner = runner or _isolated_read
        self.interval, self.timeout, self.cache_ttl = interval, timeout, cache_ttl
        self.condition = threading.Condition()
        self.active = self.waiting = 0
        self.concurrency = 1
        self.next_start = 0
        self.cache = OrderedDict()
        self.closed = False
        self.cancellations = set()

    def read(self, url, *, settings=None, cancel=None, deadline=None):
        url = normalize_url(url)
        n = (settings or {}).get('max_concurrency', 1)
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 4:
            raise ValueError('Page concurrency must be an integer from1 to4')
        # The caller can start this budget before a shared search/page gate.
        # Waiting in that gate must not grant a second ten-second fetch budget.
        deadline = min(time.monotonic() + self.timeout, deadline) if deadline is not None else time.monotonic() + self.timeout
        local_cancel = threading.Event()
        with self.condition:
            self.concurrency = n
            if self.waiting >= 4:
                raise PageError('busy')
            self.waiting += 1
            try:
                while True:
                    if self.closed or (cancel is not None and cancel.is_set()):
                        raise PageError('cancelled')
                    now = time.monotonic()
                    if now >= deadline:
                        raise PageError('timeout')
                    cached = self.cache.get(url)
                    if cached and cached[0] > now:
                        self.cache.move_to_end(url)
                        return {**cached[1], 'cached': True}
                    self.cache.pop(url, None)
                    if self.active < self.concurrency and self.next_start <= now:
                        self.active += 1
                        self.next_start = now + self.interval
                        self.cancellations.add(local_cancel)
                        break
                    self.condition.wait(min(.05, deadline - now))
            finally:
                self.waiting -= 1
        # Combine requester cancellation and service shutdown for the worker.
        class Cancellation:
            def is_set(inner):
                return local_cancel.is_set() or (cancel is not None and cancel.is_set())
            def wait(inner, seconds):
                until = time.monotonic() + seconds
                while not inner.is_set() and time.monotonic() < until:
                    local_cancel.wait(min(.025, max(0, until - time.monotonic())))
                return inner.is_set()
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PageError('timeout')
            value = self.runner(url, Cancellation(), remaining)
            if local_cancel.is_set() or (cancel is not None and cancel.is_set()):
                raise PageError('cancelled')
            if time.monotonic() >= deadline:
                raise PageError('timeout')
            with self.condition:
                if self.closed:
                    raise PageError('cancelled')
                self.cache[url] = (time.monotonic() + self.cache_ttl, dict(value))
                self.cache.move_to_end(url)
                while len(self.cache) > 64:
                    self.cache.popitem(last=False)
            return {**value, 'cached': False}
        except PageError:
            raise
        except (TimeoutError, socket.timeout):
            raise PageError('timeout') from None
        except Exception:
            raise PageError('fetch_error') from None
        finally:
            with self.condition:
                self.active -= 1
                self.cancellations.discard(local_cancel)
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            for cancel in self.cancellations:
                cancel.set()
            self.condition.notify_all()


if __name__ == '__main__':
    if sys.argv[1:] != ['--page-worker']:
        raise SystemExit('This module is an internal UI page reader.')
    _worker_main()
