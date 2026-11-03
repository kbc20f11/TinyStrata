"""Bounded server-side search for the optional UI; no model/API-server dependency.

DDGS is an optional dependency, imported only in the isolated search worker.
Brave uses its official LLM Context endpoint. We never fetch result URLs. Provider
text is untrusted data, not instructions. This module was independently written;
no Open WebUI implementation or branding is included.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
import hashlib
import http.client
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import parse_qs, urlsplit, urlunsplit

DEFAULT_SETTINGS = {'provider': 'ddgs', 'max_concurrency': 1, 'brave_api_key': ''}
MAX_RESULTS = 5
MAX_QUERY_CHARS = 500
MAX_QUERY_WORDS = 75
MAX_RESPONSE_BYTES = 1 << 20
MAX_WORKER_BYTES = 64 << 10
MAX_TITLE_CHARS = 300
MAX_SNIPPET_CHARS = 2000
DDGS_HTTP_TIMEOUT = 12
BRAVE_HTTP_TIMEOUT = 30
BRAVE_HOST = 'api.search.brave.com'
BRAVE_PATH = '/res/v1/llm/context'
_MESSAGES = {
    'invalid_query': 'Enter a search query of at most500 characters and75 words.',
    'invalid_settings': 'The search settings are invalid.',
    'missing_key': 'Configure a Brave Search API key before using Brave LLM Context.',
    'dependency_missing': 'DDGS is not installed. Install the UI search dependency and restart the UI.',
    'rate_limited': 'The search provider is rate limiting requests. Wait before trying again.',
    'quota_exceeded': 'The search provider quota is exhausted. Check the account or try later.',
    'timeout': 'The search timed out. Try a shorter query or try again later.',
    'cancelled': 'The search was cancelled.',
    'busy': 'The search queue is full or busy. Try again shortly.',
    'provider_error': 'The search provider did not return a usable response.',
    'authentication': 'The search provider rejected the API key.',
}


class SearchError(Exception):
    """Public, sanitized failure. Raw HTTP/library errors are never exposed."""
    def __init__(self, code, *, retry_after=None, retryable=False):
        self.code = code if code in _MESSAGES else 'provider_error'
        self.message = _MESSAGES[self.code]
        self.retry_after = max(0, int(math.ceil(retry_after))) if retry_after is not None else None
        self.retryable = retryable
        super().__init__(self.message)

    def to_dict(self):
        out = {'code': self.code, 'message': self.message}
        if self.retry_after is not None:
            out['retry_after'] = self.retry_after
        return out


def validate_settings(value=None):
    """Return full server-only settings. Never send this dictionary to a browser."""
    if value is None:
        value = {}
    if not isinstance(value, dict) or set(value) - set(DEFAULT_SETTINGS):
        raise ValueError('Invalid search settings')
    out = {**DEFAULT_SETTINGS, **value}
    if out['provider'] not in ('ddgs', 'brave'):
        raise ValueError('Unknown search provider')
    n = out['max_concurrency']
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 4:
        raise ValueError('Search concurrency must be an integer from1 to4')
    key = out['brave_api_key']
    if not isinstance(key, str) or len(key) > 4096 or any(c in key for c in '\r\n\0'):
        raise ValueError('Invalid Brave API key')
    out['brave_api_key'] = key.strip()
    return out


def public_settings(value=None):
    """Safe browser representation; contains neither a key nor part of one."""
    out = validate_settings(value)
    return {'provider': out['provider'], 'max_concurrency': out['max_concurrency'],
            'has_brave_key': bool(out['brave_api_key'])}


def normalize_query(query):
    if not isinstance(query, str) or any(ord(c) < 32 and c not in '\t\r\n' for c in query):
        raise SearchError('invalid_query')
    query = ' '.join(query.split())
    if not query or len(query) > MAX_QUERY_CHARS or len(query.split()) > MAX_QUERY_WORDS:
        raise SearchError('invalid_query')
    return query


def safe_url(value):
    """Accept public-looking HTTP(S) links only; does not resolve/fetch a URL."""
    if not isinstance(value, str) or len(value) > 2048 or any(c.isspace() or ord(c) < 32 for c in value) or '\\' in value:
        return None
    try:
        parts = urlsplit(value)
        host = parts.hostname
        if parts.scheme.lower() not in ('http', 'https') or not host or parts.username is not None or parts.password is not None:
            return None
        host = host.rstrip('.').encode('idna').decode('ascii').lower()
        if '%' in host or host == 'localhost' or host.startswith('localhost.') or host.endswith(('.localhost', '.local', '.localdomain', '.lan', '.home', '.home.arpa', '.internal', '.invalid', '.test')):
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # Reject abbreviated/decimal/hex IP spellings accepted by some URL
            # stacks, and single-label names. Domains are never fetched here.
            if '.' not in host or re.fullmatch(r'(?:0x[0-9a-f]+|\d+)(?:\.(?:0x[0-9a-f]+|\d+))*', host):
                return None
            if any(not label or len(label) > 63 or label.startswith('-') or label.endswith('-') or
                   any(not (c.isalnum() or c == '-') for c in label) for label in host.split('.')):
                return None
            shown = host
        else:
            if not address.is_global or address.is_multicast or address.is_unspecified or address.is_reserved:
                return None
            shown = '[' + address.compressed + ']' if address.version == 6 else address.compressed
        port = parts.port
        if port is not None and not 1 <= port <= 65535:
            return None
        if port is not None and port != (443 if parts.scheme.lower() == 'https' else 80):
            shown += ':' + str(port)
        # DuckDuckGo's known redirect wrapper is data, not a navigation request.
        if host in ('duckduckgo.com', 'www.duckduckgo.com') and parts.path in ('/l/', '/l'):
            target = parse_qs(parts.query, max_num_fields=16).get('uddg', [])
            return safe_url(target[0]) if target and target[0] != value else None
        return urlunsplit((parts.scheme.lower(), shown, parts.path or '/', parts.query, ''))
    except (ValueError, UnicodeError, RecursionError):
        return None


class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        elif tag in ('p', 'br', 'div', 'li') and not self.hidden:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in ('script', 'style') and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def plain_text(value, limit):
    if not isinstance(value, str):
        return ''
    parser = _PlainText()
    try:
        parser.feed(value[:32000])
        text = ''.join(parser.parts)
    except Exception:
        text = ''
    return ' '.join(''.join(c for c in text if ord(c) >= 32 or c.isspace()).split())[:limit]


def normalize_results(rows):
    if not isinstance(rows, list):
        raise SearchError('provider_error')
    result, seen = [], set()
    for row in rows[:50]:
        if not isinstance(row, dict):
            continue
        url = safe_url(row.get('url', row.get('href')))
        if not url or url in seen:
            continue
        snippet = row.get('snippet', row.get('body', ''))
        if 'snippets' in row:
            snippet = '\n'.join(s for s in row['snippets'][:16] if isinstance(s, str)) if isinstance(row['snippets'], list) else ''
        result.append({'title': plain_text(row.get('title'), MAX_TITLE_CHARS) or urlsplit(url).hostname,
                       'url': url, 'snippet': plain_text(snippet, MAX_SNIPPET_CHARS)})
        seen.add(url)
        if len(result) == MAX_RESULTS:
            break
    return result


def retry_delay(headers, *, now=None):
    """Honor Retry-After and exhausted finite Brave rate-limit windows."""
    headers = {k.lower(): str(v) for k, v in headers.items()}
    delays = []
    value = headers.get('retry-after')
    if value:
        try:
            delays.append(float(value))
        except ValueError:
            try:
                delays.append(parsedate_to_datetime(value).timestamp() - (time.time() if now is None else now))
            except (ValueError, OverflowError, TypeError):
                pass
    def numbers(value):
        values = []
        for item in value.split(','):
            try:
                number = float(item.strip())
                values.append(number if math.isfinite(number) else None)
            except ValueError:
                values.append(None)
        return values

    # A zero limit means unlimited, not an exhausted quota. For example a paid
    # plan can return Limit=50,0 / Remaining=49,0 / Reset=1,<month reset> on
    # HTTP 200. Inferring exhaustion from Remaining alone would block all new
    # searches until next month. Preserve tuple positions, including bad fields.
    raw_limits = headers.get('x-ratelimit-limit')
    if raw_limits is None:
        policy = headers.get('x-ratelimit-policy', '')
        raw_limits = ','.join(item.split(';', 1)[0] for item in policy.split(','))
    limits = numbers(raw_limits)
    remaining = numbers(headers.get('x-ratelimit-remaining', ''))
    reset = numbers(headers.get('x-ratelimit-reset', ''))
    delays.extend(wait for limit, left, wait in zip(limits, remaining, reset)
                  if limit is not None and limit > 0 and left is not None and left <= 0 and wait is not None)
    valid = [max(0, d) for d in delays if math.isfinite(d)]
    return max(valid) if valid else None


def _public_addresses(host, port):
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise SearchError('provider_error')
    for family, socktype, proto, canonical, target in addresses:
        ip = ipaddress.ip_address(target[0])
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            raise SearchError('provider_error')
    return addresses


class _PinnedHTTPS(http.client.HTTPSConnection):
    def connect(self):
        # Resolve once and connect to that address. TLS still validates the fixed
        # provider host; a later DNS answer cannot redirect this connection.
        last = None
        for family, socktype, proto, canonical, target in _public_addresses(self.host, self.port):
            sock = socket.socket(family, socktype, proto)
            sock.settimeout(self.timeout)
            try:
                sock.connect(target)
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                return
            except OSError as exc:
                sock.close()
                last = exc
        raise last or SearchError('provider_error')


def _brave_search(query, key):
    if not key:
        raise SearchError('missing_key')
    body = json.dumps({'q': query, 'count': MAX_RESULTS, 'maximum_number_of_urls': MAX_RESULTS,
                       'maximum_number_of_tokens': 4096, 'maximum_number_of_tokens_per_url': 1024,
                       'safesearch': 'moderate', 'enable_local': False}).encode('utf-8')
    conn = _PinnedHTTPS(BRAVE_HOST, timeout=BRAVE_HTTP_TIMEOUT, context=ssl.create_default_context())
    try:
        conn.request('POST', BRAVE_PATH, body=body, headers={'Content-Type': 'application/json', 'Accept': 'application/json',
                     'Accept-Encoding': 'identity', 'User-Agent': 'TinyStrata/1', 'X-Subscription-Token': key})
        response = conn.getresponse()
        headers = dict(response.getheaders())
        delay = retry_delay(headers)
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise SearchError('provider_error')
        if response.status in (401, 403):
            raise SearchError('authentication')
        if response.status == 429:
            try:
                quota = json.loads(data).get('error', {}).get('code') == 'QUOTA_LIMITED'
            except (ValueError, AttributeError):
                quota = False
            raise SearchError('quota_exceeded' if quota else 'rate_limited', retry_after=delay if delay is not None else 60)
        # No redirects, alternate hosts or arbitrary page-fetching are followed.
        if response.status != 200:
            raise SearchError('provider_error', retry_after=2, retryable=response.status in (500, 502, 503, 504))
        payload = json.loads(data)
        grounding = payload.get('grounding')
        if not isinstance(grounding, dict) or not isinstance(grounding.get('generic'), list):
            raise SearchError('provider_error')
        return {'results': normalize_results(grounding['generic']), 'cooldown': delay or 0}
    finally:
        conn.close()


def _ddgs_search(query):
    try:
        from ddgs import DDGS
    except ImportError:
        raise SearchError('dependency_missing') from None
    # This class attribute is process-local: no races between UI requests and no
    # multiplied provider fan-out when the global concurrency setting changes.
    DDGS.threads = 1
    try:
        with DDGS(timeout=DDGS_HTTP_TIMEOUT, verify=True) as search:
            rows = search.text(query, safesearch='moderate', max_results=MAX_RESULTS, backend='auto')
        return {'results': normalize_results(rows), 'cooldown': 0}
    except Exception as exc:
        kind = type(exc).__name__.lower()
        message = str(exc).lower()
        if 'ratelimit' in kind or any(x in message for x in ('429', 'ratelimit', 'rate limit', 'captcha', 'anomaly')):
            raise SearchError('rate_limited', retry_after=60) from None
        if 'timeout' in kind or 'timed out' in message:
            raise SearchError('timeout') from None
        raise SearchError('provider_error') from None


def _worker_main():
    """Private pipe protocol. Neither secrets nor raw errors reach stdout/logs."""
    try:
        payload = json.loads(sys.stdin.buffer.read(20000))
        settings = validate_settings(payload['settings'])
        query = normalize_query(payload['query'])
        data = _ddgs_search(query) if settings['provider'] == 'ddgs' else _brave_search(query, settings['brave_api_key'])
        out = {'ok': True, **data}
    except SearchError as exc:
        out = {'ok': False, 'error': exc.to_dict(), 'retryable': exc.retryable}
    except (TimeoutError, socket.timeout):
        out = {'ok': False, 'error': SearchError('timeout').to_dict()}
    except Exception:
        out = {'ok': False, 'error': SearchError('provider_error').to_dict()}
    encoded = json.dumps(out, ensure_ascii=False).encode('utf-8')
    if len(encoded) > MAX_WORKER_BYTES:
        encoded = json.dumps({'ok': False, 'error': SearchError('provider_error').to_dict()}).encode()
    sys.stdout.buffer.write(encoded)


def _isolated_search(query, settings, cancel, timeout):
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--search-worker']
    payload = json.dumps({'query': query, 'settings': settings}, ensure_ascii=False).encode('utf-8')
    env = dict(os.environ)
    for key in ('BRAVE_API_KEY', 'BRAVE_SEARCH_API_KEY'):
        env.pop(key, None)
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.DEVNULL,
                                   env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic() + timeout
        try:
            process.stdin.write(payload)
            process.stdin.close()
            while process.poll() is None:
                if cancel.wait(.05):
                    raise SearchError('cancelled')
                if time.monotonic() >= deadline:
                    raise SearchError('timeout')
                if os.fstat(output.fileno()).st_size > MAX_WORKER_BYTES:
                    raise SearchError('provider_error')
            if process.returncode != 0 or os.fstat(output.fileno()).st_size > MAX_WORKER_BYTES:
                raise SearchError('provider_error')
            output.seek(0)
            try:
                result = json.loads(output.read(MAX_WORKER_BYTES + 1))
            except (ValueError, UnicodeError):
                raise SearchError('provider_error') from None
            if not result.get('ok'):
                error = result.get('error', {})
                raise SearchError(error.get('code'), retry_after=error.get('retry_after'), retryable=bool(result.get('retryable')))
            return result
        finally:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)


@dataclass
class _Flight:
    query: str
    settings: dict = field(repr=False)
    identity: tuple
    done: threading.Event = field(default_factory=threading.Event)
    cancel: threading.Event = field(default_factory=threading.Event)
    waiters: int = 1
    result: dict | None = None
    error: SearchError | None = None
    thread: threading.Thread | None = None


class SearchService:
    """One instance per UI server: shared queue/rate limit/cache across models."""
    def __init__(self, *, runner=None, intervals=None, queue_timeout=20, execution_timeout=35, max_queue=4, cache_ttl=300):
        self._runner = runner or _isolated_search
        self._intervals = intervals or {'ddgs': 2.0, 'brave': 1.1}
        self._queue_timeout = queue_timeout
        self._execution_timeout = execution_timeout
        self._max_queue = max_queue
        self._cache_ttl = cache_ttl
        self._condition = threading.Condition()
        self._flights = {}
        self._cache = OrderedDict()
        self._next_start = {}
        self._cooldown = {}
        self._active = 0
        self._concurrency = 1
        self._closed = False

    def search(self, query, *, settings=None, cancel=None):
        query = normalize_query(query)
        try:
            settings = validate_settings(settings)
        except ValueError:
            raise SearchError('invalid_settings') from None
        if settings['provider'] == 'brave' and not settings['brave_api_key']:
            raise SearchError('missing_key')
        if cancel is not None and cancel.is_set():
            raise SearchError('cancelled')
        identity = (settings['provider'], hashlib.sha256(settings['brave_api_key'].encode()).digest())
        cache_key = (*identity, query)
        with self._condition:
            if self._closed:
                raise SearchError('cancelled')
            self._concurrency = settings['max_concurrency']
            self._condition.notify_all()
            cached = self._cache.get(cache_key)
            if cached and cached[0] > time.monotonic():
                self._cache.move_to_end(cache_key)
                return self._response(query, settings['provider'], cached[1], cached=True)
            self._cache.pop(cache_key, None)
            flight = self._flights.get(cache_key)
            if flight and flight.cancel.is_set():
                raise SearchError('busy', retry_after=1)
            if flight:
                flight.waiters += 1
            else:
                if len(self._flights) >= self._concurrency + self._max_queue:
                    raise SearchError('busy', retry_after=2)
                flight = _Flight(query, settings, identity)
                self._flights[cache_key] = flight
                flight.thread = threading.Thread(target=self._execute, args=(cache_key, flight), daemon=True, name='TinyStrataSearch')
                flight.thread.start()
        try:
            while not flight.done.wait(.05):
                if cancel is not None and cancel.is_set():
                    raise SearchError('cancelled')
            if cancel is not None and cancel.is_set():
                raise SearchError('cancelled')
            if flight.error:
                raise flight.error
            return self._response(query, settings['provider'], flight.result['results'], cached=False)
        finally:
            with self._condition:
                flight.waiters -= 1
                if not flight.waiters and not flight.done.is_set():
                    flight.cancel.set()
                    self._condition.notify_all()

    @staticmethod
    def _response(query, provider, rows, *, cached):
        return {'query': query, 'provider': provider, 'results': [dict(row) for row in rows], 'cached': cached}

    def _execute(self, key, flight):
        queued_until = time.monotonic() + self._queue_timeout
        run_until = None
        try:
            for attempt in range(2):
                with self._condition:
                    while True:
                        now = time.monotonic()
                        if self._closed or flight.cancel.is_set():
                            raise SearchError('cancelled')
                        cooldown = self._cooldown.get(flight.identity, 0) - now
                        if cooldown > 2:
                            raise SearchError('rate_limited', retry_after=cooldown)
                        wait = max(cooldown, self._next_start.get(flight.identity, 0) - now, 0)
                        if self._active < self._concurrency and wait <= 0:
                            self._active += 1
                            self._next_start[flight.identity] = now + self._intervals[flight.settings['provider']]
                            break
                        if now >= queued_until:
                            raise SearchError('busy', retry_after=2)
                        self._condition.wait(min(.05, queued_until - now))
                try:
                    if run_until is None:
                        run_until = time.monotonic() + self._execution_timeout
                    remaining = run_until - time.monotonic()
                    if remaining <= 0:
                        raise SearchError('timeout')
                    result = self._runner(flight.query, flight.settings, flight.cancel, remaining)
                    rows = normalize_results(result.get('results'))
                    delay = result.get('cooldown', 0)
                    with self._condition:
                        if isinstance(delay, (int, float)) and math.isfinite(delay) and delay > 0:
                            self._cooldown[flight.identity] = max(self._cooldown.get(flight.identity, 0), time.monotonic() + delay)
                        flight.result = {'results': rows}
                        self._cache[key] = (time.monotonic() + self._cache_ttl, rows)
                        self._cache.move_to_end(key)
                        while len(self._cache) > 128:
                            self._cache.popitem(last=False)
                    break
                except SearchError as exc:
                    delay = exc.retry_after or (60 if exc.code == 'rate_limited' else 2)
                    if exc.code in ('rate_limited', 'quota_exceeded') or exc.retryable:
                        with self._condition:
                            self._cooldown[flight.identity] = max(self._cooldown.get(flight.identity, 0), time.monotonic() + delay)
                    if attempt == 0 and flight.settings['provider'] == 'brave' and delay <= 2 and (exc.retryable or exc.code == 'rate_limited'):
                        queued_until = run_until
                        continue
                    raise
                finally:
                    with self._condition:
                        self._active -= 1
                        self._condition.notify_all()
        except SearchError as exc:
            flight.error = exc
        except Exception:
            flight.error = SearchError('provider_error')
        finally:
            with self._condition:
                self._flights.pop(key, None)
                flight.done.set()
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            threads = []
            for flight in self._flights.values():
                flight.cancel.set()
                if flight.thread:
                    threads.append(flight.thread)
            self._condition.notify_all()
        deadline = time.monotonic() + 3
        for thread in threads:
            thread.join(max(0, deadline - time.monotonic()))


if __name__ == '__main__':
    if sys.argv[1:] != ['--search-worker']:
        raise SystemExit('This module is an internal UI search worker.')
    _worker_main()
