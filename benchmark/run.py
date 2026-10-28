"""Run TinyStrata benchmark protocol v1; see benchmark/README.md.

Uses the installed model profiles, isolated localhost API processes and the
existing TinyStrata Python dependencies. Results stay outside the source tree.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GIB = 1 << 30
MODELS = {
    'qwen': ('config.json', 'qwen35moe', 'qwen3.6-35b-a3b'),
    'gemma': ('config-gemma4.json', 'gemma4', 'gemma-4-26b-a4b-it'),
    'lfm': ('config-lfm2.json', 'lfm2moe', 'lfm2.5-24b-a2b'),
}
ENGINE_KEYS = ('model', 'version', 'kv', 'kv_grow', 'mtp', 'vision', 'images',
               'max_context', 'context_auto', 'chunk', 'chunk_auto', 'expert_auto',
               'expert_slots', 'expert_cache_bytes', 'expert_cache_mib',
               'kv_logical_bytes', 'kv_mapped_bytes', 'kv_committed_rows', 'gpu_cc')


def save(path, value):
    with Path(path).open('w', encoding='utf-8') as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.flush()
        os.fsync(out.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def child_environment(environ=None):
    # Library discovery stays available; experimental engine tuning does not.
    source = os.environ if environ is None else environ
    discovery = {'CUDA_PATH', 'CUDA_HOME', 'HIP_PATH', 'HIP_CLANG_PATH', 'ROCM_PATH', 'ROCM_HOME'}
    tuning = ('STRATA_', 'TINYSTRATA_', 'ROCBLAS_', 'HIPBLAS', 'TENSILE_',
              'CUDA_', 'HIP_', 'HSA_', 'ROCR_', 'AMD_', 'GPU_')
    return {k: v for k, v in source.items() if k in discovery or k.startswith('CUDA_PATH_V')
            or not k.startswith(tuning)}


def controlled_config(original, model, folder, port, cases):
    file, arch, name = MODELS[model]
    if original.get('arch', 'qwen35moe') != arch:
        raise ValueError(f'{file} does not select {arch}')
    # Rebuild the native arguments rather than retain manual tuning flags.
    args = original.get('args', [])
    base = Path(original.get('cwd') or ROOT).expanduser().resolve()
    def resolved(path):
        path = Path(path).expanduser()
        return str((path if path.is_absolute() else base / path).resolve())
    def value(flag):
        if flag not in args or args.index(flag) + 1 >= len(args):
            raise ValueError(f'{file} is missing {flag}')
        return resolved(args[args.index(flag) + 1])
    native = ['--model', value('--model'), '--max-context', 'auto', '--chunk', 'auto',
              '--kv-type', 'int8', '--kv-grow']
    if model == 'gemma':
        native += ['--draft-model', value('--draft-model')]
    if model != 'lfm':
        native += ['--mtp', '3', '--vision']
    cfg = {key: copy.deepcopy(original[key]) for key in
           ('exe', 'tokenizer', 'backend', 'lib_dirs', 'gpu', 'hip_ordinal', 'cuda_device')
           if key in original}
    cfg.update(arch=arch, model_name=name, args=native, cwd=str(ROOT),
               host='127.0.0.1', port=port, log=str(Path(folder) / 'engine.log'),
               env={}, parallel=1, enable_thinking=True, max_tokens=32768,
               sampling=dict(cases['sampling'][model]), vision_mode='no' if model == 'lfm' else 'gpu')
    for key in ('exe', 'tokenizer'):
        if key in cfg:
            cfg[key] = resolved(cfg[key])
    if cfg.get('lib_dirs'):
        cfg['lib_dirs'] = [resolved(path) for path in cfg['lib_dirs']]
    if model != 'lfm':
        vision = original.get('vision')
        if not isinstance(vision, dict) or not all(vision.get(k) for k in ('exe', 'mmproj', 'model')):
            raise ValueError(f'{file} needs its installed GPU Vision model')
        cfg['vision'] = {k: resolved(vision[k]) for k in ('exe', 'mmproj', 'model')}
        cfg['vision'].update(gpu=True, elastic=False)
        cfg['vision_exe'] = cfg['vision']['exe']
    return cfg


def request_body(model, text, limit, cases, thinking=True):
    sampling = {'frequency_penalty': 0.0, **cases['sampling'][model]}
    return dict(model=MODELS[model][2], messages=[dict(role='user', content=text)],
                max_tokens=limit, stream=True, enable_thinking=thinking,
                seed=cases['seed'], **sampling)


def read_sse(response, started, clock=time.perf_counter):
    """Measure streamed output, ignoring headers, pings and empty deltas."""
    text, reasoning, first, reason = [], [], None, None
    data = []
    for raw in response:
        line = raw.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip())
        if line or not data:
            continue
        event = json.loads('\n'.join(data))
        data.clear()
        kind = event.get('type')
        if kind == 'error':
            raise RuntimeError(f'API stream error: {event.get("error")}')
        if kind in ('content_block_delta', 'content_block_start'):
            if kind == 'content_block_start':
                block = event.get('content_block', {})
                delta = {'type': 'text_delta' if block.get('type') == 'text' else 'thinking_delta',
                         'text': block.get('text'), 'thinking': block.get('thinking')}
            else:
                delta = event.get('delta', {})
            piece = delta.get('text') if delta.get('type') == 'text_delta' else \
                    delta.get('thinking') if delta.get('type') == 'thinking_delta' else None
            if isinstance(piece, str) and piece:
                if first is None:
                    first = clock() - started
                (text if delta.get('type') == 'text_delta' else reasoning).append(piece)
        elif kind == 'message_delta':
            reason = event.get('delta', {}).get('stop_reason', reason)
        elif kind == 'message_stop':
            ended = clock() - started
            if first is None:
                raise RuntimeError('The stream finished without output text or thinking')
            if not isinstance(reason, str) or not reason:
                raise RuntimeError('The stream has no completion reason')
            return dict(content=''.join(text), reasoning=''.join(reasoning),
                        first_output_s=first, elapsed_s=ended, stop_reason=reason, complete=True)
    raise RuntimeError('The stream ended before message_stop')


def validate_native(native, target=None):
    for key in ('engine_generated', 'prompt_read', 'prompt_total', 'reused', 'prompt_ms', 'decode_ms'):
        value = native.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Missing or nonfinite native timing/count: {key}')
    if native['engine_generated'] <= 0 or native['decode_ms'] <= 0 or native['prompt_ms'] <= 0:
        raise ValueError('A measured request needs generated tokens and positive native timings')
    for key in ('engine_generated', 'prompt_read', 'prompt_total', 'reused'):
        if int(native[key]) != native[key] or native[key] < (0 if key == 'reused' else 1):
            raise ValueError(f'Invalid token count: {key}')
    if native['reused'] != 0 or native['prompt_read'] != native['prompt_total']:
        raise ValueError('KV prefix was reused; this is not a fresh-input measurement')
    if target is not None and native['prompt_read'] != target:
        raise ValueError(f'Expected{target} input tokens, read{native["prompt_read"]}')
    if native.get('finish') in ('cancel', 'error'):
        raise ValueError(f'Incomplete request: {native.get("finish")}')


def exact_length_text(count_fn, paragraph, question, target):
    """Count the complete Messages request, including the model's template."""
    filler = ''.join(paragraph.format(i=i) for i in range(target // 16 + 128))
    if count_fn(filler + question) < target:
        raise ValueError('The canonical document is too short for the target')
    lo, hi, found = 0, len(filler), None
    while lo <= hi:
        middle = (lo + hi) // 2
        candidate = filler[:middle] + question
        n = count_fn(candidate)
        if n == target:
            return candidate, n
        if n < target:
            found = middle
            lo = middle + 1
        else:
            hi = middle - 1
    # BPE token counts need not be monotonic at every character boundary.
    # Check adjacent cut points, then small canonical ASCII suffixes.
    for cut in range(max(0, (found or 0) - 96), min(len(filler), (found or 0) + 96) + 1):
        candidate = filler[:cut] + question
        if count_fn(candidate) == target:
            return candidate, target
    for suffix in (' ', '\n', '.', ' x', ' 0', '\n0'):
        for n in range(1, 17):
            candidate = filler[:found or 0] + suffix * n + question
            if count_fn(candidate) == target:
                return candidate, target
    raise ValueError(f'Could not construct exactly{target} complete input tokens')


def cache_bytes(info):
    value = info.get('expert_cache_bytes')
    if value is None and info.get('expert_cache_mib') is not None:
        value = info['expert_cache_mib'] * (1 << 20)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('This engine does not report its Expert cache size; update TinyStrata')
    return int(value)


def aggregate_model(runs):
    records = [x for run in runs if run.get('complete') for x in run['records']]
    out = {}
    for category in ('generation', 'input_4k', 'input_16k'):
        selected = [r for r in records if r['category'] == category]
        if not selected:
            continue
        tokens = sum(r['native']['engine_generated' if category == 'generation' else 'prompt_read'] for r in selected)
        ms = sum(r['native']['decode_ms' if category == 'generation' else 'prompt_ms'] for r in selected)
        out[category] = dict(tokens=tokens, milliseconds=ms, tok_s=tokens * 1000 / ms,
                             requests=len(selected))
    images = [r for r in records if r['category'] == 'image']
    out['image'] = dict(requests=len(images), first_output_s=statistics.median(r['response']['first_output_s'] for r in images),
                        elapsed_s=statistics.median(r['response']['elapsed_s'] for r in images)) if images else None
    complete = [r for r in runs if r.get('complete')]
    out['expert_cache'] = {key: [cache_bytes(r[key]) for r in complete]
                           for key in ('startup_engine', 'after_16k_engine')}
    out['complete_runs'] = len(complete)
    return out


class Service:
    def __init__(self, cfg, folder, timeout):
        self.cfg, self.folder, self.timeout = cfg, Path(folder), timeout
        self.base = f'http://127.0.0.1:{cfg["port"]}'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.stdout = (self.folder / 'api.stdout.log').open('w', encoding='utf-8')
        self.stderr = (self.folder / 'api.stderr.log').open('w', encoding='utf-8')
        self.owned, self.lock, self.stop = {}, threading.Lock(), threading.Event()
        self.process = subprocess.Popen([sys.executable, '-B', '-m', 'serve.server',
            '--engine', 'tinystrata', '--config', str(self.folder / 'config.json'),
            '--host', '127.0.0.1', '--port', str(cfg['port'])], cwd=ROOT,
            stdout=self.stdout, stderr=self.stderr, env=child_environment(),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.remember()
        self.watcher = threading.Thread(target=self.watch, daemon=True)
        self.watcher.start()

    def remember(self):
        if self.process.poll() is not None:
            return
        try:
            root = psutil.Process(self.process.pid)
            for child in [root] + root.children(recursive=True):
                with self.lock:
                    self.owned[child.pid] = child.create_time()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def watch(self):
        while not self.stop.wait(.2):
            self.remember()

    def owns_listener(self):
        if self.process.poll() is not None:
            return False
        self.remember()
        with self.lock:
            identities = dict(self.owned)
        try:
            root = psutil.Process(self.process.pid)
            created = identities.get(root.pid)
            if created is None or abs(root.create_time() - created) >= .001:
                return False
            # On Windows a venv's python.exe can be a redirector: its actual
            # interpreter child, rather than the Popen PID, owns the socket.
            for pid, created in identities.items():
                try:
                    proc = psutil.Process(pid)
                    if abs(proc.create_time() - created) >= .001 or not proc.is_running():
                        continue
                    if any(c.status == psutil.CONN_LISTEN and c.laddr.port == self.cfg['port']
                           for c in proc.net_connections(kind='tcp')):
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def request(self, path, body=None):
        if not self.owns_listener():
            raise RuntimeError('The benchmark API is not listening on its owned port')
        return urllib.request.Request(self.base + path,
            data=json.dumps(body, ensure_ascii=False).encode('utf-8') if body is not None else None,
            headers={'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})

    def json(self, path, body=None, timeout=None):
        try:
            with self.opener.open(self.request(path, body), timeout=timeout or self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'{path}: HTTP{error.code}: {error.read().decode("utf-8", "replace")[:1200]}') from error

    def engine(self):
        metrics = self.json('/metrics')
        return {k: v for k, v in metrics['engine'].items() if k in ENGINE_KEYS}

    def vision_identity(self):
        if not self.cfg.get('vision'):
            return None
        self.remember()
        expected = os.path.normcase(os.path.realpath(self.cfg['vision']['exe']))
        matches = []
        with self.lock:
            owned = dict(self.owned)
        for pid, created in owned.items():
            try:
                proc = psutil.Process(pid)
                if abs(proc.create_time() - created) < .001 and proc.is_running() and \
                        os.path.normcase(os.path.realpath(proc.exe())) == expected:
                    matches.append(dict(pid=pid, created=created))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if len(matches) != 1:
            raise RuntimeError('Expected one live GPU image encoder owned by this benchmark API')
        return matches[0]

    def ready(self, timeout):
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f'Benchmark API exited; see{self.folder / "api.stderr.log"}')
            try:
                health = self.json('/health', timeout=1)
                if health.get('loaded') and (not self.cfg.get('vision') or health.get('images')):
                    return self.vision_identity()
            except (OSError, RuntimeError):
                pass
            time.sleep(.2)
        raise TimeoutError(f'API startup exceeded{timeout}s')

    def generate(self, body):
        before = self.engine()
        request = self.request('/v1/messages', body)
        started = time.perf_counter()
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                streamed = read_sse(response, started)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'Generation HTTP{error.code}: {error.read().decode("utf-8", "replace")[:1200]}') from error
        metrics = self.json('/metrics')
        if not metrics.get('requests'):
            raise RuntimeError('No native timing record after generation')
        return dict(response=streamed, native=metrics['requests'][0], engine_before=before,
                    engine_after={k: v for k, v in metrics['engine'].items() if k in ENGINE_KEYS})

    def close(self):
        self.remember()
        self.stop.set()
        self.watcher.join(timeout=2)
        targets, errors = [], []
        with self.lock:
            identities = dict(self.owned)
        for pid, created in identities.items():
            try:
                proc = psutil.Process(pid)
                if abs(proc.create_time() - created) < .001:
                    targets.append(proc)
            except psutil.NoSuchProcess:
                pass
        for proc in sorted(targets, key=lambda p: p.pid == self.process.pid):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
            except Exception as error:
                errors.append(f'PID{proc.pid}: {error!r}')
        if self.process.poll() is None:
            self.process.kill()  # Popen retains this exact process handle.
        _, alive = psutil.wait_procs(targets, timeout=5)
        errors += [f'Owned PID{p.pid} survived cleanup' for p in alive]
        self.process.wait(timeout=5)
        self.stdout.close()
        self.stderr.close()
        if errors:
            raise RuntimeError('; '.join(errors))


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def assets(cfg):
    args = cfg['args']
    paths = [Path(cfg['exe'])]
    paths += [Path(args[args.index(flag) + 1]) for flag in ('--model', '--draft-model') if flag in args]
    if cfg.get('vision'):
        paths += [Path(cfg['vision']['exe']), Path(cfg['vision']['mmproj'])]
    return list(dict.fromkeys(paths))


def render_report(report):
    lines = [f'# {report["protocol"]}', '', f'Status: {"complete" if report["complete"] else "incomplete"}.',
             f'Runs per model: {report["runs"]} ({"standard" if report["standard"] else "diagnostic"}).',
             '', '| Model | Generation tok/s | 4K input tok/s | 16K input tok/s | Image first output / complete, s | Expert cache GiB: startup → after16K |',
             '|---|---:|---:|---:|---:|---|']
    for model, data in report['models'].items():
        result = aggregate_model(data['runs'])
        def rate(key):
            return f'{result[key]["tok_s"]:.2f}' if key in result else '—'
        image = result['image']
        image_text = f'{image["first_output_s"]:.2f} / {image["elapsed_s"]:.2f}' if image else 'N/A'
        def span(values):
            values = [v / GIB for v in values]
            if not values:
                return '—'
            low, high = f'{min(values):.2f}', f'{max(values):.2f}'
            return low if low == high else f'{low}–{high}'
        cache = result['expert_cache']
        lines.append(f'| {MODELS[model][2]} | {rate("generation")} | {rate("input_4k")} | {rate("input_16k")} | '
                     f'{image_text} | {span(cache["startup_engine"])} → {span(cache["after_16k_engine"])} |')
        if data.get('error'):
            lines.append(f'\n{model}: ERROR — {data["error"]}')
    if report.get('error'):
        lines += ['', f'ERROR: {report["error"]}']
    lines += ['', 'Text rates pool actual native token counts and times. Image values are medians of client wall times.',
              'Image timings include image encoding; startup/model loading is excluded. Other-app usage is not collected.',
              'See result.json for request counts, actual output lengths, settings, hashes and individual measurements.']
    return '\n'.join(lines) + '\n'


def main():
    local = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=(*MODELS, 'all'), default='all')
    parser.add_argument('--config-dir', type=Path, default=local / 'TinyStrata-work/runtime/.tinystrata')
    parser.add_argument('--output', type=Path, default=local / 'TinyStrata-work/benchmarks' / time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--runs', type=int, default=4, help='Protocol v1 uses4; other counts are diagnostic')
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--label', default='', help='Optional GPU/system label for the report')
    parser.add_argument('--prepare-only', action='store_true', help='Validate configs and write plans without starting an API')
    options = parser.parse_args()
    if not 1 <= options.runs <= 20 or options.request_timeout <= 0 or options.startup_timeout <= 0:
        parser.error('runs must be1..20 and timeouts must be positive')
    options.config_dir, options.output = options.config_dir.resolve(), options.output.resolve()
    cases = json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))
    if cases['protocol'] != 'tinystrata-bench-v1' or cases['seed'] != 1234 or len(cases['generation']) != 3:
        parser.error('Unexpected benchmark case manifest')
    models = list(MODELS) if options.model == 'all' else [options.model]
    options.output.mkdir(parents=True, exist_ok=False)
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True,
                                        stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = dict(protocol=cases['protocol'], standard=options.runs == cases['runs'], complete=False,
                  label=options.label, source_commit=commit, benchmark_sha256=sha256(__file__),
                  cases_sha256=sha256(HERE / 'cases.json'), image_sha256=sha256(HERE / 'image.png'),
                  platform=platform.platform(), cpu=platform.processor(), ram_bytes=psutil.virtual_memory().total,
                  runs=options.runs, seed=cases['seed'], models={}, started=time.time())
    def persist():
        save(options.output / 'result.json', report)
        (options.output / 'report.md').write_text(render_report(report), encoding='utf-8')
    persist()
    all_hashes = {}
    try:
        for model in models:
            original = json.loads((options.config_dir / MODELS[model][0]).read_text(encoding='utf-8-sig'))
            data = dict(runs=[], sampling=cases['sampling'][model], assets=[])
            report['models'][model] = data
            prepared = controlled_config(original, model, options.output, free_port(), cases)
            sys.path.insert(0, str(ROOT))
            from serve import tinyconfig
            tinyconfig.validate(prepared)
            for path in assets(prepared):
                if not path.is_file():
                    raise FileNotFoundError(f'Install the model/binary before benchmarking: {path}')
                if path not in all_hashes and not options.prepare_only:
                    print(f'Hashing {path.name}', flush=True)
                    all_hashes[path] = sha256(path)
                data['assets'].append(dict(file=path.name, size=path.stat().st_size,
                                           sha256=all_hashes.get(path)))
            for number in range(1, options.runs + 1):
                folder = options.output / f'{model}-{number}'
                folder.mkdir()
                cfg = controlled_config(original, model, folder, free_port(), cases)
                tinyconfig.validate(cfg)
                save(folder / 'config.json', cfg)
                if options.prepare_only:
                    continue
                run = dict(number=number, complete=False, records=[])
                data['runs'].append(run)
                service = None
                try:
                    service = Service(cfg, folder, options.request_timeout)
                    identity = service.ready(options.startup_timeout)
                    run['vision_start'] = identity
                    run['startup_engine'] = service.engine()
                    info = run['startup_engine']
                    if info.get('kv') != 'int8' or info.get('kv_grow') != 1 or info.get('mtp') != (0 if model == 'lfm' else 3):
                        raise ValueError(f'Engine does not match protocol settings: {info}')
                    cache_bytes(info)
                    warmup = request_body(model, cases['warmup_text'], cases['warmup_max_tokens'], cases)
                    save(folder / 'warmup.json', service.generate(warmup))
                    def measure(category, name, body, target=None):
                        save(folder / f'{name}-request.json', body)
                        record = dict(category=category, name=name, **service.generate(body))
                        validate_native(record['native'], target)
                        if record['native']['engine_generated'] > body['max_tokens']:
                            raise ValueError('Engine generated more than the requested output limit')
                        if identity is not None and service.vision_identity() != identity:
                            raise RuntimeError('The GPU image encoder restarted during the benchmark')
                        if category == 'image' and not record['response']['content'].strip():
                            raise ValueError('Image test produced no description')
                        run['records'].append(record)
                        save(folder / f'{name}-result.json', record)
                        persist()
                        if category == 'image':
                            print(f'{model} run{number} image: {record["response"]["first_output_s"]:.3f}s first, '
                                  f'{record["response"]["elapsed_s"]:.3f}s complete', flush=True)
                        else:
                            native = record['native']
                            print(f'{model} run{number} {name}: {native["engine_generated"]}output, '
                                  f'PP{native["prompt_read"]}/{native["prompt_ms"]}ms, TG{native["decode_ms"]}ms', flush=True)
                    for repeat in (1, 2):
                        for prompt in cases['generation']:
                            measure('generation', f'{prompt["name"]}-{repeat}',
                                    request_body(model, prompt['text'], cases['generation_max_tokens'], cases))
                    for length, category in ((4096, 'input_4k'), (16384, 'input_16k')):
                        def count(text):
                            body = request_body(model, text, cases['prefill_max_tokens'], cases)
                            return service.json('/v1/messages/count_tokens', body)['input_tokens']
                        text, _ = exact_length_text(count, cases['long_paragraph'], cases['long_question'], length)
                        measure(category, category, request_body(model, text, cases['prefill_max_tokens'], cases), length)
                    run['after_16k_engine'] = service.engine()
                    if model != 'lfm':
                        image = dict(type='image', source=dict(type='base64', media_type='image/png',
                                     data=base64.b64encode((HERE / 'image.png').read_bytes()).decode('ascii')))
                        body = request_body(model, [image, dict(type='text', text=cases['image_prompt'])],
                                            cases['image_max_tokens'], cases, thinking=False)
                        measure('image', 'image', body)
                    run['vision_end'] = service.vision_identity()
                    run['complete'] = True
                except BaseException as error:
                    run['error'] = repr(error)
                    raise
                finally:
                    if service is not None:
                        service.close()
                    persist()
            data['aggregate'] = aggregate_model(data['runs'])
            persist()
        report['complete'] = not options.prepare_only
        report['prepared_only'] = options.prepare_only
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        report['finished'] = time.time()
        persist()
    print(f'Results: {options.output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
