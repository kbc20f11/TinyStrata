"""Measure Qwen/Clover CPU worker choices without changing saved settings.

Stop the regular TinyStrata API before running, as for benchmark/run.py. This
program starts and closes only its own isolated loopback benchmark services.
It preserves the selected profile's native arguments/environment except explicit
worker or command-line overrides, uses temperature 0, and prints a manual
recommendation only after mirrored screening and fresh ABBA/BAAB confirmation.
These results are diagnostics, not tinystrata-bench-v1 submissions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location('tinystrata_probe_protocol', HERE / 'run.py')
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)
MODELS = ('qwen', 'clover')
STATIC_FIELDS = ('model', 'version', 'kv', 'kv_grow', 'mtp', 'clover_mode', 'clover_draft',
    'clover_accept', 'clover_accept_ratio', 'clover_accept_delta', 'max_context', 'chunk',
    'vision', 'images', 'expert_slots', 'expert_total', 'expert_cache_bytes', 'expert_arena_bytes',
    'q36_mmvq_il', 'q36_mmvq_il_auto', 'q36_main_mmq', 'q36_main_mmq_auto', 'q36_mtp_mmq',
    'q36_gdn_chain', 'q36_gdn_chain_eligible_layers', 'q36_q8_pair', 'q36_q8_pair_eligible_layers',
    'q36_cpu_compact')
COUNT_FIELDS = ('engine_generated', 'output_tokens', 'prompt_total', 'prompt_read', 'reused',
                'drafts_offered', 'drafts_accepted', 'finish')
SAFE_CONFIG = ('exe', 'tokenizer', 'backend', 'lib_dirs', 'gpu', 'hip_ordinal', 'cuda_device',
    'arch', 'model_name', 'args', 'cwd', 'enable_thinking', 'max_tokens', 'vision_mode',
    'vision_exe', 'vision', 'memory_policy_version', 'chunk_policy_version',
    'generation_policy_version', 'clover_vision_policy_version', 'clover_decode_policy_version')


class _NoWatcher:
    def join(self, timeout=None):
        pass


class OwnedService(B.Service):
    """Close a partially initialized benchmark service before propagating failure."""
    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except BaseException as failure:
            cleanup_errors = []
            process = getattr(self, 'process', None)
            if process is not None:
                watcher = getattr(self, 'watcher', None)
                if watcher is None or watcher.ident is None:
                    self.watcher = _NoWatcher()
                if getattr(self, 'stop', None) is not None:
                    self.stop.set()
                try:
                    self.close()
                except BaseException:
                    # Retain the existing helper's PID+creation-time ownership
                    # rule if its normal cleanup itself failed. Never search
                    # for processes by executable name or terminate listeners.
                    for pid, created in dict(getattr(self, 'owned', {})).items():
                        try:
                            child = B.psutil.Process(pid)
                            if abs(child.create_time() - created) < .001:
                                child.kill()
                        except B.psutil.NoSuchProcess:
                            pass
                        except BaseException as error:
                            cleanup_errors.append(type(error).__name__)
                    try:
                        if process.poll() is None:
                            process.kill()  # Exact Popen process handle, not a PID lookup.
                            process.wait(timeout=10)
                    except BaseException as error:
                        cleanup_errors.append(type(error).__name__)
            for name in ('stdout', 'stderr'):
                stream = getattr(self, name, None)
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException as error:
                        cleanup_errors.append(type(error).__name__)
            if cleanup_errors and hasattr(failure, 'add_note'):
                failure.add_note('Owned benchmark initialization cleanup also reported: ' + ', '.join(cleanup_errors))
            raise


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_provenance():
    def git(*arguments):
        try:
            return subprocess.check_output(['git', '-C', str(ROOT), *arguments],
                text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    head = git('rev-parse', 'HEAD')
    status = git('status', '--porcelain', '--untracked-files=normal')
    return {'commit': head, 'commit_kind': 'source_checkout_HEAD_not_native_build',
            'source_worktree_dirty': None if status is None else bool(status),
            'tool_sha256': B.sha256(Path(__file__).resolve())}


def candidate_counts(default, supplied=None):
    require(isinstance(default, int) and not isinstance(default, bool) and default > 0,
            'The observed default worker count must be positive')
    values = {default, max(1, default // 2)}
    value = 1
    while value < default:
        values.add(value)
        value *= 2
    if supplied is not None:
        require(supplied and all(isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= default
                               for n in supplied), 'Requested workers must be in1..observed-default')
        values = {default, *supplied}
    return sorted(values)


def replace_flag(arguments, flag, value=None, aliases=()):
    result = []
    iterator = iter(arguments)
    for item in iterator:
        if item in (flag, *aliases):
            require(next(iterator, None) is not None, 'Missing native option value: ' + item)
        else:
            result.append(item)
    if value is not None:
        result += [flag, str(value)]
    return result


def preserved_environment(original, explicit):
    result = {}
    prefixes = ('STRATA_', 'TINYSTRATA_', 'CUDA_', 'HIP_', 'ROCM_', 'ROCR_', 'HSA_',
                'ROCBLAS_', 'HIPBLAS', 'TENSILE_', 'OMP_', 'OPENBLAS_', 'MKL_')
    for key, value in original.items():
        require(isinstance(key, str) and isinstance(value, str) and
                key.startswith(prefixes) and not re.search(r'API.?KEY|SECRET|PASSWORD|TOKEN', key, re.I),
                'Profile contains an environment field outside the native tuning allowlist: ' + str(key))
        require(not any(char in key + value for char in '\r\n\0'), 'Invalid native environment field')
        result[key] = value
    result.update(explicit)
    return result


def configuration(original, args, folder, workers):
    cfg = {key: copy.deepcopy(original[key]) for key in SAFE_CONFIG if key in original}
    require(all(isinstance(x, str) for x in cfg.get('args', [])), 'Native arguments must be strings')
    require(cfg.get('arch', 'qwen35moe') == B.MODELS[args.model][1], 'Profile architecture differs')
    require(not any(re.search(r'api.?key|password|secret', value, re.I)
                    for value in cfg['args'] if value.startswith('--')),
            'Credentials must not be present in native argument flags')
    base = Path(cfg.get('cwd') or ROOT).expanduser().resolve()
    def resolved(value):
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else base / path).resolve())
    for key in ('exe', 'tokenizer', 'vision_exe'):
        if key in cfg:
            cfg[key] = resolved(cfg[key])
    if cfg.get('lib_dirs'):
        cfg['lib_dirs'] = [resolved(path) for path in cfg['lib_dirs']]
    if cfg.get('vision'):
        vision = {key: value for key, value in cfg['vision'].items()
                  if key in ('exe', 'model', 'mmproj', 'gpu', 'elastic', 'min_tokens', 'max_tokens')}
        for key in ('exe', 'model', 'mmproj'):
            if key in vision:
                vision[key] = resolved(vision[key])
        cfg['vision'] = vision
    if cfg.get('vision_mode') == 'no':
        cfg.pop('vision', None)
    native = list(cfg['args'])
    for flag in ('--model', '--draft-model'):
        if flag in native:
            native[native.index(flag) + 1] = resolved(native[native.index(flag) + 1])
    native = replace_flag(native, '--cpu-threads', workers or None, aliases=('--pool-workers',))
    if args.expert_gib is not None:
        native = replace_flag(native, '--expert-vram', args.expert_gib, aliases=('--expert-slots',))
    for flag, value in (('--kv-type', args.kv_type), ('--max-context', args.context), ('--chunk', args.chunk)):
        if value is not None:
            native = replace_flag(native, flag, value)
    cfg.update(args=native, cwd=str(base), host='127.0.0.1', port=B.free_port(), log=str(folder / 'engine.log'),
               parallel=1, env=preserved_environment(original.get('env', {}), dict(args.engine_env)))
    # Private benchmark configs retain the reviewed raw argv instead of being
    # migrated to newer managed defaults; the saved application profile is read-only.
    cfg.update(memory_policy_version=B.tinyconfig.MEMORY_POLICY_VERSION,
               chunk_policy_version=B.tinyconfig.CHUNK_POLICY_VERSION,
               generation_policy_version=B.tinyconfig.GENERATION_POLICY_VERSION)
    if args.model == 'clover':
        cfg['clover_decode_policy_version'] = 2
    B.tinyconfig.validate(cfg)
    return cfg


def response_signature(record):
    response = {key: record['response'][key] for key in ('content', 'reasoning')}
    digest = hashlib.sha256(json.dumps(response, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'response_sha256': digest,
            'counts': {key: record['native'].get(key) for key in COUNT_FIELDS}}


def geometry(info):
    for key in ('expert_slots', 'expert_total', 'expert_cache_bytes', 'expert_arena_bytes', 'max_context', 'chunk'):
        value = info.get(key)
        require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                'Missing nonnegative native geometry: ' + key)
    require(info['expert_total'] > 0 and info['max_context'] > 0 and info['chunk'] > 0 and
            info['expert_slots'] <= info['expert_total'] and
            info['expert_arena_bytes'] >= info['expert_cache_bytes'], 'Invalid native geometry')
    require(info['expert_slots'] == 0 or info['expert_cache_bytes'] > 0, 'Resident Experts need a byte payload')
    return {key: info.get(key) for key in STATIC_FIELDS}


def pooled_rate(runs, category=None, prefill=False):
    rows = [row for run in runs for row in run['records']
            if category is None or row['category'] == category]
    require(rows, 'No measured rows for pooled rate')
    count = sum(row['native']['prompt_read' if prefill else 'engine_generated'] for row in rows)
    elapsed = sum(row['native']['prompt_ms' if prefill else 'decode_ms'] for row in rows)
    require(elapsed > 0, 'Nonpositive native elapsed time')
    return count * 1000 / elapsed


def noise_band(runs, category=None, floor=.03, prefill=False):
    rates = [pooled_rate([run], category, prefill) for run in runs]
    return max(floor, max(rates) / min(rates) - 1)


def pick_screen(runs, default, minimum_gain=.03):
    baseline = [run for run in runs if run['role'] == 'baseline']
    baseline_rate = pooled_rate(baseline)
    band = noise_band(baseline, floor=minimum_gain)
    counts = sorted({run['workers'] for run in runs if run['role'] == 'candidate'})
    rates = {count: pooled_rate([run for run in runs if run['role'] == 'candidate' and run['workers'] == count])
             for count in counts}
    rates[default] = baseline_rate
    best = max(rates.values())
    eligible = [count for count, rate in rates.items()
                if rate >= best / 1.02 and rate / baseline_rate - 1 > band + 1e-12 and count < default]
    return {'candidate': min(eligible) if eligible else None, 'baseline_tok_s': baseline_rate,
            'noise_band': band, 'minimum_gain': minimum_gain,
            'rates': {str(count): rate for count, rate in rates.items()}}


def confirm_result(runs, minimum_gain=.03):
    baseline = [run for run in runs if run['role'] == 'baseline']
    candidate = [run for run in runs if run['role'] == 'candidate']
    gain = pooled_rate(candidate) / pooled_rate(baseline) - 1
    band = noise_band(baseline, floor=minimum_gain)
    groups = {}
    admitted = gain > band + 1e-12
    for category in ('generation', 'input_4096', 'input_16384'):
        delta = pooled_rate(candidate, category) / pooled_rate(baseline, category) - 1
        allowed = noise_band(baseline, category, floor=minimum_gain)
        item = {'decode_gain': delta, 'baseline_noise_band': allowed}
        if category != 'generation':
            pp_delta = pooled_rate(candidate, category, True) / pooled_rate(baseline, category, True) - 1
            pp_band = noise_band(baseline, category, floor=minimum_gain, prefill=True)
            item.update(prefill_gain=pp_delta, prefill_noise_band=pp_band)
            admitted = admitted and pp_delta >= -pp_band
        admitted = admitted and delta >= -allowed
        groups[category] = item
    # Mirrored confirmation must not reverse its overall direction in one order.
    order_gains = {}
    for order in ('ABBA', 'BAAB'):
        aa = [run for run in baseline if run['order'] == order]
        bb = [run for run in candidate if run['order'] == order]
        delta = pooled_rate(bb) / pooled_rate(aa) - 1
        order_gains[order] = delta
        admitted = admitted and delta > 0
    return {'admitted': admitted, 'pooled_decode_gain': gain, 'noise_band': band,
            'orders': order_gains, 'workloads': groups}


def main():
    local = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.local/share'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--config-dir', type=Path, default=local / 'TinyStrata-work/runtime/.tinystrata')
    parser.add_argument('--output', type=Path, default=local / 'TinyStrata-work/benchmarks' / ('probe-' + time.strftime('%Y%m%d-%H%M%S')))
    parser.add_argument('--workers', help='Optional comma-separated bounded subset; AUTO default is always included')
    parser.add_argument('--default-workers', type=int, help='Plan-only AUTO count, or an additional assertion during execution')
    parser.add_argument('--baseline-workers', type=int, default=0, help='0=AUTO, otherwise explicit baseline background workers')
    parser.add_argument('--expert-gib', type=float, help='Optional fixed Expert budget; otherwise preserve the selected profile')
    parser.add_argument('--kv-type', choices=('int8', 'fp8-mma'))
    parser.add_argument('--context', type=int)
    parser.add_argument('--chunk', type=B.chunk_option)
    parser.add_argument('--engine-env', action='append', default=[], type=B.engine_environment)
    parser.add_argument('--screen-output', type=int, default=128)
    parser.add_argument('--minimum-gain', type=float, default=.03)
    parser.add_argument('--screen-only', action='store_true', help='No recommendation without full confirmation')
    parser.add_argument('--prepare-only', action='store_true', help='Write the plan without starting any process or API')
    parser.add_argument('--startup-timeout', type=float, default=300)
    parser.add_argument('--request-timeout', type=float, default=600)
    args = parser.parse_args()
    require(0 <= args.baseline_workers <= 1024 and 16 <= args.screen_output <= 256,
            'Invalid baseline workers or screen output')
    require(args.default_workers is None or 1 <= args.default_workers <= 1024, 'Invalid default worker count')
    require(args.expert_gib is None or math.isfinite(args.expert_gib) and args.expert_gib > 0, 'Invalid Expert budget')
    require(args.context is None or args.context >= 16416, 'Context must hold16K plus32 output tokens')
    require(math.isfinite(args.minimum_gain) and .01 <= args.minimum_gain <= .5 and
            math.isfinite(args.startup_timeout) and math.isfinite(args.request_timeout) and
            args.startup_timeout > 0 and args.request_timeout > 0, 'Invalid thresholds/timeouts')
    requested = None if args.workers is None else [int(value) for value in args.workers.split(',')]
    args.output = args.output.expanduser().resolve()
    require(not args.output.is_relative_to(ROOT), 'Generated results must stay outside the source checkout')
    args.output.mkdir(parents=True, exist_ok=False)
    config_dir = args.config_dir.expanduser().resolve()
    profile_paths = sorted(config_dir.glob('config*.json'))
    require(profile_paths, 'No saved model profiles found')
    profile_hashes = {str(path): B.sha256(path) for path in profile_paths}
    original = json.loads((config_dir / B.MODELS[args.model][0]).read_text(encoding='utf-8-sig'))
    cases = json.loads((HERE / 'cases.json').read_text(encoding='utf-8'))
    require(cases['protocol'] == 'tinystrata-bench-v1' and len(cases['generation']) == 3, 'Unexpected fixed workload manifest')
    report = {'protocol': 'tinystrata-worker-probe-v1', 'standard': False, 'complete': False,
        'prepared_only': args.prepare_only, 'model': args.model, **source_provenance(),
        'profiles_before': profile_hashes, 'settings_written': False, 'recommendation': None,
        'runs': [], 'sampling': 'fixed model benchmark sampler with temperature0, seed1234, checkpoint:false'}
    signatures = {}
    long_text = {}
    matched = None
    observed_default = None
    assets = None
    def persist():
        B.save(args.output / 'result.json', report)
    def profiles_intact():
        current = {str(path): B.sha256(path) for path in sorted(config_dir.glob('config*.json'))}
        require(current == profile_hashes, 'Saved profile set or bytes changed during probe')
    def body(text, limit):
        result = B.request_body(args.model, text, limit, cases, thinking=original.get('enable_thinking', True))
        result.update(temperature=0.0, strata_checkpoint=False)
        return result
    def run_one(label, workers, role, phase, order=''):
        nonlocal matched, observed_default, assets
        folder = args.output / label
        folder.mkdir()
        cfg = configuration(original, args, folder, workers)
        run = {'label': label, 'workers': workers, 'role': role, 'phase': phase, 'order': order,
               'config': cfg, 'records': [], 'complete': False}
        report['runs'].append(run)
        B.save(folder / 'config.json', cfg)
        if assets is None:
            assets = {}
            for path in B.assets(cfg):
                require(path.is_file(), 'Model/native asset missing: ' + str(path))
                assets[str(path)] = {'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns,
                                     'sha256': B.sha256(path) if not args.prepare_only else None}
            report['assets'] = assets
        if args.prepare_only:
            persist()
            return run
        service = None
        try:
            service = OwnedService(cfg, folder, args.request_timeout)
            identity = service.ready(args.startup_timeout)
            info = service.json('/metrics')['engine']
            run['startup_engine'] = info
            current = geometry(info)
            if matched is None:
                matched = current
                report['matched_geometry'] = current
            require(current == matched, 'Expert byte capacity/native settings differ across worker choices')
            if info['expert_slots'] == info['expert_total']:
                run.update(complete=True, skipped='all_experts_gpu')
                report['not_applicable'] = 'Every Expert is GPU resident; this workload has no CPU tier to tune.'
                return run
            actual = info.get('cpu_workers')
            require(isinstance(actual, int) and actual > 0 and info.get('cpu_threads') == actual + 1,
                    'Expected native background-worker count plus one participating host')
            if workers:
                require(actual == workers, 'Actual native worker count differs from requested choice')
            elif observed_default is None:
                observed_default = actual
            else:
                require(actual == observed_default, 'Automatic worker count changed during the probe')
            if args.default_workers is not None:
                require(observed_default is None or observed_default == args.default_workers, 'Observed AUTO differs from --default-workers')
            run['actual_workers'] = actual
            if phase == 'confirm':
                require(info['max_context'] >= 16416, 'Full confirmation needs16K input plus32 output capacity')
            service.generate(body(cases['warmup_text'], 32))
            jobs = [('generation', prompt['name'], prompt['text'], args.screen_output if phase == 'screen' else 128, None)
                    for prompt in cases['generation']]
            if phase == 'confirm':
                for target in (4096, 16384):
                    if target not in long_text:
                        long_text[target], _ = B.exact_length_text(
                            lambda text: service.json('/v1/messages/count_tokens', body(text, 32))['input_tokens'],
                            cases['long_paragraph'], cases['long_question'], target)
                    jobs.append(('input_' + str(target), 'input_' + str(target), long_text[target], 32, target))
            for category, name, text, limit, target in jobs:
                request = body(text, limit)
                row = {'category': category, 'name': name, 'request': request, **service.generate(request)}
                B.validate_native(row['native'], target)
                require(row['native']['engine_generated'] <= limit, 'Native generation exceeded the cap')
                after = service.json('/metrics')['engine']
                require(geometry(after) == matched and after.get('cpu_workers') == actual and
                        after.get('cpu_threads') == actual + 1, 'Native settings/Expert allocation/workers changed within a run')
                if identity is not None:
                    require(service.vision_identity() == identity, 'Owned GPU image encoder restarted')
                row['raw_engine_after'] = after
                row['signature'] = response_signature(row)
                key = phase + '/' + name
                if key not in signatures:
                    signatures[key] = row['signature']
                require(row['signature'] == signatures[key], 'Output/reasoning hash or native counts differ: ' + key)
                run['records'].append(row)
                persist()
            run['final_engine'] = service.json('/metrics')['engine']
            require(geometry(run['final_engine']) == matched, 'Final native allocation differs from startup')
            run['complete'] = True
            print(label, 'workers', actual, 'decode', round(pooled_rate([run]), 3), 'tok/s', flush=True)
            return run
        finally:
            try:
                if service is not None:
                    service.close()
            finally:
                profiles_intact()
                persist()
    persist()
    try:
        first = run_one('screen-A1', args.baseline_workers, 'baseline', 'screen')
        if report.get('not_applicable'):
            report['complete'] = True
            return
        default = args.default_workers if args.prepare_only else first['actual_workers']
        require(default is not None, '--prepare-only needs --default-workers for a bounded plan')
        candidates = candidate_counts(default, requested)
        report['worker_candidates'] = candidates
        # Three AUTO brackets surround forward and reverse candidate sweeps.
        for number, ordered in enumerate((candidates, list(reversed(candidates))), 1):
            for count in ordered:
                run_one(f'screen-{number}-W{count}', count, 'candidate', 'screen')
            run_one(f'screen-A{number+1}', args.baseline_workers, 'baseline', 'screen')
        if args.prepare_only:
            report['planned_confirmation'] = 'Best eligible worker: fresh ABBA followed by BAAB, each with3short+4K/16K.'
            report['complete'] = True
            return
        screen = pick_screen([run for run in report['runs'] if run['phase'] == 'screen'], default, args.minimum_gain)
        report['screen'] = screen
        chosen = screen['candidate']
        if chosen is not None and not args.screen_only:
            for order in ('ABBA', 'BAAB'):
                for index, arm in enumerate(order, 1):
                    run_one(f'confirm-{order}-{arm}{index}', args.baseline_workers if arm == 'A' else chosen,
                            'baseline' if arm == 'A' else 'candidate', 'confirm', order)
            confirmation = confirm_result([run for run in report['runs'] if run['phase'] == 'confirm'], args.minimum_gain)
            report['confirmation'] = confirmation
            if confirmation['admitted']:
                report['recommendation'] = {'workers': chosen, 'native_flag': '--cpu-threads ' + str(chosen),
                    'manual_only': True, 'scope': 'Recorded model/profile/environment/Expert byte geometry only',
                    'reason': 'Fewest workers within2% of the best screen; gain exceeds bracket noise and passes full mirrored confirmation.'}
        report['complete'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        profiles_intact()
        report['profiles_unchanged'] = True
        if assets is not None:
            for name, snapshot in assets.items():
                path = Path(name)
                require(path.stat().st_size == snapshot['size'] and path.stat().st_mtime_ns == snapshot['mtime_ns'],
                        'Native/model asset changed during probe: ' + path.name)
        persist()
        recommendation = report.get('recommendation')
        print('Suggested: ' + recommendation['native_flag'] if recommendation else
              'No confirmed worker recommendation; saved settings were not changed.', flush=True)


if __name__ == '__main__':
    main()
