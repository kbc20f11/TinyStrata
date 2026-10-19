"""Switch supported model profiles without restarting the HTTP server."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time

from serve.model_profiles import ModelProfiles


class ModelControlError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class ModelBundle:
    model_id: str
    config: dict
    path: Path
    fresh: object

    @property
    def engine(self):
        return self.fresh.engine

    @property
    def vision(self):
        return self.fresh.vision


# HTTP authentication, origins, listening address, telemetry and synchronization
# stay attached to the server. These fields belong to a particular model.
MODEL_FIELDS = (
    'engine', 'vision', 'tok', 'template', 'model', 'template_tokens', 'call_format',
    'literals', 'stop_ids', 'aliases', 'sampling_defaults', 'default_enable_thinking',
    'default_max_tokens', 'fit_max_tokens', 'shared', 'shared_path', 'config_effort',
    'effort_end', 'reasoning_close_retry', 'reasoning_loop_recovery',
    'reasoning_budget_tokens', 'repeat_stop_tokens', 'anthropic_think_unasked',
    'tool_call_recovery', 'codex_compaction_cache', 'codex_thread_titles',
    'slot_save_path', 'embeddings', 'conv_log', 'rate', 'live_reqs', 'history',
    'totals', 'last_timings', 'latencies', 'last_request_at', 'status',
    'restore_after_load', 'vram_reserve', 'vram_reserve_asked', 'vision_held', 'last_image_at',
)


class ModelSwitcher:
    def __init__(self, svc, cfg: dict, config_path, api):
        self.svc, self.api = svc, api
        self.profiles = ModelProfiles(config_path, cfg)
        self.active_id = self.profiles.current_id
        self._lock = threading.RLock()
        self._requests = 0
        self._closed = False
        self._worker = None
        self._busy = False
        self._state = 'idle'
        self._phase = 'idle'
        self._target = None
        self._error = None
        self._rollback = None
        self._started = self._finished = None
        self._downloads = {}
        if svc.downloads is not None:
            self._downloads[self.active_id] = svc.downloads

    def status(self) -> dict:
        with self._lock:
            return {'state': self._state, 'phase': self._phase, 'model': self.active_id,
                    'target': self._target, 'loaded': self.svc.loaded(), 'error': self._error,
                    'context': self.context_info(),
                    'rolled_back': self._rollback, 'started_at': self._started,
                    'finished_at': self._finished, 'in_flight': self._requests,
                    'switching': self._busy}

    def _open(self):
        if self._closed or self.svc.stopping.is_set():
            raise ModelControlError(503, 'the API is stopping')

    @contextmanager
    def admit(self):
        """Lease the model from request parsing through the last SSE event.

        The generation FIFO alone is too late: tokenization and parser creation
        happen before taking it. A switch starts only with no such leases.
        """
        with self._lock:
            self._open()
            if self._busy:
                raise ModelControlError(503, 'a model switch is in progress; poll GET /v1/model/switch')
            self._requests += 1
        try:
            yield
        finally:
            with self._lock:
                self._requests -= 1

    def check_model(self, req):
        name = req.get('model')
        if name is None:
            return
        if not isinstance(name, str):
            raise ValueError('model must be a string')
        with self._lock:
            try:
                target = self.profiles.resolve(name)
            except ValueError:
                if name in self.svc.model_names():
                    return
                raise ModelControlError(404, 'model not found') from None
            if target != self.active_id:
                raise ModelControlError(409, 'model is not active; POST /v1/model/switch with its model id first')

    def context_info(self, model=None):
        """Only a loaded active model has an observed allocated KV capacity."""
        with self._lock:
            try:
                target = self.profiles.resolve(model) if model is not None else self.active_id
            except ValueError:
                if model not in self.svc.model_names():
                    raise
                target = self.active_id
            context = self.profiles.context_info(target)
            if target == self.active_id and not self._busy and self.svc.loaded():
                # reported_ctx() may return a previous READY while restarting.
                # Discovery of an allocation must use this engine's actual value.
                actual = getattr(self.svc.engine, 'max_context', 0)
                if isinstance(actual, int) and not isinstance(actual, bool) and actual > 0:
                    context['allocated_tokens'] = actual
            return context

    def catalog(self):
        with self._lock:
            items = []
            for row in self.profiles.list():
                model = row['id']
                active = model == self.active_id
                loaded = active and not self._busy and self.svc.loaded()
                value = 'loaded' if loaded else 'unloaded' if row['installed'] else 'not_installed'
                if self._busy and model == self._target:
                    value = 'loading'
                context = self.context_info(model)
                items.append({'id': model, 'arch': row['arch'], 'installed': row['installed'],
                              'object': 'model', 'active': active,
                              'status': {'value': value},
                              'architecture': {'input_modalities': ['text', 'image'] if row['vision'] else ['text'],
                                               'output_modalities': ['text']},
                              'context': context,
                              'meta': {'n_ctx': context['allocated_tokens'] or 0}})
            return items

    def start(self, model) -> dict:
        target = self.profiles.resolve(model)
        with self._lock:
            self._open()
            if self._busy:
                if target == self._target:
                    return self.status()
                raise ModelControlError(409, 'another model switch is in progress')
            if target == self.active_id and self.svc.loaded():
                self._state, self._phase = 'ready', 'ready'
                self._target, self._error, self._rollback = target, None, None
                return self.status()
            if self._requests:
                raise ModelControlError(409, 'a request is preparing, running or queued; retry after it finishes')
            if any(d.status().get('state') in ('downloading', 'verifying', 'loading')
                   for d in self._downloads.values()):
                raise ModelControlError(409, 'a model download or activation is running; wait for it first')
            # Also cover non-HTTP work such as idle unload and session actions.
            if not self.svc.fifo.acquire(blocking=False):
                raise ModelControlError(409, 'the model is busy')
            self.svc.fifo.release()
            self._busy, self._state, self._phase = True, 'switching', 'preparing'
            self._target, self._error, self._rollback = target, None, None
            self._started, self._finished = time.time(), None
            self._worker = threading.Thread(target=self._switch, args=(target,),
                                            name='model-switch', daemon=True)
            self._worker.start()
            return self.status()

    def _set_phase(self, phase):
        with self._lock:
            self._open()
            self._phase = phase

    def _prepare(self, model) -> ModelBundle:
        api, svc = self.api, self.svc
        cfg = self.profiles.prepare(model)
        profile = self.profiles.get(model)
        from serve.model_setup import resolve_executable
        exe = resolve_executable(cfg)
        if exe is None:
            raise ValueError('the selected model engine has not been built')
        want_vision = profile.arch in ('qwen35moe', 'gemma4') and cfg.get('vision_mode', 'gpu') != 'no'
        if cfg.get('vision_download_required') or (want_vision and not cfg.get('vision')):
            raise ValueError('the image projector or vision executable is missing; download or build image support first')
        if cfg.get('assistant_download_required'):
            raise ValueError('the matching Gemma QAT assistant is missing or invalid; download the model bundle first')
        if (api.gpu_list(cfg) or [0]) != (getattr(svc, 'gpu_indices', None) or [0]):
            raise ValueError('live switching requires both model profiles to use this server GPU')
        tok = api.model_tokenizer(Path(cfg['tokenizer']))
        template = api.ChatTemplate(Path(cfg['tokenizer']) / 'chat_template.jinja')
        effort = api.effort_end_args(cfg, str(exe), tok)
        engine = api.StrataEngine(str(exe), api.engine_args(cfg) + (effort or []),
                                  cwd=cfg.get('cwd'), log=cfg.get('log'), env=api.child_env(cfg), lazy=True)
        engine.silence_s = api.engine_silence_s(cfg)
        vision = None
        try:
            if cfg.get('vision'):
                vision = api.Vision(cfg['vision'],
                                    log=open(cfg['log'], 'a', encoding='utf-8') if cfg.get('log') else None,
                                    env=api.vision_env(cfg, api.child_env(cfg)), lazy=True)
            fresh = api.Service(engine, tok, template, model_name=api.default_model_name(cfg),
                                vision=vision, sampling_defaults=api.sampling_defaults_from_config(cfg),
                                fit_max_tokens=cfg.get('fit_max_tokens') is True, generation_defaults=cfg)
            fresh.set_aliases(cfg.get('aliases'))
            if cfg.get('slot_save_path'):
                fresh.slot_save_path = api.slot_save_dir(cfg['slot_save_path'], cfg.get('cwd'))
            fresh.effort_end = bool(effort)
            fresh.config_effort = api.clean_shared_defaults({'reasoning_effort': cfg['reasoning_effort']}).get('reasoning_effort') if cfg.get('reasoning_effort') is not None else None
            for key in ('reasoning_close_retry', 'tool_call_recovery', 'codex_compaction_cache', 'codex_thread_titles'):
                setattr(fresh, key, cfg.get(key) is True)
            fresh.reasoning_budget_tokens = cfg.get('reasoning_budget_tokens', 0)
            fresh.reasoning_budget({})
            fresh.repeat_stop_tokens = cfg.get('repeat_stop_tokens', api.REPEAT_STOP_TOKENS)
            if isinstance(fresh.repeat_stop_tokens, bool) or not isinstance(fresh.repeat_stop_tokens, int) or fresh.repeat_stop_tokens < 0:
                raise ValueError('repeat_stop_tokens must be a nonnegative integer')
            fresh.reasoning_loop_recovery = cfg.get('reasoning_loop_recovery', False)
            if fresh.reasoning_loop_recovery is True:
                fresh.reasoning_loop_recovery = 'recover'
            if fresh.reasoning_loop_recovery not in (False, 'stop', 'recover'):
                raise ValueError('invalid reasoning_loop_recovery')
            mode = cfg.get('anthropic_thinking', 'model')
            if mode not in ('model', 'on_request'):
                raise ValueError('invalid anthropic_thinking')
            fresh.anthropic_think_unasked = mode == 'model'
            fresh.shared_path = str(profile.shared_path)
            try:
                fresh.shared = api.clean_shared_defaults(json.loads(profile.shared_path.read_text(encoding='utf-8')))
            except (OSError, ValueError):
                fresh.shared = {}
            if getattr(svc, 'telemetry', None) is not None:
                engine.gpu_busy = lambda: (svc.telemetry.snapshot()['now'].get('gpu_util') or 0) >= 10
            return ModelBundle(model, cfg, profile.path, fresh)
        except BaseException:
            engine.close()
            if vision is not None:
                vision.shutdown()
            raise

    def _activate(self, bundle):
        self.svc.check_load_policy()
        if bundle.vision is not None:
            self._set_phase('loading_vision')
            bundle.vision.restart()
        self._set_phase('loading')
        bundle.engine.restart(tries=1)
        if bundle.vision is not None and bundle.engine.info.get('vision') != 1:
            raise ValueError('the selected engine does not support image embeddings')

    def _publish(self, bundle):
        import serve.frontend as frontend
        svc = self.svc
        with svc.status_lock:
            for name in MODEL_FIELDS:
                setattr(svc, name, getattr(bundle.fresh, name))
            svc.model_available = True
        frontend.ENV_TO_USER = bundle.config.get('env_to_user') is True
        with frontend._TOOL_ORDERS_LOCK:
            frontend._TOOL_ORDERS.clear()
        self.api.responses_api._kept_prompt_tools = (None, None)

    @staticmethod
    def _dispose(engine, vision):
        try:
            engine.close()
        finally:
            if vision is not None:
                vision.shutdown()

    def _switch(self, target):
        import serve.frontend as frontend
        svc = self.svc
        bundle = None
        old_engine, old_vision = svc.engine, svc.vision
        was_loaded, was_available = svc.loaded(), svc.model_available
        old_fields = {name: getattr(svc, name) for name in MODEL_FIELDS}
        old_id, old_downloads = self.active_id, svc.downloads
        old_env_to_user = frontend.ENV_TO_USER
        released, published = False, False
        try:
            with svc.fifo:
                try:
                    bundle = self._prepare(target)
                    with self._lock:
                        downloads = self._downloader(target)
                    self._set_phase('unloading')
                    released = True  # Even a partial unload must restore the old process.
                    if hasattr(old_engine, 'unload'):
                        old_engine.unload()
                    else:
                        old_engine.close()
                    if old_vision is not None:
                        old_vision.unload()
                    svc.model_available = False
                    self._activate(bundle)
                    # Same order as download activation: FIFO, then install_lock.
                    with svc.install_lock, self._lock:
                        self._open()
                        if bundle.config.get('model_bootstrap'):
                            bundle.config.pop('model_bootstrap')
                            self.profiles.update(target, bundle.config, save=True)
                        self._publish(bundle)
                        self.active_id = target
                        svc.downloads = downloads
                        published = True
                        self._phase = 'finishing'
                    try:
                        self._dispose(old_engine, old_vision)
                    except Exception as ex:
                        print(f'[tinystrata] old model cleanup: {ex}', flush=True)
                except Exception as ex:
                    cleanup_error = None
                    if bundle is not None and not published:
                        try:
                            self._dispose(bundle.engine, bundle.vision)
                        except Exception as stop_error:
                            cleanup_error = str(stop_error)
                    # Restore fields even if publication failed part way through.
                    with self._lock, svc.status_lock:
                        for name, value in old_fields.items():
                            setattr(svc, name, value)
                        self.active_id, svc.downloads = old_id, old_downloads
                        frontend.ENV_TO_USER = old_env_to_user
                    rollback = not released
                    # The FIFO remains held until restoration completes. Never
                    # reload the old model while a failed candidate still runs.
                    if released and not self._closed and not svc.stopping.is_set() and cleanup_error is None:
                        try:
                            if was_loaded:
                                if old_vision is not None:
                                    old_vision.restart()
                                old_engine.restart(tries=1)
                            svc.model_available = was_available
                            rollback = True
                        except Exception as restore_error:
                            cleanup_error = f'old model restoration failed: {restore_error}'
                            svc.model_available = False
                    with self._lock:
                        self._state, self._phase = 'error', 'error'
                        self._rollback = rollback
                        self._error = str(ex) + (f'; {cleanup_error}' if cleanup_error else '')
                        self._finished = time.time()
                    print(f'[tinystrata] model switch failed: {self._error}', flush=True)
        finally:
            with self._lock:
                self._busy = False
                if published:
                    self._state, self._phase = 'ready', 'ready'
                    self._finished = time.time()

    def _downloader(self, model):
        if model in self._downloads:
            return self._downloads[model]
        from serve.model_download import FixedGemma4BundleDownloader, FixedModelBundleDownloader, FixedModelDownloader, manifest_for_arch
        from serve.model_setup import model_path
        profile = self.profiles.get(model)
        cfg = profile.config
        existing = model_path(cfg)
        directory = cfg['download_dir']

        def prepared(path, projector=None, assistant=None):
            # Inactive download completion only prepares its profile. The
            # selected model and native processes are changed by start().
            extra = {'assistant': assistant, 'assistant_verified': True} if profile.arch == 'gemma4' else {}
            self.profiles.prepare(model, model_path=path, projector=projector, projector_verified=True, **extra)

        if profile.arch == 'lfm2moe':
            manager = FixedModelDownloader(directory, on_ready=prepared,
                                           existing_paths=(existing,) if existing is not None else (),
                                           _manifest=manifest_for_arch(profile.arch))
        elif profile.arch == 'gemma4':
            projector = (cfg.get('vision') or {}).get('mmproj') or cfg.get('vision_mmproj')
            assistant = cfg.get('draft_model')
            manager = FixedGemma4BundleDownloader(directory, on_ready=prepared,
                                                 existing_paths=(existing,) if existing is not None else (),
                                                 projector_paths=(Path(projector),) if projector else (),
                                                 assistant_paths=(Path(assistant),) if assistant else ())
        else:
            projector = (cfg.get('vision') or {}).get('mmproj') or cfg.get('vision_mmproj')
            manager = FixedModelBundleDownloader(directory, on_ready=prepared,
                                                 existing_paths=(existing,) if existing is not None else (),
                                                 projector_paths=(Path(projector),) if projector else ())
        self._downloads[model] = manager
        return manager

    def download(self, model=None, *, start=False, cancel=False):
        with self._lock:
            self._open()
            target = self.profiles.resolve(model) if model is not None else self.active_id
            if self._busy and (start or cancel):
                raise ModelControlError(409, 'a model switch is running')
            manager = self._downloader(target)
            result = manager.cancel() if cancel else manager.start() if start else manager.status()
            return {**result, 'model_id': target, 'active': target == self.active_id}

    def close(self):
        with self._lock:
            self._closed = True
            managers = list(self._downloads.values())
            worker = self._worker
        # Never join a download while holding the generation FIFO: its active
        # callback may itself be waiting to acquire that FIFO.
        for manager in managers:
            manager.close()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=10)
