"""Pure policy helpers for opt-in native Windows Intel text routes.

The normal Qwen API/UI, lifecycle and app-local runtime have live checks. Tiel
has initial normal-thinking text/code checks; wider workload coverage remains
separate. Profiles request settings, not hardware detection or approval of
the actual selected GPU. Native device/driver and pinned GGUF validation remain
separate gates. No helper loads a DLL/model, queries hardware, or writes state.
"""
from __future__ import annotations

from copy import deepcopy
import os
from collections.abc import Mapping


QWEN_MODEL = "qwen3.6-35b-a3b"
B390_PROFILE = "b390-qwen36"
TIEL_MODEL = "tiel-coder-35b-a3b"
TIEL_PROFILE = "b390-tiel-coder"
MODEL_PROFILES = {QWEN_MODEL: B390_PROFILE, TIEL_MODEL: TIEL_PROFILE}
MAX_CONTEXT = 32768
_MODEL_NAMES = {QWEN_MODEL: QWEN_MODEL, "qwen36": QWEN_MODEL, "qwen3.6": QWEN_MODEL,
                "qwen35moe": QWEN_MODEL, TIEL_MODEL: TIEL_MODEL, "tiel": TIEL_MODEL, "tiel-coder": TIEL_MODEL}
_VALUES = {"--model", "--max-context", "--kv-type", "--chunk", "--mtp",
           "--device", "--weight-memory", "--draft-vocab"}
_OPTIMIZATIONS = ("--prefill-mkl", "--decode-dp4a", "--verify-xmx", "--prefill-int8-xmx",
                  "--vocab-xmx", "--expert-xmx-layout", "--device-state",
                  "--dense-xmx-layout", "--dense-split-k")
_SWITCHES = {*_OPTIMIZATIONS, "--mtp-stochastic", "--verify-graph", "--no-graph", "--profile", "--experimental-tiel"}


def canonical_model(cfg: dict) -> str:
    """Check the resolved model identity, not just the shared qwen35moe arch.

    An omitted legacy name denotes Qwen3.6. A caller must still check the actual
    GGUF header; an API alias alone cannot establish that its weights match.
    """
    name = cfg.get("model_name")
    if cfg.get("arch", "qwen35moe") != "qwen35moe" or (name is not None and not isinstance(name, str)):
        raise ValueError("Intel text routes support pinned Qwen3.6 and Tiel Coder identities only")
    selected = QWEN_MODEL if name is None else _MODEL_NAMES.get(name.strip().lower())
    if selected is None:
        raise ValueError("Intel text routes support pinned Qwen3.6 and Tiel Coder identities only")
    return selected


def supports_model(cfg: dict) -> bool:
    try:
        canonical_model(cfg)
        return True
    except ValueError:
        return False


def _profile(cfg):
    profile = cfg.get("intel_profile", "generic")
    if profile not in ("generic", B390_PROFILE, TIEL_PROFILE):
        raise ValueError("intel_profile must be generic, b390-qwen36 or b390-tiel-coder (requested profile only)")
    model = canonical_model(cfg)
    if ((profile == "generic" and model == TIEL_MODEL) or
            (profile != "generic" and MODEL_PROFILES[model] != profile)):
        raise ValueError("Intel profile must match its pinned model: Qwen3.6/b390-qwen36 or Tiel/b390-tiel-coder")
    return profile


def _options(cfg):
    args = cfg.get("args", [])
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("Intel args must be a list of engine argument strings")
    values, switches, seen = {}, set(), set()
    index = 0
    while index < len(args):
        flag = args[index]
        if flag in seen:
            raise ValueError(f"Duplicate Intel engine option: {flag}")
        seen.add(flag)
        if flag in _SWITCHES:
            switches.add(flag)
            index += 1
        elif flag in _VALUES and index + 1 < len(args) and not args[index + 1].startswith("--"):
            values[flag] = args[index + 1]
            index += 2
        else:
            raise ValueError(f"Unsupported or incomplete Intel engine option: {flag}")
    return values, switches


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be a whole number from {low} to {high}")
    text = str(value)
    if not text or not text.isascii() or not text.isdigit() or not low <= int(text) <= high:
        raise ValueError(f"{name} must be a whole number from {low} to {high}")
    return int(text)


def context_tokens(cfg: dict) -> int:
    """Resolve UMA auto without discrete VRAM guesses; native budget checks follow.

    Keep the saved ``auto`` spelling intact. The requested B390 profile uses
    the tested 8192 context; generic uses 4096. Never silently clamp an explicit
    context or infer capacity by adding shared GPU memory to system RAM.
    """
    values, _ = _options(cfg)
    profile = _profile(cfg)
    value = values.get("--max-context", "auto")
    return (8192 if profile in MODEL_PROFILES.values() else 4096) if value == "auto" else _integer(value, "Intel context", 16, MAX_CONTEXT)


def _chunk(cfg, values):
    value = values.get("--chunk", "auto")
    if value in ("auto", "0"):
        return 1024 if _profile(cfg) in MODEL_PROFILES.values() else 128
    return _integer(value, "Intel chunk", 1, 1024)


def _kv(values):
    value = values.get("--kv-type", "auto")
    if value not in ("auto", "int8", "fp32"):
        raise ValueError("Intel KV supports auto (actual INT8), int8 or explicit fp32")
    return "int8" if value == "auto" else value


def _profile_arguments(cfg):
    """Derive launch-only options without persisting them as user overrides."""
    args = list(cfg.get("args", []))
    values, switches = _options(cfg)
    selected = _profile(cfg)
    if "--mtp" not in values:
        _put(args, "--mtp", "3" if selected in MODEL_PROFILES.values() else "0")
    if selected in MODEL_PROFILES.values():
        for flag in _OPTIMIZATIONS:
            if flag not in switches:
                args.append(flag)
        values, switches = _options({"args": args})
        if _integer(values["--mtp"], "Intel MTP depth", 0, 7):
            if "--mtp-stochastic" not in switches:
                args.append("--mtp-stochastic")
            if "--draft-vocab" not in values:
                _put(args, "--draft-vocab", 2048)
    if canonical_model(cfg) == TIEL_MODEL and "--experimental-tiel" not in switches:
        args.append("--experimental-tiel")
    return args


def validate_config(cfg: dict, *, allow_uninstalled: bool = False) -> None:
    """Reject unsupported settings before any native launch or model unload.

    This checks configuration only, not file existence, runtime manifests,
    actual hardware, GGUF geometry/checksum, or live memory availability.
    """
    if cfg.get("backend") not in (None, "sycl"):
        raise ValueError("The Intel helper requires an explicitly selected sycl backend")
    if not supports_model(cfg):
        raise ValueError("Intel currently supports pinned Qwen3.6 and Tiel Coder text models only")
    profile = _profile(cfg)
    values, switches = _options({"args": _profile_arguments(cfg)})
    if not allow_uninstalled and (not isinstance(cfg.get("exe"), str) or not cfg["exe"]):
        raise ValueError("Set the Intel executable before starting the model")
    if not allow_uninstalled and not values.get("--model"):
        raise ValueError("Choose the selected model's pinned GGUF before starting the Intel model")
    if "--model" in values and not values["--model"]:
        raise ValueError("The Intel model path must not be empty")
    if "--experimental-tiel" in switches and canonical_model(cfg) != TIEL_MODEL:
        raise ValueError("--experimental-tiel is specific to the pinned Tiel profile")
    if cfg.get("vision") or cfg.get("images") or cfg.get("vision_mode") not in (None, "no", "none"):
        raise ValueError("Intel images are unsupported; select vision_mode=no")
    for field in ("layer_split", "vram_elastic", "slot_save_path", "expert_profile_save", "split_skip_if_fits",
                  "expert_gpu_percent", "expert_gpu_count", "expert_vram", "expert_slots", "hybrid_rows", "hip_ordinal"):
        if field in cfg and cfg[field] is not None and cfg[field] is not False and cfg[field] != "":
            raise ValueError(f"Intel does not support {field}")
    for field in ("min_free_vram_mib", "expert_profile_save_every"):
        if cfg.get(field) not in (None, 0):
            raise ValueError(f"Intel UMA does not support {field}")
    if cfg.get("effort_position") not in (None, "start"):
        raise ValueError("Intel does not support effort_position=end or tail-role tokens")
    for field in ("parallel", "replicas"):
        value = cfg.get(field, 1)
        if type(value) is not int or value != 1:
            raise ValueError("Intel supports one resident model and one request at a time")
    for field in cfg:
        if field.startswith("conversation_cache_") and cfg[field] is not None:
            raise ValueError("Intel persistent conversation snapshots are unsupported")
    if not isinstance(cfg.get("env", {}), Mapping):
        raise ValueError("Intel env must be a mapping")
    directories = cfg.get("lib_dirs", [])
    if not isinstance(directories, list) or not all(isinstance(path, str) and path for path in directories):
        raise ValueError("Intel lib_dirs must be a list of runtime directory strings")
    if cfg.get("gpu") is not None:
        gpu = _integer(cfg["gpu"], "Intel device index", 0, 1024)
        if "--device" in values and _integer(values["--device"], "Intel device index", 0, 1024) != gpu:
            raise ValueError("Intel gpu and --device select different devices")
    if "--device" in values:
        _integer(values["--device"], "Intel device index", 0, 1024)
    context_tokens(cfg)
    chunk = _chunk(cfg, values)
    kv = _kv(values)
    mtp = _integer(values.get("--mtp", "0"), "Intel MTP depth", 0, 7)
    if mtp and chunk < max(4, mtp + 1):
        raise ValueError("Intel MTP needs chunk >= max(4, depth+1)")
    if values.get("--weight-memory", "host") not in ("host", "shared", "device"):
        raise ValueError("Intel weight-memory must be host, shared or device")
    optimizations = switches.intersection({*_OPTIMIZATIONS, "--mtp-stochastic", "--verify-graph"})
    if profile not in MODEL_PROFILES.values() and (optimizations or values.get("--draft-vocab", "0") != "0"):
        raise ValueError("Intel optimized flags require the explicitly requested model-specific B390 profile")
    if "--mtp-stochastic" in switches and not mtp:
        raise ValueError("Intel --mtp-stochastic requires --mtp >= 1")
    if "--verify-graph" in switches and (not mtp or kv != "int8"):
        raise ValueError("Intel --verify-graph requires MTP and actual INT8 KV")
    for flag in ("--verify-xmx", "--prefill-int8-xmx", "--vocab-xmx", "--expert-xmx-layout", "--dense-xmx-layout"):
        if flag in switches and "--decode-dp4a" not in switches:
            raise ValueError(f"Intel {flag} requires --decode-dp4a")
    if "--dense-xmx-layout" in switches and "--prefill-mkl" not in switches:
        raise ValueError("Intel --dense-xmx-layout requires --prefill-mkl")
    if "--dense-split-k" in switches and "--dense-xmx-layout" not in switches:
        raise ValueError("Intel --dense-split-k requires --dense-xmx-layout")
    draft = _integer(values.get("--draft-vocab", "0"), "Intel draft vocabulary", 0, 4096)
    if draft and (draft < 64 or not mtp or not {"--mtp-stochastic", "--vocab-xmx"}.issubset(switches)):
        raise ValueError("Intel --draft-vocab requires 64..4096, MTP, --mtp-stochastic and --vocab-xmx")


def _put(args, flag, value):
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args.extend((flag, str(value)))


def configure_backend(cfg: dict, *, profile: str | None = None, explicit_kv: str | None = None) -> dict:
    """Return an independent explicit Intel configuration; preserve user policy.

    Persist only the profile and core defaults. Optimizations derived from the
    profile are added transiently by engine_args, so later settings updates
    from MTP3 to MTP0 cannot retain stale stochastic/draft flags. User-written
    flags remain explicit. Never retune sampling/thinking/output/auth/history.
    Actual device approval is separate from requesting this profile.
    """
    new = deepcopy(cfg)
    previous_backend = new.get("backend")
    new["backend"] = "sycl"
    if previous_backend not in (None, "sycl"):
        new.pop("gpu", None)  # Backend ordinals refer to different device lists.
        new.pop("hip_ordinal", None)
    if profile is not None:
        new["intel_profile"] = profile
    selected = _profile(new)
    new.setdefault("vision_mode", "no")
    args = new.setdefault("args", [])
    values, _ = _options(new)
    for flag, default in (("--max-context", "auto"), ("--kv-type", "auto"), ("--chunk", "auto"),
                          ("--mtp", "3" if selected in MODEL_PROFILES.values() else "0"), ("--weight-memory", "host")):
        if flag not in values:
            _put(args, flag, default)
    if explicit_kv is not None:
        _put(args, "--kv-type", explicit_kv)
    validate_config(new, allow_uninstalled=True)
    return new


def engine_args(cfg: dict, *, allow_uninstalled: bool = False) -> list[str]:
    """Produce native options; keep saved auto values distinct from actual values.

    StrataEngine supplies --serve. No CUDA/HIP placement, snapshot or multi-GPU
    flags are forwarded. Caller validates files and native device separately.
    """
    if cfg.get("backend") != "sycl":
        raise ValueError("Intel engine arguments require an explicitly selected sycl backend")
    new = configure_backend(cfg)
    new["args"] = _profile_arguments(new)
    validate_config(new, allow_uninstalled=allow_uninstalled)
    values, _ = _options(new)
    args = list(new["args"])
    _put(args, "--max-context", context_tokens(new))
    _put(args, "--kv-type", _kv(values))
    _put(args, "--chunk", _chunk(new, values))
    if "--device" not in values:
        _put(args, "--device", new.get("gpu", 0))
    return args


def runtime_environment(cfg: dict, inherited: Mapping | None = None) -> dict[str, str]:
    """Return the child environment; never modify os.environ or the input dict.

    Windows variable names are case-insensitive. User controls from either the
    inherited environment or cfg.env win over selector/filter defaults. Only
    the child's PATH is prefixed; no driver installation or SDK/global edits.
    """
    if cfg.get("backend") != "sycl":
        raise ValueError("Intel child environment requires an explicitly selected sycl backend")
    if not isinstance(cfg.get("env", {}), Mapping):
        raise ValueError("Intel env must be a mapping")
    directories = cfg.get("lib_dirs", [])
    if not isinstance(directories, list) or not all(isinstance(path, str) and path for path in directories):
        raise ValueError("Intel lib_dirs must be a list of runtime directory strings")
    environment = {str(key): str(value) for key, value in (os.environ if inherited is None else inherited).items()}
    def key_for(name):
        return next((key for key in environment if key.upper() == name.upper()), name)
    for key, value in cfg.get("env", {}).items():
        name = str(key)
        environment[key_for(name)] = str(value)
    for name, default in (("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu"), ("UR_LOADER_PRELOAD_FILTER", "1")):
        environment.setdefault(key_for(name), default)
    if directories:
        key = key_for("PATH")
        environment[key] = ";".join([*dict.fromkeys(directories), *([environment[key]] if environment.get(key) else [])])
    return environment
