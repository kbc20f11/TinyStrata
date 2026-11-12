"""Load API server configuration and validate launcher engine overrides."""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

CHUNK_POLICY_VERSION = 1

ENGINE_OPTIONS = {
    "kv_type": ("--kv-type", ("auto", "fp16", "fp32", "fp8", "fp8-mma", "int8")),
    "max_context": ("--max-context", 16, 262144, True),
    "expert_vram": ("--expert-vram", 0, None, False),
    "vram_reserve_mib": ("--vram-reserve-mib", 0, None, True),
    "mtp": ("--mtp", 0, 7, True),
    "mtp_auto": ("--mtp-auto", 0, 1, True),
    "lookup_draft": ("--lookup-draft", 0, 7, True),
    "clover_draft": ("--clover-draft", 0, 31, True),
    "clover_lookup": ("--clover-lookup", 0, 7, True),
    "clover_mode": ("--clover-mode", ("causal", "self-spec", "diffusion")),
    "clover_block_size": ("--clover-block-size", 1, 32, True),
    "clover_threshold": ("--clover-threshold", 0, 1, False),
    "clover_accept": ("--clover-accept", ("exact", "threshold")),
    "clover_accept_ratio": ("--clover-accept-ratio", 0, 1, False),
    "clover_accept_delta": ("--clover-accept-delta", 0, 1, False),
    "chunk": ("--chunk", 1, 4096, True),
    "conversation_cache_mib": ("--conversation-cache-mib", 0, 65536, True),
    "conversation_cache_disk_mib": ("--conversation-cache-disk-mib", 0, 65536, True),
    "conversation_cache_slots": ("--conversation-cache-slots", 1, 16, True),
    "conversation_cache_checkpoints": ("--conversation-cache-checkpoints", 1, 8, True),
    "conversation_cache_min_tokens": ("--conversation-cache-min-tokens", 0, 262144, True),
}
CONVERSATION_CACHE_KEYS = tuple(key for key in ENGINE_OPTIONS if key.startswith("conversation_cache_"))


def value_of(cfg: dict, key: str):
    flag = ENGINE_OPTIONS[key][0]
    args = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if flag not in args[:-1]:
        return None
    value = str(args[args.index(flag) + 1])
    try:
        return int(value) if value.isdigit() else float(value)
    except ValueError:
        return value


def check(key: str, value, cfg: dict):
    """Validate one CLI engine override; None restores its engine default."""
    if key not in ENGINE_OPTIONS:
        raise ValueError(f"Unknown engine setting: {key!r}")
    if key in CONVERSATION_CACHE_KEYS and cfg.get("arch", "qwen35moe") not in ("gemma4", "lfm2moe"):
        raise ValueError(f"{key}: conversation parking settings are supported only by Gemma 4 and LFM2.5")
    if value is None:
        return None
    if key in ("mtp_auto", "lookup_draft"):
        arch = cfg.get("arch")
        if arch not in (None, "qwen35moe", "gemma4"):
            raise ValueError(f"{key}: this option requires Qwen35MoE or Gemma 4")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key}: expected a whole number")
    if key == "clover_lookup":
        if cfg.get("arch") not in (None, "clover_moe"):
            raise ValueError("clover_lookup: this option requires a Clover model")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("clover_lookup: expected a whole number")
    spec = ENGINE_OPTIONS[key]
    if isinstance(spec[1], tuple):
        if value not in spec[1]:
            raise ValueError(f"{key}: expected one of {', '.join(spec[1])}, not {value!r}")
        if key == "kv_type" and cfg.get("backend") == "hip" and value not in ("auto", "int8", "fp8", "fp8-mma"):
            raise ValueError("The AMD HIP route supports auto, int8 or native fp8")
        return value
    if key in ("max_context", "chunk") and value == "auto":
        return value
    if key == "chunk" and not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0:
        return "auto"
    _, lo, hi, whole = spec
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            value < lo or (hi is not None and value > hi) or not math.isfinite(value) or \
            (whole and value != int(value)):
        bounds = f"at least {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {bounds}, not {value!r}")
    if key == "clover_block_size" and 32 % int(value):
        raise ValueError("clover_block_size: expected one of 1, 2, 4, 8, 16, 32")
    return int(value) if whole else value


def clover_mode(cfg: dict) -> str:
    """Resolve fresh defaults while retaining the legacy explicit draft option."""
    explicit = value_of(cfg, "clover_mode")
    if explicit is not None:
        return check("clover_mode", explicit, cfg)
    draft = value_of(cfg, "clover_draft")
    if draft is not None:
        draft = check("clover_draft", draft, cfg)
    return "self-spec" if draft is not None and draft > 0 else "causal"


CLOVER_REQUEST_KEYS = ("clover_mode", "clover_draft", "clover_lookup", "clover_block_size", "clover_threshold",
                       "clover_accept", "clover_accept_ratio", "clover_accept_delta")
SPECULATION_REQUEST_KEYS = ("mtp_auto", "lookup_draft")


def speculation_request_options(req: dict | None) -> dict:
    """Explicit target-verified speculation settings; null inherits the resident default."""
    return {key: check(key, (req or {})[key], {}) for key in SPECULATION_REQUEST_KEYS
            if (req or {}).get(key) is not None}


def clover_request_options(req: dict | None) -> dict:
    """Validate explicit JSON generation overrides without coercing strings or booleans.

    An omitted or null field keeps the resident engine's setting. Mode/draft
    compatibility also depends on that setting and is checked by Service.
    """
    out = {}
    for key in CLOVER_REQUEST_KEYS:
        value = (req or {}).get(key)
        if value is None:
            continue
        if key in ("clover_draft", "clover_lookup", "clover_block_size") and \
                (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{key}: expected a whole number")
        try:
            out[key] = check(key, value, {})
        except OverflowError:
            raise ValueError(f"{key}: expected a finite number in [0,1]") from None
    return out


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """Return a copy with validated engine overrides; leave all other settings intact."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Expected at least one engine setting")
    checked = {key: check(key, value, cfg) for key, value in changes.items()}
    new = json.loads(json.dumps(cfg))
    if "expert_vram" in checked and value_of(cfg, "expert_vram") != checked["expert_vram"]:
        # A direct CLI/API GiB override is a new manual budget. The dedicated
        # percentage control adds its descriptor after applying that override.
        new.pop("expert_gpu_percent", None)
        new.pop("expert_gpu_count", None)
        if "expert_gpu_percent" in cfg or "expert_gpu_count" in cfg:
            args = list(new.get("args") or [])
            filtered = []
            index = 0
            while index < len(args):
                if args[index] == "--expert-slots":
                    if index + 1 >= len(args) or not isinstance(args[index + 1], str) or args[index + 1].startswith("--"):
                        raise ValueError("Incomplete percentage-owned --expert-slots option")
                    index += 2
                else:
                    filtered.append(args[index])
                    index += 1
            new["args"] = filtered
    if "chunk" in checked:
        # A saved explicit choice, including 512, must survive later migrations.
        new["chunk_policy_version"] = CHUNK_POLICY_VERSION
    changed = []
    for key, value in checked.items():
        if value_of(new, key) == value:
            continue
        args = list(new.get("args") or [])
        flag = ENGINE_OPTIONS[key][0]
        if flag in args[:-1]:
            index = args.index(flag)
            if value is None:
                del args[index:index + 2]
            else:
                args[index + 1] = str(value)
        elif value is not None:
            args.extend((flag, str(value)))
        new["args"] = args
        changed.append(key)
    return new, changed


def save(path: str | Path, cfg: dict) -> Path | None:
    """Replace a config atomically, retaining an existing config as <name>.bak."""
    path = Path(path)
    backup = path.with_name(path.name + ".bak") if path.exists() else None
    if backup is not None:
        shutil.copyfile(path, backup)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return backup


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
