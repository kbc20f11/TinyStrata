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
    "chunk": ("--chunk", 1, 4096, True),
}


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
    if value is None:
        return None
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
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or \
            (whole and value != int(value)) or value < lo or (hi is not None and value > hi):
        bounds = f"at least {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {bounds}, not {value!r}")
    return int(value) if whole else value


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """Return a copy with validated engine overrides; leave all other settings intact."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Expected at least one engine setting")
    checked = {key: check(key, value, cfg) for key, value in changes.items()}
    new = json.loads(json.dumps(cfg))
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
