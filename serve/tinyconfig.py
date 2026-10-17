"""Validate the subset of upstream configuration supported by TinyStrata."""
from __future__ import annotations

from pathlib import Path

from serve import config

MEMORY_POLICY_VERSION = 1
VISION_POLICY_VERSION = 1
GENERATION_POLICY_VERSION = 1
DEFAULT_MAX_TOKENS = 32768
DEFAULT_SAMPLING = {
    "temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repetition_penalty": 1.0,
}


def migrate_generation_settings(cfg: dict) -> tuple[dict, bool]:
    """Fill omitted defaults, and upgrade the old stock greedy config once.

    A saved custom value always wins. The policy marker lets a user deliberately
    select greedy after upgrading without having it reset on the next launch.
    """
    sampling = cfg.get("sampling")
    if sampling is not None and not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    sampling = dict(sampling or {})
    if cfg.get("generation_policy_version") != GENERATION_POLICY_VERSION and sampling == {"temperature": 0}:
        sampling = {}
    defaults = dict(DEFAULT_SAMPLING)
    # Preserve the upstream spelling as well as the canonical spelling.
    if sampling.get("repeat_penalty") is not None:
        defaults.pop("repetition_penalty")
    sampling = {**defaults, **sampling}
    new = {**cfg, "sampling": sampling, "generation_policy_version": GENERATION_POLICY_VERSION}
    new.setdefault("enable_thinking", True)
    new.setdefault("max_tokens", DEFAULT_MAX_TOKENS)
    return new, new != cfg


def automatic_experts(cfg: dict) -> dict:
    """Clear fixed expert allocations while keeping every other setting."""
    new = dict(cfg)
    args = cfg.get("args")
    if isinstance(args, list):
        out = []
        seen = set()
        i = 0
        while i < len(args):
            flag = args[i]
            if flag in ("--expert-vram", "--expert-slots"):
                if flag in seen or i + 1 >= len(args) or not isinstance(args[i + 1], str) or \
                        args[i + 1].startswith("--"):
                    raise ValueError(f"Duplicate or incomplete TinyStrata engine option: {flag}")
                seen.add(flag)
                i += 2
            else:
                out.append(flag)
                i += 1
        new["args"] = out
    new["memory_policy_version"] = MEMORY_POLICY_VERSION
    return new


def migrate_memory_settings(cfg: dict) -> tuple[dict, bool]:
    """Old launcher configs used fixed allocations; migrate once to available VRAM.

    Once migrated, subsequent explicit CLI expert budgets remain user choices.
    """
    if cfg.get("memory_policy_version") == MEMORY_POLICY_VERSION:
        return cfg, False
    return automatic_experts(cfg), True


def migrate_settings(cfg: dict) -> tuple[dict, bool]:
    """Retain backend preferences while removing settings for the retired web UI."""
    new, changed = migrate_memory_settings(cfg)
    new, generation_changed = migrate_generation_settings(new)
    changed = changed or generation_changed
    # The first image-capable installer wrote CPU-only encoding unconditionally.
    # Upgrade those saved configurations too: run-api.bat must not leave old
    # installs on the CPU after a GPU-capable encoder has been installed.
    if isinstance(new.get("vision"), dict) and new["vision"] and \
            new.get("vision_policy_version") != VISION_POLICY_VERSION:
        new = {**new, "vision": {**new["vision"], "gpu": True},
               "vision_policy_version": VISION_POLICY_VERSION}
        changed = True
    retired = {"open_browser", "api_monitor"}
    if retired.intersection(new):
        new = {key: value for key, value in new.items() if key not in retired}
        changed = True
    return new, changed


def validate(cfg: dict, *, allow_uninstalled: bool = False) -> None:
    if not isinstance(cfg.get("enable_thinking", True), bool):
        raise ValueError("enable_thinking must be true or false")
    max_tokens = cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive whole number")
    if cfg.get("arch", "qwen35moe") != "qwen35moe":
        raise ValueError("TinyStrata currently supports Qwen3.6 (qwen35moe) only")
    for key in ("images", "replicas", "layer_split", "vram_elastic", "slot_save_path",
                "expert_profile_save", "split_skip_if_fits"):
        if cfg.get(key):
            raise ValueError(f"{key} is not supported by TinyStrata's Qwen3.6 engine")
    vision = cfg.get("vision")
    if vision:
        if not isinstance(vision, dict) or not all(isinstance(vision.get(key), str) and vision[key]
                                                for key in ("exe", "mmproj", "model")):
            raise ValueError("vision requires the image encoder, mmproj and model paths")
        if "gpu" in vision and not isinstance(vision["gpu"], bool):
            raise ValueError("vision.gpu must be true or false")
        if vision.get("elastic"):
            raise ValueError("TinyStrata keeps its image encoder resident before allocating the Expert cache")
    if cfg.get("parallel", 1) != 1:
        raise ValueError("TinyStrata currently runs one request at a time")
    if not cfg.get("exe") and not allow_uninstalled:
        raise ValueError("Set the TinyStrata executable with run.py --exe PATH")
    args = cfg.get("args")
    if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
        raise ValueError("args must be a list of engine argument strings")
    values = {"--model", "--max-context", "--kv-type", "--chunk", "--mtp", "--expert-vram",
              "--expert-slots", "--cpu-threads", "--hybrid-rows", "--vram-reserve-mib"}
    switches = {"--no-graph", "--no-warmup", "--hybrid-graph", "--no-hybrid-graph", "--vision"}
    seen = set()
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in seen:
            raise ValueError(f"Duplicate engine option: {flag}")
        seen.add(flag)
        if flag in switches:
            i += 1
        elif flag in values and i + 1 < len(args) and not args[i + 1].startswith("--"):
            i += 2
        else:
            raise ValueError(f"Unsupported or incomplete TinyStrata engine option: {flag}")
    if "--model" not in seen:
        raise ValueError("Choose a GGUF model with run.py --model PATH")
    for key in ("kv_type", "max_context", "expert_vram", "vram_reserve_mib", "mtp", "chunk"):
        value = config.value_of(cfg, key)
        if value is not None:
            config.check(key, value, cfg)


def check_files(cfg: dict, *, allow_uninstalled: bool = False) -> None:
    base = Path(cfg.get("cwd") or ".")
    for label, name in (("executable", cfg.get("exe")),
                        ("model", cfg["args"][cfg["args"].index("--model") + 1])):
        if not name:
            if allow_uninstalled:
                continue
            raise ValueError(f"TinyStrata {label} is not configured")
        path = Path(name)
        if not path.is_absolute():
            path = base / path
        if not path.is_file() and not (allow_uninstalled and not path.exists()):
            raise ValueError(f"TinyStrata {label} not found: {path}")
    for key in ("exe", "mmproj", "model") if cfg.get("vision") else ():
        path = Path(cfg["vision"][key])
        path = path if path.is_absolute() else base / path
        if not path.is_file():
            raise ValueError(f"TinyStrata vision {key} not found: {path}")
