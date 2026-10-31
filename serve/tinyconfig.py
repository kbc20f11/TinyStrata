"""Validate the subset of upstream configuration supported by TinyStrata."""
from __future__ import annotations

import os
from pathlib import Path

from serve import config

MEMORY_POLICY_VERSION = 1
VISION_POLICY_VERSION = 1
GENERATION_POLICY_VERSION = 1
CHUNK_POLICY_VERSION = config.CHUNK_POLICY_VERSION
DEFAULT_MAX_TOKENS = 32768
DEFAULT_SAMPLING = {
    "temperature": 0.7, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repetition_penalty": 1.0,
}
LFM2_SAMPLING = {"temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0,
                 "presence_penalty": 0.0, "repetition_penalty": 1.05}
GEMMA4_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0,
                   "presence_penalty": 0.0, "repetition_penalty": 1.0}
CLOVER_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 64, "min_p": 0.0,
                   "presence_penalty": 0.9, "frequency_penalty": 0.0, "repetition_penalty": 1.05}
MODEL_NAMES = {"qwen35moe": "qwen3.6-35b-a3b", "lfm2moe": "lfm2.5-24b-a2b",
               "gemma4": "gemma-4-26b-a4b-it", "clover_moe": "clover-35b-a3b-preview"}


def experimental_clover_vision(cfg: dict) -> bool:
    """Permit the local Vision experiment only through an explicit process opt-in.

    This does not change Clover's text-only profile or its default image mode.
    Saved experimental settings cannot enable it without the environment flag.
    """
    return cfg.get("arch") == "clover_moe" and os.environ.get("TINYSTRATA_CLOVER_VISION") == "1"


def migrate_generation_settings(cfg: dict) -> tuple[dict, bool]:
    """Fill omitted defaults, and upgrade the old stock greedy config once.

    A saved custom value always wins. The policy marker lets a user deliberately
    select greedy after upgrading without having it reset on the next launch.
    """
    sampling = cfg.get("sampling")
    if sampling is not None and not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    sampling = dict(sampling or {})
    if cfg.get("arch", "qwen35moe") == "qwen35moe" and cfg.get("generation_policy_version") != GENERATION_POLICY_VERSION and \
            sampling == {"temperature": 0}:
        sampling = {}
    defaults = dict({"lfm2moe": LFM2_SAMPLING, "gemma4": GEMMA4_SAMPLING,
                     "clover_moe": CLOVER_SAMPLING}.get(cfg.get("arch"), DEFAULT_SAMPLING))
    # Preserve the upstream spelling as well as the canonical spelling.
    if sampling.get("repeat_penalty") is not None:
        defaults.pop("repetition_penalty")
    sampling = {**defaults, **sampling}
    new = {**cfg, "sampling": sampling, "generation_policy_version": GENERATION_POLICY_VERSION}
    new.setdefault("enable_thinking", True)
    new.setdefault("max_tokens", 24576 if cfg.get("arch") == "clover_moe" else DEFAULT_MAX_TOKENS)
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


def migrate_chunk_settings(cfg: dict) -> tuple[dict, bool]:
    """Upgrade the old stock 512-token chunk once; retain custom chunk sizes.

    Old TinyStrata launcher configs did not distinguish a saved manual 512 from
    the stock value. Upgrade that value only for existing TinyStrata memory-policy
    configs. The marker preserves all explicit chunk choices made after upgrade.
    """
    if cfg.get("chunk_policy_version") == CHUNK_POLICY_VERSION:
        return cfg, False
    value = config.value_of(cfg, "chunk")
    stock = cfg.get("arch") == "qwen35moe" and cfg.get("memory_policy_version") == MEMORY_POLICY_VERSION
    if value is None or value == 0 or (value == 512 and stock):
        new, _ = config.apply(cfg, {"chunk": "auto"})
    else:
        new = {**cfg, "chunk_policy_version": CHUNK_POLICY_VERSION}
    return new, new != cfg


def check_vision_options(cfg: dict) -> str:
    """Validate launcher preferences independently of installed image assets."""
    text_only = cfg.get("arch") in ("lfm2moe", "clover_moe")
    mode = cfg.get("vision_mode", "no" if text_only else "gpu")
    if mode not in ("gpu", "cpu", "no"):
        raise ValueError("vision_mode must be gpu, cpu or no")
    if "vision_tokens" in cfg:
        tokens = cfg["vision_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("vision_tokens must be a positive whole number")
    if text_only and not experimental_clover_vision(cfg) and (mode != "no" or cfg.get("vision")):
        raise ValueError(f"{MODEL_NAMES[cfg['arch']]} is text-only; use --vision no")
    return mode


def disable_vision(cfg: dict) -> dict:
    """Remove active image configuration while retaining paths for re-enabling."""
    new = {**cfg, "vision_mode": "no"}
    previous = cfg.get("vision")
    if isinstance(previous, dict):
        if previous.get("exe") and not new.get("vision_exe"):
            new["vision_exe"] = previous["exe"]
        if previous.get("mmproj") and not new.get("vision_mmproj"):
            new["vision_mmproj"] = previous["mmproj"]
        tokens = previous.get("max_tokens")
        if "vision_tokens" not in new and isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0:
            new["vision_tokens"] = tokens
    if isinstance(new.get("args"), list):
        new["args"] = [arg for arg in new["args"] if arg != "--vision"]
    new.pop("vision", None)
    new.pop("vision_download_required", None)
    return new


def migrate_settings(cfg: dict) -> tuple[dict, bool]:
    """Retain backend preferences while removing settings for the retired web UI."""
    # Check the existing memory-policy marker before the memory migration adds it.
    new, changed = migrate_chunk_settings(cfg)
    new, memory_changed = migrate_memory_settings(new)
    changed = changed or memory_changed
    new, generation_changed = migrate_generation_settings(new)
    changed = changed or generation_changed
    # Old configs default to GPU encoding; explicit CPU/off preferences survive
    # both migration and model preparation on every subsequent launch.
    mode = check_vision_options(new)
    new = {**new, "vision_mode": mode}
    if mode == "no":
        new = disable_vision(new)
    elif isinstance(new.get("vision"), dict) and new["vision"]:
        new = {**new, "vision": {**new["vision"], "gpu": mode == "gpu"},
               "vision_policy_version": VISION_POLICY_VERSION}
    retired = {"open_browser", "api_monitor"}
    if retired.intersection(new):
        new = {key: value for key, value in new.items() if key not in retired}
        changed = True
    return new, changed or new != cfg


def validate(cfg: dict, *, allow_uninstalled: bool = False) -> None:
    check_vision_options(cfg)
    if cfg.get("backend") not in (None, "cuda", "hip"):
        raise ValueError("backend must be cuda or hip")
    if not isinstance(cfg.get("enable_thinking", True), bool):
        raise ValueError("enable_thinking must be true or false")
    max_tokens = cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive whole number")
    arch = cfg.get("arch", "qwen35moe")
    if arch not in MODEL_NAMES:
        raise ValueError("TinyStrata supports qwen35moe, lfm2moe, gemma4 and clover_moe models")
    if arch == "lfm2moe" and cfg.get("enable_thinking", True) is False:
        raise ValueError("LFM2.5-24B-A2B always uses thinking")
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
    values = {"--model", "--draft-model", "--max-context", "--kv-type", "--chunk", "--mtp", "--expert-vram",
              "--expert-slots", "--cpu-threads", "--hybrid-rows", "--vram-reserve-mib", "--clover-draft",
              "--clover-mode", "--clover-block-size", "--clover-threshold"}
    switches = {"--no-graph", "--no-warmup", "--hybrid-graph", "--no-hybrid-graph", "--vision",
                "--kv-grow", "--kv-whole", "--no-kv-grow"}
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
    if len(seen.intersection({"--kv-grow", "--kv-whole", "--no-kv-grow"})) > 1:
        raise ValueError("Choose only one KV allocation mode: --kv-grow or --kv-whole")
    for key in ("kv_type", "max_context", "expert_vram", "vram_reserve_mib", "mtp", "chunk", "clover_draft",
                "clover_mode", "clover_block_size", "clover_threshold"):
        value = config.value_of(cfg, key)
        if value is not None:
            config.check(key, value, cfg)
    kv = config.value_of(cfg, "kv_type")
    clover_options = seen.intersection({"--clover-draft", "--clover-mode", "--clover-block-size", "--clover-threshold"})
    if arch != "clover_moe" and clover_options:
        raise ValueError("Clover-only engine options: " + ", ".join(sorted(clover_options)))
    if arch == "lfm2moe":
        qwen_only = seen.intersection({"--hybrid-graph", "--no-hybrid-graph", "--hybrid-rows",
                                       "--expert-slots", "--no-warmup"})
        if qwen_only:
            raise ValueError("Unsupported LFM2.5 engine option: " + ", ".join(sorted(qwen_only)))
        if "--vision" in seen:
            raise ValueError("LFM2.5-24B-A2B is text-only; remove --vision")
        if kv not in (None, "auto", "fp16", "int8", "fp8", "fp8-mma"):
            raise ValueError("LFM2.5 supports --kv-type auto, fp16, int8, fp8 or fp8-mma")
        context = config.value_of(cfg, "max_context")
        if context not in (None, "auto") and context > 131072:
            raise ValueError("LFM2.5 supports at most 131072 context tokens")
        if config.value_of(cfg, "mtp") not in (None, 0):
            raise ValueError("LFM2.5 has no published MTP head; --mtp must be 0")
    elif arch == "gemma4":
        unsupported = seen.intersection({"--hybrid-graph", "--no-hybrid-graph", "--hybrid-rows",
                                         "--expert-slots", "--no-warmup"})
        if unsupported:
            raise ValueError("Unsupported Gemma 4 engine option: " + ", ".join(sorted(unsupported)))
        if kv not in (None, "auto", "fp16", "int8", "fp8", "fp8-mma"):
            raise ValueError("Gemma 4 supports --kv-type auto, fp16, int8, fp8 or fp8-mma")
        if config.value_of(cfg, "mtp") not in (None, 0) and "--draft-model" not in seen and not allow_uninstalled:
            raise ValueError("Gemma 4 MTP requires its matching QAT assistant (--draft-model)")
    elif arch == "clover_moe":
        mode = config.clover_mode(cfg)
        draft = config.value_of(cfg, "clover_draft")
        if mode != "self-spec" and draft not in (None, 0):
            raise ValueError("--clover-draft greater than zero requires --clover-mode self-spec")
        if mode == "self-spec" and draft == 0:
            raise ValueError("--clover-mode self-spec requires 1..31 draft tokens, or omit --clover-draft")
        if "--vision" in seen and not experimental_clover_vision(cfg):
            raise ValueError("Clover-35B-A3B-Preview is text-only; remove --vision")
        if ("--vision" in seen or cfg.get("vision") or cfg.get("vision_mode", "no") != "no") and mode != "causal":
            raise ValueError("experimental Clover Vision supports only --clover-mode causal")
        if config.value_of(cfg, "mtp") not in (None, 0):
            raise ValueError("Clover has no MTP head; use --clover-draft for self-speculative decoding")
        context = config.value_of(cfg, "max_context")
        if context not in (None, "auto") and context > 131072:
            raise ValueError("Clover supports at most 131072 context tokens")
        if kv == "fp16":
            raise ValueError("Clover does not support --kv-type fp16")
    elif kv == "fp16":
        raise ValueError("Qwen3.6 does not support --kv-type fp16")
    if arch != "gemma4" and "--draft-model" in seen:
        raise ValueError("--draft-model is only supported by the Gemma 4 engine")


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
    if cfg.get("arch") == "gemma4" and config.value_of(cfg, "mtp") not in (None, 0):
        args = cfg.get("args", [])
        name = args[args.index("--draft-model") + 1] if "--draft-model" in args[:-1] else ""
        path = Path(name) if name else None
        path = (path if path.is_absolute() else base / path) if path is not None else None
        if path is None or not path.is_file():
            if not allow_uninstalled:
                raise ValueError("Gemma 4 QAT assistant is missing; download the model bundle first")
