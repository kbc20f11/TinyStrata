"""Validate the subset of upstream configuration supported by TinyStrata."""
from __future__ import annotations

import math
import os
from pathlib import Path

from serve import config
from serve.model_download import is_tiel_model, reject_retired_model

MEMORY_POLICY_VERSION = 1
VISION_POLICY_VERSION = 1
GENERATION_POLICY_VERSION = 1
CLOVER_DECODE_POLICY_VERSION = 2
CLOVER_VISION_POLICY_VERSION = 1
CHUNK_POLICY_VERSION = config.CHUNK_POLICY_VERSION
DEFAULT_MAX_TOKENS = 32768
# Qwen3.6's official thinking-mode general-task settings. Saved sampling
# values remain explicit choices; omitted fields use these defaults.
DEFAULT_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repetition_penalty": 1.0,
}
LFM2_SAMPLING = {"temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0,
                 "presence_penalty": 0.0, "repetition_penalty": 1.05}
GEMMA4_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0,
                   "presence_penalty": 0.0, "repetition_penalty": 1.0}
CLOVER_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 64, "min_p": 0.0,
                   "presence_penalty": 0.9, "frequency_penalty": 0.0, "repetition_penalty": 1.05}
TIEL_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                 "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0}
MODEL_NAMES = {"qwen35moe": "qwen3.6-35b-a3b", "lfm2moe": "lfm2.5-24b-a2b",
               "gemma4": "gemma-4-26b-a4b-it", "clover_moe": "clover-35b-a3b-preview"}


def clover_decode_defaults(cfg: dict) -> dict:
    """The managed API/launcher profile; native direct-CLI defaults are separate."""
    if cfg.get("arch") != "clover_moe":
        return cfg
    new, _ = config.apply(cfg, {"clover_mode": "causal", "clover_draft": 0,
                                "clover_accept": "exact", "kv_type": "int8"})
    env = dict(new.get("env") or {})
    # A supplied CPU arithmetic rule wins, including an explicit process env.
    env.setdefault("STRATA_IQ_MT_MIN", os.environ.get("STRATA_IQ_MT_MIN", "1"))
    new["env"] = env
    new["clover_decode_policy_version"] = CLOVER_DECODE_POLICY_VERSION
    return new


def migrate_clover_decode_settings(cfg: dict) -> tuple[dict, bool]:
    """Return the former managed D3 default to causal without retuning the model.

    Historical same-value manual D3 settings have no provenance. Only the
    complete old stock tuple migrates once; local models and custom settings
    stay intact. Explicit CLI choices are marked before this migration, so D3
    remains opt-in on later launches. Never upgrade an old causal profile to D3.
    """
    if cfg.get("arch") != "clover_moe" or cfg.get("clover_decode_policy_version") not in (None, 1):
        return cfg, False
    stock_values = {"--model": None, "--max-context": "auto", "--kv-type": "int8",
                    "--chunk": "auto", "--clover-mode": "self-spec", "--clover-draft": "3",
                    "--clover-accept": "exact"}
    args = cfg.get("args")
    stock = (cfg.get("model_source") == "managed" and cfg.get("env") == {"STRATA_IQ_MT_MIN": "1"} and
             cfg.get("sampling") == CLOVER_SAMPLING and cfg.get("enable_thinking", True) is True and
             isinstance(args, list))
    seen = set()
    if stock:
        i = 0
        while i < len(args):
            flag = args[i]
            if flag in seen:
                stock = False
                break
            seen.add(flag)
            # Image-enabled managed profiles also carry this standalone flag.
            if flag == "--vision":
                i += 1
                continue
            if (flag not in stock_values or i + 1 >= len(args) or
                    (stock_values[flag] is not None and args[i + 1] != stock_values[flag])):
                stock = False
                break
            i += 2
        stock = stock and {"--model", "--kv-type", "--clover-mode", "--clover-draft", "--clover-accept"}.issubset(seen)
    new = config.apply(cfg, {"clover_mode": "causal", "clover_draft": 0})[0] if stock else dict(cfg)
    new["clover_decode_policy_version"] = CLOVER_DECODE_POLICY_VERSION
    return new, new != cfg


def clover_vision_defaults(cfg: dict) -> dict:
    """Enable the pinned Qwen image encoder for a fresh Clover profile."""
    if cfg.get("arch") != "clover_moe":
        return cfg
    return {**cfg, "vision_mode": "gpu", "clover_vision_policy_version": CLOVER_VISION_POLICY_VERSION}


def migrate_clover_vision_settings(cfg: dict) -> tuple[dict, bool]:
    """Upgrade the former managed D3 text-only default once; keep image choices.

    Historical automatic 'no' and an identical manual 'no' have no provenance.
    Only the complete unmarked stock tuple migrates. CLI image choices receive
    the marker before migration, so even a later explicit 'no' is retained.
    """
    if cfg.get("arch") != "clover_moe" or "clover_vision_policy_version" in cfg:
        return cfg, False
    expected = {"--model": None, "--max-context": "auto", "--kv-type": "int8", "--chunk": "auto",
                "--clover-mode": "self-spec", "--clover-draft": "3", "--clover-accept": "exact"}
    args = cfg.get("args")
    stock = (cfg.get("model_source") == "managed" and cfg.get("vision_mode", "no") == "no" and
             not cfg.get("vision") and "vision_policy_version" not in cfg and
             cfg.get("env") == {"STRATA_IQ_MT_MIN": "1"} and cfg.get("sampling") == CLOVER_SAMPLING and
             cfg.get("enable_thinking", True) is True and isinstance(args, list) and len(args) % 2 == 0)
    seen = set()
    if stock:
        for flag, value in zip(args[::2], args[1::2]):
            if flag not in expected or flag in seen or (expected[flag] is not None and value != expected[flag]):
                stock = False
                break
            seen.add(flag)
        stock = stock and {"--model", "--clover-mode", "--clover-draft", "--clover-accept"}.issubset(seen)
    new = clover_vision_defaults(cfg) if stock else {**cfg, "vision_mode": cfg.get("vision_mode", "no"),
                                                  "clover_vision_policy_version": CLOVER_VISION_POLICY_VERSION}
    return new, new != cfg


def clover_vision_enabled(cfg: dict) -> bool:
    """Select the Clover image-template adapter without replacing its tokenizer."""
    return cfg.get("arch") == "clover_moe" and cfg.get("vision_mode", "gpu") != "no"



def migrate_generation_settings(cfg: dict) -> tuple[dict, bool]:
    """Fill omitted defaults, and upgrade the old stock greedy config once.

    A saved custom value always wins. The policy marker lets a user deliberately
    select greedy after upgrading without having it reset on the next launch.
    """
    sampling = cfg.get("sampling")
    if sampling is not None and not isinstance(sampling, dict):
        raise ValueError("sampling must be an object")
    sampling = dict(sampling or {})
    if cfg.get("arch", "qwen35moe") == "qwen35moe" and not is_tiel_model(cfg.get("model_name")) and \
            cfg.get("generation_policy_version") != GENERATION_POLICY_VERSION and \
            sampling == {"temperature": 0}:
        sampling = {}
    defaults = dict({"lfm2moe": LFM2_SAMPLING, "gemma4": GEMMA4_SAMPLING,
                     "clover_moe": CLOVER_SAMPLING}.get(cfg.get("arch"), DEFAULT_SAMPLING))
    if is_tiel_model(cfg.get("model_name")):
        defaults = dict(TIEL_SAMPLING)
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
    new.pop("expert_gpu_percent", None)
    new.pop("expert_gpu_count", None)
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
    text_only = cfg.get("arch") == "lfm2moe"
    mode = cfg.get("vision_mode", "no" if text_only else "gpu")
    if mode not in ("gpu", "cpu", "no"):
        raise ValueError("vision_mode must be gpu, cpu or no")
    if "vision_tokens" in cfg:
        tokens = cfg["vision_tokens"]
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError("vision_tokens must be a positive whole number")
    if text_only and (mode != "no" or cfg.get("vision")):
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
    reject_retired_model(cfg.get("arch"), cfg.get("model_name"))
    # Check the existing memory-policy marker before the memory migration adds it.
    new, changed = migrate_chunk_settings(cfg)
    new, memory_changed = migrate_memory_settings(new)
    changed = changed or memory_changed
    new, generation_changed = migrate_generation_settings(new)
    changed = changed or generation_changed
    # Upgrade the historical D3 image signature before changing its decode mode.
    new, clover_vision_changed = migrate_clover_vision_settings(new)
    changed = changed or clover_vision_changed
    new, decode_changed = migrate_clover_decode_settings(new)
    changed = changed or decode_changed
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
    reject_retired_model(cfg.get("arch"), cfg.get("model_name"))
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
              "--clover-mode", "--clover-lookup", "--mtp-auto", "--lookup-draft", "--clover-block-size", "--clover-threshold", "--clover-accept",
              "--clover-accept-ratio", "--clover-accept-delta"}
    values.update(config.ENGINE_OPTIONS[key][0] for key in config.CONVERSATION_CACHE_KEYS)
    switches = {"--no-graph", "--no-warmup", "--hybrid-graph", "--no-hybrid-graph", "--vision",
                "--kv-grow", "--kv-whole", "--no-kv-grow"}
    seen = set()
    option_values = {}
    i = 0
    while i < len(args):
        flag = args[i]
        if flag in seen:
            raise ValueError(f"Duplicate engine option: {flag}")
        seen.add(flag)
        if flag in switches:
            i += 1
        elif flag in values and i + 1 < len(args) and not args[i + 1].startswith("--"):
            option_values[flag] = args[i + 1]
            i += 2
        else:
            raise ValueError(f"Unsupported or incomplete TinyStrata engine option: {flag}")
    if "--model" not in seen:
        raise ValueError("Choose a GGUF model with run.py --model PATH")
    if len(seen.intersection({"--kv-grow", "--kv-whole", "--no-kv-grow"})) > 1:
        raise ValueError("Choose only one KV allocation mode: --kv-grow or --kv-whole")
    for key in ("kv_type", "max_context", "expert_vram", "vram_reserve_mib", "mtp", "chunk", "clover_draft",
                "clover_mode", "clover_lookup", "clover_block_size", "clover_threshold", "clover_accept",
                "clover_accept_ratio", "clover_accept_delta", "mtp_auto", "lookup_draft") + config.CONVERSATION_CACHE_KEYS:
        value = config.value_of(cfg, key)
        if value is not None:
            config.check(key, value, cfg)
    kv = config.value_of(cfg, "kv_type")
    clover_options = seen.intersection({"--clover-draft", "--clover-mode", "--clover-block-size", "--clover-threshold",
                                       "--clover-accept", "--clover-accept-ratio", "--clover-accept-delta"})
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
        if config.value_of(cfg, "clover_lookup"):
            if mode != "causal" or draft not in (None, 0):
                raise ValueError("--clover-lookup requires Clover causal mode with zero drafts")
            temperature = (cfg.get("sampling") or {}).get("temperature", CLOVER_SAMPLING["temperature"])
            if temperature is None:
                temperature = CLOVER_SAMPLING["temperature"]
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature != 0:
                raise ValueError("--clover-lookup requires sampling.temperature 0")
        if config.value_of(cfg, "clover_accept") == "threshold" and mode != "self-spec":
            raise ValueError("--clover-accept threshold requires --clover-mode self-spec")
        if config.value_of(cfg, "clover_accept") == "threshold":
            temperature = (cfg.get("sampling") or {}).get("temperature", 0.0)
            if temperature is None:
                temperature = CLOVER_SAMPLING["temperature"]
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature != 0:
                raise ValueError("--clover-accept threshold requires sampling.temperature 0")
        if "--vision" in seen or cfg.get("vision") or check_vision_options(cfg) != "no":
            if mode not in ("causal", "self-spec") or (mode == "self-spec" and config.value_of(cfg, "clover_accept") == "threshold"):
                raise ValueError("Clover Vision supports causal or exact self-spec generation; use --vision no for diffusion or threshold acceptance")
        if config.value_of(cfg, "mtp") not in (None, 0):
            raise ValueError("Clover has no MTP head; use --clover-draft for self-speculative decoding")
        if kv == "fp16":
            raise ValueError("Clover does not support --kv-type fp16")
    elif kv == "fp16":
        raise ValueError("Qwen3.6 does not support --kv-type fp16")
    if arch != "gemma4" and "--draft-model" in seen:
        raise ValueError("--draft-model is only supported by the Gemma 4 engine")
    if config.value_of(cfg, "lookup_draft"):
        defaults = {"gemma4": GEMMA4_SAMPLING}.get(arch, DEFAULT_SAMPLING)
        temperature = (cfg.get("sampling") or {}).get("temperature", defaults["temperature"])
        if temperature is None:
            temperature = defaults["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or \
                not math.isfinite(temperature) or temperature < 0:
            raise ValueError("--lookup-draft requires a finite nonnegative sampling.temperature")


def check_files(cfg: dict, *, allow_uninstalled: bool = False) -> None:
    reject_retired_model(cfg.get("arch"), cfg.get("model_name"))
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
