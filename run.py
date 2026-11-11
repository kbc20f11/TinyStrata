"""Launch TinyStrata's API server; local state stays in the sibling work directory."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT.with_name(ROOT.name + "-work") / "runtime"


def set_arg(cfg: dict, flag: str, value) -> None:
    args = cfg.setdefault("args", [])
    if flag in args:
        args[args.index(flag) + 1] = str(value)
    else:
        args.extend((flag, str(value)))


def expert_budget(value: str):
    if value.lower() == "auto":
        return "auto"
    try:
        return float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected auto or an expert cache limit in GiB") from e


def context_limit(value: str):
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected auto or a whole-number context limit") from e


def chunk_size(value: str):
    from serve import config

    try:
        parsed = "auto" if value.lower() == "auto" else int(value)
        return config.check("chunk", parsed, {})
    except ValueError as e:
        raise argparse.ArgumentTypeError("expected auto or a whole-number input chunk from 1 to 4096") from e


def vision_tokens(value: str) -> int:
    try:
        tokens = int(value)
        if tokens > 0:
            return tokens
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("expected a positive whole-number image token limit")


def refresh_hip_device(cfg: dict) -> dict:
    """Refresh automatic Windows selection after driver/runtime device reordering."""
    if cfg.get("backend") != "hip" or cfg.get("gpu") not in (None, "", []):
        return cfg
    from serve.backend import hip_ordinal
    if os.name != "nt":
        if cfg.get("hip_ordinal") is None:
            ordinal = hip_ordinal(cfg.get("lib_dirs") or [])
            if ordinal is not None:
                return {**cfg, "hip_ordinal": ordinal}
        return cfg
    result = dict(cfg)
    masks = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    if any(os.environ.get(key) or (cfg.get("env") or {}).get(key) for key in masks):
        result.pop("hip_ordinal", None)
        return result
    ordinal = hip_ordinal(cfg.get("lib_dirs") or [], executable=cfg.get("exe"))
    if ordinal is None:
        raise ValueError("Cannot identify a usable HIP GPU with the selected TinyStrata runtime. "
                         "Check the AMD driver or select --gpu explicitly; device 0 was not assumed.")
    result["hip_ordinal"] = ordinal
    return result


def prepare(a) -> tuple[dict, Path]:
    from serve import config
    from serve.backend import configure_backend, executable_backend, hardware_backend, hip_ordinal
    from serve.tinyconfig import (CLOVER_SAMPLING, CLOVER_DECODE_POLICY_VERSION, MEMORY_POLICY_VERSION,
                                 automatic_experts, check_files, clover_decode_defaults, clover_vision_defaults,
                                 CLOVER_VISION_POLICY_VERSION, migrate_settings, validate)
    from serve.model_setup import (configure_download, configure_vision, model_architecture, model_path, prepare_installed_model,
                                   resolve_executable, save_settings, shared_library_dirs, migrate_managed_clover_model)
    from serve.model_download import is_tiel_model

    selected = getattr(a, "model_family", None)
    requested_arch = {"qwen36": "qwen35moe", "qwen3.6": "qwen35moe",
                      "lfm2": "lfm2moe", "lfm2.5": "lfm2moe",
                      "gemma4": "gemma4", "gemma-4": "gemma4", "clover": "clover_moe",
                      "gpt-oss": "gpt-oss", "gptoss": "gpt-oss",
                      "tiel": "qwen35moe", "tiel-coder": "qwen35moe"}.get(selected)
    tiel = selected in ("tiel", "tiel-coder")
    if a.model:
        local = a.model.expanduser().resolve()
        if not local.is_file():
            raise ValueError(f"TinyStrata model not found: {local}")
        actual_arch = model_architecture(local)
        if requested_arch is not None and actual_arch != requested_arch:
            raise ValueError("--model-family does not match the local GGUF architecture")
        requested_arch = actual_arch
    config_path = getattr(a, "config", None)
    path = (config_path or RUNTIME / ".tinystrata" / ("config-tiel-coder.json" if tiel else
            {"lfm2moe": "config-lfm2.json", "gemma4": "config-gemma4.json", "clover_moe": "config-clover.json",
             "gpt-oss": "config-gpt-oss.json"}
            .get(requested_arch, "config.json"))).expanduser().resolve()
    from serve.install_guard import assert_installation_ready
    assert_installation_ready(path, allow_installer=getattr(a, "configure_only", False))
    arch = requested_arch or "qwen35moe"
    fresh = not path.exists()
    cfg = config.load(path) if not fresh else {
        "arch": arch, "model_name": ("tiel-coder-35b-a3b" if tiel else
                                    {"lfm2moe": "lfm2.5-24b-a2b", "gemma4": "gemma-4-26b-a4b-it", "clover_moe": "clover-35b-a3b-preview",
                                     "gpt-oss": "gpt-oss-20b"}
                                    .get(arch, "qwen3.6-35b-a3b")),
        "cwd": str(ROOT),
        "args": ["--max-context", "auto", "--kv-type", "auto", "--chunk", "auto"] +
                (["--mtp", "0"] if arch == "gpt-oss" else [] if arch in ("clover_moe", "lfm2moe") else ["--mtp", "3"]),
        "memory_policy_version": MEMORY_POLICY_VERSION,
        "host": "127.0.0.1",
    }
    if is_tiel_model(cfg.get("model_name")):
        cfg["model_name"] = "tiel-coder-35b-a3b"
    if requested_arch is not None and cfg.get("arch", "qwen35moe") != requested_arch:
        raise ValueError("This config belongs to another model family; choose a separate --config path")
    if (tiel and cfg.get("model_name") != "tiel-coder-35b-a3b") or (
            selected in ("qwen36", "qwen3.6") and cfg.get("model_name") == "tiel-coder-35b-a3b"):
        raise ValueError("This config belongs to another model; choose a separate --config path")
    if fresh:
        cfg = clover_vision_defaults(clover_decode_defaults(cfg))
    if cfg.get("arch") == "clover_moe":
        if getattr(a, "vision", None) is not None:
            cfg["clover_vision_policy_version"] = CLOVER_VISION_POLICY_VERSION
        if a.model:
            cfg["model_source"] = "local"
        if a.model or any(getattr(a, key, None) is not None for key in (*config.CLOVER_REQUEST_KEYS, "kv_type")):
            # An explicit old-default value is still a user choice. Protect it
            # before the historical stock signature is considered.
            cfg["clover_decode_policy_version"] = CLOVER_DECODE_POLICY_VERSION
        cfg = migrate_managed_clover_model(cfg, path)
    cfg, migrated = migrate_settings(cfg)
    requested_backend = getattr(a, "backend", None)
    if requested_backend in (None, "auto") and not cfg.get("backend"):
        explicit_backend = executable_backend(a.exe, [str(p) for p in a.lib_dir or []]) if a.exe else None
        requested_backend = explicit_backend or hardware_backend()
    cfg = configure_backend(cfg, requested_backend, explicit_kv=a.kv_type)
    cfg.setdefault("port", 18084 if cfg.get("model_name") == "tiel-coder-35b-a3b" else
                   {"lfm2moe": 18081, "gemma4": 18082, "clover_moe": 18083, "gpt-oss": 18085}.get(cfg.get("arch"), 18080))
    if a.model:
        set_arg(cfg, "--model", a.model.expanduser().resolve())
        if cfg.get("arch") == "clover_moe":
            cfg["model_source"] = "local"
        # An explicit local selection keeps the existing eager-launch behavior.
        cfg.pop("model_bootstrap", None)
    if getattr(a, "draft_model", None):
        set_arg(cfg, "--draft-model", a.draft_model.expanduser().resolve())
    if a.exe:
        cfg["exe"] = str(a.exe.expanduser().resolve())
    elif not cfg.get("exe"):
        candidate = resolve_executable(cfg)
        if candidate is not None:
            cfg["exe"] = str(candidate)
    if getattr(a, "vision_exe", None):
        cfg["vision_exe"] = str(a.vision_exe.expanduser().resolve())
    if getattr(a, "vision", None) is not None:
        cfg["vision_mode"] = {"yes": "gpu", "none": "no"}.get(a.vision, a.vision)
    if getattr(a, "vision_tokens", None) is not None:
        cfg["vision_tokens"] = a.vision_tokens
    for key in ("max_context", "kv_type", "mtp", "expert_vram", "vram_reserve_mib", "chunk", "clover_draft",
                "clover_mode", "clover_lookup", "clover_block_size", "clover_threshold", "clover_accept",
                "clover_accept_ratio", "clover_accept_delta", "mtp_auto", "lookup_draft") + config.CONVERSATION_CACHE_KEYS:
        value = getattr(a, key, None)
        if value is not None:
            if key == "expert_vram" and value == "auto":
                cfg = automatic_experts(cfg)
            else:
                cfg, _ = config.apply(cfg, {key: value})
    if getattr(a, "clover_mode", None) is None and getattr(a, "clover_draft", None) is not None:
        cfg, _ = config.apply(cfg, {"clover_mode": "self-spec" if a.clover_draft > 0 else "causal"})
    elif getattr(a, "clover_mode", None) is not None and getattr(a, "clover_draft", None) is None:
        draft = config.value_of(cfg, "clover_draft")
        if a.clover_mode == "self-spec" and draft in (None, 0):
            cfg, _ = config.apply(cfg, {"clover_draft": 3})
        elif a.clover_mode != "self-spec" and draft not in (None, 0):
            cfg, _ = config.apply(cfg, {"clover_draft": 0})
    if cfg.get("arch") == "clover_moe":
        temperature = (cfg.get("sampling") or {}).get("temperature")
        if temperature is None:
            temperature = CLOVER_SAMPLING["temperature"]
        if getattr(a, "clover_accept", None) is None and config.value_of(cfg, "clover_accept") == "threshold" and (
                config.clover_mode(cfg) != "self-spec" or temperature != 0):
            cfg, _ = config.apply(cfg, {"clover_accept": "exact"})
    if cfg.get("arch") == "clover_moe" and getattr(a, "vision", None) is None and (
            getattr(a, "clover_mode", None) == "diffusion" or getattr(a, "clover_accept", None) == "threshold"):
        # Explicit text-only experimental modes remain usable without requiring
        # a second flag to undo the fresh image default. Explicit GPU/CPU image
        # choices still fail validation for these unsupported combinations.
        cfg["vision_mode"] = "no"
    if getattr(a, "kv_grow", None) is not None:
        cfg["args"] = [x for x in cfg["args"] if x not in ("--kv-grow", "--kv-whole", "--no-kv-grow")]
        cfg["args"].append("--kv-grow" if a.kv_grow else "--kv-whole")
    if a.lib_dir:
        cfg["lib_dirs"] = [str(p.expanduser().resolve()) for p in a.lib_dir]
    if a.port is not None:
        cfg["port"] = a.port
    if a.host is not None:
        cfg["host"] = a.host
    if a.api_key is not None:
        cfg["api_key"] = a.api_key
    if a.gpu is not None:
        cfg["gpu"] = a.gpu
        cfg.pop("hip_ordinal", None)
    from serve.hip_runtime import migrate_installed_runtime
    cfg = migrate_installed_runtime(cfg, path)
    cfg = shared_library_dirs(cfg)
    native_backend = executable_backend(cfg["exe"], cfg.get("lib_dirs") or []) if cfg.get("exe") else None
    if native_backend and native_backend != cfg["backend"]:
        raise ValueError(f"The selected executable uses {native_backend}, but the configuration selects {cfg['backend']}; rebuild/install the matching backend")
    cfg = refresh_hip_device(cfg)
    cfg = configure_download(cfg, path)
    model = model_path(cfg)
    if cfg.get("vision_mode") == "no":
        cfg = configure_vision(cfg, model)
    validate(cfg, allow_uninstalled=True)
    check_files(cfg, allow_uninstalled=True)
    if a.model and not model.is_file():
        raise ValueError(f"TinyStrata model not found: {model}")
    cfg["log"] = str(path.parent / ("engine-tiel-coder.log" if cfg.get("model_name") == "tiel-coder-35b-a3b" else
                    {"lfm2moe": "engine-lfm2.log", "gemma4": "engine-gemma4.log", "clover_moe": "engine-clover.log",
                     "gpt-oss": "engine-gpt-oss.log"}
                    .get(cfg.get("arch"), "engine.log")))
    if model.is_file():
        cfg = prepare_installed_model(cfg, path, model)
    else:
        cfg["model_bootstrap"] = True
        cfg.pop("tokenizer", None)
        save_settings(path, cfg)
    if migrated:
        print("Migrated legacy settings to the API server configuration.", flush=True)
    return cfg, path


def main() -> int:
    from serve import config
    from serve.stdio import configure_utf8_stdio

    configure_utf8_stdio()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, help="existing supported GGUF (the local header selects its model family)")
    ap.add_argument("--model-family", choices=("qwen36", "qwen3.6", "lfm2", "lfm2.5", "gemma4", "gemma-4", "clover", "tiel", "tiel-coder",
                             "gpt-oss", "gptoss"),
                    help="Qwen3.6 (default), LFM2.5, Gemma4, Clover, Tiel Coder or gpt-oss; settings are separate per model")
    ap.add_argument("--draft-model", type=Path, help="existing Gemma4 QAT assistant GGUF for MTP")
    ap.add_argument("--exe", type=Path, help="built tinystrata executable")
    ap.add_argument("--backend", choices=("auto", "cuda", "hip"), default="auto",
                    help="GPU backend (default: saved setting or native executable); HIP supports native FP8 on RDNA4")
    ap.add_argument("--vision-exe", type=Path, help="built tinystrata-vision image encoder")
    ap.add_argument("--vision", choices=("gpu", "cpu", "no", "yes", "none"),
                    help="image encoder: gpu (default), cpu or no; yes = gpu, none = no; saved for later starts")
    ap.add_argument("--vision-tokens", type=vision_tokens, metavar="N",
                    help="maximum image tokens per picture (default: 4096); saved for later starts")
    ap.add_argument("--config", type=Path, help="configuration path (default: a separate file per model family)")
    ap.add_argument("--max-context", type=context_limit, metavar="auto|TOKENS",
                    help="choose context capacity from total GPU VRAM (default: auto), or set a token limit")
    ap.add_argument("--kv-type", choices=("auto", "fp16", "fp32", "fp8", "fp8-mma", "int8"),
                    help="auto: Qwen/LFM use native FP8 when supported, otherwise INT8; Gemma uses INT8")
    kv_allocation = ap.add_mutually_exclusive_group()
    kv_allocation.add_argument("--kv-grow", dest="kv_grow", action="store_true", default=None,
                               help="grow physical KV storage with the conversation (default); saved for later starts")
    kv_allocation.add_argument("--kv-whole", "--no-kv-grow", dest="kv_grow", action="store_false",
                               help="allocate the whole context's KV at load; saved for later starts")
    ap.add_argument("--mtp", type=int)
    ap.add_argument("--mtp-auto", type=int, choices=(0, 1), help="adapt MTP depth using measured throughput; default off")
    ap.add_argument("--lookup-draft", type=int, choices=range(8), help="Qwen/Gemma/gpt-oss exact target-sampled suffix proposals, 0-7; default off")
    ap.add_argument("--clover-draft", type=int, metavar="N",
                    help="legacy Clover self-speculative draft tokens, 0-31; 0 selects causal, positive values self-spec")
    ap.add_argument("--clover-lookup", type=int, metavar="N",
                    help="opt-in exact repetition lookup, 0-7 proposed tokens (default: 0/off); Clover causal and temperature 0 only")
    ap.add_argument("--clover-mode", choices=("causal", "self-spec", "diffusion"),
                    help="Clover generation mode (default: saved profile; fresh profiles use causal D0, INT8 KV and MT1); "
                         "self-spec D3 and native LowConfidence diffusion are opt-in alternatives")
    ap.add_argument("--clover-block-size", type=int, metavar="N",
                    help="Clover diffusion block size: 1, 2, 4, 8, 16 or 32 (default: 32)")
    ap.add_argument("--clover-threshold", type=float, metavar="P",
                    help="Clover diffusion confidence threshold in [0,1] (default: 0.95)")
    ap.add_argument("--clover-accept", choices=("exact", "threshold"),
                    help="Clover self-speculative acceptance (default: inherited profile): exact or experimental "
                         "threshold (greedy only)")
    ap.add_argument("--clover-accept-ratio", type=float, metavar="P",
                    help="Clover threshold acceptance probability ratio in [0,1] (default: 0.9)")
    ap.add_argument("--clover-accept-delta", type=float, metavar="P",
                    help="Clover threshold acceptance entropy factor in [0,1] (default: 0.6)")
    ap.add_argument("--expert-vram", type=expert_budget, metavar="auto|GiB",
                    help="fit experts to available VRAM (default: auto), or set an advanced GiB limit; 0 = CPU experts")
    ap.add_argument("--vram-reserve-mib", type=int,
                    help="override the automatic VRAM reserve (default: 768 MiB; upstream near-fit path: 256 MiB)")
    ap.add_argument("--chunk", type=chunk_size, metavar="auto|TOKENS",
                    help="use the upstream engine's automatic chunk selection (default: auto), or fix 1-4096 tokens")
    for key in config.CONVERSATION_CACHE_KEYS:
        ap.add_argument(config.ENGINE_OPTIONS[key][0], type=int, metavar="N",
                        help="Gemma4/LFM2.5 conversation parking setting; omitted keeps the engine default")
    ap.add_argument("--gpu", type=int, help="GPU index")
    ap.add_argument("--lib-dir", type=Path, action="append", help="GPU runtime DLL directory (repeatable)")
    ap.add_argument("--port", type=int, help="API port (default: saved setting or 18080)")
    ap.add_argument("--host", help="API bind address (default: 127.0.0.1)")
    ap.add_argument("--api-key", help="API key required by clients (default: unset)")
    ap.add_argument("--configure-only", action="store_true")
    a = ap.parse_args()
    from serve.install_guard import assert_installation_ready
    try:
        assert_installation_ready(a.config or RUNTIME / ".tinystrata/config.json",
                                  allow_installer=a.configure_only)
    except ValueError as error:
        ap.error(str(error))
    missing = [m for m in ("jinja2", "regex", "psutil") if importlib.util.find_spec(m) is None]
    if missing:
        ap.error("Missing API dependencies: " + ", ".join(missing) +
                 f'. Install with: "{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"')
    try:
        cfg, path = prepare(a)
    except (ValueError, OSError, KeyError) as e:
        ap.error(str(e))
    print(f"TinyStrata config: {path}", flush=True)
    if a.configure_only:
        return 0
    command = [sys.executable, "-B", "-m", "serve.server", "--engine", "tinystrata", "--config", str(path),
               "--port", str(cfg.get("port", 18080))]
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
