"""Explicit Intel entry point using the normal API.

This does not migrate existing profiles or select Intel automatically. Model
switching and persistent conversation snapshots are not enabled.
Image input is opt-in and requires the matching Intel image encoder/projector.
MTP is a separate experimental opt-in and defaults to zero.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import threading


def context_option(value):
    if value == "auto":
        return value
    if not value.isascii() or not value.isdigit() or not 16 <= int(value) <= 262144:
        raise argparse.ArgumentTypeError("context must be auto or a whole number from 16 to 262144")
    return int(value)


def main(argv=None, *, stop_event: threading.Event | None = None) -> int:
    from serve.stdio import configure_utf8_stdio
    from serve.tinyconfig import DEFAULT_MAX_TOKENS
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--vision-mode", choices=("no", "gpu", "cpu"), default="no")
    parser.add_argument("--vision-exe", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--lib-dir", type=Path, action="append", default=[])
    from serve.intel_models import MODEL_PROFILES
    parser.add_argument("--intel-profile", choices=("generic", *MODEL_PROFILES.values()), default=None)
    parser.add_argument("--model-family", choices=("qwen36", "tiel", "gemma4", "lfm2", "clover"), default=None)
    parser.add_argument("--context", type=context_option, default="auto",
                        help="auto lets native choose the largest context fitting actual memory after loading; explicit values are retained")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="default output limit; omitted request limits fit the remaining context")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--weight-memory", choices=("host", "shared", "device"), default="host")
    parser.add_argument("--device-state", action="store_true", help="candidate device USM for GPU-only state/scratch")
    parser.add_argument("--verify-graph", action="store_true", help="candidate MTP verification command graphs")
    parser.add_argument("--kv-type", choices=("fp32", "int8"), default=None)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--mtp", type=int, choices=tuple(range(8)), default=0)
    parser.add_argument("--draft-vocab", type=int, default=0, help="candidate exact-MTP proposal vocabulary size, 0 or 64..4096")
    parser.add_argument("--mtp-stochastic", action="store_true", help="candidate exact speculative rejection sampling")
    parser.add_argument("--decode-dp4a", action="store_true")
    parser.add_argument("--verify-xmx", action="store_true", help="candidate INT8 XMX verification")
    parser.add_argument("--prefill-int8-xmx", action="store_true", help="candidate INT8 XMX short prefill")
    parser.add_argument("--vocab-xmx", action="store_true", help="candidate lossless Q6 vocabulary interleave")
    parser.add_argument("--expert-xmx-layout", action="store_true", help="candidate lossless expert interleave with Q8 activations")
    parser.add_argument("--dense-split-k", action="store_true", help="candidate four-way split for width-4096 Q6 outputs")
    parser.add_argument("--dense-xmx-layout", action="store_true", help="candidate lossless Q6 dense interleave")
    parser.add_argument("--verify-mkl-attention", action="store_true", help="candidate grouped FP32 verification attention")
    parser.add_argument("--prefill-mkl", action="store_true")
    parser.add_argument("--prefill-npu", action="store_true")
    parser.add_argument("--prefill-xmx", action="store_true", help="candidate FP16-multiplicand XMX prefill")
    parser.add_argument("--port", type=int, default=8095)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--work-dir", type=Path, default=root.with_name(root.name + "-work") / "intel-runtime")
    args = parser.parse_args(argv)
    if os.name != "nt":
        parser.error("The Intel entry point currently requires native Windows")
    if not 1 <= args.port <= 65535 or args.device < 0 or not 1 <= args.chunk <= 1024 or args.max_tokens < 1:
        parser.error("Invalid context, port, device index, chunk or output token limit")
    if args.mtp and args.chunk < max(4, args.mtp + 1):
        parser.error("Experimental MTP needs chunk >= max(4, depth+1)")
    if args.draft_vocab and (not 64 <= args.draft_vocab <= 4096 or not args.mtp_stochastic or not args.mtp or not args.vocab_xmx):
        parser.error("--draft-vocab requires --mtp-stochastic, --mtp, --vocab-xmx and 64..4096 rows")
    if args.mtp_stochastic and not args.mtp:
        parser.error("--mtp-stochastic requires --mtp >= 1")
    if args.verify_graph and (not args.mtp or args.kv_type != "int8" or args.verify_mkl_attention):
        parser.error("--verify-graph requires MTP, INT8 KV and native attention")
    if args.verify_xmx and not args.decode_dp4a:
        parser.error("--verify-xmx requires --decode-dp4a")
    if args.prefill_int8_xmx and not args.decode_dp4a:
        parser.error("--prefill-int8-xmx requires --decode-dp4a")
    if args.vocab_xmx and not args.decode_dp4a:
        parser.error("--vocab-xmx requires --decode-dp4a")
    if args.expert_xmx_layout and not args.decode_dp4a:
        parser.error("--expert-xmx-layout requires --decode-dp4a")
    if args.dense_split_k and not args.dense_xmx_layout:
        parser.error("--dense-split-k requires --dense-xmx-layout")
    if args.dense_xmx_layout and (not args.decode_dp4a or not args.prefill_mkl):
        parser.error("--dense-xmx-layout requires --decode-dp4a and --prefill-mkl")
    if args.verify_mkl_attention and (not args.prefill_mkl or args.chunk < 64):
        parser.error("--verify-mkl-attention requires --prefill-mkl and chunk >= 64")
    if not args.model.is_file() or not args.exe.is_file():
        parser.error("The GGUF and Intel executable must already exist")
    if args.vision_mode != "no" and (args.vision_exe is None or args.mmproj is None or
            not args.vision_exe.is_file() or not args.mmproj.is_file()):
        parser.error("Image input requires existing --vision-exe and --mmproj files")
    from serve.gguf_reader import GGUFFile
    metadata = GGUFFile(args.model).metadata
    from serve.intel_models import (QWEN_MODEL, TIEL_MODEL, GEMMA_MODEL, LFM_MODEL, CLOVER_MODEL,
                                    TARGET_ONLY_MODELS)
    from serve.model_download import manifest_for_arch
    arch, name = metadata.get("general.architecture"), str(metadata.get("general.name", ""))
    selected = ({"gemma4": GEMMA_MODEL, "lfm2moe": LFM_MODEL, "clover_moe": CLOVER_MODEL}.get(arch) or
                (QWEN_MODEL if arch == "qwen35moe" and "qwen3.6" in name.lower() else
                 TIEL_MODEL if arch == "qwen35moe" and name == "Ornith-1.5-35B" else None))
    if selected is None:
        parser.error("The GGUF does not identify a pinned Intel model family")
    requested = {"qwen36": QWEN_MODEL, "tiel": TIEL_MODEL, "gemma4": GEMMA_MODEL,
                 "lfm2": LFM_MODEL, "clover": CLOVER_MODEL}.get(args.model_family)
    if requested is not None and requested != selected:
        parser.error("--model-family does not match the actual GGUF identity")
    from serve.intel_backend import context_tokens
    try:
        context_tokens({"arch": arch, "model_name": selected, "args": ["--max-context", str(args.context)]})
    except ValueError as error:
        parser.error(str(error))
    if selected in TARGET_ONLY_MODELS or selected == TIEL_MODEL:
        pinned = manifest_for_arch(arch, selected)
        if args.model.name != pinned.filename or args.model.stat().st_size != pinned.size:
            parser.error("The selected Intel family requires its pinned model artifact")
    args.kv_type = args.kv_type or ("int8" if selected in TARGET_ONLY_MODELS else "fp32")
    if selected == QWEN_MODEL and args.intel_profile is None and (args.context == "auto" or args.context > 32768):
        # Preserve the diagnostic Qwen option surface, but never send auto or
        # a larger explicit request to an old fixed-context executable.
        from serve.intel_capabilities import probe_capabilities, validate_capabilities
        proof = probe_capabilities({"backend": "sycl", "exe": str(args.exe.resolve()),
                                    "lib_dirs": [str(path.resolve()) for path in args.lib_dir]})
        validate_capabilities(proof.report, model=selected, profile="generic", allow_legacy=True,
                              requested_context=args.context)
    from serve.tokenizer import Tokenizer, extract
    from serve.frontend import ChatTemplate
    from serve.server import Service, StrataEngine, Vision, vision_config, vision_env, serve
    from serve.tinyconfig import migrate_generation_settings
    work = args.work_dir.resolve()
    tokenizer_path = work / "tokenizer"
    # Model identity is explicit for this entry point; always extract the
    # selected GGUF's vocabulary/template rather than reusing another model's.
    work.mkdir(parents=True, exist_ok=True)
    extract(args.model.resolve(), work)
    tokenizer = Tokenizer.from_dir(tokenizer_path)
    template = ChatTemplate(tokenizer_path / "chat_template.jinja")
    env = dict(os.environ)
    # Windows UR otherwise filters devices after loading adapters and may
    # enumerate unrelated SDK CPU OpenCL ICDs registered on the machine.
    env.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
    env.setdefault("UR_LOADER_PRELOAD_FILTER", "1")
    env["PATH"] = os.pathsep.join(str(path.resolve()) for path in args.lib_dir) + os.pathsep + env.get("PATH", "")
    native_args = (["--model", str(args.model.resolve()), "--serve",
        "--max-context", str(args.context), "--kv-type", args.kv_type, "--mtp", str(args.mtp), "--device", str(args.device),
        "--weight-memory", args.weight_memory, "--chunk", str(args.chunk)] + (["--prefill-xmx"] if args.prefill_xmx else [])
        + (["--prefill-mkl"] if args.prefill_mkl else []) + (["--prefill-npu"] if args.prefill_npu else []) + (["--decode-dp4a"] if args.decode_dp4a else [])
        + (["--verify-xmx"] if args.verify_xmx else []) + (["--prefill-int8-xmx"] if args.prefill_int8_xmx else [])
        + (["--vocab-xmx"] if args.vocab_xmx else []) + (["--verify-mkl-attention"] if args.verify_mkl_attention else [])
        + (["--mtp-stochastic"] if args.mtp_stochastic else []) + (["--expert-xmx-layout"] if args.expert_xmx_layout else [])
        + (["--device-state"] if args.device_state else []) + (["--verify-graph"] if args.verify_graph else [])
        + (["--dense-xmx-layout"] if args.dense_xmx_layout else [])
        + (["--dense-split-k"] if args.dense_split_k else [])
        + (["--draft-vocab", str(args.draft_vocab)] if args.draft_vocab else []))
    generation = {"arch": arch, "model_name": selected, "enable_thinking": True, "max_tokens": args.max_tokens}
    if selected == TIEL_MODEL:
        generation["reasoning_effort"] = "medium"
    if selected in TARGET_ONLY_MODELS or selected == TIEL_MODEL or args.intel_profile is not None or args.vision_mode != "no":
        from serve.intel_backend import configure_backend, engine_args
        from serve.model_setup import validate_intel_executable
        cfg = configure_backend({**generation, "backend": "sycl", "vision_mode": args.vision_mode,
              "intel_profile": args.intel_profile or "generic", "cwd": str(root), "exe": str(args.exe.resolve()),
              "lib_dirs": [str(path.resolve()) for path in args.lib_dir],
              "args": [item for item in native_args if item != "--serve"]})
        validate_intel_executable(cfg, allow_legacy=selected == QWEN_MODEL)
        if args.vision_mode != "no":
            from serve.model_setup import configure_vision
            cfg.update(vision_exe=str(args.vision_exe.resolve()),
                       download_dir=str(args.mmproj.resolve().parent))
            cfg = configure_vision(cfg, args.model.resolve(), projector=args.mmproj.resolve())
            if not cfg.get("vision"):
                parser.error("The projector or Intel image runtime did not pass validation")
        native_args = engine_args(cfg)
        generation.update(cfg)
    engine = vision = httpd = service = None
    stopping = stop_event if stop_event is not None else threading.Event()
    if threading.current_thread() is threading.main_thread():
        for event in (signal.SIGINT, signal.SIGTERM):
            signal.signal(event, lambda *_: stopping.set())
    try:
        if args.vision_mode != "no":
            vision = Vision(vision_config(generation, env), env=vision_env(generation, env),
                            log=open(work / "vision.log", "a", encoding="utf-8"))
            template = ChatTemplate(tokenizer_path / "chat_template.jinja",
                                    clover_vision=arch == "clover_moe")
        engine = StrataEngine(str(args.exe.resolve()), native_args,
                             cwd=str(root), log=str(work / "native.log"), env=env)
        if vision is not None and engine.info.get("vision") != 1:
            raise RuntimeError("The Intel engine did not enable its image input contract")
        service = Service(engine, tokenizer, template, model_name=selected, vision=vision,
            sampling_defaults=migrate_generation_settings(generation)[0]["sampling"], generation_defaults=generation)
        service.backend = "sycl"
        httpd = serve(service, "127.0.0.1", args.port)
        print(f"Intel API: http://127.0.0.1:{args.port} ({args.kv_type} KV, MTP={args.mtp}, vision={args.vision_mode})", flush=True)
        stopping.wait()
    finally:
        if httpd is not None:
            service.stopping.set()
            httpd.shutdown()
            httpd.server_close()
            if getattr(service, "telemetry", None) is not None:
                service.telemetry.close()
        if engine is not None and engine.alive():
            engine._send("STOP")
        if engine is not None:
            engine.close()
        if vision is not None:
            vision.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
