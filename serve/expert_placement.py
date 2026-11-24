"""Header-only routed-Expert cache geometry for the native model families.

The byte budget follows native slot alignment and guard tails, not GGUF file
size. Shared/dense weights, KV, vision, and Gemma's dense QAT assistant are
outside this budget. No tensor payload or executable is read or started.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
import re
import struct

from serve.gguf_reader import BLOCK_GEOMETRY, GGUFFile
from serve.model_setup import model_path

_QWEN_TYPES = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 40, 42}
_OTHER_TYPES = {2, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 29, 39, 42}
_FAMILIES = {"qwen35moe", "clover_moe", "gemma4", "lfm2moe"}
_INKLING_ARCHES = {"inkling2", "inkling2_loop"}
_INKLING_CUDA_RAW_TYPES = {0, 1, 2, 6, 8, 10, 11, 12, 13, 14, 18, 20, 23, 30, 39}
_INKLING_HIP_RAW_TYPES = _INKLING_CUDA_RAW_TYPES - {6}
_INKLING_SK_BLOCK_BYTES = {4: 134, 6: 198, 8: 262}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Missing or invalid Expert geometry: {name}")
    return value


def _headers(path):
    first = GGUFFile(path)
    count = first.metadata.get("split.count", 1)
    _positive(count, "split.count")
    if first.metadata.get("split.no", 0) != 0:
        raise ValueError("Choose the first GGUF shard for Expert geometry")
    files = [first]
    if count > 1:
        if count > 1024:
            raise ValueError("Too many GGUF shards")
        tail = f"-00001-of-{count:05d}.gguf"
        if not str(path).endswith(tail):
            raise ValueError("The first split GGUF filename does not match split.count")
        stem = str(path)[:-len(tail)]
        for index in range(1, count):
            shard_path = Path(f"{stem}-{index + 1:05d}-of-{count:05d}.gguf")
            if not shard_path.is_file():
                raise ValueError(f"Missing GGUF shard: {shard_path.name}")
            shard = GGUFFile(shard_path)
            if shard.metadata.get("split.no", index) != index or shard.metadata.get("split.count", count) != count:
                raise ValueError("Inconsistent GGUF shard metadata")
            files.append(shard)
    return files


def _payload(tensor, supported):
    block = BLOCK_GEOMETRY.get(tensor.type_name)
    if tensor.type_id not in supported or block is None:
        raise ValueError(f"Unsupported Expert tensor type: {tensor.name} ({tensor.type_name})")
    if len(tensor.shape) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in tensor.shape):
        raise ValueError(f"Invalid routed-Expert tensor shape: {tensor.name}")
    # GGML quantization blocks are rows, not one flat sequence across rows.
    if tensor.shape[0] % block[0]:
        raise ValueError(f"Expert rows do not match quantization blocks: {tensor.name}")
    return math.prod(tensor.shape) // block[0] * block[1], block[1]


def _inkling_raw_body_bytes(gate, down, count):
    # The native loader keeps each plane's GGUF row layout (including mixed
    # IQ3_XXS gate/up and IQ4_NL down), with a 32-byte codec header and
    # 32-byte boundary between planes. HIP additionally aligns whole bodies.
    gate_bytes, _ = _payload(gate, _INKLING_CUDA_RAW_TYPES)
    down_bytes, _ = _payload(down, _INKLING_CUDA_RAW_TYPES)
    down_offset = (32 + gate_bytes // count + 31) & ~31
    return (down_offset + down_bytes // count + 31) & ~31


def _inkling_sk_block_bytes(type_id, bits):
    if not bits:
        if type_id in {0, 1, 8, 9, 15, 30}:
            bits = 8
        elif type_id in {6, 7, 13, 14}:
            bits = 6
        else:
            bits = 4
    return _INKLING_SK_BLOCK_BYTES[bits]


def _measure_inkling(metadata, tensors, arch, backend, hip_transcode):
    """Mirror the fused GGUF loader and the backend's per-body allocation."""
    if backend not in ("cuda", "hip"):
        raise ValueError("Inkling Expert budgets require a CUDA or HIP runtime")
    prefix = arch + "."
    count = _positive(metadata.get(prefix + "expert_count"), "Inkling expert_count")
    width = _positive(metadata.get(prefix + "embedding_length"), "Inkling embedding_length")
    hidden = _positive(metadata.get(prefix + "expert_feed_forward_length"), "Inkling expert_feed_forward_length")
    blocks = _positive(metadata.get(prefix + "block_count"), "Inkling block_count")
    if arch == "inkling2" and "loop_res_query.weight" in tensors:
        start = metadata.get(prefix + "loop.core_start", metadata.get(prefix + "loop.start", 6))
        end = metadata.get(prefix + "loop.core_end", metadata.get(prefix + "loop.end", 18))
        loops = _positive(metadata.get(prefix + "num_loops", metadata.get(prefix + "loop.count", 3)), "Inkling loop count")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("Invalid Inkling loop geometry")
        if blocks > end + 6:
            blocks -= (loops - 1) * (end - start)
    if (blocks, count, width, hidden) != (24, 128, 2304, 1152):
        raise ValueError("Unsupported Inkling 2 Mini Expert geometry")
    total = 0
    force_sk, sk_bits = hip_transcode
    for layer in range(blocks):
        layer_prefix = f"blk.{layer}.ffn_"
        gate = tensors.get(layer_prefix + "gate_up_exps.weight")
        down = tensors.get(layer_prefix + "down_exps.weight")
        if gate is None or down is None or gate.shape != [width, 2 * hidden, count] or down.shape != [hidden, width, count]:
            raise ValueError(f"Invalid fused Inkling Expert tensors in layer {layer}")
        if gate.type_id == down.type_id == 40:
            for plane in ("gate_up", "down"):
                if layer_prefix + plane + "_exps.scale" not in tensors:
                    raise ValueError(f"Missing NVFP4 Expert scales in Inkling layer {layer}")
            # Fused NVFP4 is exactly repacked to SF4: 16-byte header, then
            # one 4-bit code per weight and one byte of scale per 16 weights.
            body = 16 + 2 * hidden * (width // 2 + width // 16) + width * (hidden // 2 + hidden // 16)
        else:
            body = _inkling_raw_body_bytes(gate, down, count)
            if backend == "hip" and (force_sk or not {gate.type_id, down.type_id} <= _INKLING_HIP_RAW_TYPES):
                # HIP's S_K transcode uses paired down rows, so 1152 need
                # not divide 256 as long as their combined width does.
                gate_block = _inkling_sk_block_bytes(gate.type_id, sk_bits)
                down_block = _inkling_sk_block_bytes(down.type_id, sk_bits)
                down_offset = (2 * hidden + 2 * hidden * (width // 256) * gate_block + 63) & ~63
                body = down_offset + (width // 2) * ((2 * hidden) // 256) * down_block
        if backend == "hip":
            body = (body + 63) & ~63
        total += body * count
    return {"total_expert_bytes": total, "expert_total": blocks * count, "experts_per_layer": count}


def _measure(files, cfg, *, hip_transcode=(False, 0)):
    metadata = files[0].metadata
    arch = metadata.get("general.architecture")
    if arch not in _FAMILIES | _INKLING_ARCHES:
        raise ValueError(f"Unsupported Expert model architecture: {arch}")
    profile_arch = "inkling2_loop" if arch in _INKLING_ARCHES else arch
    if cfg.get("arch") not in (None, profile_arch):
        raise ValueError("Configured architecture does not match the GGUF")
    tensors = {}
    for header in files:
        for tensor in header.tensors:
            if tensor.name in tensors:
                raise ValueError(f"Duplicate GGUF tensor: {tensor.name}")
            tensors[tensor.name] = tensor
    if arch in _INKLING_ARCHES:
        return _measure_inkling(metadata, tensors, arch, cfg.get("backend", "cuda"), hip_transcode)
    count = _positive(metadata.get(f"{arch}.expert_count"), f"{arch}.expert_count")
    blocks = _positive(metadata.get(f"{arch}.block_count"), f"{arch}.block_count")
    nextn = metadata.get(f"{arch}.nextn_predict_layers", 0)
    if isinstance(nextn, bool) or not isinstance(nextn, int) or nextn not in (0, 1) or nextn >= blocks:
        raise ValueError("Unsupported next-token Expert layer geometry")
    qwen = arch in {"qwen35moe", "clover_moe"}
    # Qwen loads its one inline MTP block even with drafts disabled. Gemma's
    # external gemma4-assistant is dense, and never contributes routed Experts.
    loaded = blocks if qwen else blocks - nextn
    if not qwen and nextn:
        args = cfg.get("args", [])
        mtp = args[args.index("--mtp") + 1] if "--mtp" in args[:-1] else 0
        if str(mtp) not in {"0", "0.0"}:
            raise ValueError("Inline Gemma/LFM MTP Expert geometry is unsupported")
    total = expert_total = 0
    supported = _QWEN_TYPES if qwen else _OTHER_TYPES
    leading_dense = metadata.get(f"{arch}.leading_dense_block_count", 0)
    if arch == "lfm2moe" and (isinstance(leading_dense, bool) or not isinstance(leading_dense, int) or not 0 <= leading_dense < loaded):
        raise ValueError("Invalid leading dense layer count")
    for layer in range(loaded):
        prefix = f"blk.{layer}."
        if arch == "lfm2moe" and layer < leading_dense:
            continue
        if arch == "gemma4" and prefix + "ffn_gate_inp.weight" not in tensors:
            continue  # Gemma chooses its routed layers by router presence
        fused = tensors.get(prefix + "ffn_gate_up_exps.weight")
        gate = tensors.get(prefix + "ffn_gate_exps.weight")
        up = tensors.get(prefix + "ffn_up_exps.weight")
        down = tensors.get(prefix + "ffn_down_exps.weight")
        if not any((fused, gate, up, down)):
            raise ValueError(f"Missing routed Experts in layer {layer}")
        if down is None or (fused is None and (gate is None or up is None)):
            raise ValueError(f"Incomplete routed-Expert projections in layer {layer}")
        if qwen and fused is not None:
            raise ValueError("Qwen/Clover native Experts require separate gate/up tensors")
        down_bytes, down_block = _payload(down, supported)
        if down.shape[2] != count:
            raise ValueError("Expert count does not match routed tensor shapes")
        width, hidden = down.shape[1], down.shape[0]
        if fused is not None:
            gu_bytes, gu_block = _payload(fused, supported)
            if fused.shape != [width, 2 * hidden, count]:
                raise ValueError("Fused gate/up Expert shape does not match down")
            gu_type = fused.type_id
        else:
            gate_bytes, gu_block = _payload(gate, supported)
            up_bytes, _ = _payload(up, supported)
            if gate.type_id != up.type_id or gate.shape != [width, hidden, count] or up.shape != gate.shape:
                raise ValueError("Gate/up Expert shapes or quantization types differ")
            gu_bytes, gu_type = gate_bytes + up_bytes, gate.type_id
        declared_width = metadata.get(f"{arch}.embedding_length")
        declared_hidden = metadata.get(f"{arch}.expert_feed_forward_length")
        if ((declared_width is not None and declared_width != width) or
                (declared_hidden is not None and declared_hidden != hidden)):
            raise ValueError("Expert tensor shape does not match model metadata")
        if gu_bytes % count or down_bytes % count:
            raise ValueError("Expert payload is not divisible by expert_count")
        payload = (gu_bytes + down_bytes) // count
        if qwen:
            if width % 256:
                raise ValueError("Qwen/Clover Expert input width must be divisible by 256")
            if gu_type == 40 or down.type_id == 40:
                payload += 16  # NativeExpertLayout's three NVFP4 scales plus padding
            unit = math.lcm(gu_block, down_block, 16)
            stride = (payload + unit - 1) // unit * unit
            total += stride * count + 4096  # kMmqTail, once per layer arena
        else:
            unit = math.lcm(gu_block, down_block, 8)
            stride = (payload + 256 + unit - 1) // unit * unit
            total += stride * count
        expert_total += count
    if not expert_total:
        raise ValueError("The model has no routed Experts")
    if not qwen:
        total += 256  # common cache-arena tail, outside all per-Expert slots
    return {"total_expert_bytes": total, "expert_total": expert_total, "experts_per_layer": count}


def _signatures(path):
    """Find the standard shard names without re-reading the first header."""
    paths = [path]
    split = re.fullmatch(r"(.*)-([0-9]{5})-of-([0-9]{5})\.gguf", str(path))
    if split:
        number, count = int(split[2]), int(split[3])
        if number != 1 or not 1 <= count <= 1024:
            raise ValueError("Choose the first GGUF shard for Expert geometry")
        paths = [Path(f"{split[1]}-{index:05d}-of-{count:05d}.gguf") for index in range(1, count + 1)]
    out = []
    for item in paths:
        try:
            info = item.stat()
            out.append((str(item), info.st_size, info.st_mtime_ns))
        except FileNotFoundError:
            # A standalone model can have a split-looking name. The first
            # header decides whether the absent siblings are actually needed.
            out.append((str(item), -1, -1))
    return tuple(out)


def _hip_transcode_options(cfg):
    # Match the two native getenv/atoi options, including config overrides.
    def option(name):
        value = (cfg.get("env") or {}).get(name, os.environ.get(name, ""))
        match = re.match(r"\s*([+-]?[0-9]+)", str(value))
        return int(match.group(1)) if match else 0

    bits = option("INK_GGUF_SK_BITS")
    return bool(option("INK_HIP_FORCE_SK_TRANSCODE")), bits if bits in _INKLING_SK_BLOCK_BYTES else 0


@lru_cache(maxsize=16)
def _cached_geometry(signatures, arch, mtp, backend, hip_transcode):
    path = Path(signatures[0][0])
    result = _measure(_headers(path), {"arch": arch, "args": ["--mtp", mtp], "backend": backend},
                      hip_transcode=hip_transcode)
    if _signatures(path) != signatures:
        raise ValueError("The GGUF changed while Expert geometry was being read; try again")
    # Do not return the cached mapping itself: endpoint callers may extend it.
    return tuple(result[key] for key in ("total_expert_bytes", "expert_total", "experts_per_layer"))


def geometry(cfg):
    """Full native cache budget and Expert count, using only local GGUF headers.

    `total_expert_bytes` includes slot padding and native arena guard tails so
    a percentage can be translated into the engine's `--expert-vram` budget.
    Actual residence can be lower: native allocation retains its VRAM reserve
    and physical page granularity. This is not total model memory or GPU load.

    `experts_per_layer` supplies the native per-layer slot target when that
    engine supports safe physical clamping of fixed slots.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Expected a model configuration")
    try:
        path = model_path(cfg)
        if path is None or not path.is_file():
            raise ValueError("The configured GGUF model is not available")
        args = cfg.get("args", [])
        mtp = str(args[args.index("--mtp") + 1]) if "--mtp" in args[:-1] else "0"
        arch, backend = cfg.get("arch"), cfg.get("backend", "cuda")
        transcode = _hip_transcode_options(cfg) if backend == "hip" and arch in (None, "inkling2_loop") else (False, 0)
        total, experts, per_layer = _cached_geometry(_signatures(path), arch, mtp, backend, transcode)
        return {"total_expert_bytes": total, "expert_total": experts, "experts_per_layer": per_layer}
    except (OSError, struct.error, KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Unable to read routed-Expert geometry: {error}") from error
