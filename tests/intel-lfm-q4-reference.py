"""Independent, bounded Q4_K rows via gguf/NumPy; never loads a model engine.

Usage: python tests/intel-lfm-q4-reference.py --model PINNED.gguf --output NEW.bin
       python tests/intel-lfm-q4-reference.py --synthetic --output NEW.bin
Only 17 rows of dense tensors and four selected experts are streamed. Unselected
expert rows in the local numerical fixture are zero; the source GGUF is read-only.
The source's full SHA is not recomputed or claimed by this bounded extractor.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.gguf_reader import GGUFFile

MAGIC = 0x31324634514D464C
IDS = (0, 63, 17, 42)
ROUTES = (IDS, (63, 42, 0, 17), (17, 0, 42, 63))
PINNED_BYTES = 14526258464
NAMES = ('blk.0.shortconv.in_proj.weight', 'blk.0.ffn_gate.weight', 'blk.2.attn_k.weight',
         'blk.2.ffn_gate_exps.weight', 'blk.2.ffn_up_exps.weight', 'blk.5.ffn_down_exps.weight')


def case(name, width, raw, experts):
    rows, row_bytes = 17, width // 256 * 144
    decoded = dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(experts * rows, row_bytes),
                         GGMLQuantizationType.Q4_K).reshape(experts, rows, width).astype('<f4')
    x = np.stack([np.sin(np.arange(width, dtype=np.float64) * (.127 + v * .019) + v) for v in range(4)]).astype('<f4')
    references = []
    if experts == 1:
        references = [decoded[0].astype(np.float64) @ vector.astype(np.float64) for vector in x]
        values = decoded[0]
    else:
        for route in ROUTES:
            for separate in (False, True):
                references.extend(decoded[expert].astype(np.float64) @ x[p if separate else 0].astype(np.float64)
                                  for p, expert in enumerate(route))
        values = decoded[list(IDS)]
    assert np.isfinite(decoded).all() and np.isfinite(x).all()
    return name, width, experts, raw, x, values.astype('<f4'), np.asarray(references, dtype='<f8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if bool(args.model) == args.synthetic:
        parser.error('Select exactly one of --model or --synthetic')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error('Use new output paths; old numerical evidence is preserved')
    cases, evidence, streamed = [], [], 0
    if args.model:
        source = args.model.resolve()
        if source.stat().st_size != PINNED_BYTES:
            raise ValueError('Expected the unchanged pinned LFM Q4_K_M file size')
        model = GGUFFile(source)
        if model.metadata.get('general.architecture') != 'lfm2moe':
            raise ValueError('Expected the genuine LFM model')
        by_name = {tensor.name: tensor for tensor in model.tensors}
        with source.open('rb') as stream:
            for name in NAMES:
                tensor = by_name[name]
                width, full_rows = tensor.shape[:2]
                experts = tensor.shape[2] if len(tensor.shape) == 3 else 1
                if tensor.type_id != 12 or width % 256 or full_rows < 17 or experts not in (1, 64):
                    raise ValueError('Bounded fixture geometry/type differs from the pinned target')
                row_bytes = width // 256 * 144
                raw = bytearray(experts * 17 * row_bytes)
                ranges = []
                for expert in (IDS if experts == 64 else (0,)):
                    offset = model.data_start + tensor.offset + expert * full_rows * row_bytes
                    stream.seek(offset)
                    payload = stream.read(17 * row_bytes)
                    if len(payload) != 17 * row_bytes:
                        raise ValueError('Truncated real Q4_K tensor slice')
                    raw[expert * 17 * row_bytes:(expert + 1) * 17 * row_bytes] = payload
                    streamed += len(payload)
                    if streamed > 1024 * 1024:
                        raise ValueError('Bounded extraction exceeded its one-MiB source-payload limit')
                    ranges.append({'expert': expert, 'offset': offset, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
                cases.append(case(name, width, raw, experts))
                evidence.append({'tensor': name, 'source_shape': tensor.shape, 'fixture_rows': 17, 'ranges': ranges})
    else:
        for experts in (1, 64):
            width = 256
            raw = bytearray(experts * 17 * 144)
            rng = np.random.default_rng(90210 + experts)
            for expert in (IDS if experts == 64 else (0,)):
                for row in range(17):
                    at = (expert * 17 + row) * 144
                    block = bytearray(rng.integers(0, 256, 144, dtype=np.uint8).tobytes())
                    block[:4] = np.array([0 if row == 0 else .0078125, .00390625], dtype='<f2').tobytes()
                    if row == 1:
                        block[4:16] = b'\xff' * 12  # All split six-bit scales/minima at63.
                    raw[at:at + 144] = block
            cases.append(case('synthetic-all-groups-' + str(experts), width, raw, experts))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as out:
        out.write(struct.pack('<QI', MAGIC, len(cases)))
        for name, width, experts, raw, x, values, references in cases:
            label = name.encode('utf-8')
            out.write(struct.pack('<IIIIQI', 17, width, experts, 4, len(raw), len(label)))
            out.write(label)
            for payload in (raw, x.tobytes(), values.tobytes(), references.tobytes()):
                out.write(payload)
    receipt = {'schema': 1, 'independent_decoder': 'gguf.quants.dequantize Q4_K NumPy', 'cases': len(cases),
               'synthetic': args.synthetic, 'source_payload_bytes_streamed': streamed, 'source_read_only': True,
               'full_model_sha_recomputed': False, 'source': str(args.model) if args.model else None,
               'selected_experts': list(IDS), 'rows': 17, 'ranges': evidence,
               'fixture_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
               'device_or_queue_or_model_engine_started': False}
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
