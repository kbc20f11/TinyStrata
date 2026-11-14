"""Generate an independent NumPy/gguf reference from unchanged GGUF tensors.

Usage: python tests/intel-quant-reference.py MODEL.gguf OUTPUT.bin
Requires gguf and numpy in a developer environment. No downloads or writes to
the model. The binary fixture is a local validation artifact, not a weight file
to distribute. Its floating-point rows come from gguf's NumPy dequantizers.
"""
import math
import pathlib
import struct
import sys

import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from serve.gguf_reader import GGUFFile, BLOCK_GEOMETRY


def generate(source: pathlib.Path, destination: pathlib.Path) -> None:
    model = GGUFFile(source)
    selected = {}
    for tensor in model.tensors:
        selected.setdefault(tensor.type_id, []).append(tensor)
    cases = []
    with source.open("rb") as fh:
        for type_id, tensors in sorted(selected.items()):
            for tensor in (tensors[0], tensors[len(tensors) // 2], tensors[-1]):
                width = tensor.shape[0]
                rows = min(17, math.prod(tensor.shape[1:]))
                block, size = BLOCK_GEOMETRY[tensor.type_name]
                row_bytes = width // block * size
                fh.seek(model.data_start + tensor.offset)
                raw = fh.read(rows * row_bytes)
                if len(raw) != rows * row_bytes:
                    raise ValueError("Truncated tensor")
                data = np.frombuffer(raw, dtype=np.uint8).reshape(rows, row_bytes)
                decoded = dequantize(data, GGMLQuantizationType(type_id)).astype("<f4")
                x = np.sin(np.arange(width, dtype=np.float64) * .127).astype("<f4")
                expected = decoded.astype(np.float64) @ x.astype(np.float64)
                cases.append((type_id, rows, width, raw, x, decoded, expected))
                print(f"{tensor.name}: {tensor.type_name} {rows}x{width}")
    with destination.open("wb") as fh:
        fh.write(struct.pack("<I", len(cases)))
        for type_id, rows, width, raw, x, decoded, expected in cases:
            fh.write(struct.pack("<iiiQ", type_id, rows, width, len(raw)))
            fh.write(raw)
            fh.write(x.tobytes())
            fh.write(decoded.tobytes())
            fh.write(expected.astype("<f8").tobytes())
    print(f"Wrote {len(cases)} cases: {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: intel-quant-reference.py MODEL.gguf OUTPUT.bin")
    generate(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
