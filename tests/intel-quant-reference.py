"""Generate an independent NumPy/gguf reference from unchanged GGUF tensors.

Usage: python tests/intel-quant-reference.py MODEL.gguf OUTPUT.bin
       python tests/intel-quant-reference.py --type18-slices MANIFEST.json OUTPUT.bin
Requires gguf and numpy in a developer environment. No downloads or writes to
the model. The binary fixture is a local validation artifact, not a weight file
to distribute. Its floating-point rows come from gguf's NumPy dequantizers.
"""
import math
import hashlib
import json
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
    write_cases(cases, destination)


def write_cases(cases, destination: pathlib.Path) -> None:
    with destination.open("wb") as fh:
        fh.write(struct.pack("<I", len(cases)))
        for type_id, rows, width, raw, x, decoded, expected in cases:
            fh.write(struct.pack("<iiiQ", type_id, rows, width, len(raw)))
            fh.write(raw)
            fh.write(x.tobytes())
            fh.write(decoded.tobytes())
            fh.write(expected.astype("<f8").tobytes())
    print(f"Wrote {len(cases)} cases: {destination}")


def generate_type18_slices(manifest_path: pathlib.Path, destination: pathlib.Path) -> None:
    """Use bounded, hashed raw slices; never require or fetch a full Tiel model."""
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("Slice manifest exceeds its bound")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures, ranges = manifest["fixtures"], manifest["ranges"]
    if not fixtures or len(fixtures) > 3:
        raise ValueError("Expected at most three Type18 slices")
    hashes = {item["file"]: item["sha256"] for item in ranges}
    cases, total = [], 0
    for fixture in fixtures:
        rows, width = fixture["rows"], fixture["width"]
        name = fixture["raw_file"]
        if pathlib.Path(name).name != name or not 1 <= rows <= 17 or not 256 <= width <= 8192 or width % 256:
            raise ValueError("Invalid bounded Type18 slice")
        source = manifest_path.parent / name
        expected_bytes = rows * (width // 256) * 98
        total += expected_bytes
        if total > 1024 * 1024 or source.stat().st_size != expected_bytes:
            raise ValueError("Type18 slice payload exceeds its bound")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != hashes[name]:
            raise ValueError("Type18 slice SHA256 mismatch")
        data = np.frombuffer(raw, dtype=np.uint8).reshape(rows, width // 256 * 98)
        decoded = dequantize(data, GGMLQuantizationType.IQ3_XXS).astype("<f4")
        if not np.isfinite(decoded).all():
            raise ValueError("Nonfinite Type18 reference")
        x = np.sin(np.arange(width, dtype=np.float64) * .127).astype("<f4")
        expected = decoded.astype(np.float64) @ x.astype(np.float64)
        cases.append((18, rows, width, raw, x, decoded, expected))
        print(f"{fixture['tensor']}: IQ3_XXS bounded slice {rows}x{width}")
    write_cases(cases, destination)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--type18-slices":
        generate_type18_slices(pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
    elif len(sys.argv) == 3:
        generate(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
    else:
        raise SystemExit("Usage: intel-quant-reference.py MODEL.gguf OUTPUT.bin")
