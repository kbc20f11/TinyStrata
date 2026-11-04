"""Native Expert cache geometry: no model payload, GPU or inference process."""
from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import expert_placement as E
from serve.gguf_reader import TensorInfo, GGML_TYPES


def tensor(name, shape, dtype):
    return TensorInfo(name, shape, dtype, GGML_TYPES[dtype], 0)


def model(arch="qwen35moe", blocks=2, experts=4, width=2048, hidden=512, gate_type=18, down_type=21, fused=False, dense=0, nextn=0):
    metadata = {"general.architecture": arch, f"{arch}.block_count": blocks, f"{arch}.expert_count": experts,
                f"{arch}.embedding_length": width, f"{arch}.expert_feed_forward_length": hidden,
                f"{arch}.nextn_predict_layers": nextn}
    if arch == "lfm2moe":
        metadata[f"{arch}.leading_dense_block_count"] = dense
    tensors = []
    for layer in range(dense, blocks):
        p = f"blk.{layer}."
        tensors.append(tensor(p + "ffn_gate_inp.weight", [width, experts], 0))
        if fused:
            tensors.append(tensor(p + "ffn_gate_up_exps.weight", [width, 2 * hidden, experts], gate_type))
        else:
            tensors += [tensor(p + name + "_exps.weight", [width, hidden, experts], gate_type) for name in ("ffn_gate", "ffn_up")]
        tensors.append(tensor(p + "ffn_down_exps.weight", [hidden, width, experts], down_type))
    return SimpleNamespace(metadata=metadata, tensors=tensors)


class ExpertGeometryTests(unittest.TestCase):
    def setUp(self):
        E._cached_geometry.cache_clear()

    def test_qwen_and_clover_native_alignment(self):
        for arch in ("qwen35moe", "clover_moe"):
            with self.subTest(arch=arch):
                header = model(arch)
                payload = 2048 * 512 // 256 * (98 * 2 + 110)
                unit = math.lcm(98, 110, 16)
                stride = (payload + unit - 1) // unit * unit
                self.assertEqual(E._measure([header], {}), {"total_expert_bytes": 2 * (4 * stride + 4096), "expert_total": 8,
                                                          "experts_per_layer": 4})

    def test_qwen_inline_mtp_remains_loaded_when_drafts_off(self):
        header = model(blocks=3, nextn=1)
        result = E._measure([header], {"args": ["--mtp", "0"]})
        self.assertEqual(result["expert_total"], 12)
        self.assertEqual(result["total_expert_bytes"], E._measure([model(blocks=2)], {})["total_expert_bytes"] // 2 * 3)

    def test_qwen_nvfp4_scale_tail_is_included(self):
        header = model(gate_type=40, down_type=40)
        payload = 2048 * 512 // 64 * 36 * 3 + 16
        unit = math.lcm(36, 16)
        stride = (payload + unit - 1) // unit * unit
        self.assertEqual(E._measure([header], {})["total_expert_bytes"], 2 * (stride * 4 + 4096))

    def test_gemma_qat_fused_layout_excludes_dense_scales(self):
        header = model("gemma4", width=2816, hidden=704, gate_type=2, down_type=2, fused=True)
        header.tensors.append(tensor("blk.0.ffn_down_exps.scale", [4], 0))
        header.tensors.append(tensor("blk.0.ffn_gate.weight", [2816, 11264], 8))
        payload = 2816 * 704 // 32 * 18 * 3
        unit = math.lcm(18, 8)
        slot = (payload + 256 + unit - 1) // unit * unit
        self.assertEqual(E._measure([header], {}), {"total_expert_bytes": 2 * 4 * slot + 256, "expert_total": 8,
                                                  "experts_per_layer": 4})

    def test_lfm_leading_dense_and_mixed_quantization(self):
        header = model("lfm2moe", blocks=4, dense=2, width=2048, hidden=1536, gate_type=12, down_type=14)
        payload = 2048 * 1536 // 256 * (144 * 2 + 210)
        unit = math.lcm(144, 210, 8)
        slot = (payload + 256 + unit - 1) // unit * unit
        self.assertEqual(E._measure([header], {}), {"total_expert_bytes": 2 * 4 * slot + 256, "expert_total": 8,
                                                  "experts_per_layer": 4})

    def test_shards_merge_without_weight_reads(self):
        full = model()
        first = SimpleNamespace(metadata=full.metadata, tensors=full.tensors[:3])
        second = SimpleNamespace(metadata={}, tensors=full.tensors[3:])
        self.assertEqual(E._measure([first, second], {}), E._measure([full], {}))

    def test_missing_unsupported_and_inconsistent_headers_rejected(self):
        for mutate in (
            lambda h: h.metadata.update({"general.architecture": "unknown"}),
            lambda h: h.metadata.update({"qwen35moe.expert_count": 0}),
            lambda h: h.tensors.pop(),
            lambda h: h.tensors.append(h.tensors[0]),
            lambda h: setattr(h.tensors[1], "type_id", 999),
            lambda h: h.tensors[1].shape.__setitem__(0, 2049),
            lambda h: h.tensors[1].shape.__setitem__(2, 3),
        ):
            header = model()
            mutate(header)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                E._measure([header], {})

    def test_public_geometry_reads_headers_only(self):
        header = model()
        with patch.object(E, "model_path", return_value=Path(__file__)), patch.object(E, "GGUFFile", return_value=header) as reader:
            self.assertEqual(E.geometry({})["expert_total"], 8)
            reader.assert_called_once_with(Path(__file__))

    def test_missing_model_and_wrong_config_rejected(self):
        with self.assertRaises(ValueError):
            E.geometry({})
        with self.assertRaises(ValueError):
            E.geometry(None)
        with self.assertRaises(ValueError):
            E._measure([model()], {"arch": "gemma4"})

    def test_missing_routed_weights_are_not_misclassified_as_dense(self):
        for arch in ("gemma4", "lfm2moe"):
            header = model(arch)
            header.tensors = [t for t in header.tensors if not t.name.startswith("blk.1.ffn_") or t.name == "blk.1.ffn_gate_inp.weight"]
            with self.subTest(arch=arch), self.assertRaisesRegex(ValueError, "Missing routed"):
                E._measure([header], {})

    def test_inline_legacy_mtp_fails_explicitly_instead_of_miscounting(self):
        header = model("gemma4", nextn=1)
        with self.assertRaisesRegex(ValueError, "Inline"):
            E._measure([header], {"args": ["--mtp", "1"]})

    def test_cache_reuses_header_and_returns_independent_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"header fixture")
            cfg = {"args": ["--model", str(path)]}
            with patch.object(E, "GGUFFile", return_value=model()) as reader:
                first = E.geometry(cfg)
                first["expert_total"] = -1
                self.assertEqual(E.geometry(cfg)["expert_total"], 8)
                reader.assert_called_once_with(path.resolve())
                E.geometry({**cfg, "args": cfg["args"] + ["--mtp", "3"]})
                self.assertEqual(reader.call_count, 2)
                E.geometry({**cfg, "arch": "qwen35moe"})
                self.assertEqual(reader.call_count, 3)

    def test_cache_invalidates_path_size_and_nanosecond_mtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            other = Path(temporary) / "other.gguf"
            path.write_bytes(b"header")
            other.write_bytes(b"header")
            with patch.object(E, "GGUFFile", return_value=model()) as reader:
                cfg = {"args": ["--model", str(path)]}
                E.geometry(cfg)
                original = path.stat()
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1000000000))
                E.geometry(cfg)
                self.assertEqual(reader.call_count, 2)
                path.write_bytes(b"longer header")
                os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1000000000))
                E.geometry(cfg)
                self.assertEqual(reader.call_count, 3)
                E.geometry({"args": ["--model", str(other)]})
                self.assertEqual(reader.call_count, 4)

    def test_cache_invalidates_when_any_shard_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            first_path = Path(temporary) / "model-00001-of-00002.gguf"
            second_path = Path(temporary) / "model-00002-of-00002.gguf"
            first_path.write_bytes(b"first")
            second_path.write_bytes(b"second")
            full = model()
            first = SimpleNamespace(metadata={**full.metadata, "split.no": 0, "split.count": 2}, tensors=full.tensors[:4])
            second = SimpleNamespace(metadata={"split.no": 1, "split.count": 2}, tensors=full.tensors[4:])
            with patch.object(E, "GGUFFile", side_effect=lambda path: first if path == first_path else second) as reader:
                cfg = {"args": ["--model", str(first_path)]}
                self.assertEqual(E.geometry(cfg)["expert_total"], 8)
                E.geometry(cfg)
                self.assertEqual(reader.call_count, 2)
                second_path.write_bytes(b"second changed")
                E.geometry(cfg)
                self.assertEqual(reader.call_count, 4)
                second_path.unlink()
                with self.assertRaisesRegex(ValueError, "Missing GGUF shard"):
                    E.geometry(cfg)

    def test_failed_geometry_is_not_cached(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"header")
            bad = model()
            bad.metadata["general.architecture"] = "unknown"
            with patch.object(E, "GGUFFile", side_effect=[bad, model()]) as reader:
                cfg = {"args": ["--model", str(path)]}
                with self.assertRaises(ValueError):
                    E.geometry(cfg)
                self.assertEqual(E.geometry(cfg)["expert_total"], 8)
                self.assertEqual(reader.call_count, 2)

    def test_files_changed_during_header_read_are_not_cached(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            path.write_bytes(b"header")
            def read(chosen):
                chosen.write_bytes(b"header changed")
                return model()
            cfg = {"args": ["--model", str(path)]}
            with patch.object(E, "GGUFFile", side_effect=read):
                with self.assertRaisesRegex(ValueError, "changed while"):
                    E.geometry(cfg)
            with patch.object(E, "GGUFFile", return_value=model()) as reader:
                self.assertEqual(E.geometry(cfg)["expert_total"], 8)
                reader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
