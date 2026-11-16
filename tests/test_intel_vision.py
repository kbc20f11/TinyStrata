"""Startup adapter identity and explicit encoder-backend policy, without GPUs."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from serve.intel_vision import prepare_encoder


class IntelVisionStartup(unittest.TestCase):
    def test_public_features_require_both_encoder_and_native_image_readiness(self):
        from serve.server import intel_feature_fields
        for encoder, native, expected in ((True, 1, True), (False, 1, False), (True, 0, False), (True, None, False)):
            svc = SimpleNamespace(backend="sycl", model="qwen3.6-35b-a3b",
                                  engine=SimpleNamespace(info={"vision": native}),
                                  vision_available=lambda: encoder)
            self.assertEqual(intel_feature_fields(svc)["features"]["images"], expected)

    def config(self):
        return {"backend": "sycl", "arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b",
                "exe": "native.exe", "vision_mode": "gpu", "gpu": 1,
                "args": ["--model", "model.gguf", "--device", "1"]}

    def capabilities(self):
        return {"backend": "sycl", "sycl": True, "cpu": False, "device_luid": True,
                "warmup_before_ready": True, "input_protocol": "enc-tabs"}

    def device(self):
        return {"schema": 1, "backend": "sycl", "device_index": 1, "vendor_id": 0x8086,
                "device_id": 0xB082, "adapter_luid": "00000000:00001234", "adapter_node_mask": 1}

    def test_actual_language_adapter_is_bound_before_encoder_start(self):
        cfg, vision = self.config(), {"exe": "vision.exe", "model": "model.gguf", "mmproj": "proj.gguf", "gpu": True}
        before = deepcopy((cfg, vision))
        with patch("serve.intel_vision._report", side_effect=[self.capabilities(), self.device()]) as query:
            selected = prepare_encoder(cfg, vision, {})
        self.assertEqual(query.call_args_list[0].args[1], ["--capabilities"])
        self.assertEqual(query.call_args_list[1].args[1], ["--device-info", "--device", "1"])
        self.assertEqual(selected["device_luid"], "00000000:00001234")
        self.assertEqual(selected["input_protocol"], "enc-tabs")
        self.assertFalse(selected["elastic"])
        self.assertEqual((cfg, vision), before)

    def test_wrong_adapter_or_missing_identity_never_falls_back(self):
        for field, bad in (("vendor_id", 0x10DE), ("device_index", 0), ("adapter_luid", "B390"),
                           ("adapter_node_mask", 0)):
            device = self.device(); device[field] = bad
            with self.subTest(field=field), patch("serve.intel_vision._report", side_effect=[self.capabilities(), device]):
                with self.assertRaisesRegex(ValueError, "adapter identity"):
                    prepare_encoder(self.config(), {"exe": "v.exe", "gpu": True}, {})

    def test_gpu_request_rejects_other_backend_or_missing_warmup(self):
        for field, bad in (("backend", "cuda"), ("sycl", False), ("device_luid", False), ("warmup_before_ready", False)):
            caps = self.capabilities(); caps[field] = bad
            with self.subTest(field=field), patch("serve.intel_vision._report", return_value=caps):
                with self.assertRaisesRegex(ValueError, "matching SYCL"):
                    prepare_encoder(self.config(), {"exe": "v.exe", "gpu": True}, {})

    def test_cpu_is_explicit_and_does_not_query_a_gpu(self):
        with patch("serve.intel_vision._report", return_value=self.capabilities()) as query:
            with self.assertRaisesRegex(ValueError, "GPU-only"):
                prepare_encoder(self.config(), {"exe": "v.exe", "gpu": False}, {})
            self.assertEqual(query.call_count, 1)
        with patch("serve.intel_vision._report", return_value={"backend": "cpu"}) as query:
            selected = prepare_encoder(self.config(), {"exe": "v.exe", "gpu": False}, {})
            self.assertEqual(selected["backend"], "cpu")
            self.assertEqual(query.call_count, 1)

    def test_dimension_count_requires_actual_pinned_encoder_capability(self):
        for advertised, pin, expected in ((True, "3cf03257f219afbe7334045ff7c6a06ac68c627d", True),
                (False, "3cf03257f219afbe7334045ff7c6a06ac68c627d", False), (True, "unknown", False)):
            caps = self.capabilities() | {"count_dimensions":advertised,"llama_base_commit":pin}
            with self.subTest(advertised=advertised,pin=pin), \
                 patch("serve.intel_vision._report", side_effect=[caps,self.device()]):
                selected = prepare_encoder(self.config(), {"exe":"v.exe","gpu":True,"count_dimensions":True}, {})
            self.assertIs(selected["count_dimensions"], expected)


class IntelProjectorReuse(unittest.TestCase):
    def test_local_projector_is_pinned_and_explicit_selection_is_retained(self):
        from serve.model_setup import configure_vision
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            local = base / "models"; local.mkdir()
            model = local / "model.gguf"; model.write_bytes(b"model")
            encoder = base / "vision.exe"; encoder.write_bytes(b"encoder")
            projector = local / "mmproj.gguf"; projector.write_bytes(b"verified")
            artifact = SimpleNamespace(filename=projector.name, size=8,
                                       sha256=hashlib.sha256(b"verified").hexdigest())
            cfg = {"backend": "sycl", "arch": "qwen35moe", "model_name": "qwen3.6-35b-a3b",
                   "exe": str(base / "native.exe"), "vision_exe": str(encoder),
                   "vision_mode": "gpu", "download_dir": str(base / "managed"),
                   "args": ["--model", str(model)]}
            proof = SimpleNamespace(report={}, executable=cfg["exe"], sha256="a"*64)
            with patch("serve.model_setup.validate_intel_executable", return_value=proof), \
                 patch("serve.intel_capabilities.supports_images", return_value=True), \
                 patch("serve.model_download.projector_for_arch", return_value=artifact):
                ready = configure_vision(cfg, model, verified=True)
                self.assertEqual(ready["vision"]["mmproj"], str(projector.resolve()))
                self.assertEqual(ready["vision_mmproj"], str(projector.resolve()))
                # A downloaded target's verified flag cannot bless another file.
                projector.write_bytes(b"tampered")
                bad = configure_vision(cfg, model, verified=True)
                self.assertNotIn("vision", bad)
                self.assertTrue(bad["vision_download_required"])
                explicit = configure_vision(cfg | {"vision_mmproj": str(base / "chosen.gguf")}, model)
                self.assertEqual(explicit["vision_mmproj"], str(base / "chosen.gguf"))
                self.assertNotIn("vision", explicit)


if __name__ == "__main__":
    unittest.main()
