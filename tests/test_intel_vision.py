"""Startup adapter identity and explicit encoder-backend policy, without GPUs."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from serve.intel_vision import prepare_encoder


class IntelVisionStartup(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
