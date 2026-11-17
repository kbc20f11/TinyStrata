"""Legacy image-generation config must not obstruct image understanding or change user data."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import tinyconfig
from serve.model_switch import MODEL_FIELDS


class GeneratorRetirement(unittest.TestCase):
    def test_only_root_generator_setting_is_removed_before_backend_validation(self):
        for backend in ('cuda', 'hip', 'sycl'):
            for value in ({'exe': r'Z:\missing\sd-cli.exe', 'vae': 'retained-file.gguf'}, True, None, 'malformed'):
                with self.subTest(backend=backend, legacy=value):
                    cfg = {'backend': backend, 'vision': {'exe': 'vision.exe', 'mmproj': 'mmproj.gguf'},
                           'vision_mode': 'gpu', 'images': value, 'api_key': 'private',
                           'private_state': {'images': [{'url': 'data:image/png;base64,AAAA'}]}}
                    before = deepcopy(cfg)
                    after, changed = tinyconfig.migrate_image_generation_settings(cfg)
                    self.assertTrue(changed)
                    self.assertEqual(after, {key: val for key, val in before.items() if key != 'images'})
                    self.assertEqual(cfg, before)
                    again, changed = tinyconfig.migrate_image_generation_settings(after)
                    self.assertFalse(changed)
                    self.assertEqual(again, after)

    def test_full_settings_migration_preserves_vision_sampling_and_nested_images(self):
        for backend in ('cuda', 'hip', 'sycl'):
            for mode in ('gpu', 'cpu', 'no'):
                with self.subTest(backend=backend, mode=mode):
                    cfg = {'backend': backend, 'arch': 'qwen35moe', 'model_name': 'qwen3.6-35b-a3b',
                           'args': ['--max-context', '32768'], 'vision_mode': mode,
                           'vision': {'exe': 'vision.exe', 'mmproj': 'mmproj.gguf', 'model': 'model.gguf',
                                      'gpu': mode == 'gpu'},
                           'sampling': {'temperature': 0.7}, 'api_key': 'private',
                           'private_state': {'images': [{'url': 'data:image/png;base64,AAAA'}]}}
                    expected, _ = tinyconfig.migrate_settings(deepcopy(cfg))
                    cfg['images'] = {'exe': 'old-generator.exe', 'diffusion_model': 'old.gguf'}
                    before = deepcopy(cfg)
                    after, changed = tinyconfig.migrate_settings(cfg)
                    self.assertTrue(changed)
                    self.assertNotIn('images', after)
                    self.assertEqual(after, expected)
                    self.assertEqual(cfg, before)
                    self.assertEqual(after['private_state'], before['private_state'])
                    self.assertEqual(after['sampling']['temperature'], 0.7)
                    again, changed = tinyconfig.migrate_settings(after)
                    self.assertFalse(changed)
                    self.assertEqual(again, after)

    def test_generator_only_restore_hook_is_no_longer_adopted_on_model_switch(self):
        self.assertNotIn('restore_after_load', MODEL_FIELDS)
        self.assertIn('vision', MODEL_FIELDS)
        self.assertIn('last_image_at', MODEL_FIELDS)


if __name__ == '__main__':
    unittest.main(verbosity=2)
