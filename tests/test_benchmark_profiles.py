"""CPU-only benchmark profile migration tests; no services, files or GPU work."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import config, tinyconfig

spec = importlib.util.spec_from_file_location("benchmark_profile_test_module", ROOT / "benchmark/run.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
CASES = json.loads((ROOT / "benchmark/cases.json").read_text(encoding="utf-8"))


def original(model):
    cfg = {"arch": bench.MODELS[model][1], "exe": "engine.exe", "tokenizer": "tokenizer",
           "args": ["--model", "model.gguf", "--max-context", "32768", "--chunk", "512"],
           "vision": {"exe": "vision.exe", "mmproj": "projector.gguf", "model": "model.gguf"}}
    if model == "gemma":
        cfg["args"] += ["--draft-model", "draft.gguf"]
    return cfg


def controlled(model):
    return bench.controlled_config(original(model), model, ROOT / "unused-test-output", 18001, CASES)


class BenchmarkProfileMigration(unittest.TestCase):
    def test_benchmark_requests_are_one_shot_without_parking(self):
        cases = json.loads((ROOT / 'benchmark/cases.json').read_text(encoding='utf-8'))
        for model in bench.MODELS:
            with self.subTest(model=model):
                body = bench.request_body(model, 'fresh input', 128, cases)
                self.assertIs(body['strata_checkpoint'], False)
                self.assertTrue(body['stream'])

    def test_explicit_expert_vram_survives_api_startup_migration(self):
        for model in bench.MODELS:
            with self.subTest(model=model):
                cfg = controlled(model)
                cfg["args"] += ["--expert-vram", "6"]
                after, _ = tinyconfig.migrate_settings(cfg)
                self.assertEqual(config.value_of(after, "expert_vram"), 6)

    def test_explicit_expert_slots_survive_api_startup_migration(self):
        cfg = controlled("qwen")
        cfg["args"] += ["--expert-slots", "64"]
        after, _ = tinyconfig.migrate_settings(cfg)
        at = after["args"].index("--expert-slots")
        self.assertEqual(after["args"][at + 1], "64")

    def test_manual_chunk512_is_not_migrated_to_auto(self):
        for model in bench.MODELS:
            with self.subTest(model=model):
                cfg = controlled(model)
                # A caller may already have protected its fixed Expert budget.
                cfg["memory_policy_version"] = tinyconfig.MEMORY_POLICY_VERSION
                cfg["args"][cfg["args"].index("--chunk") + 1] = "512"
                after, _ = tinyconfig.migrate_settings(cfg)
                self.assertEqual(config.value_of(after, "chunk"), 512)

    def test_markers_follow_the_current_policy_versions(self):
        with patch.object(tinyconfig, "MEMORY_POLICY_VERSION", 17), \
             patch.object(tinyconfig, "CHUNK_POLICY_VERSION", 18), \
             patch.object(tinyconfig, "GENERATION_POLICY_VERSION", 19):
            cfg = controlled("qwen")
            self.assertEqual((cfg.get("memory_policy_version"), cfg.get("chunk_policy_version"),
                              cfg.get("generation_policy_version")), (17, 18, 19))

    def test_sampling_and_automatic_defaults_are_retained_without_mutating_source(self):
        before = original("qwen")
        snapshot = copy.deepcopy(before)
        cfg = bench.controlled_config(before, "qwen", ROOT / "unused-test-output", 18001, CASES)
        cfg["sampling"].update(temperature=0.37, presence_penalty=0.12)
        after, _ = tinyconfig.migrate_settings(cfg)
        self.assertEqual(after["sampling"], cfg["sampling"])
        self.assertEqual(config.value_of(after, "chunk"), "auto")
        self.assertIsNone(config.value_of(after, "expert_vram"))
        self.assertEqual(before, snapshot)


if __name__ == "__main__":
    unittest.main()
