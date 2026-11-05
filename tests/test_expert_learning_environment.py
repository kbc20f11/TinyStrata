"""Managed Expert hints use runtime state and respect explicit user overrides."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve.server import child_env


class ExpertLearningEnvironment(unittest.TestCase):
    def test_managed_models_use_runtime_without_creating_state(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            base = Path(temporary) / "runtime"
            for arch in ("qwen35moe", "clover_moe"):
                env = child_env({"arch": arch, "log": str(base / "engine.log")})
                self.assertEqual(env["TINYSTRATA_Q36_EXPERT_PROFILE_DIR"], str(base / "expert-profiles"))
                self.assertEqual(env["TINYSTRATA_Q36_EXPERT_TOPIC"], "1")
            self.assertFalse(base.exists())

    def test_explicit_process_and_configuration_overrides_win(self):
        with patch.dict(os.environ, {"TINYSTRATA_Q36_EXPERT_PROFILE_DIR": "chosen", "TINYSTRATA_Q36_EXPERT_TOPIC": "0"}, clear=True):
            cfg = {"arch": "clover_moe", "log": "engine.log", "env": {"TINYSTRATA_Q36_EXPERT_PROFILE": "off"}}
            env = child_env(cfg)
            self.assertEqual(env["TINYSTRATA_Q36_EXPERT_PROFILE_DIR"], "chosen")
            self.assertEqual(env["TINYSTRATA_Q36_EXPERT_TOPIC"], "0")
            self.assertEqual(env["TINYSTRATA_Q36_EXPERT_PROFILE"], "off")
            self.assertNotIn("TINYSTRATA_Q36_EXPERT_PROFILE", os.environ)

    def test_other_families_and_unmanaged_configs_keep_existing_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            for cfg in ({"arch": "gemma4", "log": "gemma.log"},
                        {"arch": "lfm2moe", "log": "lfm.log"}, {"arch": "qwen35moe"}):
                env = child_env(cfg)
                self.assertNotIn("TINYSTRATA_Q36_EXPERT_PROFILE_DIR", env)
                self.assertNotIn("TINYSTRATA_Q36_EXPERT_TOPIC", env)


if __name__ == "__main__":
    unittest.main()
