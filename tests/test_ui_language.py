"""Language switching, persistence and recoverable catalog failures."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LanguageBehavior(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_language_switch_persistence_and_recovery(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/ui-language.cjs")],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS 11 language behavior cases", result.stdout)


if __name__ == "__main__":
    unittest.main()
