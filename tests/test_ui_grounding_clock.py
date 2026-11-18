"""Local clock grounding has no server, Python-runtime or network dependency."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GroundingClock(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_timezone_dst_and_single_instant(self):
        result = subprocess.run(
            [shutil.which("node"), str(ROOT / "tests/ui-grounding-clock.cjs")],
            capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertRegex(result.stdout, r"PASS \d+ clock checks")


if __name__ == "__main__":
    unittest.main()
