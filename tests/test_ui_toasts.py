"""Production toast timing/dismissal tests without a browser dependency."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ToastBehavior(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_timing_actions_dismissal_and_nonmodal_events(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/ui-toasts.cjs")],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS 10 toast behavior cases", result.stdout)


if __name__ == "__main__":
    unittest.main()
