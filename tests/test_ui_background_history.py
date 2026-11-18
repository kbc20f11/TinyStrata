"""Local changes in another chat must not alter an active generation's transcript."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BackgroundHistory(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for UI controller checks")
    def test_local_changes_are_scoped_and_new_requests_remain_locked(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/ui-background-history.cjs")],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt["result"], "PASS")
        self.assertEqual(receipt["model_requests"], 0)


if __name__ == "__main__":
    unittest.main()
