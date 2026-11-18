"""Production ownership helpers under deterministic asynchronous boundaries."""
from pathlib import Path
import shutil
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BackgroundChatOwnership(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_background_chat_ownership_and_gated_sse(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/ui-background-chat.cjs")],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS background chat ownership", result.stdout)


if __name__ == "__main__":
    unittest.main()
