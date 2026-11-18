"""UI tool discovery, choice snapshots and bounded combined web lookups."""
from pathlib import Path
import shutil
import subprocess
import unittest
ROOT = Path(__file__).resolve().parents[1]
class GroundingTools(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_protocol_and_disabled_network_tools(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/ui-local-tools.cjs")],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("local-tool protocol checks", result.stdout)
    @unittest.skipUnless(shutil.which("node"), "Node.js is required")
    def test_runtime_identity_and_tool_context_keep_their_owner(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/ui-runtime-model.cjs")],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
if __name__ == "__main__":
    unittest.main()
