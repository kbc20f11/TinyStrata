"""CPU GGUF/path consumers, with independent Python Unicode fixtures and payloads.

Set TINYSTRATA_UNICODE_PATH_TEST_EXE to the CMake unicode_paths_test executable.
No model engine, GPU or product service is run.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

EXE = Path(os.environ.get("TINYSTRATA_UNICODE_PATH_TEST_EXE", ""))


def gguf_bytes():
    def string(value):
        data = value.encode("utf-8")
        return struct.pack("<Q", len(data)) + data
    # An independently encoded GGUF v3: one string KV, one four-element F32 tensor.
    header = struct.pack("<IIQQ", 0x46554747, 3, 1, 1)
    metadata = string("general.name") + struct.pack("<I", 8) + string("independent unicode fixture")
    tensor = string("weight") + struct.pack("<IQIQ", 1, 4, 0, 0)
    prefix = header + metadata + tensor
    return prefix + bytes((-len(prefix)) % 32) + struct.pack("<ffff", 1, -2, 3.5, 0)


@unittest.skipUnless(EXE.is_file(), "build unicode_paths_test and set TINYSTRATA_UNICODE_PATH_TEST_EXE")
class UnicodePaths(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tinystrata-paths-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_consumer(self, mode, path="", *, env=None):
        result = subprocess.run([str(EXE.resolve()), mode], input=(str(path) + "\n").encode("utf-8"),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=10,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        self.assertEqual(result.stdout.decode("utf-8").strip(), "OK")

    def fixture(self, dirname, filename):
        path = self.root / dirname / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gguf_bytes())
        return path

    def test_ascii_control(self):
        self.run_consumer("mapped", self.fixture("control", "model.gguf"))

    def test_japanese_directory_and_model_name(self):
        self.run_consumer("mapped", self.fixture("日本語のモデル保存先", "会話モデル.gguf"))

    def test_space_and_non_ansi_unicode_in_directory_and_file(self):
        path = self.fixture("利用者 名前 🧪 𠮷", "画像 対応モデル 🌸.gguf")
        for mode in ("mapped", "fopen", "roundtrip"):
            with self.subTest(mode=mode):
                self.run_consumer(mode, path)

    def test_utf8_file_write_and_independent_readback(self):
        path = self.root / "保存 結果 🧪.txt"
        self.run_consumer("write", path)
        self.assertEqual(path.read_bytes(), b"independent output\n")

    def test_missing_model_names_the_original_path(self):
        self.run_consumer("missing", self.root / "存在しない 🌸.gguf")

    def test_reopen_after_missing_file_is_created(self):
        path = self.root / "再試行 モデル 🧪.gguf"
        self.run_consumer("missing", path)
        path.write_bytes(gguf_bytes())
        self.run_consumer("mapped", path)

    def test_split_discovery_from_middle_and_last_shard(self):
        parent = self.root / "複数 分割モデル 🌸"
        parent.mkdir()
        parts = [parent / f"検証モデル-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
        for path in parts:
            path.write_bytes(gguf_bytes())
        for path in (parts[1], parts[2]):
            with self.subTest(path=path.name):
                self.run_consumer("split", path)

    def test_missing_shard_and_retry(self):
        parent = self.root / "未完了 分割モデル 🧪"
        parent.mkdir()
        parts = [parent / f"資料-0000{i}-of-00003.gguf" for i in (1, 2, 3)]
        for path in (parts[0], parts[2]):
            path.write_bytes(gguf_bytes())
        self.run_consumer("split-missing", parts[0])
        parts[1].write_bytes(gguf_bytes())
        self.run_consumer("split", parts[2])

    def test_unicode_environment_path(self):
        path = str(self.root / "環境変数 保存先 🧪")
        self.run_consumer("env", path, env={**os.environ, "TINYSTRATA_UTF8_TEST_PATH": path})

    def test_empty_environment_value(self):
        self.run_consumer("env", "", env={**os.environ, "TINYSTRATA_UTF8_TEST_PATH": ""})

    def test_unset_environment_value(self):
        env = dict(os.environ)
        env.pop("TINYSTRATA_UTF8_TEST_PATH", None)
        self.run_consumer("env-unset", env=env)

    def test_invalid_encodings_and_embedded_nul_are_rejected(self):
        self.run_consumer("invalid")


if __name__ == "__main__":
    unittest.main()
