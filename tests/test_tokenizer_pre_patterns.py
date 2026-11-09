"""Declared pre-tokenizers must not silently inherit another model's BPE splits."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from serve.tokenizer import (BYTE_TO_UNICODE, GEMMA4_PATTERN, GPT4O_PATTERN, LFM2_PATTERN,
                             QWEN2_PATTERN, QWEN35_PATTERN, Tokenizer, extract)


MISSING = object()
CORPUS = ["", "a\u0301.b", "e\u0301_j", "\u65e5\u672c\u8a9e", "\u0645\u0631\u062d\u0628\u0627",
          "\U0001f600\U0001f680", "a\r\n\r\nb", "1234567890", "  leading and trailing  ",
          "\x00\x01\x7f", "def f(x):\n\treturn x + 1", "I've said it's fine."]


def byte_vocabulary():
    return [BYTE_TO_UNICODE[i] for i in range(256)]


def metadata(pre=MISSING):
    value = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.tokens": byte_vocabulary(),
             "tokenizer.ggml.merges": [], "tokenizer.ggml.token_type": [1] * 256,
             "tokenizer.chat_template": "{{ messages[0]['content'] }}"}
    if pre is not MISSING:
        value["tokenizer.ggml.pre"] = pre
    return value


def write_cache(path, meta):
    path.mkdir()
    (path / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(byte_vocabulary())}), encoding="utf-8")
    (path / "merges.txt").write_text("", encoding="utf-8")
    (path / "token_type.json").write_text(json.dumps([1] * 256), encoding="utf-8")
    if meta is not MISSING:
        (path / "tokenizer.json").write_text(json.dumps(meta), encoding="utf-8")


class DeclaredPreTokenizers(unittest.TestCase):
    def test_qwen2_combining_mark_boundary_changes_actual_token_ids(self):
        # Only Qwen35 may merge the letter with its combining accent. Both
        # tokenizations round-trip, so a round-trip-only test misses this bug.
        accent = "\u0301".encode("utf-8")
        first, second = (BYTE_TO_UNICODE[i] for i in accent)
        tokens = byte_vocabulary() + ["a" + first, "a" + first + second]
        merges = ["a " + first, "a" + first + " " + second]
        qwen2 = Tokenizer(tokens, merges, pre="qwen2")
        qwen35 = Tokenizer(tokens, merges, pre="qwen35")
        self.assertEqual(qwen2.encode("a\u0301"), [ord("a"), *accent])
        self.assertEqual(qwen35.encode("a\u0301"), [257])
        self.assertEqual(qwen2.decode(qwen2.encode("a\u0301")), "a\u0301")
        self.assertEqual(qwen35.decode(qwen35.encode("a\u0301")), "a\u0301")

    def test_qwen2_declared_splits_match_independent_examples(self):
        tokenizer = Tokenizer(byte_vocabulary(), [], pre="qwen2")
        self.assertEqual(tokenizer._re.findall("a\u0301.b"), ["a", "\u0301.", "b"])
        self.assertEqual(tokenizer._re.findall("12345"), ["1", "2", "3", "4", "5"])
        self.assertEqual(tokenizer._re.findall("I've"), ["I", "'ve"])
        self.assertEqual(tokenizer._re.findall("a\r\n\r\nb"), ["a", "\r\n\r\n", "b"])

    def test_qwen2_unicode_and_whitespace_round_trip(self):
        tokenizer = Tokenizer(byte_vocabulary(), [], pre="qwen2")
        for text in CORPUS:
            with self.subTest(text=text):
                self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_existing_pre_patterns_keep_their_distinct_semantics(self):
        qwen = Tokenizer(byte_vocabulary(), [])
        lfm = Tokenizer(byte_vocabulary(), [], pre="lfm2")
        gemma = Tokenizer(["a", "b", "\n"], [], pre="gemma4")
        self.assertEqual(qwen.pre_pattern, QWEN35_PATTERN)
        self.assertEqual(lfm.pre_pattern, LFM2_PATTERN)
        self.assertEqual(gemma.pre_pattern, GEMMA4_PATTERN)
        self.assertEqual(qwen._re.findall("a\u0301.b"), ["a\u0301", ".b"])
        self.assertEqual(lfm._re.findall("12345"), ["123", "45"])
        self.assertEqual(gemma._re.findall("ab\n\nba"), ["ab", "\n\n", "ba"])

    def test_gpt4o_splits_case_runs_numbers_and_round_trips(self):
        # gpt-oss declares "gpt-4o": words split where case changes, numbers in threes, a leading space joins its word
        import regex
        self.assertEqual(regex.findall(GPT4O_PATTERN, "Hello, world! I'm 12345."),
                         ["Hello", ",", " world", "!", " I'm", " ", "123", "45", "."])
        self.assertEqual(regex.findall(GPT4O_PATTERN, "CamelCaseWord"), ["Camel", "Case", "Word"])
        tokenizer = Tokenizer(byte_vocabulary(), [], [1] * 256, pre="gpt-4o")
        for text in CORPUS:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_unknown_and_non_string_pre_are_rejected_before_vocabulary_work(self):
        for pre in ("unknown", "qwen355", "minimax-m2", "", None, True, 1, [], {}):
            with self.subTest(pre=pre), self.assertRaisesRegex(ValueError, "Unsupported pre-tokenizer"):
                Tokenizer(["duplicate", "duplicate"], [], pre=pre)

    def test_gguf_explicit_unsupported_pre_is_rejected(self):
        for pre in ("unknown", None, 1, []):
            with self.subTest(pre=pre), patch("serve.gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=metadata(pre))):
                with self.assertRaisesRegex(ValueError, "Unsupported pre-tokenizer"):
                    Tokenizer.from_gguf("unused.gguf")

    def test_gguf_missing_pre_retains_qwen35_compatibility(self):
        with patch("serve.gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=metadata())):
            tokenizer = Tokenizer.from_gguf("unused.gguf")
        self.assertEqual(tokenizer.pre, "qwen35")
        self.assertEqual(tokenizer.pre_pattern, QWEN35_PATTERN)

    def test_cache_explicit_unsupported_pre_fails_without_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            for i, pre in enumerate(("unknown", None, 1, [], {})):
                path = Path(temporary) / str(i)
                write_cache(path, {"pre": pre})
                before = {p.name: p.read_bytes() for p in path.iterdir()}
                with self.subTest(pre=pre), self.assertRaisesRegex(ValueError, "Unsupported pre-tokenizer"):
                    Tokenizer.from_dir(path)
                self.assertEqual({p.name: p.read_bytes() for p in path.iterdir()}, before)

    def test_old_cache_missing_pre_or_metadata_keeps_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            for i, meta in enumerate(({}, MISSING)):
                path = Path(temporary) / str(i)
                write_cache(path, meta)
                restored = Tokenizer.from_dir(path)
                self.assertEqual(restored.pre, "qwen35")
                self.assertEqual(restored.pre_pattern, QWEN35_PATTERN)

    def test_extract_records_qwen2_and_restores_the_same_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache"
            with patch("serve.gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=metadata("qwen2"))):
                exported = extract("unused.gguf", path)
            self.assertEqual(exported["pre"], "qwen2")
            self.assertEqual(exported["pre_pattern"], QWEN2_PATTERN)
            self.assertIn("LLAMA_VOCAB_PRE_TYPE_QWEN2", exported["pre_pattern_source"])
            restored = Tokenizer.from_dir(path / "tokenizer")
            for text in CORPUS:
                self.assertEqual(restored.decode(restored.encode(text)), text)

    def test_extract_invalid_pre_does_not_create_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-created"
            with patch("serve.gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=metadata("unsupported"))):
                with self.assertRaisesRegex(ValueError, "Unsupported pre-tokenizer"):
                    extract("unused.gguf", path)
            self.assertFalse(path.exists())

    def test_extract_invalid_pre_preserves_existing_cache_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "existing"
            (path / "tokenizer").mkdir(parents=True)
            marker = path / "tokenizer/vocab.json"
            marker.write_text("existing valid cache", encoding="utf-8")
            with patch("serve.gguf_reader.GGUFFile", return_value=SimpleNamespace(metadata=metadata(None))):
                with self.assertRaisesRegex(ValueError, "Unsupported pre-tokenizer"):
                    extract("unused.gguf", path)
            self.assertEqual(marker.read_text(encoding="utf-8"), "existing valid cache")
            self.assertEqual([p.name for p in (path / "tokenizer").iterdir()], ["vocab.json"])


if __name__ == "__main__":
    unittest.main()
