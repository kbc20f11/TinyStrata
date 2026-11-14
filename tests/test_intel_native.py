"""Opt-in native Intel integration checks on the real unchanged Qwen GGUF.

Set TINYSTRATA_INTEL_EXE and TINYSTRATA_INTEL_GGUF in an initialized oneAPI
environment. No download, installation, or saved-profile mutation is performed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import unittest

from serve.tokenizer import Tokenizer

EXE = os.environ.get("TINYSTRATA_INTEL_EXE")
MODEL = os.environ.get("TINYSTRATA_INTEL_GGUF")
HEADER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


@unittest.skipUnless(EXE and MODEL, "requires Intel executable and real Qwen model")
class IntelNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_gguf(MODEL)
        cls.stderr = tempfile.TemporaryFile(mode="w+b")
        cls.proc = subprocess.Popen([EXE, "--model", MODEL, "--serve", "--max-context", "1024", "--kv-type", os.environ.get("TINYSTRATA_INTEL_KV_TYPE", "fp32"), "--chunk", os.environ.get("TINYSTRATA_INTEL_CHUNK", "128")]
            + ["--mtp", os.environ.get("TINYSTRATA_INTEL_MTP", "0")]
            + (["--verify-xmx"] if os.environ.get("TINYSTRATA_INTEL_VERIFY_XMX") == "1" else [])
            + (["--prefill-int8-xmx"] if os.environ.get("TINYSTRATA_INTEL_PREFILL_INT8_XMX") == "1" else [])
            + (["--vocab-xmx"] if os.environ.get("TINYSTRATA_INTEL_VOCAB_XMX") == "1" else [])
            + (["--verify-mkl-attention"] if os.environ.get("TINYSTRATA_INTEL_VERIFY_MKL_ATTENTION") == "1" else [])
            + (["--mtp-stochastic"] if os.environ.get("TINYSTRATA_INTEL_MTP_STOCHASTIC") == "1" else [])
            + (["--expert-xmx-layout"] if os.environ.get("TINYSTRATA_INTEL_EXPERT_LAYOUT") == "1" else [])
            + (["--device-state"] if os.environ.get("TINYSTRATA_INTEL_DEVICE_STATE") == "1" else [])
            + (["--verify-graph"] if os.environ.get("TINYSTRATA_INTEL_VERIFY_GRAPH") == "1" else [])
            + (["--dense-xmx-layout"] if os.environ.get("TINYSTRATA_INTEL_DENSE_LAYOUT") == "1" else [])
            + (["--dense-split-k"] if os.environ.get("TINYSTRATA_INTEL_DENSE_SPLIT_K") == "1" else [])
            + (["--draft-vocab", os.environ["TINYSTRATA_INTEL_DRAFT_VOCAB"]] if os.environ.get("TINYSTRATA_INTEL_DRAFT_VOCAB") else [])
            + (["--decode-dp4a"] if os.environ.get("TINYSTRATA_INTEL_DECODE_DP4A") == "1" else [])
            + (["--prefill-mkl"] if os.environ.get("TINYSTRATA_INTEL_PREFILL_MKL") == "1" else []) + (["--prefill-xmx"] if os.environ.get("TINYSTRATA_INTEL_PREFILL_XMX") == "1" else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=cls.stderr, text=True, encoding="utf-8",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        cls.lines = queue.Queue()
        cls.transcript = []
        def read():
            for line in cls.proc.stdout:
                cls.transcript.append(line.rstrip())
                cls.lines.put(line.rstrip())
            cls.lines.put(None)
        cls.reader = threading.Thread(target=read, daemon=True)
        cls.reader.start()
        cls.addClassCleanup(cls.close)
        while True:
            line = cls.next_line()
            if line.startswith("READY "):
                break

    @classmethod
    def next_line(cls):
        try:
            line = cls.lines.get(timeout=120)
        except queue.Empty:
            raise AssertionError("Intel engine stopped responding") from None
        if line is None:
            cls.stderr.seek(0)
            raise AssertionError("Intel engine exited: " + cls.stderr.read().decode("utf-8", errors="replace")[-2000:])
        return line

    @classmethod
    def close(cls):
        if cls.proc.poll() is None:
            cls.proc.stdin.write("QUIT\n")
            cls.proc.stdin.flush()
            try:
                cls.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
                cls.proc.wait(timeout=10)
        cls.reader.join(timeout=5)
        cls.stderr.seek(0)
        if output := os.environ.get("TINYSTRATA_INTEL_TEST_LOG"):
            Path(output).write_text(json.dumps({"stdout": cls.transcript,
                "stderr": cls.stderr.read().decode("utf-8", errors="replace"), "exit": cls.proc.returncode}, indent=2), encoding="utf-8")
        cls.stderr.close()

    @classmethod
    def send(cls, line):
        cls.proc.stdin.write(line + "\n")
        cls.proc.stdin.flush()

    @classmethod
    def prompt(cls, question="What is 2 + 3? Answer with one number."):
        return cls.tokenizer.encode("<|im_start|>user\n" + question + "<|im_end|>\n" + HEADER, parse_special=True)

    @classmethod
    def generate(cls, ids, limit=8, options="temperature=0"):
        cls.send(f"GEN {limit} {options} " + ",".join(map(str, ids)))
        return cls.result()

    @classmethod
    def result(cls):
        ids, reused = [], None
        while True:
            line = cls.next_line()
            if line.startswith("ERR "):
                raise AssertionError(line)
            if line.startswith("T "):
                ids.append(int(line[2:]))
            elif line.startswith("RESUME "):
                reused = int(line.split()[1])
            elif line.startswith("DONE "):
                fields = line.split()
                return ids, reused, fields

    def text(self, ids):
        eos = self.tokenizer.special_ids["tokenizer.ggml.eos_token_id"]
        return self.tokenizer.decode([token for token in ids if token != eos]).strip()

    def test_arithmetic_and_terminal(self):
        ids, _, done = self.generate(self.prompt())
        self.assertEqual(self.text(ids), "5")
        self.assertEqual(done[5], "stop")
        self.assertEqual(int(done[1]), len(ids))
        self.assertLessEqual(int(done[6]), int(done[7]))
        if int(os.environ.get("TINYSTRATA_INTEL_MTP", "0")):
            self.assertGreater(int(done[7]), 0)
        else:
            self.assertEqual((int(done[6]), int(done[7])), (0, 0))

    def test_continuation_and_edit(self):
        prompt = self.prompt()
        output, _, _ = self.generate(prompt)
        next_turn = prompt + output + self.tokenizer.encode("\n<|im_start|>user\nWhat is that number plus 1? Answer with one number.<|im_end|>\n" + HEADER, parse_special=True)
        answer, reused, _ = self.generate(next_turn)
        self.assertGreaterEqual(reused, len(prompt))
        self.assertEqual(self.text(answer), "6")
        answer, reused, _ = self.generate(self.prompt("What is 2 + 4? Answer with one number."))
        self.assertEqual(reused, 0)
        self.assertEqual(self.text(answer), "6")

    def test_invalid_requests_preserve_state(self):
        prompt = self.prompt()
        self.generate(prompt, 1)
        for command in ("GEN 1 99999999", "GENI 1 invalid 1", "GEN 1 mtp_auto=1 1", "GEN 1024 1,2"):
            self.send(command)
            self.assertTrue(self.next_line().startswith("ERR "), command)
        output, reused, _ = self.generate(prompt, 1)
        self.assertEqual(reused, len(prompt))
        self.assertEqual(self.text(output), "5")

    def test_prefill_cancellation_and_recovery(self):
        prompt = self.prompt() + self.tokenizer.encode(" hello" * 600)
        self.send("GEN 8 temperature=0 " + ",".join(map(str, prompt)))
        while True:
            line = self.next_line()
            self.assertFalse(line.startswith("ERR "))
            if line.startswith("PP "):
                self.send("STOP")
                break
        _, _, done = self.result()
        self.assertEqual(done[5], "cancel")
        processed = int(done[8]) + int(done[14])
        self.assertLess(processed, len(prompt))
        # The cancelled prefill skipped its output head. Reusing exactly that
        # prefix must calculate fresh logits, not use the preceding request's.
        cached, reused, done = self.generate(prompt[:processed], 1)
        self.assertEqual(reused, processed)
        self.assertEqual(int(done[1]), 1)
        output, _, _ = self.generate(self.prompt())
        self.assertEqual(self.text(output), "5")
        # Recompute the same retained prefix from a different conversation;
        # the resumed hidden/recurrent state must agree with a clean replay.
        replayed, reused, _ = self.generate(prompt[:processed], 1)
        self.assertEqual(reused, 0)
        self.assertEqual(cached, replayed)

    def test_decode_cancellation_and_recovery(self):
        if os.environ.get("TINYSTRATA_INTEL_VERIFY_GRAPH") != "1":
            self.skipTest("explicit verification-graph cancellation coverage")
        prompt = self.prompt("List the integers from 1 to 200 in order, separated by commas. No explanation.")
        self.send("GEN 256 temperature=0 " + ",".join(map(str, prompt)))
        emitted = []
        while len(emitted) < 3:
            line = self.next_line()
            self.assertFalse(line.startswith(("ERR ", "DONE ")), line)
            if line.startswith("T "):
                emitted.append(int(line[2:]))
        # Interrupt an active decode window, including its graph path. The
        # retained prefix must still resume like a replay from a clean state.
        time.sleep(.03)
        self.send("STOP")
        tail, _, done = self.result()
        emitted += tail
        self.assertEqual(done[5], "cancel")
        self.assertTrue(emitted)
        prefix = prompt + emitted
        resumed, reused, _ = self.generate(prefix, 4)
        self.assertGreaterEqual(reused, len(prompt))
        self.generate(self.prompt(), 1)
        replayed, reused, _ = self.generate(prefix, 4)
        self.assertEqual(reused, 0)
        self.assertEqual(resumed, replayed)

    def test_seeded_sampling_replay(self):
        options = "temperature=0.7 top_p=0.95 top_k=20 seed=123 penalty_repeat=1 penalty_present=0 penalty_last_n=64"
        first, _, _ = self.generate(self.prompt(), 16, options)
        second, _, _ = self.generate(self.prompt(), 16, options)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
