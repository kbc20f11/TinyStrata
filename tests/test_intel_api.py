"""Opt-in real Intel inference through TinyStrata's existing HTTP handlers.

Uses the same TINYSTRATA_INTEL_EXE/GGUF variables as test_intel_native.py.
No mocks replace the engine, tokenizer, templates, sampler or HTTP transport.
"""
import http.client
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import time
import traceback
import unittest

EXE = os.environ.get("TINYSTRATA_INTEL_EXE")
MODEL = os.environ.get("TINYSTRATA_INTEL_GGUF")


@unittest.skipUnless(EXE and MODEL, "requires Intel executable and real Qwen model")
class IntelAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serve import intel
        cls.work = tempfile.TemporaryDirectory(prefix="tinystrata-intel-api-")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            cls.port = reservation.getsockname()[1]
        cls.stop = threading.Event()
        cls.errors = []
        cls.results = []
        def start():
            try:
                intel.main(["--model", MODEL, "--exe", EXE, "--context", "4096", "--port", str(cls.port),
                    "--work-dir", cls.work.name, "--chunk", os.environ.get("TINYSTRATA_INTEL_CHUNK", "128")]
                    + ["--mtp", os.environ.get("TINYSTRATA_INTEL_MTP", "0")]
                    + ["--kv-type", os.environ.get("TINYSTRATA_INTEL_KV_TYPE", "fp32")]
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
            + (["--prefill-mkl"] if os.environ.get("TINYSTRATA_INTEL_PREFILL_MKL") == "1" else []) + (["--prefill-xmx"] if os.environ.get("TINYSTRATA_INTEL_PREFILL_XMX") == "1" else []), stop_event=cls.stop)
            except BaseException as error:
                cls.errors.append(traceback.format_exc())
        cls.thread = threading.Thread(target=start)
        cls.thread.start()
        cls.addClassCleanup(cls.close)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if cls.errors:
                raise AssertionError(cls.errors)
            try:
                connection = http.client.HTTPConnection("127.0.0.1", cls.port, timeout=2)
                connection.request("GET", "/health")
                reply = connection.getresponse()
                reply.read()
                connection.close()
                if reply.status == 200:
                    return
            except OSError:
                pass
            time.sleep(.2)
        raise AssertionError("Intel HTTP server did not start")

    @classmethod
    def close(cls):
        cls.stop.set()
        cls.thread.join(timeout=40)
        if cls.thread.is_alive():
            raise AssertionError("Intel API shutdown did not finish")
        if output := os.environ.get("TINYSTRATA_INTEL_API_TEST_LOG"):
            native = Path(cls.work.name, "native.log")
            Path(output).write_text(json.dumps({"responses": cls.results, "errors": cls.errors,
                "native": native.read_text(encoding="utf-8", errors="replace") if native.exists() else ""}, ensure_ascii=False, indent=2), encoding="utf-8")
        cls.work.cleanup()
        if cls.errors:
            raise AssertionError(cls.errors)

    def post(self, path, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=180)
        connection.request("POST", path, json.dumps(body, ensure_ascii=False).encode("utf-8"), {"Content-Type": "application/json"})
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        connection.close()
        self.results.append({"path": path, "request": body, "status": response.status, "response": raw})
        return response.status, raw

    def chat(self, text, **extra):
        return {"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": text}],
            "max_tokens": 128, "temperature": 0, "presence_penalty": 0, "enable_thinking": False, **extra}

    def test_japanese_source_arithmetic(self):
        question = "資料: 参加者は12人、参加費は1人800円です。会場費は3200円です。参加費の総額から会場費を引いた残額を、数値と円だけで答えてください。"
        status, raw = self.post("/v1/chat/completions", self.chat(question))
        self.assertEqual(status, 200, raw)
        answer = json.loads(raw)["choices"][0]["message"]["content"]
        self.assertEqual(answer.strip().replace(",", ""), "6400円")

    def test_streaming(self):
        status, raw = self.post("/v1/chat/completions", self.chat("What is 2 + 3? Answer with one number.", stream=True, max_tokens=16))
        self.assertEqual(status, 200, raw)
        self.assertIn("data: [DONE]", raw)
        chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
        content = "".join(chunk["choices"][0]["delta"].get("content") or "" for chunk in chunks if chunk.get("choices"))
        self.assertEqual(content.strip(), "5")

    def test_official_sampling_defaults(self):
        question = "資料: 参加者は12人、参加費は1人800円です。会場費は3200円です。参加費の総額から会場費を引いた残額を、数値と円だけで答えてください。"
        body = self.chat(question, seed=123)
        body.pop("temperature")
        body.pop("presence_penalty")
        status, raw = self.post("/v1/chat/completions", body)
        self.assertEqual(status, 200, raw)
        answer = json.loads(raw)["choices"][0]["message"]["content"]
        self.assertEqual(answer.strip().replace(",", ""), "6400円")

    def test_tool_call(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        body = self.chat("Call get_weather for Tokyo. Do not guess the weather.", tools=tools, tool_choice="required")
        status, raw = self.post("/v1/chat/completions", body)
        self.assertEqual(status, 200, raw)
        calls = json.loads(raw)["choices"][0]["message"].get("tool_calls")
        self.assertTrue(calls, raw)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        arguments = json.loads(calls[0]["function"]["arguments"])
        self.assertIn(arguments["city"].lower(), ("tokyo", "東京"))

    def test_images_are_rejected(self):
        body = self.chat("Describe this image.")
        body["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,invalid"}}]
        status, raw = self.post("/v1/chat/completions", body)
        self.assertIn(status, (400, 422, 501), raw)


if __name__ == "__main__":
    unittest.main()
