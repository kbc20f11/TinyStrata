"""Removed output-image endpoints stay absent while text and image-input capabilities work."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serve import server
from serve.frontend import ChatTemplate

ROOT = Path(__file__).resolve().parents[1]


class RemovedImageGenerationAPI(unittest.TestCase):
    def setUp(self):
        tokenizer = server.ByteTokenizer()
        engine = server.MockEngine(tokenizer, "retained text service", max_context=8192)
        engine.info = {"vision": 1}
        vision = SimpleNamespace(alive=lambda: True, startup_error="", stopped=False, gpu=False, elastic=False)
        self.svc = server.Service(engine, tokenizer, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                                  model_name="test-model", vision=vision)
        self.http = server.Server(("127.0.0.1", 0), server.make_handler(self.svc))
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.thread.join, 3)
        self.addCleanup(self.http.shutdown)

    def request(self, method, path, body=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.http.server_port, timeout=5)
        try:
            connection.request(method, path, json.dumps(body) if body is not None else None,
                               {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_retired_routes_cannot_start_a_worker(self):
        with patch.object(server, "start_contained", side_effect=AssertionError("must not spawn a generator")) as spawn:
            for path in ("/v1/images/generations", "/v1/images/generations/"):
                status, body = self.request("POST", path, {"prompt": "a landscape", "n": 2, "size": "1024x1024"})
                self.assertEqual(status, 404, body)
                self.assertNotIn("setup", body["error"]["message"])
            for path in ("/v1/images/files/" + "a" * 32 + ".png", "/v1/images/files/arbitrary.png"):
                self.assertEqual(self.request("GET", path)[0], 404)
            spawn.assert_not_called()

    def test_health_metrics_and_models_keep_only_image_input(self):
        status, health = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["images"])
        self.assertNotIn("image_generation", health)
        status, metrics = self.request("GET", "/metrics")
        self.assertEqual(status, 200)
        self.assertNotIn("image_generation", metrics)
        self.assertNotIn("image_generation", metrics.get("engine", {}))
        status, models = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(len(models["data"]), 1)
        self.assertEqual(models["data"][0]["architecture"],
                         {"input_modalities": ["text", "image"], "output_modalities": ["text"]})

    def test_text_still_answers_after_a_removed_endpoint_request(self):
        self.assertEqual(self.request("POST", "/v1/images/generations", {"prompt": "test"})[0], 404)
        status, answer = self.request("POST", "/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "テスト"}], "max_tokens": 100,
            "chat_template_kwargs": {"enable_thinking": False}})
        self.assertEqual(status, 200, answer)
        self.assertEqual(answer["choices"][0]["message"]["content"], "retained text service")


if __name__ == "__main__":
    unittest.main()
