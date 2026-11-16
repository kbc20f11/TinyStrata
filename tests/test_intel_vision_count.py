"""CPU-only resident-encoder token planning; no image decoding or GPU work."""
import base64
import hashlib
import io
from pathlib import Path
import queue
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image
from serve.frontend import ChatTemplate, openai_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, Vision


def picture(width=320, height=640, format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (width, height)).save(output, format=format)
    return output.getvalue()


def resident(reply="TOKENS 200 10 20\n", *, enabled=True, maximum=4096):
    vision = object.__new__(Vision)
    vision.backend = "sycl"
    vision.count_dimensions_enabled = enabled
    vision.max_tokens, vision.min_tokens = maximum, 0
    vision._count_pending, vision._dimension_counts = None, {}
    vision.lock = threading.Lock()
    vision.stopped = False
    vision.cache = {"unchanged": ("kept.sve", 7)}
    vision.proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(reply), poll=lambda: None,
                                  kill=mock.Mock(side_effect=AssertionError("Counting killed an encoder")))
    return vision


class DimensionCountTests(unittest.TestCase):
    def test_inline_headers_need_no_pixel_decode_fetch_file_or_model_load(self):
        for format in ("PNG", "JPEG", "GIF", "BMP"):
            data = picture(format=format)
            url = "data:image/" + format.lower() + ";base64," + base64.b64encode(data).decode()
            with self.subTest(format=format), \
                 mock.patch.object(Image.Image, "load", side_effect=AssertionError("Pixel decode")), \
                 mock.patch.object(Vision, "load", side_effect=AssertionError("File/network load")), \
                 mock.patch.object(Vision, "restart", side_effect=AssertionError("Model start")):
                self.assertEqual(Vision.inline_dimensions(data), (320, 640))
                self.assertEqual(Vision.inline_dimensions(url), (320, 640))

    def test_unsupported_remote_local_and_malformed_inputs_keep_old_bound(self):
        for source in ("https://invalid.example/image.png", "file:///C:/image.png", "C:/image.png",
                       "data:image/png;base64,???", b"not an image", picture(format="TIFF")):
            with self.subTest(source=str(source)[:50]), \
                 mock.patch.object(Vision, "load", side_effect=AssertionError("Unexpected image load")):
                self.assertIsNone(Vision.inline_dimensions(source))

    def test_only_count_command_and_bounded_dimension_cache_are_used(self):
        vision, data = resident(), picture()
        self.assertEqual(vision.count_tokens(data), 200)
        self.assertEqual(vision.count_tokens(data), 200)
        self.assertEqual(vision.proc.stdin.getvalue(), "COUNT\t320\t640\n")
        self.assertEqual(vision.cache, {"unchanged": ("kept.sve", 7)})
        vision.proc.kill.assert_not_called()

    def test_unknown_old_unloaded_and_busy_producers_are_not_queried(self):
        data = picture()
        for state in ("old", "unloaded", "busy"):
            vision = resident(enabled=state != "old")
            if state == "unloaded":
                vision.stopped = True
            if state == "busy":
                vision.lock.acquire()
            with self.subTest(state=state):
                self.assertIsNone(vision.count_tokens(data))
                self.assertEqual(vision.proc.stdin.getvalue(), "")

    def test_invalid_replies_never_underestimate_or_kill(self):
        for reply in ("ERR unsupported\n", "TOKENS 0 0 0\n", "TOKENS 5000 100 50\n",
                      "TOKENS 200 10 19\n", "READY\n", ""):
            with self.subTest(reply=reply):
                vision = resident(reply)
                self.assertIsNone(vision.count_tokens(picture()))
                self.assertFalse(vision._dimension_counts)
                vision.proc.kill.assert_not_called()

    def test_unanswered_count_keeps_pipe_owned_without_start_or_kill(self):
        vision = resident()
        vision._count_pending = (vision.proc, queue.Queue())
        self.assertIsNone(vision.count_tokens(picture()))
        self.assertEqual(vision.proc.stdin.getvalue(), "")
        vision.proc.kill.assert_not_called()

    def test_late_count_reply_is_drained_before_an_encode_cache_hit(self):
        vision, data = resident(), picture()
        reply = queue.Queue(); reply.put("TOKENS 200 10 20\n")
        vision._count_pending = (vision.proc, reply)
        key = hashlib.sha256(data).hexdigest()[:32]
        vision.cache = {key: (Path("kept.sve"), 200)}
        self.assertEqual(vision.encode(data), (Path("kept.sve"), 200))
        self.assertIsNone(vision._count_pending)
        self.assertEqual(vision.proc.stdin.getvalue(), "")


class ServiceCountTests(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, "unused", max_context=4096)
        self.service = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.service.backend = "sycl"
        self.service.vision = resident("TOKENS 1036 28 37\n")
        data = picture(888, 1176)
        self.inline = "data:image/png;base64," + base64.b64encode(data).decode()

    def request(self, images):
        return {"model":self.service.model,"messages":[{"role":"user","content":
            [{"type":"text","text":"Describe the shapes"}] +
            [{"type":"image_url","image_url":{"url":image}} for image in images]}]}

    def raw_count(self, request):
        messages, tools, kwargs = openai_to_messages(request)
        return len(self.service.encode_prompt(messages, tools, kwargs))

    def test_ready_intel_inline_count_is_exact_without_gpu_or_saved_cache_changes(self):
        request = self.request([self.inline])
        with mock.patch.object(self.engine, "generate", side_effect=AssertionError("GPU turn")), \
             mock.patch.object(Vision, "restart", side_effect=AssertionError("Start")), \
             mock.patch.object(Image.Image, "load", side_effect=AssertionError("Decode pixels")):
            counted = self.service.chat_token_count(request)
        self.assertEqual(counted["prompt_tokens"], self.raw_count(request) + 1036 - 1)
        self.assertTrue(counted["exact"])
        self.assertEqual(counted["images_counted"], 1)
        self.assertEqual(counted["images_estimated"], 0)
        self.assertEqual(self.service.vision.cache, {"unchanged": ("kept.sve", 7)})

    def test_mixed_unknown_image_retains_its_full_bound_without_fetching(self):
        request = self.request([self.inline, "https://invalid.example/image.png"])
        with mock.patch("serve.server.urllib.request.urlopen", side_effect=AssertionError("Fetch")):
            counted = self.service.chat_token_count(request)
        self.assertEqual(counted["prompt_tokens"], self.raw_count(request) + 1036 - 1 + 4096 - 1)
        self.assertFalse(counted["exact"])
        self.assertEqual(counted["images_estimated"], 1)

    def test_cuda_text_and_gemma_keep_existing_policy(self):
        request = self.request([self.inline])
        self.service.backend = "cuda"
        with mock.patch.object(self.service.vision, "count_tokens", side_effect=AssertionError("CUDA COUNT")):
            counted = self.service.chat_token_count(request)
        self.assertEqual(counted["prompt_tokens"], self.raw_count(request) + 4096 - 1)
        self.assertFalse(counted["exact"])


if __name__ == "__main__":
    unittest.main()
