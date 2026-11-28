"""Exercise the managed downloader against a real, isolated HTTP Range server."""
from contextlib import suppress
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import re
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.model_download import FixedModelDownloader, _Manifest, _RangeUnsupported


class SmallRanges(FixedModelDownloader):
    _RANGE_BYTES = 4096

    def _transfer(self, offset):
        if self.manifest.size - offset < 2 * self._RANGE_BYTES:
            self._download_single(offset)
            return
        try:
            self._download_parallel(offset, self.manifest.size)
        except _RangeUnsupported:
            self._download_single(self.partial.stat().st_size)


class SmallAdaptive(FixedModelDownloader):
    _RANGE_BYTES = SmallRanges._RANGE_BYTES
    _PARALLEL_MIN_BYTES = 8 * _RANGE_BYTES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel_rates = iter(())

    def _download_single(self, offset, end=None):
        super()._download_single(offset, end)
        stop = self.manifest.size if end is None else end + 1
        return (stop - offset) / self._RANGE_BYTES

    def _download_parallel(self, offset, stop):
        super()._download_parallel(offset, stop)
        return (stop - offset) / (self._RANGE_BYTES * next(self.parallel_rates, 2))


class RangeSource(ThreadingHTTPServer):
    def __init__(self, payload):
        super().__init__(("127.0.0.1", 0), RangeHandler)
        self.payload = payload
        self.ignore_ranges = False
        self.bad_start = None
        self.bad_header = None
        self.hold_start = None
        self.release = threading.Event()
        self.later_served = threading.Event()
        self.requests = []
        self.lock = threading.Lock()


class RangeHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        source = self.server
        byte_range = self.headers.get("Range")
        with source.lock:
            source.requests.append(byte_range)
        length = len(source.payload)
        if byte_range is None or source.ignore_ranges:
            status, start, end = 200, 0, length - 1
        else:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", byte_range)
            if not match:
                self.send_error(416)
                return
            start = int(match[1])
            end = int(match[2]) if match[2] else length - 1
            if not 0 <= start <= end < length:
                self.send_error(416)
                return
            status = 206

        advertised_end, advertised_length = end, end - start + 1
        if status == 206 and start == source.bad_start:
            if source.bad_header == "range":
                advertised_end -= 1
            else:
                advertised_length += 1
        self.send_response(status)
        self.send_header("Content-Length", str(advertised_length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{advertised_end}/{length}")
        self.end_headers()

        if status == 206 and start == source.hold_start and not source.release.wait(8):
            return
        with suppress(BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.wfile.write(source.payload[start:end + 1])
            if status == 206 and source.hold_start is not None and start > source.hold_start:
                source.later_served.set()


class ModelDownloadRanges(unittest.TestCase):
    def setUp(self):
        self.segment = SmallRanges._RANGE_BYTES
        self.payload = random.Random(814).randbytes(43 * self.segment + 131)
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = RangeSource(self.payload)
        thread = threading.Thread(target=self.source.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(self.source.server_close)
        self.addCleanup(thread.join, 3)
        self.addCleanup(self.source.shutdown)
        self.manifest = _Manifest(
            "fixture", "fixture", "model.gguf", len(self.payload), hashlib.sha256(self.payload).hexdigest(),
            f"http://127.0.0.1:{self.source.server_port}/model.gguf",
        )
        self.ready = []

    def tearDown(self):
        self.source.release.set()

    def downloader(self, downloader=SmallRanges):
        result = downloader(self.directory, on_ready=self.ready.append, _manifest=self.manifest,
                            _chunk_size=1024, _timeout=3, _reserve_bytes=0)
        self.addCleanup(result.close)
        return result

    def finished(self, download, expected):
        deadline = time.monotonic() + 7
        while download.running and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(download.running, download.status())
        result = download.status()
        self.assertEqual(result["state"], expected, result)
        return result

    def assert_complete(self, download):
        result = self.finished(download, "ready")
        self.assertEqual(result["bytes_downloaded"], len(self.payload))
        self.assertEqual(result["progress"], 1)
        self.assertEqual(download.target.read_bytes(), self.payload)
        self.assertFalse(download.partial.exists())
        self.assertEqual(self.ready, [download.target])
        self.assertEqual(download.start()["state"], "ready")
        self.assertEqual(self.ready, [download.target])

    def bounded_requests(self, start, stop):
        return [f"bytes={offset}-{min(offset + self.segment, stop) - 1}"
                for offset in range(start, stop, self.segment)]

    def test_parallel_responses_can_finish_out_of_order_without_holes(self):
        download = self.downloader()
        prefix = 73
        download.partial.write_bytes(self.payload[:prefix])
        self.source.hold_start = prefix + self.segment
        download.start()
        self.assertTrue(self.source.later_served.wait(3), self.source.requests)
        saved = download.partial.read_bytes()
        self.assertGreaterEqual(len(saved), prefix)
        self.assertLessEqual(len(saved), self.source.hold_start)
        self.assertEqual(saved, self.payload[:len(saved)])
        self.source.release.set()
        self.assert_complete(download)
        self.assertCountEqual(self.source.requests, self.bounded_requests(prefix, len(self.payload)))

    def test_bad_range_and_length_keep_prefix_for_retry(self):
        for bad_header in ("range", "length"):
            with self.subTest(bad_header=bad_header):
                download = self.downloader()
                prefix = 101
                download.partial.write_bytes(self.payload[:prefix])
                self.source.bad_start = prefix + self.segment
                self.source.bad_header = bad_header
                download.start()
                failure = self.finished(download, "error")
                self.assertFalse(download.target.exists())
                self.assertEqual(self.ready, [])
                saved = download.partial.read_bytes()
                self.assertEqual(failure["bytes_downloaded"], len(saved))
                self.assertGreaterEqual(len(saved), prefix)
                self.assertLessEqual(len(saved), self.source.bad_start)
                self.assertEqual(saved, self.payload[:len(saved)])
                request_count = len(self.source.requests)
                self.source.bad_start = None

                retry = self.downloader()
                retry.start()
                self.assert_complete(retry)
                self.assertIn(f"bytes={len(saved)}-{len(saved) + self.segment - 1}",
                              self.source.requests[request_count:])
                retry.target.unlink()
                self.ready.clear()

    def test_ignored_range_restarts_a_saved_prefix_and_verifies(self):
        self.source.ignore_ranges = True
        download = self.downloader()
        prefix = 101
        download.partial.write_bytes(self.payload[:prefix])
        download.start()
        self.assert_complete(download)
        self.assertIn(f"bytes={prefix}-{prefix + self.segment - 1}", self.source.requests)

    def test_small_remaining_prefix_uses_the_original_open_ended_range(self):
        download = self.downloader(SmallAdaptive)
        prefix = len(self.payload) - self.segment
        download.partial.write_bytes(self.payload[:prefix])
        download.start()
        self.assert_complete(download)
        self.assertEqual(self.source.requests, [f"bytes={prefix}-"])

    def test_adaptive_pilot_continues_on_parallel_only_for_a_meaningful_gain(self):
        prefix = 73
        single_stop = prefix + 2 * self.segment
        pilot_stop = single_stop + SmallAdaptive._RANGE_WORKERS * self.segment
        pilot = self.bounded_requests(single_stop, pilot_stop)
        for parallel_rate, keep_parallel in ((0.8, False), (1.05, False), (1.1, False), (2, True)):
            with self.subTest(parallel_rate=parallel_rate):
                download = self.downloader(SmallAdaptive)
                download.partial.write_bytes(self.payload[:prefix])
                download.parallel_rates = iter((parallel_rate,))
                download.start()
                self.assert_complete(download)

                self.assertEqual(self.source.requests[0], f"bytes={prefix}-{single_stop - 1}")
                self.assertCountEqual(self.source.requests[1:1 + len(pilot)], pilot)
                if keep_parallel:
                    continuation = self.bounded_requests(pilot_stop, len(self.payload))
                else:
                    continuation = [f"bytes={pilot_stop}-"]
                self.assertCountEqual(self.source.requests[1 + len(pilot):], continuation)
                download.target.unlink()
                self.source.requests.clear()
                self.ready.clear()

    def test_adaptive_parallel_rechecks_each_window_and_backs_off(self):
        download = self.downloader(SmallAdaptive)
        download.parallel_rates = iter((2, 1.05, 0.8))
        single_stop = 2 * self.segment
        pilot_bytes = SmallAdaptive._RANGE_WORKERS * self.segment
        pilot_stop = single_stop + pilot_bytes
        first_stop = pilot_stop + 4 * pilot_bytes
        second_stop = first_stop + 4 * pilot_bytes

        download.start()
        self.assert_complete(download)
        self.assertEqual(self.source.requests[0], f"bytes=0-{single_stop - 1}")
        next_request = 1
        for start, stop in ((single_stop, pilot_stop), (pilot_stop, first_stop), (first_stop, second_stop)):
            expected = self.bounded_requests(start, stop)
            self.assertCountEqual(self.source.requests[next_request:next_request + len(expected)], expected)
            next_request += len(expected)
        self.assertEqual(self.source.requests[next_request:], [f"bytes={second_stop}-"])

    def test_adaptive_parallel_finishes_a_short_tail_in_one_request(self):
        download = self.downloader(SmallAdaptive)
        parallel_bytes = SmallAdaptive._RANGE_WORKERS * self.segment
        parallel_stop = len(self.payload) - self.segment
        prefix = parallel_stop - 2 * self.segment - 5 * parallel_bytes
        download.partial.write_bytes(self.payload[:prefix])
        download.start()
        self.assert_complete(download)
        self.assertEqual(self.source.requests[-1], f"bytes={parallel_stop}-")

    def test_adaptive_pilot_falls_back_if_ranges_are_ignored(self):
        self.source.ignore_ranges = True
        download = self.downloader(SmallAdaptive)
        prefix = 101
        download.partial.write_bytes(self.payload[:prefix])
        download.start()
        self.assert_complete(download)
        self.assertEqual(self.source.requests, [f"bytes={prefix}-{prefix + 2 * self.segment - 1}", f"bytes={prefix}-"])

    def test_cancel_during_parallel_transfer_can_resume(self):
        download = self.downloader()
        prefix = 199
        download.partial.write_bytes(self.payload[:prefix])
        self.source.hold_start = prefix + self.segment
        download.start()
        self.assertTrue(self.source.later_served.wait(3), self.source.requests)
        self.assertTrue(download.cancel()["can_cancel"])
        self.source.release.set()
        cancelled = self.finished(download, "cancelled")
        saved = download.partial.read_bytes()
        self.assertEqual(cancelled["bytes_downloaded"], len(saved))
        self.assertGreaterEqual(len(saved), prefix)
        self.assertLess(len(saved), len(self.payload))
        self.assertEqual(saved, self.payload[:len(saved)])
        self.assertFalse(download.target.exists())
        self.assertEqual(self.ready, [])
        download.start()
        self.assert_complete(download)

    def test_sha_mismatch_never_activates_then_retry_restarts(self):
        self.source.payload = b"!" + self.payload[1:]
        self.assertNotEqual(self.source.payload, self.payload)
        download = self.downloader()
        download.start()
        failed = self.finished(download, "error")
        self.assertIn("SHA-256", failed["error"])
        self.assertFalse(download.target.exists())
        self.assertEqual(download.partial.stat().st_size, len(self.payload))
        self.assertEqual(self.ready, [])

        self.source.payload = self.payload
        request_count = len(self.source.requests)
        download.start()
        self.assert_complete(download)
        self.assertIn(f"bytes=0-{self.segment - 1}", self.source.requests[request_count:])


if __name__ == "__main__":
    unittest.main()
