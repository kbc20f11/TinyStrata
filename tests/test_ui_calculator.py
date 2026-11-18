"""Decimal tool's known answers, bounds and browser-origin HTTP boundary."""
from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from ui.calculator import calculate, MAX_EXPRESSION, PRECISION
from ui.server import Handler, UIServer


class CalculatorTests(unittest.TestCase):
    def test_invoice_decimal_arithmetic_without_binary_float_drift(self):
        cases = {
            "0.1 + 0.2": "0.3",
            "(1980 * 3 + 2480 * 2) * 1.1 - 500": "11490",
            "1360000 / 26": "52307." + "692307" * 15 + "69231",
            "1.23e3 + .005 * 2": "1230.01",
            "2 ** -3": "0.125",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                result = calculate(expression)
                self.assertEqual(result["result"], expected)
                self.assertEqual(result["precision"], PRECISION)
                self.assertEqual(result["rounded"], expression == "1360000 / 26")

    def test_large_integer_precision_and_operator_precedence(self):
        self.assertEqual(calculate("9007199254740993 + 7")["result"], "9007199254741000")
        self.assertEqual(calculate("-2 ** 2 + (2 + 3) * 4")["result"], "16")
        self.assertEqual(calculate("2 ** 256")["result"], "115792089237316195423570985008687907853269984665640564039457584007913129639936")
        self.assertFalse(calculate("2 ** 256")["rounded"])

    def test_floor_remainder_follow_python_sign_and_avoid_cancellation(self):
        for expression, expected in {"-7 // 3": "-3", "-7 % 3": "2", "7 % -3": "-2", "7.5 % 2": "1.5", "1e1000 % 3": "1"}.items():
            with self.subTest(expression=expression):
                self.assertEqual(calculate(expression)["result"], expected)

    def test_rounding_and_pc_independent_determinism_are_explicit(self):
        result = calculate("1 / 3")
        self.assertEqual(result["result"], "0." + "3" * 100)
        self.assertTrue(result["rounded"])
        self.assertTrue(calculate("(1 / 3) * 3")["rounded"])
        self.assertFalse(calculate("1 / 8")["rounded"])
        # A caller's unrelated Decimal context must not change tool results.
        from decimal import Inexact, ROUND_DOWN, localcontext
        with localcontext() as context:
            context.prec = 2; context.rounding = ROUND_DOWN; context.traps[Inexact] = True
            self.assertEqual(calculate("2 / 3")["result"], "0." + "6" * 99 + "7")

    def test_only_arithmetic_not_python_or_host_access(self):
        expressions = ["__import__('os').getcwd()", "open('private.txt').read()", "(1).__class__", "abs(-2)", "[1][0]", "True + 1", "1j", "0xff", "1_000", "2 << 3", "1 if 1 else 2", "1; 2", "x := 2", "1 # comment", "1\n+2"]
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                calculate(expression)

    def test_invalid_undefined_and_resource_bounds(self):
        expressions = [None, True, 1, "", " " * 4, "1 / 0", "1 // 0", "1 % 0", "0 ** 0", "2 ** .5", "2 ** 1001", "2 ** -1001", "1e1001", "1e-1001", "1e1000 * 10", "1e-1000 / 10", "9" * 101, "1" * (MAX_EXPRESSION + 1), "+" * 40 + "1", "+".join(["1"] * 60), "(" * 300 + "1" + ")" * 300]
        for expression in expressions:
            with self.subTest(expression=str(expression)[:50]), self.assertRaises(ValueError):
                calculate(expression)

    def test_valid_magnitude_edges_have_bounded_plain_decimal_strings(self):
        self.assertEqual(calculate("1e1000")["result"], "1" + "0" * 1000)
        self.assertEqual(calculate("1e-1000")["result"], "0." + "0" * 999 + "1")
        self.assertEqual(calculate("-0.00")["result"], "0")


class CalculatorHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tinystrata-calculator-")
        self.server = UIServer(0, 9, Path(self.temp.name))
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.logs = mock.patch.object(Handler, "log_message"); self.logs.start()

    def tearDown(self):
        self.server.shutdown(); self.thread.join(3); self.server.server_close()
        self.logs.stop(); self.temp.cleanup()
        self.assertFalse(self.thread.is_alive())

    def request(self, body=None, *, method="POST", headers=None, raw=None):
        request_headers = {"Origin": self.origin, "Content-Type": "application/json", "Connection": "close", **(headers or {})}
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, "/local-ui/calculate", body=json.dumps(body) if raw is None and body is not None else raw, headers=request_headers)
            response = connection.getresponse(); data = response.read()
            return response.status, json.loads(data) if "application/json" in response.getheader("Content-Type", "") else data
        finally:
            connection.close()

    def test_browser_request_is_instant_and_does_not_prepare_python_or_search(self):
        with mock.patch.object(self.server.sandbox_runtime, "prepare", side_effect=AssertionError("No runtime download")), mock.patch.object(self.server.search, "search", side_effect=AssertionError("No search")):
            status, value = self.request({"expression": "(1980 * 3 + 2480 * 2) * 1.1 - 500"})
        self.assertEqual(status, 200); self.assertEqual(value["result"], "11490")
        self.assertFalse(value["rounded"])
        self.assertFalse(self.server.sandbox_runtime.path.exists())

    def test_host_origin_and_content_type_checks_apply_before_calculation(self):
        for headers, expected in [({"Host": "attacker.invalid"}, 403), ({"Origin": "https://attacker.invalid"}, 403), ({"Content-Type": "text/plain"}, 415)]:
            with self.subTest(headers=headers), mock.patch("ui.server.calculate", side_effect=AssertionError("Origin boundary bypassed")):
                self.assertEqual(self.request({"expression": "1+1"}, headers=headers)[0], expected)

    def test_bad_arguments_fail_without_leaking_host_data_and_retry_works(self):
        for body in [{}, {"expression": "1", "code": "ignored"}, {"expression": 1}, {"expression": "1/0"}, {"expression": "open('private.txt').read()"}]:
            with self.subTest(body=body):
                status, value = self.request(body)
                self.assertEqual(status, 400); self.assertIn("message", value["error"])
                self.assertNotIn("private.txt", value["error"]["message"])
        self.assertEqual(self.request({"expression": "1/8"})[1]["result"], "0.125")

    def test_json_and_method_boundaries(self):
        self.assertEqual(self.request(raw=b"[]")[0], 400)
        self.assertEqual(self.request(raw=b"{broken")[0], 400)
        self.assertEqual(self.request(raw=b"", headers={"Content-Length": "12289"})[0], 400)
        self.assertEqual(self.request(method="GET")[0], 404)


if __name__ == "__main__":
    unittest.main()
