"""CPU-only checks of Intel API defaults and the shared context budget policy."""
from contextlib import ExitStack, redirect_stderr
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

from serve import intel
from serve.server import ByteTokenizer, CTX_SLACK, MockEngine
from serve.tinyconfig import DEFAULT_MAX_TOKENS, DEFAULT_SAMPLING


class IntelOutputOptions(unittest.TestCase):
    def entry_point(self, *extra, context=4096, environment=None, expected_error=None):
        """Exercise the real entry point and Service; replace native/file IO only."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            model, executable = root / "model.gguf", root / "engine.exe"
            model.touch()
            executable.touch()
            tokenizer = ByteTokenizer()
            engine = MockEngine(tokenizer, "", max_context=context)
            engine.alive = Mock(return_value=False)
            engine.close = Mock()
            inherited_environment = dict(os.environ if environment is None else environment)
            stack.enter_context(patch.object(intel, "os", SimpleNamespace(
                name="nt", environ=inherited_environment, pathsep=os.pathsep)))
            stack.enter_context(patch("serve.stdio.configure_utf8_stdio"))
            gguf = stack.enter_context(patch("serve.gguf_reader.GGUFFile", return_value=SimpleNamespace(
                metadata={"general.architecture": "qwen35moe", "general.name": "Qwen3.6"})))
            stack.enter_context(patch("serve.tokenizer.extract"))
            stack.enter_context(patch("serve.tokenizer.Tokenizer.from_dir", return_value=tokenizer))
            stack.enter_context(patch("serve.frontend.ChatTemplate", return_value=SimpleNamespace(call_format="function")))
            native = stack.enter_context(patch("serve.server.StrataEngine", return_value=engine))
            http = stack.enter_context(patch("serve.server.serve"))
            stack.enter_context(patch("serve.intel.signal.signal"))
            stopping = threading.Event()
            stopping.set()
            args = ["--model", str(model), "--exe", str(executable), "--context", str(context),
                    "--work-dir", str(root / "runtime"), *extra]
            if expected_error is not None:
                errors = io.StringIO()
                with redirect_stderr(errors), self.assertRaises(SystemExit) as error:
                    intel.main(args, stop_event=stopping)
                self.assertEqual(error.exception.code, 2)
                self.assertRegex(errors.getvalue(), expected_error)
                gguf.assert_not_called()
                native.assert_not_called()
                http.assert_not_called()
                return None
            self.assertEqual(intel.main(args, stop_event=stopping), 0)
            self.native_command = native.call_args.args[1]
            self.native_environment = native.call_args.kwargs["env"]
            self.inherited_environment = inherited_environment
            service = http.call_args.args[0]
            # Fixed prompt tokens isolate context admission from template/tokenizer IO.
            service.encode_prompt = Mock(return_value=list(range(64)))
            service.prompt_turn_at = Mock(return_value=None)
            return service

    def test_normal_default_fits_remaining_context_without_1024_cap(self):
        service = self.entry_point()
        self.assertEqual(service.default_max_tokens, DEFAULT_MAX_TOKENS)
        self.assertTrue(service.default_enable_thinking)
        self.assertEqual(service.sampling_defaults, DEFAULT_SAMPLING)
        request = service.with_shared({}, "openai")
        _, thinking, output = service.prepare([{"role": "user", "content": "question"}], None, {},
                                              request["max_tokens"], req=request)
        self.assertTrue(thinking)
        self.assertEqual(output, 4096 - CTX_SLACK - 64)
        self.assertGreater(output, 1024)

    def test_explicit_cli_default_is_used(self):
        service = self.entry_point("--max-tokens", "2048", context=8192)
        _, thinking, output = service.prepare([{"role": "user", "content": "question"}], None, {})
        self.assertEqual(output, 2048)
        self.assertTrue(thinking)
        self.assertEqual(service.sampling_defaults, DEFAULT_SAMPLING)

    def test_explicit_client_limit_still_refuses_context_overflow(self):
        service = self.entry_point()
        request = service.with_shared({"max_tokens": 5000}, "openai")
        with self.assertRaisesRegex(ValueError, "exceeds the context"):
            service.prepare([{"role": "user", "content": "question"}], None, {},
                            request["max_tokens"], req=request)

    def test_nonpositive_cli_default_is_refused_before_engine_start(self):
        for limit in ("0", "-1"):
            with self.subTest(limit=limit):
                self.entry_point("--max-tokens", limit, expected_error="output token limit")

    def test_draft_vocabulary_default_and_explicit_zero_are_disabled(self):
        for arguments in ((), ("--draft-vocab", "0"),
                          ("--mtp", "3", "--mtp-stochastic", "--vocab-xmx", "--decode-dp4a", "--draft-vocab", "0")):
            with self.subTest(arguments=arguments):
                self.entry_point(*arguments)
                self.assertNotIn("--draft-vocab", self.native_command)

    def test_draft_vocabulary_boundaries_and_size_are_forwarded_to_native(self):
        for limit in ("64", "1024", "4096"):
            with self.subTest(limit=limit):
                service = self.entry_point("--mtp", "3", "--mtp-stochastic", "--vocab-xmx", "--decode-dp4a",
                                           "--draft-vocab", limit)
                command = self.native_command
                self.assertEqual(command.count("--draft-vocab"), 1)
                self.assertEqual(command[command.index("--draft-vocab") + 1], limit)
                self.assertEqual(command[command.index("--mtp") + 1], "3")
                for dependency in ("--mtp-stochastic", "--vocab-xmx", "--decode-dp4a"):
                    self.assertEqual(command.count(dependency), 1)
                self.assertEqual(service.sampling_defaults, DEFAULT_SAMPLING)
                self.assertTrue(service.default_enable_thinking)
                self.assertEqual(service.default_max_tokens, DEFAULT_MAX_TOKENS)

    def test_draft_vocabulary_out_of_range_is_rejected_before_model_io(self):
        for limit in ("63", "4097", "-1"):
            with self.subTest(limit=limit):
                self.entry_point("--mtp", "3", "--mtp-stochastic", "--vocab-xmx", "--decode-dp4a",
                                 "--draft-vocab", limit, expected_error="--draft-vocab requires")

    def test_draft_vocabulary_requires_positive_mtp_depth(self):
        for depth in ((), ("--mtp", "0")):
            with self.subTest(depth=depth):
                self.entry_point(*depth, "--mtp-stochastic", "--vocab-xmx", "--decode-dp4a", "--draft-vocab", "64",
                                 expected_error="--draft-vocab requires")

    def test_draft_vocabulary_requires_exact_stochastic_sampling(self):
        self.entry_point("--mtp", "3", "--vocab-xmx", "--decode-dp4a", "--draft-vocab", "64",
                         expected_error="--draft-vocab requires")

    def test_draft_vocabulary_requires_interleaved_vocab(self):
        self.entry_point("--mtp", "3", "--mtp-stochastic", "--decode-dp4a", "--draft-vocab", "64",
                         expected_error="--draft-vocab requires")

    def test_draft_vocabulary_interleaved_vocab_still_requires_dp4a(self):
        self.entry_point("--mtp", "3", "--mtp-stochastic", "--vocab-xmx", "--draft-vocab", "64",
                         expected_error="--vocab-xmx requires --decode-dp4a")

    def test_disabled_draft_vocabulary_preserves_other_dependency_checks(self):
        self.entry_point("--draft-vocab", "0", "--mtp-stochastic",
                         expected_error="--mtp-stochastic requires --mtp")
        self.entry_point("--draft-vocab", "0", "--vocab-xmx",
                         expected_error="--vocab-xmx requires --decode-dp4a")

    def test_child_defaults_filter_adapters_without_changing_parent_environment(self):
        inherited = {"PATH": "inherited-runtime"}
        self.entry_point(environment=inherited)
        self.assertEqual(self.native_environment["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu")
        self.assertEqual(self.native_environment["UR_LOADER_PRELOAD_FILTER"], "1")
        self.assertEqual(self.inherited_environment, inherited)

    def test_explicit_user_device_controls_are_preserved(self):
        inherited = {"PATH": "inherited-runtime", "ONEAPI_DEVICE_SELECTOR": "level_zero:gpu:1",
                     "UR_LOADER_PRELOAD_FILTER": "0"}
        self.entry_point(environment=inherited)
        self.assertEqual(self.native_environment["ONEAPI_DEVICE_SELECTOR"], "level_zero:gpu:1")
        self.assertEqual(self.native_environment["UR_LOADER_PRELOAD_FILTER"], "0")
        self.assertEqual(self.inherited_environment, inherited)


if __name__ == "__main__":
    unittest.main()
