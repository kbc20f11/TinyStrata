"""Request-bound live observations and bounded rolling rates; CPU mocks only."""
from collections import deque
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from serve import server as api
from serve.frontend import ChatTemplate, Event, ToolCall

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def tick(self, seconds=.004):
        self.now += seconds


class TimedEngine(api.MockEngine):
    def __init__(self, tok, script, clock, *, cancel_at=None):
        super().__init__(tok, script)
        self.clock, self.cancel_at = clock, cancel_at

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for index, token in enumerate(super().generate(ids, max_new, sampling, cancel, embeddings)):
            self.clock.tick()
            yield token
            if self.cancel_at is not None and index + 1 == self.cancel_at:
                cancel.set()


def fixture(script, clock, **kw):
    tok = api.ByteTokenizer()
    engine = TimedEngine(tok, script, clock, **kw)
    svc = api.Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
    # Statistics must not request global metrics or initialize hardware telemetry.
    svc.metrics = lambda *a, **k: (_ for _ in ()).throw(AssertionError('global metrics consulted'))
    svc.start_telemetry = lambda: (_ for _ in ()).throw(AssertionError('hardware consulted'))
    return svc, engine, tok.encode('user prompt', parse_special=True)


class RollingRates(unittest.TestCase):
    def test_high_rate_retains_two_seconds_instead_of_lifetime_fallback(self):
        samples = deque(maxlen=api.RATE_SAMPLES)
        first = 1000.0
        samples.append((first, 0))
        generated = 0
        # Four seconds at 2000/s, then four at 1000/s. A lifetime mean is 1500/s.
        for step in range(8000):
            generated += 2 if step < 4000 else 1
            api.note_rate_sample(samples, first + (step + 1) / 1000, generated)
        rate = api.measured_token_rate(generated, first, samples, first + 8)
        self.assertAlmostEqual(rate, 1000, delta=1)
        self.assertLessEqual(len(samples), 64)
        self.assertGreater(samples[-1][0] - samples[0][0], 2)

    def test_quiet_tail_decays_and_expires_without_new_tokens(self):
        samples = deque(maxlen=api.RATE_SAMPLES)
        samples.append((10., 0))
        for n in range(1, 1001):
            api.note_rate_sample(samples, 10 + n / 1000, n)
        active = api.measured_token_rate(1000, 10, samples, 11)
        quiet = api.measured_token_rate(1000, 10, samples, 11.5)
        self.assertGreater(active, quiet)
        self.assertEqual(api.measured_token_rate(1000, 10, samples, 13.01), 0)

    def test_young_rate_is_bounded_and_idle_is_zero(self):
        self.assertEqual(api.measured_token_rate(1, 10, [(10, 0)], 10.000001), 4)
        self.assertEqual(api.measured_token_rate(0, None, [], 10), 0)
        self.assertEqual(api.Service._rate_of({'busy': False}, []), 0)

    def test_parallel_rates_are_sum_of_separate_request_windows(self):
        svc = object.__new__(api.Service)
        svc.status_lock = threading.Lock()
        svc.live_reqs = {
            1: ({'busy': True, 'first_token': 10., 'generated': 200}, [(10., 0), (12., 200)]),
            2: ({'busy': True, 'first_token': 11., 'generated': 200}, [(11., 0), (12., 200)])}
        with patch.object(api.time, 'time', return_value=12.):
            self.assertEqual(svc._tok_s(), 300)


class StreamStatistics(unittest.TestCase):
    def run_chunks(self, script, *, enabled=True, thinking=False, tools=None, cancel_at=None):
        clock = Clock()
        svc, engine, ids = fixture(script, clock, cancel_at=cancel_at)
        req = {'stream': True, **({'return_live_stats': True} if enabled else {})}
        with patch.object(api.time, 'time', side_effect=lambda: clock.now), \
             patch.object(api.time, 'monotonic', side_effect=lambda: clock.now), redirect_stdout(StringIO()):
            chunks = list(api.openai_chunks(svc, req, ids, thinking, tools, 4096, threading.Event()))
        return chunks, engine, svc

    def test_opt_out_adds_no_wire_fields(self):
        chunks, _, _ = self.run_chunks('hello', enabled=False)
        self.assertTrue(chunks)
        self.assertFalse(any('strata_stats' in item for item in chunks if item))
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 6)

    def test_counts_are_tokens_not_visible_text_and_final_usage_is_unchanged(self):
        chunks, engine, _ = self.run_chunks('日本語🙂')
        stats = [item['strata_stats'] for item in chunks if item and 'strata_stats' in item]
        self.assertEqual(stats[0]['generated'], 0)
        self.assertIsNone(stats[0]['tok_s'])
        self.assertEqual(stats[1]['generated'], 1)
        self.assertEqual(stats[-1]['generated'], len(engine.script))
        self.assertTrue(stats[-1]['done'])
        self.assertEqual(stats[-1]['generated'], chunks[-1]['usage']['completion_tokens'])
        self.assertNotEqual(stats[-1]['generated'], len('日本語🙂'))
        self.assertEqual(set(stats[-1]), {'generated', 'tok_s', 'elapsed_s', 'first_token_s', 'done'})
        for item in chunks:
            if item and 'strata_stats' in item:
                self.assertEqual(item['choices'][0]['delta'], {})
                self.assertIsNone(item['choices'][0]['finish_reason'])

    def test_updates_are_throttled_except_first_and_final(self):
        chunks, _, _ = self.run_chunks('a different word. ' * 20)
        stats = [item['strata_stats'] for item in chunks if item and 'strata_stats' in item]
        # Initial zero, first observed token and terminal are explicit boundaries.
        ordinary = stats[1:-1]
        self.assertGreater(len(ordinary), 3)
        for before, after in zip(ordinary, ordinary[1:]):
            self.assertGreaterEqual(after['elapsed_s'] - before['elapsed_s'], .099)
        self.assertLess(len(stats), 25)
        self.assertTrue(all(a['generated'] <= b['generated'] for a, b in zip(stats, stats[1:])))

    def test_reasoning_and_tool_syntax_are_counted_when_not_visible_content(self):
        tool = {'name': 'lookup', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}}}
        script = 'Think carefully.</think><tool_call>\n<function=lookup>\n<parameter=query>weather</parameter>\n</function>\n</tool_call>'
        chunks, engine, _ = self.run_chunks(script, thinking=True, tools=[tool])
        stats = [item['strata_stats'] for item in chunks if item and 'strata_stats' in item]
        self.assertEqual(stats[-1]['generated'], len(engine.script))
        self.assertTrue(any(item['choices'][0]['delta'].get('reasoning_content') for item in chunks if item))
        self.assertTrue(any(item['choices'][0]['delta'].get('tool_calls') for item in chunks if item))

    def test_cancel_reports_only_observed_tokens(self):
        chunks, _, _ = self.run_chunks('long response text', cancel_at=5)
        stats = [item['strata_stats'] for item in chunks if item and 'strata_stats' in item]
        self.assertEqual(stats[-1]['generated'], 5)
        self.assertEqual(chunks[-1]['usage']['completion_tokens'], 5)
        self.assertTrue(stats[-1]['done'])

    def test_request_isolation_and_no_stats_in_native_sampling_keys(self):
        clock = Clock()
        svc, _, ids = fixture(['first response', 'x'], clock)
        totals = []
        with patch.object(api.time, 'time', side_effect=lambda: clock.now), \
             patch.object(api.time, 'monotonic', side_effect=lambda: clock.now), redirect_stdout(StringIO()):
            for _ in range(2):
                chunks = list(api.openai_chunks(svc, {'stream': True, 'return_live_stats': True}, ids,
                                                False, None, 4096, threading.Event()))
                stats = [item['strata_stats'] for item in chunks if item and 'strata_stats' in item]
                self.assertEqual(stats[0]['generated'], 0)
                totals.append(stats[-1]['generated'])
        self.assertEqual(totals, [15, 2])
        self.assertEqual(api.StrataEngine.sampling_keys({'return_live_stats': True}), '')

    def test_boolean_validation_is_strict(self):
        for value in (True, False, None):
            api.check_request_sampling({'return_live_stats': value})
        for value in ('true', 1, 0, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'return_live_stats'):
                api.check_request_sampling({'return_live_stats': value})

    def test_optional_stats_preserve_nonstream_collect_and_native_timings(self):
        clock = Clock()
        svc, _, ids = fixture('unused', clock)
        timings = {'predicted_n': 3, 'predicted_ms': 15., 'predicted_per_second': 200.}
        run = [('stats', {'generated': 3, 'done': True}), ('event', Event('content', 'ok')),
               ('done', {'finish': 'stop', 'completion_tokens': 3, 'timings': timings})]
        chunks = list(api.openai_chunks(svc, {'return_live_stats': True}, ids, False, None, 10,
                                        threading.Event(), run=iter(run)))
        self.assertEqual(chunks[-1]['timings'], timings)
        collected = api.openai_collect(iter(chunks))
        self.assertEqual(collected['choices'][0]['message']['content'], 'ok')
        self.assertEqual(collected['usage']['completion_tokens'], 3)
        self.assertNotIn('strata_stats', collected)

    def test_mcp_stats_accumulate_every_native_round_without_reset(self):
        clock = Clock()
        class Service:
            rounds = 0
            def run(self, *args):
                self.rounds += 1
                count = 3 if self.rounds == 1 else 4
                for n in (0, 1, count):
                    clock.tick(.11)
                    yield 'stats', {'generated': n, 'done': n == count}
                if self.rounds == 1:
                    yield 'event', Event('tool_call', call=ToolCall(name='lookup', arguments={}, id='call1'))
                yield 'done', {'completion_tokens': count, 'finish': 'stop'}
            def prepare(self, *args, **kw):
                return [1, 2, 3], False, 8
        hub = SimpleNamespace(settings={'max_rounds': 2}, routes=lambda: {},
            call=lambda *args: {'ok': True, 'chars': 2, 'text': 'ok', 'truncated': False, 'ms': 0})
        with patch.object(api.time, 'monotonic', side_effect=lambda: clock.now), redirect_stdout(StringIO()):
            events = list(api.run_with_mcp(Service(), hub, [], None, {}, [1], False, 8, 8,
                                          {'return_live_stats': True}, threading.Event(), {'lookup'}))
        stats = [value for kind, value in events if kind == 'stats']
        self.assertTrue(all(a['generated'] <= b['generated'] for a, b in zip(stats, stats[1:])))
        self.assertEqual(stats[-1]['generated'], 7)
        self.assertEqual(sum(value['done'] for value in stats), 1)
        self.assertTrue(stats[-1]['done'])
        self.assertEqual(events[-1], ('done', {'completion_tokens': 7, 'finish': 'stop', 'prompt_tokens': 3}))


if __name__ == '__main__':
    unittest.main()
