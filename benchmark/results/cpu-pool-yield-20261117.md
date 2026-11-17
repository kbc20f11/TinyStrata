# CPU Expert pool scheduling — 2026-11-17

The parked CPU workers now yield approximately every 50 microseconds while
retaining the existing 20 ms sleep deadline. This prevents an idle worker
from monopolizing a processor needed by a runnable worker. Expert selection,
quantized kernels, reductions and the host completion watchdog are unchanged.
The scheduling idea follows [Basalt febc72de](https://github.com/jesdga95/basalt/commit/febc72de27efb5884c3c339bba39153f3082de74).

## CUDA model check

Ryzen 9 7945HX3D (16 cores / 32 logical processors), RTX PRO 4000 Blackwell
SFF 24 GB, CUDA 13.2.86. Qwen3.6-35B-A3B-UD-IQ3_XXS used a 4 GiB Expert
cache, an 8,192-token context, its existing automatic KV policy, native default
CPU worker count, temperature 0, seed 42 and MTP disabled to isolate scheduling.
Each arm warmed the same request once, then generated 256 tokens from the same
Japanese document prompt. The timed request reused 4,926 input tokens and
read seven new tokens.

| Available CPU processors | Before, tok/s | After, tok/s | Output |
|---|---:|---:|---|
| Normal process affinity | 64.7 | 64.8 | Identical text and token counts |
| Eight logical processors, restricted after pool startup | 5.9 | 61.9 | Identical text and token counts |

The second row deliberately oversubscribes the already-created worker pool.
Its 10.49× result demonstrates recovery from scheduler contention. The normal
row shows essentially unchanged throughput; one paired run does not establish
a small percentage improvement. These are focused scheduling diagnostics.

The accepted candidate changes only the worker park loop. An earlier private
candidate also sampled host watchdog clocks more often; it was excluded from
this table. Baseline and accepted executables share the same guarded CUDA
objects and CPU kernel objects, with only the pool translation unit rebuilt.

## Correctness and portability

CUDA Qwen and Gemma API workflows also covered short and long source documents,
typed tool arguments, out-of-order parallel tool results, client cancellation
and a following request. Gemma additionally read an image and reused its
encoded cache. Source facts, units, dates and conditions were checked against
independent expected answers.

The public native Expert-pool test passed 2,112 batches across the control
and accepted builds, including serial/fused parity, sleep/wake and teardown
under CPU oversubscription. Another 200 controlled cadence batches retained
bitwise-identical outputs. The Python suite passed 712 tests; three separately
run CMake source-selection checks also passed.

The CPU scheduler is shared by CUDA and HIP. The Python process, image and
tool fixes are shared with SYCL. HIP and SYCL hardware execution was not
performed in this refresh.
