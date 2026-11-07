# Decode diagnostics (not part of tinystrata-bench-v1)

Diagnostic aids for the Gemma/LFM decode investigation (issue #7). None of this changes engine behavior unless the
environment variable is set; none of it is a benchmark result.

## Engine instrumentation

These variables are read once per process and the state is global: they are meant for **one engine on one device**.
They are ignored when the engine is one of several cards (`split_`), and the added work is skipped when they are unset.

- `TINYSTRATA_HOST_PROF=1` (Gemma and LFM): host-side `std::chrono` around `Engine::moe()` for one-token calls whose
  picks were all cached. It adds no events and no syncs, so it can run with Graph enabled. It prints the time waiting
  for the router event, routing + preparing, and issuing the experts' launches. With the variable unset, the clock is
  not read.
- `TINYSTRATA_PHASE_PROF=1` (Gemma only, use `--no-graph`): GPU events around the phases of a layer, around each
  launch of the experts' chain, and around the device routing and the launches inside `moe_card()`. The extra events
  and syncs slow the run, so use it to localize time, not to quote throughput.

Example (engine run directly):

```powershell
$env:TINYSTRATA_HOST_PROF = '1'
tinystrata-gemma4.exe --model gemma-4-26B_q4_0-it.gguf --tokens (a list of token ids) --n 600 --max-context 8192 `
  --kv-type int8 --mtp 0 --kv-grow
```

## Standalone programs

All timings are stream spans between two events (they include submission gaps), not kernel-active times, and they use
the same event pattern in every mode.

- `gemma_mmvq_micro.cpp`
  - `shapes [calls] [host_idle_us]`: `gemma_native_mmvq` and `gemma_native_mmvq_slots` on the real Gemma 4 Q4_0 shapes,
    random weights rotated through at least 256 MB of contiguous memory.
  - `layout [steps] [host_idle_us] [layers]`: the routed-expert step as `Engine::moe()` issues it for one token: 8 picks
    among the slots of a layer, slot = [gate | up | down] back to back, the same activation for every gate+up pick,
    separate rows for the down picks. It runs on a contiguous `hipMalloc` arena and on a VMM arena mapped like the
    expert cache (64 MiB pieces), back to back and, if `host_idle_us` is non-zero, with the host spinning that long
    after the stream has drained before each step.
- `hip_launch_micro.cpp`: cost of empty kernel launches (back to back, onto an idle stream, and as a graph).

Build them with the HIP compiler (`-std=c++17`) against the repository's `include/` and the built `tinystrata_gemma4.lib`,
`strata_kernels.lib`, `tinystrata_gemma4_mmq.lib` and `strata_hip_support.lib` (`hip_launch_micro` needs only the HIP
runtime).
