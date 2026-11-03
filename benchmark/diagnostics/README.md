# Decode diagnostics (not part of tinystrata-bench-v1)

Diagnostic aids for the Gemma/LFM decode investigation (issue #7). None of this changes engine behavior unless the
environment variable is set; none of it is a benchmark result.

## Engine instrumentation

- `TINYSTRATA_HOST_PROF=1` (Gemma and LFM): host-side `std::chrono` around `Engine::moe()` for one-token calls whose
  picks were all cached. It adds no events and no syncs, so it can run with Graph enabled. It prints the time waiting
  for the router event, routing + preparing, and issuing the experts' launches.
- `TINYSTRATA_PHASE_PROF=1` (Gemma only, use `--no-graph`): GPU events around the phases of a layer and around each
  launch of the experts' chain. The extra events and syncs slow the run, so use it to localize time, not to quote
  throughput.

Example (engine run directly):

```powershell
$env:TINYSTRATA_HOST_PROF = '1'
tinystrata-gemma4.exe --model gemma-4-26B_q4_0-it.gguf --tokens (a list of token ids) --n 600 --max-context 8192 `
  --kv-type int8 --mtp 0 --kv-grow
```

## Standalone programs

- `gemma_mmvq_micro.cpp`: times `gemma_native_mmvq` and `gemma_native_mmvq_slots` on the real Gemma 4 Q4_0 shapes with
  random weights rotated through at least 256 MB. `gemma_mmvq_micro.exe [iterations] [host_idle_us]`; with a
  non-zero `host_idle_us` the host waits that long before each call, with the stream idle.
- `hip_launch_micro.cpp`: cost of empty kernel launches (back to back, onto an idle stream) and of a graph of empty
  kernels.

Both are built with the HIP compiler against the repository's `include/` and the built `tinystrata_gemma4.lib`,
`strata_kernels.lib`, `tinystrata_gemma4_mmq.lib` and `strata_hip_support.lib` (the second needs only the HIP runtime).
