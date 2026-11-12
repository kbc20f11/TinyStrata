# GPT-OSS support and reintroduction

GPT-OSS is retired from TinyStrata's active model lineup as of 2026-11-12. Its download, launcher, model-picker and benchmark targets are no longer supported. GPT-OSS-specific native/MXFP4 build targets have also been removed. Historical measurements remain available; they describe earlier builds and are not current launch instructions.

This policy applies specifically to GPT-OSS reintroduction.

## Reason for retirement

[PR #14](https://github.com/kbc20f11/TinyStrata/pull/14) established the initial engine and Harmony integration. The subsequent optimization, debugging and validation consumed almost two full days of maintainer time. Follow-up work included HIP MXFP4 MMQ/device routing, bias and attention fusions, full-vocabulary GPU sampling, repeated Expert-weight reuse during verification, and correction of a CUDA MXFP4/Q8 activation-path defect. The [historical HIP adoption report](../benchmark/results/gptoss-adoption-20261110.md) records the settings and limits of those measurements.

After that work, the integration's performance, answer completeness and maintenance/build cost did not establish a sufficiently clear practical role alongside the existing LFM2.5 and Qwen options. Long-document checks still omitted prerequisites or confused required quantities with purchase quantities.

This is not an established limitation of the model itself. An independent llama.cpp reference passed the same document test with the same GGUF and input token IDs. RNG and KV representation differed, so those sampled answers do not isolate a single TinyStrata implementation defect. That integration discrepancy remains unresolved. Raising reasoning effort or reordering attached documents did not establish a general fix.

## Requirements for a future GPT-OSS PR

A reintroduction PR will be declined unless it already supplies all of the following:

- **Complete implementation and correctness evidence.** Compare with an independent implementation using the same weights and tokenized inputs. Declare numerical tolerances, KV representation, sampler/RNG differences and supported capabilities. Include actual final-answer checks for quantities, units, owners, deadlines, inherited conditions, revisions, unknown values and failure/retry behavior. Numerical checks and reasoning drafts do not replace delivered-answer validation. Keep failed results and distinguish evaluator ambiguity from factual errors.
- **A demonstrated practical benefit.** Provide reproducible short, 4K and 16K results across the intended 8/12/16 GB conditions, alongside relevant existing models. Record hardware, settings, actual output counts, total processing time, memory use and quality. Clearly label simulated capacity limits and isolated kernel measurements. A useful advantage in a stated workload or memory range is sufficient; review does not require one model to lead every benchmark.
- **Finished integration and regression coverage.** Validate fresh installation, updates, download/first load, model switching/unloading, API streaming, Harmony channels, reasoning effort, tools and the actual UI. Check shared-engine regressions and every advertised backend; document and enforce unverified capability limits. Model catalogs, installers, tests, benchmark commands, documentation and translations must agree.

The implementation and evidence must be included before acceptance so review has a bounded scope. Reintroduction must not depend on another open-ended maintainer optimization/debugging cycle. Existing numerical-quality and Expert-completeness policies still apply.

## Historical records

The [initial HIP follow-up](../benchmark/results/gptoss-adoption-20261110.md), [GPU sampling checks](../benchmark/results/gptoss-gpu-sampling-20261110.md), [fusion work](../benchmark/results/gptoss-fusions-20261110.md), [EAGLE-3 experiments](../benchmark/results/gptoss-eagle3-20261110.md) and saved benchmark results are retained as historical evidence. Retirement does not rewrite their measurements or turn unverified cases into passes.
