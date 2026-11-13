# Qwen UI identity question: reproduction and llama.cpp comparison

Measured on 2026-11-12, RTX PRO 4000 Blackwell SFF 24 GB, driver 597.30. The question was exactly `あなたはどのようなモデルですか`, sent from a fresh TinyStrata chat with the default settings: thinking and web search on, Python available. The effective Qwen sampler was temperature 0.7, top-p 0.95, top-k 20, presence penalty 1.5, repetition penalty 1, seed 0. The model was the installed Qwen3.6-35B-A3B-UD-IQ3_XXS GGUF.

## Observations

| Condition | Input tokens | Generated tokens | Response time | Observed result |
|---|---:|---:|---:|---|
| Original TinyStrata UI, first model request | 663 | 5,974 | 51.1 s | Repeated uncertainty about its own identity, then an unnecessary Python call |
| Original UI, complete exchange | 663, then 791 | 6,677 total | 59.2 s | Final Japanese answer after Python and another generation |
| TinyStrata, MTP disabled, original request | 663 | 5,974 | 60.8 s | Same long reasoning and the same unnecessary Python call |
| TinyStrata, INT8 KV, original request | 663 | 560 | 4.9 s | Finished, but incorrectly identified itself as developed by Google |
| Official llama-server, original UI messages and tools | 681 | 417 | 4.8 s | Normal Qwen answer, no tool call |
| Official llama.cpp, exact TinyStrata input IDs | 663 | 260 | 2.9 s | Normal Qwen answer, no tool call |
| Improved TinyStrata UI, default FP8/MTP settings | 690 | 576 | 4.8 s | Correct runtime model ID, no tool call |

The original Python call only printed `Thinking process simulation complete.`; it did not help answer the question. Both the original and improved UI exchanges eventually stopped normally: the original was excessive reasoning and an unnecessary tool, rather than an observed permanent hang. The nearby wording `あなたのモデル名を教えてください` finished in about 1.2 seconds without the UI improvement.

The improved UI was tested twice. The content and reasoning matched; the second request reused 685 cached input tokens, so it is a cache-reuse regression check rather than a second independent cold performance sample.

## Independent implementation

The comparison used the [official llama.cpp b12318 Windows CUDA 13.5 release](https://github.com/ggml-org/llama.cpp/releases/tag/b12318), with both downloaded ZIP hashes matched to GitHub's published digests. No system CUDA files were replaced.

The same GGUF, Japanese question, original system text and `web_search` / `run_python` definitions were used. Thinking and the effective numeric sampling settings were retained. The official server used a 65,536-token context, a 32,768-token output cap, one slot, full GPU offload, Flash Attention and F16 KV. TinyStrata used its default FP8 KV and 256K context. TinyStrata uses `mt19937_64`; the [official b12318 sampler uses `mt19937`](https://github.com/ggml-org/llama.cpp/blob/b12318/src/llama-sampler.cpp). These differences, sampler order and numerical kernels prevent a bit-identical sampled-output expectation. This comparison does not establish that a specific TinyStrata kernel is incorrect, or that the problem is intrinsic to the GGUF/model.

The initial 18-token prompt difference was traced to tool serialization: the official chat path rendered the API `type/function` wrapper, while TinyStrata rendered the normalized function objects. Both used the same 8,057-byte embedded chat template. A second llama.cpp control therefore sent the exact 663 TinyStrata input IDs through `/completion`, eliminating that renderer difference. It still finished normally. The native completion control includes the tool definitions in the prompt; it bypasses the official chat response parser and tool grammar.

The 663-token input's int32 little-endian SHA-256 was `773120b0ad882fcc9d3f638852fc7b90abafa2707c7011d781bef899fdac6b53`. The official server SHA-256 was `7cbf01f6cd175ffd2ffad22a266eb7ebef06072c13c582f3b9cc3afa81d7ffa3`.

## UI change and scope

The UI adds one factual line identifying the model ID reported by the validated `/health` response. It does not guess from the picker, old chat messages, a filename, a provider or a capability list. Bootstrap/unknown health is omitted; custom model IDs are quoted literally. A request keeps the same model snapshot during token counting, compaction, retries and local-tool rounds, including manual KV compaction.

This is general deployment context, without matching the user's question or supplying a fixed answer. Native decoding, API defaults, temperature, KV policy, MTP and tool availability are unchanged. It is a verified UI improvement for this case, not proof that the implementation-level difference between the original TinyStrata and llama.cpp samples has been resolved.

Real Playwright checks confirmed the default Qwen exchange and its repetition, normal sandboxed Python use (sum of squares from 1 to 10,000: **333383335000**), and a Qwen-to-Gemma switch whose following count/completion requests used the Gemma model ID. Screenshots were inspected. Relevant Node checks cover serializer snapshots, context compaction, history/tool roles, live statistics, model selection and the homepage. Original installed profiles remained unchanged, and owned test servers/models were stopped. AMD hardware was not re-tested in this session.

Private requests, SSE, answers, captures, executable provenance and comparison records are retained under `TinyStrata-work/qwen-ui-identity-20261112`; they are not included as full reasoning transcripts in this report.

## Follow-up: effective temperature and official settings (2026-11-13)

The original UI request omitted temperature, top-p and top-k. The API startup log reported the saved Qwen defaults as **0.7 / 0.95 / 20**; the server merges those defaults before generation. The previous UI displayed **1.0 / 0.95 / 20** as an implicit-default marker, without forwarding it. Even manually selecting that complete tuple was indistinguishable from the marker. Thus the original reproduction was at effective temperature **0.7**, despite the UI display. The saved native `GEN` line in the private token-control fixture is a CPU reconstruction, not a captured native stdin trace.

The [official Qwen3.6-35B-A3B recommendations](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#best-practices), checked on November 13, distinguish thinking/general (**temperature 1.0, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5, repetition penalty 1**), thinking/precise coding (temperature 0.6, presence penalty 0) and non-thinking (temperature 0.7, top-p 0.8). The follow-up used thinking/general, retaining the original system/user/tool definitions without the added runtime model ID.

| KV | Temperature | Input / output tokens | Request time | Observed result |
|---|---:|---:|---:|---|
| FP8 | 0.7 | 663 / 5,974 | 51.1 s | Original long reasoning and unnecessary Python call |
| INT8 | 0.7 | 663 / 560 | 4.9 s | Incorrect Google identity |
| FP8 | 1.0 | 663 / 410 | 3.3 s | Correct Qwen identity, normal stop, no tool call |
| INT8 | 1.0 | 663 / 1,756 | 18.6 s | Generic AI assistant answer, normal stop, no tool call; no specific model ID |

Both new arms explicitly requested the same official numeric settings, with MTP 3, 2048-token chunks, 262,144 context capacity and all Experts on GPU. The requested output cap was 32,768. An explicit API seed of 0 is not forwarded by the current adapter, but the native sampler's existing default is also 0. TinyStrata's penalty window remains 64 tokens; matching these numeric settings does not make its RNG or penalty semantics identical to every official serving framework. New runs were cold replay requests through the API, not another real-UI verification. Original profiles remained unchanged during this comparison and both test models unloaded normally.

This case improved with the official thinking/general temperature even with FP8. It does not establish a universal cure for repetition or prove a KV implementation defect. The user then chose those official numeric values as Qwen's default. The application change and actual UI verification are recorded separately below; historical benchmark inputs and recorded temperatures remain unchanged.

Private captures and cleanup receipts are retained under `TinyStrata-work/qwen-official-chat-20261113`.

### Applied default and real UI validation

Following the user's instruction, new/omitted Qwen sampling defaults now use the official thinking/general numeric settings. Other models, KV/MTP policies and fixed v1 benchmark inputs remain unchanged. Existing saved values remain explicit settings: old automatic 0.7 and an intentional 0.7 have no distinguishing provenance. This PC's Qwen profile was separately updated to 1.0, with all other profile bytes retained. Non-thinking and precise-coding sampler presets were not added by this change.

The UI now records whether sampling follows the model or is custom, instead of comparing numbers to a hidden sentinel. It displays the current model's actual `/props` defaults, sends manually edited values literally, and resets to model defaults. Unknown defaults display an em dash with disabled sliders until a successful retry; stale model responses cannot replace a staged edit or reset.

Real Playwright/model checks confirmed default Qwen **1.00 / 0.95 / 20**, a normal Qwen self-introduction (**556 output tokens, approximately 5.4 seconds**) and literal manual temperature 1.0 with a privately changed server default of 0.7. An actual LFM switch displayed **0.1 / 1.0 / 50** and Qwen restored 1.0. Delayed metadata after Reset and an intentionally failed metadata request/retry were also checked. Light/dark screenshots were inspected; no horizontal overflow or clipped sampling labels was observed.

A separate 6,019-character document contained an obsolete order, 80 unrelated records and a final correction. Through the actual UI, the default Qwen read **4,812 input tokens** and returned all **eight** independently specified fields correctly in one generation: project, owner, quantity, unit price, total, deadline/time zone, delivery location and approval condition. It took approximately **6.4 seconds**, without an unnecessary search or tool round. These bounded checks do not claim that every high-temperature answer or long conversation is correct. Model-switch metric polls returned temporary HTTP 503 while the models loaded; both switches recovered.

Relevant Python profile/benchmark and Node sampling/thinking/runtime/live-stat checks passed. The original native executable was reused; no CUDA/HIP kernel was changed. AMD was not re-run for this configuration/UI change. Private UI data, request/SSE captures, expected answers and cleanup records are retained alongside the official-setting replay.

## Bounded FP8 numerical audit (2026-11-13)

The audit forced the same 128 generated token IDs from the independent correct llama.cpp continuation, recording 129 full-vocabulary logit rows. FP8 and INT8 were compared with the existing main MMQ route both enabled and disabled. All recorded logits were finite. With MMQ enabled, FP8 and INT8 agreed on top-1 at **126/129** rows; the disagreements were small-margin choices, rather than non-finite output or a discontinuity at the checked cache boundary. With MMQ disabled they agreed at 128/129 rows.

An independent offline analysis at temperature **0.7** gave mean KL(INT8 to FP8) **0.005170** and mean continuation NLL **0.574404 / 0.575850**. These are softmax(logits / 0.7) measurements without top-k, top-p or penalties, not the complete production sampling distribution and not an answer-quality score. The audit also retains raw temperature-1 metrics.

A first-row FP32-KV control used matched 65,536 context capacity and chunk 2048 for all three KV formats. The FP8 and INT8 first rows were byte-identical to their respective 262,144-context rows. FP8's distribution was farther from FP32 than INT8's: offline temperature-0.7 KL(FP32 to candidate) was **0.003108 / 0.002510**. FP32 KV does not make the entire network FP32; storage quantization and the attention implementation both change between these arms. This first-row check cannot establish later-token equivalence.

An independent pinned llama.cpp F16-KV logit reference also completed at prefix positions 0, 64 and 128. Its prefix batching and recurrent accumulation differ from TinyStrata's incremental forced decoding, so differences cannot be attributed solely to KV quantization. The first reference attempt correctly rejected a fixture whose text had been converted from LF to CRLF; that failed attempt is retained. A separate corrected LF fixture verified identical token IDs before running.

Finally, the **current CUDA FP8 attention kernels** passed a synthetic same-stored-cache comparison against independent CPU FP64 calculations for **521 and 663 rows**, including ragged/causal tails, cache offsets, split boundaries and graph replay/rollback. This isolates arithmetic using the same stored FP8 codes and scales. It does not validate the quantization quality of every model cache or prove whole-model answer correctness. No product kernel, MMQ policy or KV default was changed based on this audit.

Private source/build provenance, full logits, independent reports and execution receipts are retained under `TinyStrata-work/qwen-fp8-numerics-20261113`.
