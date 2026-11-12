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
