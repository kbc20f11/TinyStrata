# Tiel Coder 35B-A3B support — 2026-11-06

Tiel uses TinyStrata's existing Qwen35MoE engine. This addition supplies a
separate model profile, pinned model/projector downloads, its own Sharp chat
template and the normal model-selection controls. No new native kernel is
required. The tested native executable is unchanged from the previous Expert
learning update: SHA256
`413a73b0c307bf0f60264cd9fe53cc10161465d3613589fefa85c1337e29a311`.

## Artifacts and model behavior

The pinned source is
[`peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP`](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP/tree/bb5f3f441849fc24c2d9037c310bc995404bfa0c),
revision `bb5f3f441849fc24c2d9037c310bc995404bfa0c`, MIT licensed.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `Tiel-Coder-35B-A3B-MTP-UD-IQ3_XXS.gguf` | 13,600,579,424 | `a4d8ffc7f030466c67db289543871a8f9e6d5340241b300c09fbaf16642273ca` |
| `mmproj-BF16.gguf` | 902,822,016 | `d9ce31026d1cb1f3f8d5152e2e2a014d9d2b302b6c93a7dc07bb0a0487f52837` |

Total download size is **14.503 GB**. The actual GGUF architecture is
`qwen35moe`: 40 main layers plus one MTP layer, 2,048 hidden dimensions,
256 Experts per layer, top-8 routing and 262,144-token context capacity.
The IQ3_XXS file is mixed quantization: main gate/up projections use IQ2_S,
37 down projections use IQ3_XXS and three use IQ4_XS. The trained MTP Expert
projections use Q3_K and its dense projections use Q8_0. These formats already
have native CPU/GPU paths.

Tiel's vocabulary, merges and token types match the installed Qwen3.6 tokenizer;
its **chat template differs**. TinyStrata retains Tiel's own Sharp template
(30,517 bytes, SHA256
`a41d3a6495c7531e85a45b665ae99c0251a68eaef5fada66cfa283827d3f6422`).
Live API checks confirmed system roles, XML function calls, parallel tool calls
and preserved reasoning. The template defaults to medium reasoning; explicit
high maps to its xhigh mode, and thinking can be disabled.

The supplied BF16 projector is Ornith's `qwen3vl_merger`, with 2,048-wide output,
27 vision blocks and no deep-stack outputs. It is downloaded as Tiel's pinned
asset rather than assumed identical to the Qwen projector. The model card's
[sampling guidance](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP/blob/bb5f3f441849fc24c2d9037c310bc995404bfa0c/README.md)
is temperature 1.0, top-p 0.95 and top-k 20; its coding experiments use
temperature 0.6. Tiel's profile keeps those defaults separate from Qwen's.

## Bounded MTP comparison

These are development diagnostics, **not `tinystrata-bench-v1` results** and not
physical 8/12 GB GPU simulations. They do not replace the standard README tables.

- Windows, Radeon RX 9060 XT 16 GB, driver 32.0.32015.7029, HIP/ROCm.
- One isolated API start for each MTP value; the same three short coding/design
  inputs generate 128 tokens each. Thinking is enabled at medium effort, so these
  capped outputs contain reasoning and do not measure completed coding tasks.
- FP8-MMA KV, 262K context **capacity**, chunk 2,048, 15 CPU workers plus the host,
  and a GPU-resident vision encoder. This is not a 262K-input test.
- Requested Expert budget 6 GiB; all three runs actually retain **5,658 Experts,
  6,279,337,824 bytes (5.848 GiB)**. Profile persistence is off and topic adaptation
  is on. All nine measured requests have zero reused KV tokens.
- Greedy requests use temperature 0, top-p 1, top-k 20, seed 1234 and no repetition,
  presence or frequency penalties. MTP is a native launch setting.

| MTP drafts | LRU tok/s | Intervals tok/s | Index design tok/s | Pooled tok/s | Accepted/offered drafts | Total request wall time |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40.0 | 42.5 | 44.2 | **42.15** | — | 10.455 s |
| 1 | 63.2 | 62.2 | 65.3 | **63.54** | 171/206 (83.0%) | 7.423 s |
| 3 | 64.9 | 79.6 | 75.4 | **72.76** | 239/405 (59.0%) | 6.698 s |

Pooled rates divide all 384 generated tokens by summed native decode time;
loading and prefill are excluded. The wall-time column includes each request's
prefill and API work, but excludes engine/encoder startup. In this screen MTP3
is 72.6% faster than MTP0 and 14.5% faster than MTP1 on pooled decode, supporting
MTP3 as the initial managed profile default. Higher acceptance alone does not
mean higher speed. Other hardware and workloads can favor a different setting.

This is one pass per configuration, without statistical confidence intervals.
Output text/reasoning differs across all three arms; adaptive Expert placement
and CPU/GPU arithmetic mean this is a complete-policy comparison, not a proof of
identical numerical work. A preliminary automatic-budget screen is excluded
from the MTP comparison because its Expert capacities differed between runs.

## Functionality checks

Both the isolated MTP1 profile and the installed managed **MTP3** profile passed
all seven smoke checks with thinking disabled:

- Exact arithmetic `3973`, correctly typed sorted JSON and Japanese `緑 青 赤`.
- Generated interval-merging and bracket-checking functions passed **13 behavioral
  cases** in isolated Python subprocesses, including empty/negative/touching inputs,
  Unicode and non-mutation. Execution was limited to AST-checked pure functions
  with a three-second timeout.
- The canonical image was identified as red square A, blue circle B and green
  triangle C. A text-only follow-up answered B and reused **1,004 prompt tokens**,
  exceeding the 948-token image-prefix threshold.

The installed API catalog lists five models, including Tiel independently from
Qwen despite their shared architecture. A real switch into Tiel loaded its MTP3
profile and exact Sharp template. A forced `lookup` call was parsed as a
structured tool call with `{"city":"Tokyo"}`. The test then switched back to
Clover and left it loaded with a healthy UI. All four pre-existing model
configuration hashes and every native executable remained unchanged.

These are bounded correctness and image-continuation checks, not broad coding,
vision, long-context or quantization-quality evaluations. The model card's own
coding benchmark claims use Q4_K_XL and are not attributed to this IQ3_XXS build.

Private evidence is in `TinyStrata-work/tiel-support-20261106/`: the `fixed6-mtp*`
timing reports, `mtp1/smoke`, `installed-mtp3-smoke`, `installed-switch-qa.json`,
fixed launch configurations, raw API responses and
`independent-runtime-audit.json`. The audit reconciles native timing/token counts,
capacity, template/source hashes, JSON types and original Python-check results;
it does not rerun GPU inference. The recorded harness SHA is
`2ad024e0a186fa74d4244b069999e9a2a37ba3ddb15c753147f269b62ebff002` and fixture SHA is
`b962e3a1762cc86cf73320190418f199746c42ee1339c833b7284c1cf1927caa`.
