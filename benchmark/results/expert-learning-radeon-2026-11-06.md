# Learned Expert placement and topic adaptation on Radeon — 2026-11-06

These are single-pair engineering comparisons, **not `tinystrata-bench-v1`
results**. They do not replace the historical README benchmark tables and do
not establish a speed improvement on every model, workload or Radeon.

## Changes and reference

- Save and restore each layer's complete Expert ranking. The file stores no
  conversation text, KV cache or fixed VRAM capacity. At load, current memory
  planning chooses how many ranked Experts fit. Invalid or mismatched files
  fall back to ordinary placement.
- Seed admission from the newly read prompt: normalize routing counts to a
  64-token equivalent, retain 0.3 of prior usage and seed up to 32 new
  nonresident candidates **per layer**. Resident candidates retain their signal.
  Existing slot capacities remain unchanged. Topic mode also enables fair
  layer scanning on HIP unless explicitly disabled; it is a policy comparison,
  not an isolated arithmetic-kernel comparison.
- Managed Qwen/Clover launches enable profile persistence and topic adaptation;
  explicit profile `off` and topic `0` remain respected. Direct native launches
  require explicit selection. The short-prompt smoothing experiment below is
  not retained.

The design references Basalt at
[`bbf2f5bb783adc5ffe5b49043f82c9399313bb46`](https://github.com/jesdga95/basalt/commit/bbf2f5bb783adc5ffe5b49043f82c9399313bb46):
[profile ranking](https://github.com/jesdga95/basalt/blob/bbf2f5bb783adc5ffe5b49043f82c9399313bb46/src/core/expert_cache.cpp),
[profile persistence](https://github.com/jesdga95/basalt/blob/bbf2f5bb783adc5ffe5b49043f82c9399313bb46/src/program/serve/tier_profile.cpp)
and [prompt seeding](https://github.com/jesdga95/basalt/blob/bbf2f5bb783adc5ffe5b49043f82c9399313bb46/src/program/serve/tiers.cpp).
TinyStrata keeps its per-layer capacities rather than importing Basalt's global
Expert tier. RAM-complement storage is outside this change.

## Machine, scope and provenance

- Windows x64, AMD Radeon RX 9060 XT 16 GB, HIP/ROCm. GPU power limit was not
  captured; no power-normalized result is claimed.
- Every measurement here was newly executed after the November 5 reboot
  (approximately 23:19:26 JST). The reported sequences began November 6,
  05:08–05:49 JST. No pre-reboot performance values were reused.
- **3/6 GiB are fixed Expert budgets on this physical 16 GB card. They are not
  physical or simulated 8/12 GB GPU benchmarks.**
- Native text-only execution, no resident image encoder, context 32,768,
  automatic chunk resolved to 2,048, 15 CPU workers plus the participating host.
  Qwen uses MTP3; Clover uses causal D0/MTP0, `STRATA_IQ_MT_MIN=1` and
  `STRATA_EXP_MODE=8`. KV format is identified in each table.
- Greedy sampling: temperature 0, top-p 1, top-k 64, seed 1234, repetition 1,
  presence/frequency 0. Thinking is enabled for performance fixtures and disabled
  for the four answer checks. Prefix checkpoint reuse is disabled.
- Each comparison has one fresh OFF/ON pair, with no confidence interval or
  repeated statistical ranking. Native startup warm-up remains; there is
  intentionally no additional harness warm-up, to expose first-request effects.
  Rates count actual native generated tokens over native decode time. Pooled
  rates divide summed tokens by summed time, not the mean of fixture rates.

Source baseline:
[`375830f541dbaccec83abaf533cbfe00f81e1b47`](https://github.com/kbc20f11/TinyStrata/commit/375830f541dbaccec83abaf533cbfe00f81e1b47).
The timed development executables are identified separately; a final build or
source commit must not be described as having been timed merely by association.

| Development executable | SHA256 | Evidence folders |
|---|---|---|
| v1 | `a02c36216314ca507f0747866587e1911b90b2bf577d8f25a600b32ab5d88d50` | `qwen-topic-v1` |
| v2 | `8fb26c13ecdd43b792c8c1c78510df5b35ce9ac4a89fa0b46b0b50ddd5133b52` | `clover-topic-v2`, both `*-profile-v2` |
| v3 | `868510fe075d10489f2571ee56589d4346a5e32ac7c198301f05fa99f8966c1b` | both `*-combined3-fp8-v3`, both `*-quality-fp8-v3` |
| v4, rejected smoothing experiment | `2243d137cab0d2b2d38a008b561283ef7d8ae747bcb65f4381925d40269e4ee4` | both `*-short-smooth-v4` |
| final native candidate | `413a73b0c307bf0f60264cd9fe53cc10161465d3613589fefa85c1337e29a311` | `qwen-topic-16k-fp8-final`; earlier timing rows remain tied to their own hashes |

Each OFF/ON pair uses the same recorded executable, model, byte capacity,
KV/MTP/chunk/workers and frozen input IDs. Expert mappings and often generated
answers intentionally differ. Independent audits reconciled raw `T`/`DONE`
records and prompt/decode counters; saved application configurations were
unchanged. Local evidence is under
`TinyStrata-work/expert-learning-20261106/<folder>/`, including `result.json`,
`prompts.json`, native logs and the six primary `independent-audit.json` files.
The separate final 16K screen also has a passing independent integrity audit.

## Reproducing the workload

Use the existing pinned Qwen3.6 UD-IQ3_XXS and Clover AD-IQ3_S-IQ3_XXS GGUFs
and their own tokenizers/templates. Native controls are `--expert-profile off`
versus a profile file, and `--expert-topic-cache 0/1`; keep all other settings
above constant. Recreate each pair from fresh engines, never carrying a mutable
trial profile into its control.

- **Profile training:** two requests capped at 128 outputs, in this order:
  `Implement an LRU cache in Python with O(1) get/put. Explain invariants, then show five tests.`
  and `Design a local document indexing service. Explain its hash tables, queues, cache invalidation and failure recovery in detail.`
  Freeze the saved ranking after training and copy that immutable file for every
  trial.
- **A, 4K/16K conditioning:** render a thinking-enabled user request containing a
  fictional software archive. Entry `i` is
  `Archive {i:05d}: A Python service stores bounded caches, hashes keys, evicts least-recent entries and records verification tests.\n`.
  Generate `N // 20 + 100` entries for target `N=4096` or `16384`, prepend
  `The following is a fictional software engineering archive.\n`, append
  `\nSummarize the engineering workflow and provide an implementation with tests.`
  and retain token IDs `ids[:N-256] + ids[-256:]`. Cap output at 32.
- **B:** `Explain how a refrigerator works. Cover compression, condensation, expansion and evaporation, and explain where heat goes.`
- **Held B:** `Explain how a heat pump heats a room in winter. Compare its indoor and outdoor coils with a refrigerator.`
- **Held code:** `Implement interval merging in Python. Handle unsorted input, touching intervals, negative endpoints and empty input. Explain complexity and show five tests.`
  These three short fixtures are capped at 256 outputs.

Both model tokenizers produced the following input-ID hashes (comma-separated
decimal IDs, SHA256); this also fixes the rendered template, not just the text.

| Fixture | Tokens | SHA256 |
|---|---:|---|
| A | 4,096 | `b757b758709ca3712285eeed600470eaa27dbf6dc8e85248045b8fc972ed1fb5` |
| A, 16K extension | 16,384 | `842f9792116171a3924d79d0b09a8b77842b7f5f40d1c49b205c9dc0586450f7` |
| B | 34 | `e718dac5b6bab35089210a1a26b6aa95a13eed99d0e308e77dc657c6fabe8e12` |
| Held B | 32 | `96646766763de7e0da701269f141ad01160317406a38ef5a312f3b17e012d3d2` |
| Held code | 37 | `a759210f20b3cb6927ad3991c1bde1db643870ae08fbb69752b1c990a30bc9cc` |

The v1/v2/v3 private harness SHA256 is
`1791e42cf223c7c730148c3f90fcbdaf52f8bb091ef27efe7830bc93a8ecdb40`.
The scorer-corrected/v4 harness is
`860cd19ad01c53a8bf52b47d15efa2cf124d920f0d8ba97cce58f737b352b288`.

## Topic adaptation after a long different topic

Run A → B → held B, with persistence OFF and a 3 GiB Expert cap. Every request
has zero KV reuse. Rates below are decode tok/s, OFF → topic ON.

| Model / KV | A | B | Held B |
|---|---:|---:|---:|
| Qwen / INT8 | 62.92 → 62.87 (-0.08%) | 59.50 → 73.39 (**+23.33%**) | 66.67 → 78.06 (+17.08%) |
| Clover / FP8-MMA | 30.93 → 31.99 (+3.44%) | 30.27 → 35.45 (**+17.12%**) | 33.38 → 36.40 (+9.05%) |

**Qwen B is the strongest same-output comparison:** all 256 native token IDs
match, with the same 169 accepted / 245 offered MTP drafts. GPU-served
decode Expert entries rise from 23.07% to 64.92%. This is an Expert-entry share,
not GPU utilization. Total request wall time falls 4.524 → 3.752 s (-17.08%),
while first output increases 221.7 → 262.9 ms: cache adaptation has a cost.

Qwen A/held B and Clover B/held B produce different token sequences; their
figures are whole-policy observations, not identical-work kernel gains. Clover
A has identical output IDs. Clover B's first output similarly increases
186.7 → 227.1 ms despite faster decode. These three-request screens do not prove
that all topic changes improve throughput or latency.

## Restored profiles and capacity independence

Training, OFF and restored-profile arms use a 6 GiB Expert budget and topic
adaptation OFF. First measured request is held code; B follows it.

| Model / KV | Applied Experts / total | Held-code prompt ms, OFF → restored | Pooled decode tok/s, OFF → restored |
|---|---:|---:|---:|
| Qwen / INT8 | 5,494 / 10,496 | 239.7 → 171.3 (-28.54%) | 84.83 → 90.82 (+7.06%) |
| Clover / FP8-MMA | 4,920 / 10,240 | 217.4 → 120.2 (-44.71%) | 36.49 → 37.45 (+2.64%) |

All four measured profile-restoration answers differ from their controls, and
Qwen B's MTP work differs. Cold prompt/GPU-entry improvements are observed;
there is no claim of bitwise-neutral answer behavior or faster model loading.

The frozen profiles contain all 256 ranks per layer: Qwen 41 rows / 42,968 bytes,
Clover 40 rows / 41,922 bytes. Checksums and every row permutation were audited.
Their training capacities are not serialized. Reusing these same immutable
6 GiB-trained files at a 3 GiB budget applies **2,665 Qwen / 2,440 Clover**
Experts, rather than restoring the old slot count. Actual Expert payload bytes
are constant within each comparison:

| Model | 3 GiB cap | 6 GiB cap |
|---|---:|---:|
| Qwen | 3,107,409,760 | 6,406,044,736 |
| Clover | 3,156,384,000 | 6,364,512,000 |

Profile identity binds model-shard metadata/sampled content and quantized layout;
it is a hint identity, **not a full-file weight checksum**. The profile stores
rankings, never physical GPU pointers or numerical model state.

## Combined profile + topic screen at 3 GiB, FP8-MMA

Use held code → B → held B, 256 outputs each. Both models restore the frozen
6 GiB profile into the smaller current capacity. OFF disables both features.

| Model | Held code decode | B decode | Held B decode | Pooled decode |
|---|---:|---:|---:|---:|
| Qwen | +2.27% | +12.18% | +2.73% | 72.72 → 76.91 tok/s (**+5.76%**) |
| Clover | -1.59% | +5.76% | -3.63% | 35.00 → 35.06 tok/s (**+0.17%**) |

All six outputs differ; Qwen's MTP work also differs. Clover is a mixed result,
not a general speedup: two of three decode fixtures regress. Pooled request wall
time changes -6.09% for Qwen and -0.70% for Clover. Including engine startup,
Clover is 36.803 → 37.040 s, also preventing an unconditional latency claim.

## Final-binary fresh 16K conditioning screen

`qwen-topic-16k-fp8-final` uses the final native candidate above, Qwen FP8-MMA,
3 GiB Experts and persistence OFF. It runs fresh 16K A → unrelated fresh B →
fresh held B, with zero KV reuse on every request. The Expert placement learned
from A persists; its 16K KV context does not. This measures placement adaptation
after conditioning, **not decode with a retained 16K context**. It is a separate
fresh pair, not a continuation or retiming of the earlier binaries.

| Fixture | Output tokens | Decode tok/s, OFF → topic | Decode change | Request wall seconds, OFF → topic |
|---|---:|---:|---:|---:|
| 16K A | 32 | 64.22 → 54.09 | **-15.77%** | 55.086 → 54.957 |
| B | 256 | 61.75 → 72.43 | +17.30% | 4.383 → 3.804 |
| Held B | 256 | 61.96 → 75.58 | +21.99% | 4.373 → 3.574 |

All three output token sequences and MTP work differ. The conditioning request's
32-token decode regresses; its 16K input processing is almost flat,
300.15 → 301.38 tok/s (+0.41%). Including that full conditioning cost, the
complete sequence is **63.843 → 62.335 s (-2.36% wall time)**, excluding engine
startup. The two later requests alone are 8.757 → 7.378 s (-15.74%). Reporting
only their +17–22% decode rates would overstate the whole-context benefit.

## Quality and state checks; rejected smoothing

FP8 checks compare OFF and both features at 3 GiB: multiplication (`3973`),
sorting three JSON objects, reversing the three Japanese color words, and an
exact named-value lookup (`7319`). Both models pass 4/4 in each arm: **16/16**.
This is a limited answer/format/completion screen, not PPL/KLD, broad reasoning,
vision-quality or long-conversation quality validation.

The original private scorer counted rendered terminal EOS as answer content.
It was corrected to strip only known trailing native EOS controls and outer
whitespace. All 16 actual terminal token IDs were verified as 248046; raw
tokens/text/timing were not changed, original grades/reports were retained and
nine scorer checks passed. This rescoring did not rerun GPU inference.

Separate v4 fixed-membership save/restore checks pass for both models: all layers
retain the same resident Expert sets (65 per layer for Qwen, 61 for Clover),
with identical native output IDs on the 64-token and 5-token fixtures, MTP counts
and phase CPU/GPU work. Slot order may change during serialization.

Final-candidate `cache-prefix-qa-final-r2` also passes for both models: 11 standard
cases each, 16 observed requests and four cancellations per model, covering
cancel/retry, edited turns, synthetic-image boundaries and a real reused KV
prefix followed by one fresh token. Fixed membership produces exact retry/edit
outputs; canceled requests publish no prompt histogram or seed. For Qwen, the
one-fresh-token request physically executes 352 Expert entries but collects only
328 new-prompt entries, excluding the 24 entries from three old MTP repair rows.
Clover collects the expected 320 entries. Frozen profiles and installed
configuration hashes remain unchanged. Synthetic image embeddings check state
boundaries; this does not establish actual image-understanding quality.

An optional v4 experiment avoided upweighting prompts shorter than 64 tokens
and adjusted prior retention per newly read token. It preserved ordinary
admission aging to isolate the seed change. At the same 3 GiB FP8 profile+topic
setting, pooled decode changes **76.67 → 74.86 tok/s (-2.36%) for Qwen** and
**35.27 → 36.13 tok/s (+2.43%) for Clover**; all outputs differ. The result is
mixed, so the helper, extra tests and tuning knob were removed from product
sources. Its source/test hashes and results remain in the private experiment.

Profile persistence and topic adaptation are retained as functionality with
observed workload-dependent benefits. Their native OFF controls remain available;
the evidence does not justify advertising a universal tok/s increase.

## Installed application check

The final native executable and managed API defaults were deployed locally.
Clover correctly identified the canonical image as a red square A, blue circle B
and green triangle C. A text-only follow-up correctly answered B and reused
1,082 prompt tokens. After unloading and reloading, the automatically saved
ranking loaded and applied successfully, and the image answer remained correct.
The application was left with Clover loaded and image input enabled. Saved model
configurations and the Gemma/LFM/vision executables were unchanged; the temporary
CMake output override was removed. This is an application smoke check, not a
broad visual-quality benchmark. Private evidence: `installed-qa.json` and
`deployment.json`.
