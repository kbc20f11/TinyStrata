# Upstream refresh and validation — 2026-11-06

TinyStrata baseline:
[`0585f998039c493a4ca3f4532dfc038821f6f39c`](https://github.com/kbc20f11/TinyStrata/commit/0585f998039c493a4ca3f4532dfc038821f6f39c).
This review updates the four reference checkouts and selects changes for the
five existing supported models. It does not merge another engine wholesale.

## Reference revisions

| Project | Updated revision | Relevant result |
|---|---|---|
| [Strata](https://github.com/Niko1221/Strata) | [`9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e`](https://github.com/Niko1221/Strata/commit/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e) | Q5_0 dot-product and streamed tool-argument corrections; bounded experiment of RDNA4 staging barriers |
| [PolyStrata](https://github.com/VecSzn/PolyStrata) | [`a82999fd60c2b6b3dbfb3924df60f8ff756c4371`](https://github.com/VecSzn/PolyStrata/commit/a82999fd60c2b6b3dbfb3924df60f8ff756c4371) | Existing-model/server changes reviewed; legacy batch-protocol ownership and wider engine changes deferred |
| [MoMoE](https://github.com/aflin/momoe) | [`c8f7c3c8c3128b6c5f6ffc620db8ce0bcece5d14`](https://github.com/aflin/momoe/commit/c8f7c3c8c3128b6c5f6ffc620db8ce0bcece5d14) | Explicit Qwen2 pre-tokenizer handling; most new engine work concerns larger model architectures |
| [Basalt](https://github.com/jesdga95/basalt) | [`ca7d07418cdf6ee08462c560102a7798c94886cc`](https://github.com/jesdga95/basalt/commit/ca7d07418cdf6ee08462c560102a7798c94886cc) | Tolerant native stdout decoding; new model/router/peer-tier architecture not imported |

The reference checkouts preserve their local state; clean branches were
fast-forwarded, and the Strata reference remains a detached checkout at its
updated revision. No model downloads were needed for this review.

## Correctness and reliability changes

### Q5_0 offset uses the quantized activation sum

Adapted from Strata
[`aa678f10d0e51ac17d301c75cbb6b80783437e8b`](https://github.com/Niko1221/Strata/commit/aa678f10d0e51ac17d301c75cbb6b80783437e8b).
The shared Expert/`iq_mmvq` Q5_0 product subtracts the constant weight offset
using the sum of the stored quantized activations. Previously that term used the Q8_1 block's
original-input sum, mixing two activation-rounding conventions.

The HIP regression test passes **97 `iq_mmvq` matrix/grouped Expert cases**
against an independent scalar reference of the stored Q5_0/Q8_1 values. The old formula
fails 48,957 checked output comparisons; the selected formula's largest
absolute reference error is 0.0009765625, within the test's sum-magnitude-scaled
bound. Zero-weight rows remain exactly zero, and guards and inputs remain
unchanged. This is a format-specific numerical fix, not a tok/s claim or an
assertion of broader answer quality.

The separate dense `native_q5_0_mmvq` implementation in `native_mmvq.cu` uses
its own arithmetic convention and is outside this patch and these 97 cases.
This update does not claim to have changed every Q5_0 matrix-vector path.

### Native diagnostic bytes no longer stop the stdout reader

Following Basalt
[`09f785308d88d1c8cdcb3711e8ca68bff8468d0f`](https://github.com/jesdga95/basalt/commit/09f785308d88d1c8cdcb3711e8ca68bff8468d0f),
language-engine and image-encoder pipes explicitly replace malformed UTF-8
diagnostic bytes. Valid Unicode and native protocol delivery are preserved.

Four subprocess regression cases cover bad bytes before READY and between
tokens, EOF notification instead of a hanging request, slot-queue EOF delivery,
and a real synthetic vision-vector reply. This does not make corrupt protocol
numbers valid; it prevents a diagnostic decoding error from killing the reader.

### The declared pre-tokenizer is selected explicitly

Following MoMoE
[`7304c605a65d30484294946c079b733dbe18179d`](https://github.com/aflin/momoe/commit/7304c605a65d30484294946c079b733dbe18179d),
`qwen2` uses its own combining-mark split rules. Unknown or non-string `pre`
values raise a clear error instead of silently inheriting Qwen35's tokenizer.
Missing legacy metadata still defaults to Qwen35. Extraction validates the
tokenizer before creating or replacing cache files.

All **12 focused CPU tests** pass, including an actual BPE token-ID difference
that a round-trip check alone would miss. A separate read-only audit of the five
installed model caches passes **400 old/new token-array comparisons and 200
Unicode round trips**, with every configuration/cache hash unchanged. Existing
Qwen35, LFM and Gemma pattern strings are unchanged. No existing-model speed
improvement is claimed for this dispatch fix.

### Literal closing tags inside streamed tool arguments

This port includes Strata's prerequisite
[`49b880c01f55a248d0485b6cd244ae053191d928`](https://github.com/Niko1221/Strata/commit/49b880c01f55a248d0485b6cd244ae053191d928)
and streamed-argument correction
[`3bbf0f399e2a1b3d38642eca4342721fc7994201`](https://github.com/Niko1221/Strata/commit/3bbf0f399e2a1b3d38642eca4342721fc7994201).

A tool such as `write_file` can legitimately receive documentation containing
`</parameter>`, `</function>` and `</tool_call>` as file content. The old parser
could interpret these literal strings as the end of the tool argument when
their closing sequence arrived in separate streaming pieces. The reference
comparison reproduced 1,044 failures in 1,662 corpus/chunking cases, while
latest Strata passed all 1,662.

The complete prerequisite keeps ambiguous closes pending until the turn ends,
another call establishes a boundary, or the bounded look-ahead decides. The
follow-up fix also treats an already delimited body as complete when scanning
its streamed arguments. Both pieces are required and included in this port.

All **21 tool-parser tests** pass, including the **1,662 corpus/chunking
cases** with complete arguments preserved. An independent audit passes
**410 additional probes**, covering bare and truncated calls, declared value
types, Unicode, multiple arguments, and the Gemma/LFM parsing paths. These are
parser correctness checks; they do not claim better model tool selection.

### Final native and CPU checks

After removing the LDS experiment, all three HIP language-engine binaries
build successfully. The final Q5_0 native regression still passes all **97
cases** with the error and guard results above. The touched native source
also compiles for CUDA SM120; there was no physical NVIDIA test in this run.
The focused CPU suite passes **37 tests**, and the separate tool-parser suite
passes the **21 tests** described above.

Final HIP artifact SHA256 values:

| Artifact | SHA256 |
|---|---|
| `tinystrata.exe` | `aa8dfec0563988dc2bedb1938ea4e615cf005a3f67f1f7869c31d19c191cf9a2` |
| `tinystrata-lfm2.exe` | `e703c610e7ad44c3c02c2e20582b1400b4c818e78d48f257eb07d941a11045fe` |
| `tinystrata-gemma4.exe` | `6885c0e15fd03f9c6d847d6e72badb067f2f26381db8a583697f1da3dcf1cd22` |

The installed-binary smoke check returns `42` for the arithmetic request on
all five models. Clover correctly identifies a red square A, blue circle B,
and green triangle C from the image; its text-only follow-up answers `B` and
reuses **1,082 cached tokens**. Existing model configurations and the image
encoder binary remain unchanged. These short checks establish basic
deployment behavior, not benchmark throughput or general answer quality.

## RDNA4 LDS barrier experiment

The source pattern is Strata's
[`1f8ca8de1e870f285892f6d7f9a67f917923df77`](https://github.com/Niko1221/Strata/commit/1f8ca8de1e870f285892f6d7f9a67f917923df77),
crediting bsorensen110's PRs #3259/#3264. TinyStrata tests it only at the three
mode-8 fused Expert gate/up staging barriers. It does not replace arbitrary
global-memory synchronization. The experimental specialized instructions compile
only for gfx1200/gfx1201; other targets use full barriers.

HIP correctness passes **270 cases**, 1,620 eager/capture launches and three
paired Graph replays per case: gate/up intermediate data and final outputs are
bit-identical, guards hold, and inputs remain immutable. The touched native
source also compiles for CUDA SM120. CUDA compilation is not a physical NVIDIA
measurement; the GPU tests here ran on Radeon.

The experiment was **not retained**. Its source changes, environment control
and product test target were removed after the comparisons below failed to
establish a consistent benefit. Sources, hashes and results remain in the
private experiment evidence. No LDS barrier change is enabled in the product.

### Measurement conditions

- Windows x64; Radeon RX 9060 XT 16 GB, HIP ordinal 1; Ryzen 9 7945HX3D,
  64 GB system RAM. Tests are after the current boot.
- FP8-MMA KV, 32,768-token context capacity, automatic chunk resolved to 2,048,
  15 CPU workers plus the participating host; no resident image encoder.
- Qwen/Tiel use MTP3; Clover uses causal D0. Expert profile persistence is off,
  topic adaptation is on, and KV reuse is zero.
- One OFF/ON pair per cell, using the same executable and changing only
  `TINYSTRATA_HIP_LDS_BARRIERS=0/1`. A short interval-merging request emits 256
  tokens; a fresh exact 4K input emits 32. Thinking is included in token counts.
- The 3/6 GiB settings are fixed Expert budgets on the physical 16 GB card,
  **not physical or simulated 8/12 GB GPU benchmarks**. One pair gives a point
  estimate, not statistical significance or a universal speed ranking.

Timed experimental executable SHA256:
`ca38760671614b5b12bc86f36a7d382885f7f693e8ae14cb5c987b2196165dd8`.
This identifies the experimental timing binary; it is not the final executable
after the LDS experiment is removed.

### Matched comparison results

| Model / Expert budget | GPU Expert slots | Short decode OFF → ON, tok/s | Decode after 4K OFF → ON, tok/s | Pooled decode change | Whole-request wall change |
|---|---:|---:|---:|---:|---:|
| Qwen / 3 GiB | 2,665 | 71.85 → 69.88 (-2.74%) | 85.22 → 87.00 (+2.09%) | -2.30% | +1.33% |
| Qwen / 6 GiB | 5,494 | 86.77 → 85.37 (-1.61%) | 92.33 → 93.16 (+0.90%) | -1.35% | +1.13% |
| Clover / 3 GiB | 2,440 | 33.85 → 33.57 (-0.81%) | 30.94 → 31.81 (+2.81%) | -0.39% | +1.11% |
| Tiel / 3 GiB | 2,829 | 57.90 → 58.85 (+1.63%) | 56.44 → 54.15 (-4.04%) | +0.96% | -0.08% |

Pooled decode divides 288 emitted tokens per arm by summed native decode time.
Whole-request wall includes the two requests' prefill and protocol work, but
not engine startup. Actual cache byte counts, identical within each pair, are
3,107,409,760; 6,406,044,736; 3,156,384,000; and 3,139,668,912 respectively.

All **eight paired output token arrays** are identical, as are MTP accepted/
offered counts, prompt/decode CPU/GPU work, swaps, slot fingerprints and
geometry. Raw native `T`/`DONE` lines were reconciled independently with the
saved results. The actual LDS activation message appears only in each ON
process's stderr. Thus the controls exercised the intended path; the rejection
is based on mixed measured outcomes, not an inactive flag or changed workload.

The 4K prefill times also do not establish a gain: OFF → ON is 11,456.1 →
11,568.0 ms (Qwen 3 GiB), 8,023.6 → 8,102.6 ms (Qwen 6 GiB), 13,014.4 →
13,219.4 ms (Clover), and 10,828.5 → 10,863.0 ms (Tiel). These are single-pair
observations; no generic decode, prefill or end-to-end speedup is claimed.

## Changes deferred

- Strata's reply-only penalty scope,
  [`31c3a16f4fd61a1db36f8027a96c69cbdd10fe37`](https://github.com/Niko1221/Strata/commit/31c3a16f4fd61a1db36f8027a96c69cbdd10fe37),
  is coupled to continuation bookkeeping;
  changing a penalty slice alone would alter generation semantics. It needs a
  complete continuation implementation and explicit quality checks.
- PolyStrata's `BDONE` ownership change concerns a legacy batched native
  protocol that the current TinyStrata engines do not use. It is not imported
  into the existing serial generation path without a matching batch protocol.
- MoMoE's new MiMo/MiniMax/GLM paths, their multi-card layer split and model pool
  rules are not drop-in improvements for the existing five TinyStrata models.
  TinyStrata already accounts for each current layer's allocation/VMM rounding
  and searches tensors across GGUF shards.
- Basalt's new model, serving-router, peer-tier and broader memory policies need
  their own architectural validation. A measured gain on another model/device
  is not treated as a default for AMD or smaller GPUs.
- Expert omission or residency-biased routing is not introduced. Every Expert
  selected by the model is still computed.

Private evidence is under `TinyStrata-work/upstream-sync-20261106/`:
`native_q5_offset_test.json`, `native_lds_barriers_test.json`, per-cell native
logs/results and `independent-audit.json`. Tokenizer cache parity is recorded
under `upstream-refresh-20261106/tokenizer-pre-installed-parity.json`.

## Final application verification

The final HIP builds of the Qwen/Clover/Tiel, Gemma and LFM executables were
deployed together with the Python fixes. All five models returned the expected
answer and terminated normally in controlled arithmetic requests. Clover then
read the canonical red-square/blue-circle/green-triangle image correctly, and
its follow-up reused 1,082 tokens. All five saved model configurations retained
their original SHA256; the vision executable was unchanged.

After the final parser deployment and API restart, a real streaming Clover
request called `store_text` with a string containing `</parameter>`,
`</function>` and `</tool_call>` on separate lines. Reconstructed SSE argument
JSON matched the complete requested string exactly. Clover remains loaded with
images enabled and the UI healthy. HIP and CUDA build-output overrides were
removed. Final Python checks pass 37 general regression tests and 21 tool-parser
tests. Evidence: `installed-qa.json`, `live-tool-qa.json`,
`python-regression.json`, `deployment.json` and `parser-deployment.json` in the
private validation directory.
