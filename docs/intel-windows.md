# Experimental Intel UMA backend on native Windows

The installer now supports the validated **Arc B390 / Core Ultra X7 358H / 32 GB RAM** configuration through a prebuilt, hash-pinned Intel runtime. It supports experimental **Qwen3.6 and Tiel Coder text**, with model-specific profiles and separate settings. Images, Gemma, LFM and Clover remain unsupported on Intel; the broader [Issue #11](https://github.com/kbc20f11/TinyStrata/issues/11) remains open.

On 2026-11-14, the actual shipping installer prepared a fresh Python environment, imported 65 runtime/license files, created the Intel desktop shortcut and started the real API/UI on the B390-only PC. It used the local source checkout and existing verified GGUF rather than downloading a second model copy. Its normal Qwen request used 84 fresh input tokens and returned 1,268 tokens with thinking enabled: **107.2 tok/s prefill, 34.4 tok/s decode**, terminal stop and the correct six-field answer. No request output cap or sampler override was supplied. This earlier chunk512 checkpoint is retained separately. Real Qwen/Tiel switching, native unload/reload and live Monitor/About checks subsequently passed, preserving settings and keys with no compiler-SDK DLLs loaded.

For ordinary installation, use the UI installer linked in the README. A local checkout accepts `install-ui.ps1 -Backend sycl`; a verified offline stage may instead be supplied with `-IntelRuntimeDirectory`. NVIDIA/AMD selection retains priority on mixed systems. Only the validated B390 device/target is accepted, with no generic fallback to untested Intel hardware. Runtime files and Microsoft VC++ release DLLs are app-local; the installer does not install an Intel SDK, replace a driver or edit global PATH.

Qwen uses `b390-qwen36` / `config-sycl.json`; Tiel uses `b390-tiel-coder` / `config-tiel-coder-sycl.json`. Both resolve saved context `auto` to 8,192, KV `auto` to actual INT8, and chunk `auto` to 1,024. Explicit chunk settings, including 512, are retained. The native diagnostic CLI's default chunk128 is separate. The Intel shortcut normally uses UI 3005/API 18085. The `--intel-text` [benchmark variant](../benchmark/README.md) explicitly uses a private 32,768-cell context for its 16K input and labels its differences from fixed v1.

Tiel's native medium-thinking tests returned the correct reservation calculation at **135.60 / 30.95 tok/s** (prefill/decode), and generated binary-search code at **130.57 / 29.21 tok/s**, passing 5,100 independent cases. Its no-think arithmetic error was reproduced by the independent llama.cpp SYCL reference; the owner explicitly excludes that known behavior from the adoption gate. The error remains recorded in the [measurement artifact](../benchmark/results/intel-uma-windows-20261114.json).

The implementation began with a Qwen3.6-only causal-text proof of concept and a separate diagnostic API. The following engineering milestones retain those early measurements; the normal installer and model-specific profiles described above supersede that initial entry-point limitation.

The native diagnostic CLI defaults MTP to zero; the normal B390 profiles select experimental MTP3. `--capabilities` retains an empty formal `model_families` list while separately declaring the two experimental text models, profile contracts and compiled quantization paths. Images, Clover, Gemma and LFM are not enabled on Intel. The wider gates in the issue remain open.

Owner clarification on 2026-11-14: a working prototype is **not sufficient for adoption**. Upstream Coder already exceeds 20 tok/s with a model larger than 20 GiB. Prefill must reach at least 100 tok/s, with a dedicated-engine generation aim of 30–40 tok/s and the issue's separate 42.5 tok/s MTP reference retained. Keep model and speculation differences visible. Practical throughput and answer quality must both pass.

## Test machine and dependencies

Recorded on 2026-11-14:

- MouseComputer ISeNEi-16FHX31; Core Ultra X7 358H, 16 cores/16 threads.
- Arc B390, PCI `8086:b082`; the SYCL runtime identifies `intel_gpu_ptl_h` (Panther Lake/Xe3). No B-series marketing-name heuristic is used.
- Installed RAM: 32 GiB LPDDR5X, configured 7467 MT/s. Windows reports 33,873,752,064 physical bytes after hardware reservations.
- Windows 11 Home, build 26200; graphics driver 32.0.101.9175. Level Zero device driver string: `1.17.39894+6`.
- Intel DPC++/C++ 2026.2.0; SYCL runtime 2026.2.0.0; installed UR loader/adapters 0.12.0. Visual Studio 2022 C++ Build Tools 17.14.45, Windows SDK 10.0.26100.0.
- Level Zero Windows SDK 1.35.0; the system loader reports 1.33.1. The probe uses the existing installed graphics driver.
- CMake 4.4.4, Ninja 1.13.2, Python 3.12.10. Optional prefill experiments link oneMKL 2026.2; the optional NPU experiment uses OpenVINO 2026.4.1 from a private environment.

The model is TinyStrata's unchanged `Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf`, revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d`, 14,069,266,720 bytes, SHA-256 `36f9ec0e4c775f6efd3a61c1ea76f0875128469c3d012e3d9e2495a90e7a7150`. Its filename does not mean every tensor is IQ3_XXS: this file has F32, Q8_0, Q2_K, Q3_K, Q6_K, IQ2_S, IQ3_S, IQ4_XS and BF16 tensors.

## What the implementation does

`TINYSTRATA_BACKEND=SYCL` builds `tinystrata-intel`, `tinystrata-intel-probe` and `tinystrata-intel-weights`. It requires 64-bit native Windows and oneAPI 2026.2 or later. CUDA/HIP remain separate build branches.

The device probe selects an Intel Level Zero GPU, obtains the integrated flag from Level Zero, and queries the runtime architecture and subgroup capabilities. It reads available physical RAM, remaining commit and the selected adapter's WDDM process budget. Identical PCI IDs on multiple adapters are treated as an ambiguous WDDM match and refused for allocation probes/loading; another adapter's budget is not borrowed.

The weight loader validates the existing GGUF geometry/layout. It closes all GGUF memory mappings before allocating one persistent compressed arena. Host/shared USM is filled directly from the file. Device USM uses an 8 MiB staging buffer which is released before returning. All three modes retain one compressed weight store, without a complete CPU replica and duplicate expert-cache arena. The current file occupies 14,058,283,008 arena bytes including alignment. Draft tensors are resident in the same arena and execute only when MTP is explicitly selected.

Physical RAM and Windows shared GPU memory are not added together. The loader leaves 4 GiB of physical/commit headroom, respects the WDDM budget and maximum allocation size, and rechecks memory pressure during reading. A sampled byte check on the GPU covers each tensor at 64 KiB intervals plus its last byte. This is a visibility/transfer check, not a second cryptographic file hash or proof against every possible corruption.

The engine uses unchanged compressed weights and FP32 KV/recurrent state. It implements the complete 40-layer causal Qwen path: embedding, normalization, projections, four-tap convolution, gated delta recurrence, rotary attention, routing, all eight selected experts, the shared expert and output projection. No experts are omitted. Prompt processing uses configurable causal batches and stable expert grouping. Decode uses subgroup matrix products and a captured graph whose attention extent grows with the live context. This is TinyStrata's model path, not a llama.cpp subprocess.

Two faster paths remain explicit candidates. `--prefill-mkl` dequantizes one bounded matrix at a time to FP16 scratch and uses FP16 multiplicands with FP32 GEMM accumulation; its attention QK/PV products use FP32 multiplicands. It does not create a persistent FP16 model replica. `--decode-dp4a` temporarily quantizes projection inputs to blockwise Q8 and uses ESIMD integer dot products with the original compressed weights. Other activations and recurrent/KV state remain FP32. These changed arithmetic paths have separate numerical and real-answer checks; passing those limited checks does not establish full CUDA/HIP quality equivalence.

The stdin protocol supports text `GEN`, streaming `T`, progress `PP`, `DONE`, `STOP` and `QUIT`. An exact live prefix is reused; an edit replays from a clean recurrent state. Each cancellable prefill batch checkpoints recurrent/convolution state and the last hidden vector; interruption restores the committed prefix, leaving future KV cells invisible. Invalid requests leave the resident prefix intact. Sampling follows the existing Qwen host policy. Disk snapshots, multiple concurrent native slots and dynamic KV growth are not implemented here.

## Build and run

Install Visual Studio C++ Build Tools, Intel oneAPI DPC++/C++ 2026.2, CMake, Ninja and the [Level Zero Windows SDK](https://github.com/oneapi-src/level-zero/releases). The SDK directory supplied below contains `include/level_zero/ze_api.h` and `lib/ze_loader.lib`.

From the repository:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\cmake\build-intel.ps1 `
  -BuildDirectory ..\tinystrata-work\build-intel `
  -LevelZeroRoot C:\path\to\level-zero -Inspect
```

The script initializes both the MSVC and oneAPI environments in the build process. `setvars.bat` alone did not supply the MSVC include paths on this installation. It does not rewrite the user's global PATH. Add `-MklPrefill` to build the optional oneMKL path; select it at runtime with `--prefill-mkl`. A build that links oneMKL needs its runtime DLLs in the launching process's environment, including when launched from Python.

In an initialized oneAPI command prompt, with the project's Python requirements installed, start the normal HTTP handlers through the explicit Intel entry point:

```text
python -m serve.intel --model C:\path\to\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --exe ..\tinystrata-work\build-intel\tinystrata-intel.exe --context 4096 --port 8095
```

The API binds to `127.0.0.1`. It uses the GGUF's tokenizer/template and the project's official Qwen sampling defaults. `--work-dir` chooses its private tokenizer/log directory. Existing settings, model profiles, keys and chat history are not migrated. Model management and the ordinary UI installer are not enabled by this entry point. Images are rejected explicitly. Ctrl+C stops the API and releases the native model.

`--max-tokens` sets the API's default output budget and defaults to the normal TinyStrata value of 32,768. When a client omits its output limit, the shared API logic fits that default to the actual remaining context. An explicitly oversized client limit is still refused. Earlier experimental builds imposed a separate 1,024-token default even with thinking enabled; that restriction has been removed without changing thinking or sampling settings.

For token-level diagnostics:

```text
tinystrata-intel --model MODEL.gguf --tokens-file prompt-ids.txt --n 128 --max-context 4096 --kv-type fp32 --weight-memory host
tinystrata-intel-probe --inspect
tinystrata-intel-probe --kind shared --large --rounds 6
tinystrata-intel-weights MODEL.gguf host
```

`--weight-memory` accepts `host`, `shared` or `device`. Host USM remains the initial default; a single allocation-mode comparison is not a universal tuning result. The native context is explicit and currently limited to 32,768 cells; the requested prompt plus output must fit. `--chunk` accepts 1–1024, with default 128. Scratch allocation and memory checks account for the requested chunk/context and optional MTP KV/state buffers. `--kv-type fp32|int8` is explicit and defaults to `fp32`; `auto`, FP8 and image requests are rejected rather than silently changed. Experimental `--mtp 1..7` requires a draft-equipped GGUF and chunk >= max(4, depth+1); it is available through both native and explicit Intel HTTP entry points.

INT8 KV stores one FP16 scale per 32 signed bytes and accumulates attention in FP32. Small attention batches read it directly; larger prefill batches expand the visible prefix into one shared FP32 scratch pair reused across layers. FP32 KV is not kept alongside every INT8 cache. Nonfinite inputs and unrepresentable scales set a checked error flag. Independent tests cover partial writes, quantization bounds and FP64 attention. Both FP32 causal and INT8 MTP3 passed the ten native/HTTP integration tests.

Build option `-Int8Xmx` enables the experimental INT8 DPAS kernels. Runtime `--verify-xmx --decode-dp4a` selects them for grouped expert verification; original-layout dense projections retain DP4A. The separate `--prefill-int8-xmx --decode-dp4a` candidate extends grouped expert processing to 9–128 input tokens and remains under evaluation. Extending selection to 512 did not beat the oneMKL control in the focused test, so that selection was withdrawn. The underlying grouped kernel still has independent 129-token checks. Activation scratch is bounded to 4 MiB of signed bytes plus 512 KiB of FP32 scales; oversized shapes are refused before dispatch. Both preserve the original compressed GGUF weights, use explicitly quantized activations and accumulate in FP32. Neither is enabled by default.

`--vocab-xmx --decode-dp4a` is a separate optional vocabulary projection candidate, requiring `-Int8Xmx`. It interleaves Q6_K words across 16 rows without changing any quantization byte. Loading converts bounded batches inside an 8 MiB staging buffer and retains only the converted head in the persistent arena. Padding increases this model's arena by 3,973,120 bytes, to 14,062,256,128. XMX selects M=1/2/4/8 for the actual input count. Independent NumPy/FP64 fixtures, lossless-byte checks, native/HTTP integration and the three additional Python code tasks passed. The standalone prototype's temporary duplicate head was used only for a kernel comparison and is not part of the engine's memory policy.

`--expert-xmx-layout --decode-dp4a` extends lossless interleaving to eligible Q6_K/IQ2_S/IQ3_S expert trios. In this GGUF, 111 expert tensors are converted and 12 retain their original layout, including the three main layers with IQ4_XS down projections and the Q2_K/Q3_K MTP trio. Grouped expert input may contain up to 1,024 tokens, with bounded scratch and no CPU expert fallback. Together with the vocabulary layout, the arena occupies 14,295,040,000 bytes. Q2_K/Q3_K DP4A decoding is implemented separately for the original-layout draft tensors. Independent real-tensor references cover these paths.

The later `build-packed-iq` follow-up gathers aligned DWORDs across 16 rows for interleaved IQ2_S/IQ3_S with 2–8 input tokens, recovering identical code/sign bytes in registers. Quantization, selected experts, DPAS and FP32 accumulation order are unchanged. Single-token, larger-prefill, original-layout and Q6 paths retain their earlier loads. Its 54 standalone comparisons preserved output bits and tail canaries; adverse single-token wall timings were retained, rather than selecting that path. The fresh build also passed CPU3, the full 27-reference/capacity suite, native/API11 and the three closed-thinking code tasks.

`--dense-xmx-layout` additionally requires `--prefill-mkl`. It converts selected two-dimensional Q6_K projections, preserving the embedding's original layout. Selection follows measured matrix costs: width <= 2,048 and at least 2,048 output rows; small gate/K/V and width-4,096 output projections retain their original paths. Short batches use integer XMX. Larger prefill batches decode into bounded, transposed FP16 scratch for oneMKL with FP32 accumulation. The original all-dense candidate slowed short prefill to 58–60 tok/s and is not the selected policy. This flag remains experimental.

The separate opt-in `--dense-split-k --dense-xmx-layout` candidate also packs Q6 projections with width 4,096 and 2,048 output rows. It divides each dot product's input range into four GPU jobs and adds their FP32 partial sums. All weights and selected experts remain in use. The additional partial scratch is bounded to 1 MiB and included in the memory budget. Other matrix shapes keep their previous path; splitting them was slower in the standalone tests. Independent actual-tensor/FP64 checks cover 1/2/3/4/5/7/8 rows, and a separate exact-coefficient fixture checks the full scratch capacity, output tail and oversized-shape refusal.

After reboot, one same-binary normal-settings comparison measured 26.54 and 27.14 accepted decode tok/s in the controls, versus 27.66 with split-K. All three final reservation JSON answers were correct, but output lengths were 1,373 / 1,658 / 1,373; this does not establish a precise speedup percentage. Native/API 11 tests and the three fixed Python coding tasks passed. Cold first prefill was 35.39 tok/s in the first control, versus 96.33 and 101.68 in later runs; preserve the cold/warm distinction. Evidence: `dense-splitk-normal-paired`, `dense-splitk-capacity-quant.log`, `dense-splitk-integration.log`, `code-quality-dense-splitk`.

`--device-state` puts GPU-only state/scratch in device USM while leaving CPU-visible inputs, logits, counts and errors shared. It does not duplicate the weight arena. `--verify-graph` separately records MTP verification windows, requires INT8 KV, and cannot be combined with grouped oneMKL verification attention. Graph caches are bounded to 16 entries, with 128 MiB budget headroom. Native/API tests include interruption during decode and clean replay. Verification graphs did not outperform the eager control in the measured short-code comparison, so they remain opt-in.

INT8 decode attention processes up to eight independent causal query rows in four launches for the whole window, rather than four launches per row. Scores and split-value scratch are bounded by min(chunk, 8), and included in the allocation budget. Independent FP64 fixtures cover 1/3/4/7/8 rows, different positions, the 255/256 and 511/512 boundaries, and a final partial split at context 1,031. The engine retains the FP32 attention path separately. Large packed-expert prefill uses oneMKL at 512 or more tokens: the existing output scratch temporarily restores one compressed expert before FP16 dequantization, then becomes the GEMM output again. It never retains a second complete expert/model store.

`--verify-mkl-attention --prefill-mkl` is an experimental FP32 grouped-query verification-attention candidate, requiring chunk >= 64. It combines heads sharing each KV head into GEMMs and reuses bounded prefill scratch. INT8 KV is expanded into the existing shared scratch pair. Matrix extents round up to 256 cells; the engine zeroes the future KV tail and softmax masks it before the value product. Single-token decode retains its existing path. Independent FP64 attention fixtures cover causal positions and 1/4/7-row groups. Performance evaluation remains separate from correctness checks.

The MTP candidate warms the draft attention cache with the preceding main-model normalized hidden states. It checks each proposal against a causal main-model verification window. Sampled requests draw only from valid target rows, in order; a mismatch or terminal token ends the window without sampling discarded rows. A partial acceptance restores the starting recurrent/convolution state and replays the kept rows' saved inputs. Accepted draft KV is repaired with main-model seeds before later extensions. Prefill cancellation also restores the main hidden-state seed. No approximate acceptance is implemented.

The separate `--mtp-stochastic` candidate samples draft proposals and applies exact speculative rejection sampling: accept with min(1, p(x)/q(x)), and draw a rejection replacement from normalized positive p-q. Both distributions include the configured sampling filters and penalties. Draft history and RNG are separate from committed target history; this preserves the target distribution rather than the same seed-to-text mapping across modes. Six CPU cases covering 2.4 million draws, native/API seeded replay and cancellation checks passed. A normal-settings 1,024-token trial measured 17.11 tok/s versus 16.82 for target-only verification; both exhausted their output budget during thinking. This does not establish practical completion or the generation target.

Initial MTP observations: a 51-token prime-function prompt produced the same 89 greedy output IDs as the causal FP32 reference, with all 66 proposals accepted. The FP32-input verification candidate measured 15.08 accepted decode tok/s; the Q8-input four-row candidate measured 22.84. Both MTP candidates passed the native/HTTP ten-test set, including cancellation/recovery and tool calls. These limited examples do not establish full answer-quality equivalence or the 30–40 tok/s target. Private evidence: `mtp-code-fp32`, `mtp-code-dp4a-batch`, `mtp-integration.log`, `mtp-batch-integration.log` and `mtp-batch-quant.log`.

### Selected draft-vocabulary candidate

`--draft-vocab N` defaults to 0 (off); nonzero values 64–4096 require `--serve`, `--mtp-stochastic`, a positive `--mtp` depth and `--vocab-xmx`. It evaluates only selected real vocabulary IDs for proposal distribution q, chosen from the current target logits and recent history. Target p still evaluates the full vocabulary with unchanged sampling filters/penalties; sampled requests retain exact min(1, p/q) acceptance and positive-residual rejection. All eight selected experts still execute. This changes the proposal, not the target policy or GGUF quantization, and remains opt-in.

The fresh AOT build passed three CPU CTest cases, 27 actual-tensor numerical references, selected-head tails of 1/15/17 rows, capacity 4096 with tail/refusal checks, and the split-K 1 MiB boundary fixture. Native/API 11 tests and all three independently checked Python coding tasks passed. The measured configuration retains FP32 recurrent state, explicit INT8 KV and one 14,305,198,080-byte compressed/interleaved weight arena; 2,048 selected IDs require another 8 KiB. `gpu_upload_bytes=0` counts only loader-explicit copies, not all driver/page traffic. Conditions, results and private log references are recorded in the [Intel measurement artifact](../benchmark/results/intel-uma-windows-20261114.json).

These earlier kernel milestones describe native diagnostic defaults. The normal B390 profiles now derive the measured optimization flags, MTP3, stochastic verification and draft vocabulary2048 for their child engines. A saved explicit MTP0/draft0 remains a user choice.

### Runtime staging and UMA telemetry

The initial app-local staging trial completed an actual normal-settings API request: 84 prompt tokens, 1,658 output tokens, 95.8 prefill / 26.5 decode tok/s, with the six final JSON fields correct. However, the native process loaded the compiler installation's `intelocl64.dll`; the isolated-runtime check therefore **failed**. Preserve `staged-api-validation-initial/result.json` and its logs as a failed isolation trial, rather than a distribution success. The normal UI location pointer was unchanged.

The Intel entry point now defaults its child environment to `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` and `UR_LOADER_PRELOAD_FILTER=1`, preserving explicitly supplied values and leaving the parent/global environment unchanged. A small staged queue test first demonstrated the correction. A later fresh full-model API request completed with 1,268 output tokens and terminal stop, with no compiler-SDK DLL observed. However, its UI capture failed before Monitor rendered, so `staged-api-draft-vocab-ui-final/result.json` remains failed overall. That failure was retained for the later live retry below.

UMA telemetry uses actual Level Zero integrated/LUID/node identity and an exact Windows adapter match. Shared GPU usage overlaps system RAM; dedicated-VRAM totals stay null and unmeasured expert placement is hidden. Fourteen focused CPU tests plus six Intel option and seven staging tests initially passed, followed by the expanded parser/telemetry/staging CPU set of 35. Existing AMD/HIP/status/stream Python tests passed 74 cases, and six existing Node UI suites passed 396 checks; `app.js` syntax also passed. Evidence: `intel-uma-telemetry-cpu.log`, `draft-vocabulary-cpu/api-options.log`, `intel-uma-shared-regression-python.log`, `intel-uma-shared-regression-node.json`. The fresh API snapshot confirmed matching native/DXGI LUID, node 1, an available shared-memory counter and null legacy VRAM totals.

A model-free replay of that saved snapshot subsequently rendered the real Monitor and About views in private Edge, showing shared RAM and hiding unmeasured placement. CDP recorded zero runtime exceptions, console errors and network failures; browser/server cleanup and normal UI state checks passed. This is explicitly **recorded rendering, not live native/API validation**. The first replay's private helper waited for hidden tabs before navigating and timed out; that failed evidence is retained. No production startup exception was reproduced. Evidence: `ui-recorded-replay-20261114-172622-525569` (failed helper predicate), `ui-recorded-replay-20261114-172814-735064` (corrected replay).

The remaining recent-request VRAM-hit column was then hidden for UMA, including its header, row cells and empty-row span; CUDA/HIP retain the existing column. Fourteen focused telemetry/UI CPU tests and the six existing UI suites (396 checks) passed. The later live capture also confirmed eight visible headers and eight cells in its populated request row; the empty-row span was covered by the CPU fixture.

The final **live** run used `runtime-stage-packed-iq` with draft limit 2,048 and official sampling/thinking, without a client output-cap override: 84 prompt / 1,268 output tokens, 95.7 prefill / 33.0 generation tok/s, TTFT 0.922 seconds, thinking completed, terminal stop/SSE DONE and six correct JSON facts. Its short-prompt prefill remains below the 100 tok/s floor. It verified actual UMA identity/counters, no compiler-SDK DLL loading, real Monitor/About rendering, and zero CDP runtime exceptions, console errors or network failures. All owned API/UI/native/browser processes exited normally without force; ordinary history/settings/location were unchanged. Evidence: `staged-api-packed-iq-ui-20261114-173910/result.json`, `summary.json`, `ui-cdp.json`, `ui-monitor.png`, `ui-about.png`. This completes that explicit staged text API/UI check, not ordinary installer/backend selection, model switching, images or other-model validation; issue #11 remains open.

At the earlier staging checkpoint, draft-vocabulary actual-device validation was still pending. Later measurements superseded that pending status; it is now selected by the explicit B390 profiles. The broader issue #11 remains open.

## Normal explicit Intel launch

The normal launcher/server now also has an experimental Intel route:

```powershell
python -B run.py --backend sycl --intel-profile b390-qwen36 `
  --model C:\path\to\Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf `
  --exe C:\path\to\intel-runtime\tinystrata-intel.exe `
  --lib-dir C:\path\to\intel-runtime
```

This selects `config-sycl.json` and API port 18085. The Intel UI uses separate data/settings and a separate shortcut. Existing CUDA/HIP automatic selection and saved files are retained. The B390 profile resolves saved context `auto` to 8,192 cells and saved KV `auto` to actual INT8. Explicit MTP0 and user sampling/thinking/auth settings remain choices. Derived optimization flags are generated for the child, so they do not remain active after switching MTP off. Tiel uses its own model-specific profile; Gemma, LFM, Clover and images are refused before unloading the current model or starting a download job.

The normal `run.py` configuration → `serve.server` → actual Monitor/About path measured **96.1 prefill / 32.7 accepted decode tok/s**, with the same 1,268 correct normal-thinking output tokens. Installed-generation tests passed authentication rejection, eleven unsupported operations with the resident PID retained, and MTP3→0→3 reload/recovery. A private NoStart import exercised all 53 runtime files, licenses, immutable generation and config publication. Its Python-bootstrap boundary reused the existing private environment; fresh Python download and actual user Desktop installation were not exercised. Separate CPU installer/update/shortcut fixtures and exact evidence are recorded in the public results JSON.

The first actual import correctly refused an SDK symbolic-link size recorded as zero for `umf.dll`, although its copied payload was 209,720 bytes. Staging now records the opened payload's size. The old failed stage/import is preserved; a fresh v2 stage and the relocated runtime passed. Short-prompt prefill remains below 100 tok/s, and the broader issue, other model families and images remain open.

## Performance experiments

The `build-ptlh-aot` full-model executable completed one normal-settings, thinking-enabled reservation question with INT8 KV and MTP3: 84 prompt tokens, 1,373 output tokens, 91.75 prefill / 26.13 accepted decode tok/s, and all six final JSON fields correct. Native startup to READY took 16.891 seconds; model loading is excluded from the prefill/decode rates. This single AOT observation establishes neither a speedup nor the practical throughput targets. Its output length also differs from the staged API trial above. Evidence: `ptlh-aot-normal/results.json`, `ptlh-aot-normal.log`.

The fresh `build-draft-vocab` same-binary sequence used the official thinking-enabled sampler and 84-token reservation prompt, INT8 KV and MTP3. Generation rates count accepted output tokens minus the first output, divided by native decode time; rejected drafts and model loading are excluded.

| Proposal limit | Prefill tok/s | Generation tok/s | Output tokens |
| --- | ---: | ---: | ---: |
| 0, control before | 93.34 | 25.94 | 1,658 |
| 256 | 97.69 | 30.28 | 574 |
| 2,048 | 84.00 | 31.80 | 1,268 |
| 0, control after | 100.90 | 28.04 | 1,658 |

All four stopped with the correct six-field JSON answer. Different proposal distributions produced different thinking/output lengths, so these observations establish neither a precise percentage speedup nor universal generation above 30 tok/s. In the first sparse-draft build, a normal-thinking binary-search coding request with limit 2,048 completed 1,061 output tokens at 87.76 prefill / 29.16 generation tok/s and passed 5,100 independent cases. Three separate closed-thinking Python coding tasks also passed. Evidence: `draft-vocab-normal-paired/results.json`, `code-quality-draft-vocab-thinking/results.json`, `code-quality-draft-vocab/results.json`.

The packed-IQ follow-up then measured normal-thinking binary search at **29.09 / 32.80 / 30.77 generation tok/s** for old-before / new / old-after. All three produced identical 1,061 output IDs, terminal stop and 5,100 passing independent cases. Their prefill rates were 86.83 / 86.43 / 91.16, still below 100. A separate native R17 run measured 95.32 prefill / 33.055 generation tok/s with the same 1,268 output IDs as the earlier sparse-2,048 run. These are bounded same-output observations, not all-workload adoption results. See `packed_iq_load_candidate` in the [measurement artifact](../benchmark/results/intel-uma-windows-20261114.json).

Long source inputs used limit 2,048, chunk 1,024, temperature 0 and presence penalty 1.5, with the existing closed-thinking token files. Both returned the six correct facts, 59 output tokens and terminal stop:

| Fresh input / context cells | Prefill tok/s | Generation tok/s |
| --- | ---: | ---: |
| 4,096 / 8,192 | 340.60 | 33.70 |
| 16,384 / 32,768 | 306.00 | 25.98 |

These long-input sampler penalties differ from the older direct-CLI greedy trials, which used presence penalty 0; do not interpret the rows as a direct speedup comparison. Evidence: `draft-vocab-long-20261114-171305-379061/results.json` and the [recorded measurement conditions](../benchmark/results/intel-uma-windows-20261114.json).

The following historical measurements use the unchanged 14,069,266,720-byte IQ3_XXS GGUF above. They measure real prompts, with model loading excluded and prefill/decode timed separately. They are development observations, not a completed adoption benchmark. These cases use FP32 KV, all selected experts, and no MTP.

| Candidate and input | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: |
| oneMKL prefill, Q8-input DP4A decode, 512 input tokens, warm replay | 194.98 | 16.28 |
| oneMKL prefill, FP32-input packed decode, 4,096 input tokens, chunk 1,024 | 220.47 | 9.78 |
| Same path, 16,384 input tokens, chunk 1,024 | 112.57 | 8.32 |
| FP32 oneMKL attention + Q8-input DP4A decode, 4,096 input tokens, chunk 1,024 | 249.98 | 10.78 |
| Same path, 16,384 input tokens, chunk 1,024 | 199.04 | 7.74 |

The rows use different attention/decode paths; their numbers must not be combined into one measured configuration. The source-question cases returned the six expected JSON facts. Native/HTTP checks also cover Japanese arithmetic, streaming, tool calls, prefix edits and cancellation. Broader coding/reasoning/long-context quality comparisons remain open. Raw evidence: private `dp4a-512`, `attention-long`, `blas-attention-long`, `dp4a-integration.log`, `blas-attention-integration.log`.

Later MTP3 + expert INT8 XMX + tiled GDN observations reached 29.91 decode tok/s on the short prime-function case. INT8 KV measured 29.83 on that case, but a subsequent longer-prompt run measured 140.91 prefill / 15.24 decode tok/s at 4K and 115.29 / 8.89 at 16K, with all six requested facts correct. These do not establish sustained 30–40 tok/s generation. MTP7 was slower than MTP3 on the short case. Raw evidence: `gdn-tiled-code`, `kv-int8-code`, `kv-int8-long/results.json`.

The later vocabulary-interleave candidate with tiled GDN throughout prefill measured 31.82–34.01 accepted decode tok/s on the 51-token prime prompt in three runs. Its warm prefill was 104.02–106.20 tok/s; the first run was 93.06. The same binary's original-head control measured 30.01–30.25 decode and 98.48–98.99 warm prefill. All six generated prime functions passed 10,021 independent inputs. Longer source prompts measured 378.01 prefill / 25.59 decode at 4K and 221.04 / 11.56 at 16K, both with six correct facts. These observations do not establish 30–40 tok/s sustained coding or long-context generation. Raw evidence: `vocab-xmx-paired`, `vocab-xmx-integration.log`, `code-quality-vocab-xmx`, `vocab-xmx-long`.

The selective dense/expert layout plus device-state candidate measured 37.99–38.17 decode tok/s on the same short greedy prime case. Prefill was 85.63–99.47, below the control's 101.61–108.09; preserve this regression. All six prime functions passed 10,021 inputs, native/API 11 tests passed, and the three fixed Python tasks passed. With batched INT8 attention and the packed-expert oneMKL path, the later 4K source case measured 358.63 prefill / 34.10 decode, and 16K measured 201.96 / 12.60, both with six correct final facts. Earlier candidates without packed-expert oneMKL measured as low as 148.12 / 17.53 at 4K; a faster single run does not establish stability. Evidence: `dense-selective-code`, `dense-selective-integration.log`, `code-quality-dense-selective`, `q8-batch-long`, `packed-mkl-long`.

Earlier normal settings were materially slower. A separate 84-token reservation-arithmetic question, using the official thinking-enabled template and temperature 1 / top-p .95 / top-k 20 / presence penalty 1.5 / last-n 64 / seed 2026, completed with six correct JSON fields after 1,797 output tokens. MTP depth 1 with exact rejection sampling measured only 19.75 prefill / 14.70 decode tok/s and about 122 seconds of decode. This is neither a sustained-speed success nor equivalent to the closed-thinking greedy source cases. Evidence: `packed-mkl-normal/depth-1.json`.

Practical coding quality remains an adoption gate. A closed-thinking greedy C99 merge-sort request failed independent array tests in native FP32, Q8 causal and MTP variants. The same GGUF in llama.cpp SYCL passed once (425 output tokens, 20.53 tok/s) and failed a subsequent run with token/probability diagnostics enabled (390 output tokens, 22.81 tok/s). Preserve both; do not infer general model quality from one run or equate differing token sequences with faulty kernels. Three additional fixed Python coding tasks passed independent tests in all three paths: first-index binary search (5,100 cases), nonmutating merge sort (100 cases), and topological sorting (32 DAG cases and three cycles). The tiled-GDN native variants passed those tasks again. Evidence: `merge-sort-comparison`, `merge-sort-independent`, `merge-sort-independent-token-ids`, `merge-first-divergence-actual.json`, `code-quality`, `code-quality-gdn`.

The FP32 oneMKL attention candidate regressed at chunk 512 (137.94 tok/s on the 4,096-token input); selection is therefore restricted to batches of 1,024 tokens. Later repeated 512-token measurements varied substantially even with the unchanged control binary (approximately 7.7–17.0 tok/s decode). Level Zero Sysman subsequently reported thermal/current-limit flags and momentary 100 MHz GPU frequency during an active run. This is a performance investigation lead, not proof that every earlier slowdown had that cause; the power guard's heartbeat continued. No thermal/current safety limits were changed. Keep repeated-run and telemetry evidence alongside the faster individual observations.

The owner's [Windows SYCL reference](https://jishuzhan.net/article/2109547317837979650) reports approximately 97 tok/s prefill and 23–24 tok/s generation on 358H/B390, using a different UD-Q3_K_M GGUF and Q8 KV; prefill input length is unspecified. The 100 tok/s floor is reached in the measured longer-prompt candidates, but is not stable across short prompts. Some normal-thinking coding and long-context decode cases still fall below the dedicated-engine target.

The requested GPU/NPU split was tested on the actual Intel NPU. At 512 tokens, the older GPU path achieved 23.89 tok/s prefill, the NPU dense-projection candidate 27.69, and the oneMKL GPU candidate 164.80. Combining NPU with oneMKL reached 113.87, below GPU oneMKL alone. The NPU path is therefore optional, not the adopted default. Internal NPU MatMul output rounds to FP16. The handwritten XMX candidate also underperformed and has not been selected by default. Keep the slower cases in the evidence.

## Validation and current limits

- Host/shared/device USM each passed six 64 MiB and six 4 GiB + 64 KiB allocation/addressing rounds. Every 4 KiB page is tagged, and host/shared allocations are checked in both CPU/GPU directions. Individual dispatches are bounded to avoid a single long Windows job.
- The entire real GGUF loaded and unloaded in all three memory modes. GPU sampled visibility passed for all 753 tensors; host/shared reported zero explicit GPU weight-upload bytes, device mode uploaded one tensor payload. Process/RAM/WDDM readings are separate observations of overlapping memory, not capacities to sum.
- FP64 CPU references cover RMS normalization, Q8 matrix products, GDN outputs/state, split inputs and state restoration. Independently generated NumPy/gguf references cover 27 real tensor cases across all nine types in this GGUF. Their dequantized samples matched exactly; matrix products passed the recorded numerical tolerances.
- Native integration covers terminal tokens, continuation, edited prompts, invalid requests without state loss, cancellation/recovery and seeded sampling. HTTP integration uses the actual model and the existing tokenizer/template/handlers, including Japanese source arithmetic, SSE and a parsed `get_weather` call. Images fail with the expected unsupported-vision response.
- These observations do not prove answer-quality equivalence for every existing CUDA/HIP workload. A successful kernel check is not an answer-quality evaluation. Gemma, LFM, Clover and images remain unvalidated on Intel.

The private evidence is in `tinystrata-work/intel-20261114`: hardware/version records, full model checksum, raw probe and loader logs, numerical fixtures/results, native/HTTP transcripts and real generated answers. The initial installer collision, incorrect tokenizer-cache path and early baseline failures were investigated separately from successful inference.

### Power state and benchmark interpretation

During the first runs, Windows logged repeated Modern Standby entries from `Idle Timeout` and `Lid`. The actual active plan still had lid action `Sleep` on AC/DC, unattended sleep 120 seconds, display-off 300/180 seconds, and ordinary automatic sleep disabled. Those early timings are retained under `pre-power-fix`; they are not clean performance evidence.

The machine was subsequently set to the Windows **Best performance** AC overlay (`ded574b5-45a0-4f42-8737-46345c09c238`). Lid action and unattended/ordinary sleep timers were set to zero, while display-off remained 300/180 seconds. No S3 firmware support was inferred or forced. The native Intel driver now holds `SystemRequired` and `ExecutionRequired` requests during loading/inference and clears them afterwards, including cancellation. It never requests `DisplayRequired`.

Later owner instructions also disabled display-off and selected an Ultimate Performance base plan. The owner ran the administrator-only hibernation command and set `PlatformAoAcOverride=0`, then rebooted. On resumption, `powercfg /a` reported S0/S1/S2/S3 and hibernation unavailable; the existing firmware did not gain S3 support. A private, user-specific keep-awake helper holds display/system/execution requests after sign-in. These machine settings are separate from the native engine's product defaults. The closed-lid, suspended physical placement was retained as requested; it is not established as the cause of throttling.

A Modern Standby entry event can mark display-off/NoCS entry, rather than prove that inference stopped. After the change, real inference continued after the 01:20:09 display-idle entry without an intervening resume event. Microsoft describes the [display-off/NoCS phases](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/prepare-software-for-modern-standby) and [power-request behavior](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-powersetrequest), including the separate limitations on battery power and explicit user-initiated sleep.

Same-machine llama.cpp b12360 (`2d8a32502909ca065c41c4cf5f2fb1c427c047fc`) SYCL/Vulkan/CPU comparison binaries were obtained. The first SYCL default-load run was stopped when free physical RAM fell below 1 GiB; this is not a driver-crash diagnosis. Direct I/O allowed the diagnostic run to finish with ample headroom. llama-bench rejected FP32 KV, so those engineering references use F16 KV. They are random-token kernel measurements and are not directly comparable to this engine's real-prompt, FP32-KV timings or evidence of answer quality.

Before adoption, finish larger-context/pressure and sustained-quality checks, batching/fusion and attention/kernel tuning, and ordinary install/update/backend-selection integration. Explicit INT8 KV and the staged text API/UI have the checks above; broader lifecycle and workload coverage remain open. The experimental MTP route needs broader acceptance/performance and quality evidence; images and other model families need their own ports. Issue #11 stays open.


## Completed canonical Intel text measurements

| Model | Generation tok/s | 4K prefill tok/s | 16K prefill tok/s |
|---|---:|---:|---:|
| Qwen3.6 | 31.65 | 292.40 | 264.32 |
| Tiel Coder | 25.65 | 238.07 | 218.12 |

These are completed four-start, two-pass `tinystrata-bench-v1-intel-text` results, `standard: false`, with exact canonical text inputs, seed1234, INT8 KV/MTP3, private context32768 and chunk1024. Qwen uses the fixed-v1 sampler; Tiel uses its official sampler. Image is unsupported/N/A. Saved configurations remained unchanged.

The completed earlier chunk512 Qwen control was generation30.84, 4K prefill159.60 and 16K prefill85.72 tok/s. The new auto chunk1024 uses FP32 oneMKL attention for full1024-token batches; explicit chunk512 remains a user choice. Native weights, selected experts and the runtime ZIP are unchanged.

The actual installed app, refreshed through its Desktop shortcut, returned the correct uncapped normal-thinking Qwen answer at111.0 prefill/35.3 generation tok/s (84 fresh input/1268 output tokens). Independent16K normal-thinking full-answer tests passed all six facts for both models: Qwen263.44/26.02 and Tiel212.82/25.52 tok/s. Long generation30 tok/s, the42.5 aspiration, and universal short-prefill100 are not achieved. Gemma/LFM/Clover/images remain unsupported on Intel; Issue #11 stays open.
