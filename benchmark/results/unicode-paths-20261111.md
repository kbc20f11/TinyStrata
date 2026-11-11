# Windows Unicode model paths — 2026-11-11

Windows native argument and file APIs mixed the ANSI code page with UTF-8. On this CP932 Windows machine, even a Japanese-only model path reproduced failure: the image worker could not open its projector, and printing the decoded failure to Python's CP932 log raised `UnicodeEncodeError`, terminating API startup. The same failure was reproduced with spaces, emoji and a supplementary Japanese character.

The four native executables now recover Unicode command-line arguments with `CommandLineToArgvW` and retain UTF-8 internally. TinyStrata-owned model mapping, split discovery, embedding/token files, output/profile files and adjacent MMQ DLL loading use explicit Windows wide paths. GGML/MTMD already consume UTF-8 paths. Executable Python entrypoints write UTF-8 diagnostics. The supplied `run-ui.bat` uses a small PowerShell bootstrap to retain Unicode interpreter/install/work paths, read the UTF-8 location pointer, preserve arguments and propagate failure codes. The independently installable UI includes its own stdlib-only diagnostic helper for compatibility with older core installations. No system code page, sampling policy, weights, KV format or Expert arithmetic is changed.

## Actual model verification

RX 9060 XT 16 GB, gfx1200, Ryzen 9 7945HX3D, frozen managed HIP 7.17.26391, verified 64 KiB VMM. Model, executable, tokenizer, configuration and working paths included Japanese, spaces, emoji and `𠮷`. Existing model files were reused through private hard links; user settings and model data were preserved. These are functional checks, not new throughput benchmarks.

| Model | Two-document factual check | Additional coverage |
| --- | --- | --- |
| Qwen3.6 35B-A3B | Passed, including long appendix | Missing-file error then retry; image; unload/reload; persisted Expert-profile path |
| Gemma 4 26B-A4B | Passed, including long appendix | Separate MTP and projector; actual UI image, unload/restart/reload and history restoration |
| LFM2.5 24B-A2B | Passed | Unload/load and a fresh process reopening the same Unicode paths, mandatory thinking preserved |
| Clover 35B-A3B Preview | Passed | Image projector shared with Qwen, causal default |
| Tiel Coder 35B-A3B | Passed | Image and existing MTP route |
| gpt-oss 20B | Passed | Existing reasoning route through the shared Gemma executable |

The independent oracle checks corrected quantities, stock shortage, units, owners, deadlines, inherited safety conditions, a 0.70 MPa / 15 minute test, unknown fields remaining null, the ¥68,800 pre-tax total, pending approval and no execution. All six saved answers agree with the source documents. Correct JSON was sometimes fenced as Markdown. The two long cases include an unrelated 18 KB appendix; this is not a maximum-context stress test.

The actual supplied BAT was also run from a Japanese/space/emoji install directory with a similarly named work directory and private Python. Playwright uploaded the two Japanese-named source files and appendix, generated a Gemma answer, and compared it with the oracle. The three SQLite attachment blobs exactly match the original bytes. Image facts were correct. After UI/API restart through the BAT, all attachments, answers and the image reappeared; the reloaded model answered a follow-up. Screenshots were inspected for text, placement and clipping. UI business generation took 60.7 seconds including thinking; image follow-up took 4.5 seconds and restart follow-up 6.2 seconds. No search or tool execution was required.

## Build, regressions and limits

All four HIP executables were rebuilt and exercised. CPU discovery passed 457 of 458 tests, with one Windows symlink-privilege skip; the final two-line bootstrap output cleanup passed the ten targeted BAT tests again. Eight CUDA compile-only checks passed, including SM86/SM120 and Vision's C++17 compilation. Native path/argument tests cover Japanese, spaces, supplementary characters, quotes, empty arguments, long arguments, mapped GGUF data, split discovery, read/write and failures. The build's whole-tree snapshot changed while Python tests were being added; the compiled native source closure is separately matched to the final source. Python and bootstrap changes are bound by their later tests and actual UI run.

Preserved trial failures are not hidden: an early non-default Gemma greedy/4096-token diagnostic exhausted its thinking budget; the normal-sampling answer subsequently passed. One oracle initially rejected the source's literal approval value `未了`; it was independently reviewed and the captured answer rechecked without regeneration. Initial Qwen and LFM harness errors used an unsupported managed-API option and an unsupported thinking-off request respectively; the supported retry paths passed. Original failed receipts remain intact.

Physical NVIDIA and POSIX runtime execution, maximum Windows path lengths, a new OS user account, full installer execution in a Unicode directory, and optional EAGLE dump output were not exercised here. Split weights were checked with valid small CPU GGUF fixtures, not a second full split model. Native desktop Save dialogs were unchanged and were not tested. A separately opt-in HIP tuning-table environment path still uses the legacy ANSI code page; an unrepresentable table name falls back to ordinary BLAS and does not prevent model loading.

Raw failed/successful runs, original documents, independent oracle, source/build hashes, CPU/CUDA receipts, UI database and screenshots are retained under `TinyStrata-work/unicode-paths-20261111`. The [compact validation receipt](unicode-paths-20261111.json) records their hashes and verified scope. The existing [RX 9060 XT benchmark table](rx-9060-xt.md) is unchanged.
