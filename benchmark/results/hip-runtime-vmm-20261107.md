# Verified Windows HIP runtime and minimum VMM backing — 2026-11-07

## English

Automatic AMD installation and updates now select the private **ROCm `10.2.0a20260930` SDK / HIP 7.17** and use the driver's reported VMM minimum when the loaded runtime passes verification. **Old, mismatched or unverified runtimes retain the 4 MiB allocation floor.** CUDA behavior and model sampling defaults are unchanged.

Adoption follows the completed local stability/reproduction checks and a consistent SDK/allocation policy. It is not a blanket speed improvement: the development matrix's generation aggregate improves only **0.37%**, with individual regressions. The adoption rationale is runtime consistency and a shared driver-granularity policy after the stability checks, rather than a minimum throughput gain. This does not establish that every customer environment is fixed.

### Selection, verification and recovery

Automatic setup prefers the pinned private SDK over global `HIP_PATH` or older system SDKs. Explicit `-HipDirectory` remains supported; an older custom runtime can stay coarse. Older SDK/native generations remain available to profiles referencing them. Ordinary installed legacy profiles can migrate to a new immutable native generation using a locally verified private pin, without downloading or initializing a GPU during migration.

The installer puts matching **HIP / COMGR / Kpack** files beside the native executables, leaving math DLLs and their assets in the selected SDK. A JSON manifest records core/math hashes and paths; an ASCII proof records three core identities. Native code independently checks the actual loaded HIP module path, all three hashes, and **runtime version = compiled HIP version, at least 7.17**, before using the minimum. Python checks the bundle, proof and effective math-library search order. GPU Vision and model switching retain the same SDK identity. Drivers, System32 and persistent system PATH are unchanged.

Updates stage source/Python/native changes and publish one validated configuration, with preparation rollback and a startup guard. The original settings backup is retained when settings and runtime migrate together. Altered/missing managed dependencies fail clearly rather than silently mixing SDKs.

For coarse recovery, set `$env:TINYSTRATA_HIP_VMM_COARSE='1'` before launch. **Any nonempty value forces coarse backing**; remove the variable to restore automatic selection. There is no force-small override. Native INFO reports verification, driver minimum, actual quantum, runtime/compiled versions and reason. Verification can still be 1 during coarse recovery: the actual quantum matters.

### Conditions and method

Base: TinyStrata `b3fc0c1`. Windows, **RX 9060 XT 16 GB**, Ryzen 9 7945HX3D, 64 GB RAM, display connected to the Radeon before Windows boot. Three arms separate runtime alignment from allocation size:

| Arm | Runtime | Windows HIP backing |
|---|---|---|
| Control | Existing System32 HIP 7.12 | 4 MiB |
| SDK 4 MiB | Matching app-local HIP 7.17 core and SDK math | 4 MiB |
| SDK minimum | Same private candidate and HIP 7.17 bundle | Driver minimum, **64 KiB here** |

Both SDK arms use the same executable/bundle; a private diagnostic control changes only quantum. Actual language/Vision module paths and hashes, and effective quantum, are checked. System HIP is not tested with models at minimum grain: its fresh low-level comparison already fails there.

The completed matrix has **90 non-warmup measurements**, all from one Windows boot/session; earlier-boot results are not reused. Five models run in each arm: three generation cases once each (128-token limits), exact 4K/16K inputs (32-token limits), Clover 64K additionally, and one image request (128-token limit) for each image-capable model. Original starts have a 32-token warmup. Saved sampling, automatic Expert placement and physical KV growth remain active; supported models keep Vision resident. Profiles retain Qwen/Gemma MTP 3, Clover causal D0, and lookup/automatic-MTP off.

These are **one-pass development comparisons, not repeated standard v1 benchmarks**. Clover needed extra process starts for teardown investigation within the same boot. Later warmup-only unload checks do not replace measured requests. Frozen requests, native counts, placement/KV geometry, module maps and attempt history remain in the evidence.

### Throughput and costs

Generation is the geometric mean of three generation cases. The last two columns compare **SDK 4 MiB → SDK minimum**; prefill uses the 4K/16K geometric mean.

| Model | System 4 MiB tok/s | SDK 4 MiB tok/s | SDK minimum tok/s | Generation change | Prefill change |
|---|---:|---:|---:|---:|---:|
| Clover | 41.90 | 40.65 | 37.25 | **−8.36%** | −3.17% |
| Qwen3.6 | 87.47 | 86.43 | 90.85 | +5.12% | −0.98% |
| Tiel Coder | 70.10 | 70.06 | 71.59 | +2.19% | −2.35% |
| Gemma 4 | 72.58 | 71.82 | 75.58 | +5.22% | −0.18% |
| LFM2.5 | 94.63 | 98.91 | 97.25 | −1.68% | −1.47% |

Across five complete models, changing only runtime at 4 MiB gives generation **−0.18%**, prefill **+0.08%**. Removing the floor with the aligned runtime gives generation **+0.37%**, prefill **−1.64%**. Small one-pass differences are not established gains. Changed placement and positive-temperature outputs prevent treating every row as equal-answer latency.

Minimum grain has costs. Clover's first measured SDK startup rises **39.10→68.03 seconds**; later minimum-grain startups are about 51.5 seconds. Its image first-token/total time changes **2.52 / 4.64→6.71 / 10.05 seconds**. Smaller handles reduce padding but add mapping/release work; these tradeoffs are included in the adoption decision.

### Reproducing the failing operations

- Fresh **System HIP + 64 KiB** comparison: **245,760 / 262,144 pattern words mismatch**, cleanup succeeds. Matching SDK 64 KiB and 4 MiB checks pass. This supports retaining the old-runtime fallback, not enabling minimum backing everywhere.
- Frozen pre-interruption Clover replay with aligned SDK and 64 KiB: short/4K/16K cases with **256-token limits** complete, saved settings unchanged.
- Earlier error-719 Qwen first-prefill setup: INT8 KV, context 8,192, chunk 512, MTP 0, KV growth, no warmup, PP64/TG16; **0 and 4 GiB Expert budgets**, three built-in repetitions each. Both processes exit **0**, aligned SDK/64 KiB observed. This replays the local application failure, not another machine's BSOD.
- One bounded probe maps **128 × 64 KiB handles (8 MiB)** with zeroing queued between maps. GPU writes/reads and **2,097,152 words** pass with zero mismatches, followed by reverse unmap/release/free and exit **0**. This adds mapping/queued-access interleaving; it is not general endurance or retained-handle stress coverage.

### Shutdown failures found and corrected

The first minimum-grain Clover attempt completed seven measured requests, but the private harness forcibly killed API/native processes and treated survival beyond **five seconds** as failure. They were gone on immediate follow-up. That cleanup failure remains recorded; it is not silently counted as a clean lifecycle.

Normal `/unload` then returned after **24.156 seconds**, with language-process exit **1**: the 20-second graceful-QUIT budget was insufficient. A separate **90-second bounded observation**, without forced termination, obtains a natural **exit 0 after 30.482143 seconds**. This distinguishes slow release from simply declaring a timed-out process healthy.

Product graceful QUIT is now **60 seconds only when native INFO proves verified HIP with actual quantum below 4 MiB**. Legacy/CUDA/coarse paths retain 20 seconds; terminate and kill retain 20 seconds each. A process still stuck raises an error. CPU mocks cover the selection and bounded fallback stages.

The resumed diagnostic harness uses the corresponding 60-second cleanup allowance for its proven SDK/64 KiB route. All **30 minimum-grain inference cases** complete. Normal unload is independently observed with known **exit 0** for applicable language/Vision children for all five models: approximately **29.72 / 31.43 / 31.32 / 1.83 / 1.72 seconds** for Clover/Qwen/Tiel/Gemma/LFM. Clover's successful unload verification is a later warmup-only attempt, separate from its earlier measurements/failures. No whole-PC restart occurs in this current-boot matrix. Earlier failed cleanup/QUIT attempts remain failures.

### Output review, tests and limits

The matrix flags **40 paired decoded-output differences** and **43 paired geometry changes**. Sampling and changed placement can alter floating-point execution order; raw-token identity is not asserted. Qwen and Tiel misread the image heading in **all three arms**, while identifying the colored shapes. This limited review is not a broad quality evaluation or a guarantee of identical precision/answers.

Completed verification:

- **68 native CPU policy/guard checks**; **four real-HIP host metadata/IAT checks**. Valid proof selects 64 KiB; missing/tampered proof and coarse recovery select 4 MiB. Host checks submit no GPU work.
- **46 HIP runtime/integration Python tests**, **14 installation-guard tests**, **19 UI-launch tests**, **60 option/Clover decode/vision tests**, **10 saved-profile benchmark tests**, and **14 existing 64K benchmark tests**.
- Both **Windows PowerShell 5.1 and PowerShell 7**: entrypoints **99**, HIP/proof **64**, update/rollback **158** fixture checks. Fixtures use files/mocks, no models/GPU/network. PowerShell 5.1 bypass is process-only, not a persistent policy change.
- CUDA compile-only checks pass for five changed-code objects, including Qwen SM86/SM120 and Gemma host code. No CUDA device execution or performance result is claimed by this AMD validation.
- Matrix protected settings/history/key files compare unchanged. Native verification failure falls back coarse; managed-bundle corruption is reported before normal child startup.

The local application failure and the other user's [BSOD issue #1](https://github.com/kbc20f11/TinyStrata/issues/1) are distinct evidence. Issue #1 is to be reopened for a shipped-default retest; this owner's adoption does not wait for that response. The separate [AMD iGPU display issue #10](https://github.com/kbc20f11/TinyStrata/issues/10) remains unresolved: boot with the monitor on the Radeon, not motherboard output. No universal crash fix is claimed.

Private evidence: `TinyStrata-work/hip-validation-20261107`. Matrix session `20261107-101823-2ae20521`; SHA-256 `2e0ea69066155d01044ffe57d41cb3bd1a9fb369e7747c363c00f78ba7d03207`. Runtime-versus-quantum attribution uses the diagnostic matrix, not the selected-product run below.

### Final selected-product benchmark

The selected product build was compiled from the integration of base `b3fc0c1` and the guarded default-runtime policy. A **separate fresh one-pass saved-profile run** then completed all five models: **29 measured requests**, plus excluded warmups. Every engine reported `hip_vmm_verified=1`, runtime/compiled version `71726391`, and a 64 KiB quantum, without a diagnostic force-small override. All measured prefixes were fresh; every run completed normal unload with no forced fallback. Settings and model files were preserved.

The [updated RX 9060 XT table](rx-9060-xt.md#current-saved-settings--2026-11-07-one-pass) records the selected results: Qwen **90.30**, Gemma **72.67**, LFM **97.77**, Clover causal D0 **41.36**, and Tiel **70.54 tok/s**. These pooled rates retain the actual saved samplers and resident Vision, and do not replace the diagnostic matrix with a repeated/statistical comparison. Selected native SHA-256 identities:

| Executable | SHA-256 |
|---|---|
| `tinystrata.exe` | `2de23e57416fdb383400378624eba822daf609453410cc46a35b80757063f478` |
| `tinystrata-gemma4.exe` | `f1d3b5c53d4158e2de18a0bea1bc77f27a833bd6b3d95c3e2921405483b0935e` |
| `tinystrata-lfm2.exe` | `b34429dffd0f84399efc5561ab4039d5a82ce8c21caf51326459c9c2f278ca7f` |
| `tinystrata-vision.exe` | `cf71e294ef2c6e0c242c48f7a610fd68052c43ed2ae73d0a3d841abad1462cef` |

The shared proof SHA-256 is `22200da232d8564c707aa9333a4b0f7bd80cc3fceb313ee00fcb76dcf9d16e53`. Private build, source/diff, request, timing and runtime manifests remain in `TinyStrata-work/hip-validation-20261107/final-candidate2` and `final-benchmark`.

A passive event-log check after the final benchmark confirms the same Windows boot, with no new Kernel-Power 41, unexpected-shutdown, WHEA/display-driver, TinyStrata application-crash or LiveKernelEvent records in the test window. This is limited to the completed local run.

## 日本語

AMD の既定を private **ROCm `10.2.0a20260930`／HIP 7.17** の統一環境にしました。app-local HIP・COMGR・Kpack のハッシュ、実際の読み込み元、実行時／コンパイル時バージョンを確認できた場合だけ、ドライバーの最小粒度（この PC は **64 KiB**）を使います。古い・不一致・未検証の環境は **4 MiB の下限**を維持し、ドライバーや System32 は変更しません。

同じ Windows 起動中に 3 条件・5 モデル、ウォームアップを除く **90 件**を検証しました。SDK 4 MiB から最小粒度へ変えると、生成の全モデル集計は **+0.37%**、Clover は **−8.36%**でした。Clover の初回起動も **39.10→68.03 秒**となり、一律の高速化ではありません。採用理由は、安定性を確認した SDK の既定化と、割り当て判断をドライバーの報告値へ統一することです。一律の高速化を保証する変更ではありません。

過去にエラー719を起こした PP64／TG16・Expert 0／4 GiB 各3回、Clover の256トークン・短文／4K／16K、128個の64 KiBハンドルのマップと書き込みの交互実行でも正常終了を確認しました。一方、旧 System HIP の64 KiBでは読み戻し不一致が出るため、小粒度へ変更していません。

終了処理にも問題がありました。強制終了後の5秒待機では足りず、その後の通常アンロードもQUITの20秒で打ち切られていました。独立観測では強制終了なしに **30.482143秒で終了コード0**でした。この結果から、**検証済みかつ4 MiB未満のHIPだけ正常終了待ちを60秒**にしました。再開後は全モデルの通常アンロードで終了コード0を確認しています。Cloverの最初の測定と、後の終了確認だけの起動は分けて記録しました。

復旧用には起動前に `$env:TINYSTRATA_HIP_VMM_COARSE='1'` を指定できます。空でない値はすべて粗い粒度を強制するため、自動へ戻す場合は変数を削除します。MTP・温度・Clover因果D0の既定は維持します。出力・配置が異なる条件もあり、品質の同一性や全ユーザー環境での安全性を保証する検証ではありません。

他ユーザーのBSODを扱うissue #1には再検証を依頼しますが、返信待ちで今回の採用を止めません。マザーボード側の映像出力で発生したissue #10は別の未解決問題であり、Windows起動前からRadeon側へ画面を接続する制限を維持します。

最終製品ビルドでも、保存済み設定を使う独立した1回のベンチを5モデル・29測定で完走しました。全モデルが検証済みHIP・64 KiBを選択し、通常アンロードまで完了しています。最新値は[RX 9060 XT表](rx-9060-xt.md)を参照してください。
