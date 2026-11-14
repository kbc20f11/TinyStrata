# 開発メモ

更新日: 2026-11-14

## Intel B390 / Windowsの初期移植と電源状態

- **最終C1024規定テキスト計測完了**: 両モデル4 starts/3課題各2passes・seed1234・private context32768・実INT8 KV/MTP3・chunk1024で完走し、保存設定不変。generation/4K/16Kは qwen:31.65/292.40/264.32; tiel:25.65/238.07/218.12 tok/s。`standard:false`・画像N/Aで固定v1全体合格とはしない。旧C512対照4回（30.84/159.60/85.72）は保持。実installer/65payload/freshvenv/UI/Desktop、Qwen↔Tiel/Native再読込/SDK0/状態保護、C1024更新後実Qwen111.0/35.3と16K両モデル最終6facts正解を確認。長文生成26.02/25.52は30未達、42.5/他3モデル/画像は残件。詳細は公開測定JSON。

- **19時台のprefill候補とType18検証**: 新 `build-type18-prefill` は同一84入力・1268出力のold/new/oldでprefill93.98/99.22/97.02、decode32.32/34.05/34.32 tok/s。全出力ID一致・最終6項目正解・全exit0。生成の上積みは断定せず、full-answerの短prefill100未満も保持する。別のGEN1診断は無関係promptを挟み毎回RESUME0、new冷間105.08・温間118.37/118.85/112.27、old冷間98.87・温間80.84/84.84/106.61。これはprefillだけの診断で、完答品質や生成速度の測定に代えない。実IQ3_XXS Type18(98B→100B lossless packed)を追加し、Qwen27/Tiel21実tensor参照とrouting/tail/boundaryをGPUで通過。最初のconst ESIMD view compile失敗は保持し最小修正後のretry成功。Tiel実GGUFは13,600,579,424 bytes/SHA a4d8ffc7f030466c67db289543871a8f9e6d5340241b300c09fbaf16642273caを検証済みだが、fullmodel品質とpublic採用は別ゲート。新native capabilities実probeと53file strict私有importも通過（`installer-live-smoke/run-20261114-192124-5d41cc67`）。根拠は[測定JSON](../benchmark/results/intel-uma-windows-20261114.json)の新3section。
- **ユーザーの公開指示**: 目標到達まで最適化→本人規定ベンチ→Intelオンボードだけの環境で選択・実導入できるinstaller→main push→Issue #11に進捗を記録する。固定v1にはGPU Vision常駐等があるため、未実装差を隠して標準合格にしない。ローカルcheckpoint `5389ea8` 済み、remote publishはまだ。
- **Tielのfullmodel品質／追加指示**: 私有native検証 `tiel-native-sanity-quality-20261114-1942` で通常medium思考のR17はprefill135.60/decode30.95 tok/s・101出力・6項目正解、二分探索は130.57/29.21・151出力・独立5100ケース合格。全owned自然exit0、SDK DLLなし、GGUF/exe不変。closed-thinkingは残金6800（正解6400）だったが、独立llama.cpp SYCLでも同一prompt IDsから全同文の誤答。ユーザーが「Tielは過去もnothinkの暗算系ミスりやすいので無視していい」と明示したため、この既知の暗算ミスを採否の障害から外し、結果は保持して通常medium/コードを重視する。初回weight進捗・DONE accepted/offeredを誤解したprivate parserの失敗と修正履歴も測定JSONへ保存。public Tiel/profile接続は実装中。加えてユーザーがmain push前のこのPCでの実installer導入を明示、準備のmockで代えずbootstrap/UI/Desktop/起動生成を確認する。

- **通常経路と実導入後の復帰が通過**: `unified-intel-api-ui-20261114-180929` で通常run.py→serve.server→実UI、84入力/1268出力・prefill96.1/decode32.7 tok/s・公式設定・最終6項目正解、SDK隔離/LUID/UMA/全owned exit0/通常状態不変を確認。新installerはSYCLを明示選択し、config-sycl/API18085/ui-sycl/専用shortcutを分離。新60 PS＋7 Python、既存launcher26/entrypoint114/UI7/desktop34/transaction159/updater12は各通過。私有NoStart実導入は53実fileのhash/license/immutable generation/config publish/guard cleanupを通過（`installer-live-smoke/run-20261114-182701-34b4f6c0`）。Python初期化だけ既存Rvenvを返す境界は明記し、新規Python取得・ユーザーDesktop実配置と混同しない。その実generationで未認証2・unsupported11の拒否時に同resident PIDを保持し、MTP3→0→3再load全正答・最大resident1・全11owned自然exit0・元cfg/stage/UI不変を通過（`unified-intel-state-ops-20261114-182941`）。最初の監視failedは40.7msのcaps-onlyプロセスをresident扱いしたprivate分類で、厳密args/creation/exit区別後に再確認。旧failed保持。画像/他モデル/短prefill100未満/全課題性能は残件でIssue全体は未完了
- **実importでstage size不備を修正**: SDKのumf.dllはsymlinkでFileInfo.Length=0、実payloadは209720B。初回importerの厳格拒否を保持し（`installer-live-smoke/run-20261114-182328-31f662a2`）、stagingのsizeをstream.Lengthへ修正。alias metadata0回帰を含む8検査通過、新 `runtime-stage-packed-iq-v2` から実importと再配置後APIを通した。旧stage/manifestは上書きしていない
- **通常起動への統合を継続中（18時台）**: engineのlive33.0 tok/s通過後もIssue完了にせず、`run.py --backend sycl --intel-profile b390-qwen36`、通常server/profile、既存installerの明示Intel分岐を追加している。専用 `config-sycl.json` / API18085とIntel UI領域で既存CUDA/HIP設定・鍵・履歴を上書きしない。`serve/intel_backend.py` はQwen3.6限定・auto contextはB3908192/generic4096・保存KV autoと実INT8を区別し、profile派生fast/stochastic/draft flagsは起動用copyだけへ追加する（MTP3→0→3もCPU確認）。helper21検査、launcher9と既存HIP/profile/retiredを含む前段93検査、実GGUF＋app-local stageのconfigure-onlyが通過。通常serverからの実生成・installer rollback/UI選択の最終検証はまだ行っていない。129検査の途中回帰ではconstructorを通さない既存ModelSwitcher fixtureの新属性参照を検出し、server担当が互換修正後の回帰を実行中。通常起動の未確認状態を先のstandalone成功で埋めず、最終検証を続ける
- **packed-IQ follow-upの同一出力比較**: interleaved IQ2_S/IQ3_Sの2〜8入力だけaligned DWORD gatherへ変更し、register内で同じcode/sign byteを復元する。単token・大prefill・元layout・Q6は既存loadを保持し、量子化・選択expert・DPAS/FP32加算順は不変。単体54比較は全出力bit/tail canary一致、単tokenの遅いwall結果も残し適用しない。`build-packed-iq` はfresh CPU3/full GPU27＋capacity/native API11/greedy3コードを再通過。通常thinking binary-searchのold前→new→old後は29.09→32.80→30.77 tok/s、全1061出力IDs一致・stop・5100例合格。prefill86.83/86.43/91.16で100未満は残る。R17 newはprefill95.32/decode33.055・1268 IDsが旧sparse2048と一致。[公開測定JSON](../benchmark/results/intel-uma-windows-20261114.json) の `packed_iq_load_candidate` を根拠とし、単体kernel名や1課題だけから全体採用を主張しない
- **最終LIVE app-local API/UI通過**: `runtime-stage-packed-iq` / draft2048を使う `staged-api-packed-iq-ui-20261114-173910/result.json` はpassed=true。通常thinkingで84入力/1268出力・prefill95.7(<100)/decode33.0 tok/s・TTFT0.922秒、stop/SSE DONE・1024超・最終JSON6項目正解。actual LUID/node一致、shared counter取得・legacy VRAM null、compiler SDK DLLなし、実Monitor/AboutのDOM/PNGとCDP例外/console error/network failure各0を確認。UMAのVRAM hit列hidden・visible header8・実行行8cellも通過（空行colspan8はCPU検査）。全owned API/UI/native/browserのexit0・強制停止なし、通常history/settings/location不変。これは保存値replayではなくlive検証。下記の実UI待ちや初回failedは当時の履歴として保持し、この限定staged text API/UI検査を後続で完了した意味である。通常installer/選択/model switch・画像・他モデル・全課題30以上の採用ゲートは未完了、Issue #11は継続
- **UMAの残るVRAM表記**: recent-requestのVRAMヒット率列をUMAだけ非表示にし、header・実行行のcell数・空行colspanを8列へ一致。非UMAでは従来の9列/値を保持し、架空の代替指標は追加しない。focused telemetry/UI14検査と既存Node UI6組396項目、JS構文検査が通過（`intel-uma-hit-column-cpu.log`, `intel-uma-hit-column-node.json`）。rootのgeneration測定中はbrowser replayを追加実行しておらず、最終live UI captureで確認待ち
- **fresh APIの部分確認／保存値UI replay**: `staged-api-draft-vocab-ui-final` の実APIは1268出力・stop、compiler SDK DLLなし、native/DXGI LUID一致・node1・shared counter取得・legacy VRAM nullまで確認。ただしMonitor表示前のcapture失敗でresult.json全体はfailedのまま保持し、live API/UI通しのretryは未完了。実データ保存値だけのmodel-free replayでは、最初のCDP helperが未表示tabの描画を先に待つ条件でtimeout（`ui-recorded-replay-20261114-172622-525569`）。private helperを修正したreplayはMonitor/Aboutの実ブラウザー描画・例外/console error/network failure各0、Edge/Node＋16子processのexit0/強制停止なし、HTTP/UI終了、通常UI状態不変まで通過（`ui-recorded-replay-20261114-172814-735064`）。共有RAM14.3GBは保存snapshot値であり、replay中の新しいGPU計測値ではない。production起動例外は再現しておらず、replay成功をlive native/API/UIの合格へ読み替えない
- **draft-vocabの設計・fresh検査**: 既定0/off、非zero 64〜4096は `--serve --mtp-stochastic --mtp --vocab-xmx` が必要。直近のtarget logitsと履歴から実IDを選び、proposal qのheadだけを縮小する。target pは全語彙、sampler/penalty・正確なmin(1,p/q)＋正残差拒否・選択8 expertsを維持し、GGUF再量子化なし。fresh AOTのCPU CTest3・実tensor27参照＋selected head tail1/15/17・上限4096/tail/refusal・split-K1MiB境界、native/API11、独立Python3コード課題が通過。現在の単一arena14,305,198,080 byte、2048 IDsは追加8KiB、FP32 recurrent/INT8 KV。`gpu_upload_bytes=0` はloaderの明示copy計数だけで物理driver/page traffic全体のゼロを意味しない。[測定条件と結果](../benchmark/results/intel-uma-windows-20261114.json)、`draft-vocab-quant.log`, `draft-vocab-integration.log`。下の「draft-vocab未検証」は以前の時点の履歴であり、この検証が後続する。opt-in候補のままで既定採用・Issue完了とはしない
- **draft-vocabの通常設定比較**: 同binary・思考あり・84入力のcontrol前→limit256→2048→control後はdecode25.94/30.28/31.80/28.04 tok/s、出力1658/574/1268/1658。全てstop・最終JSON6項目正解。prefill93.34/97.69/84.00/100.90で短入力100以上は安定しない。proposal変更に伴う出力長差があるため正確な改善率や全課題30以上を主張しない。limit2048の通常thinking binary-searchは69入力/1061出力、prefill87.76/decode29.16、stop・5100独立例合格。`draft-vocab-normal-paired/results.json`, `code-quality-draft-vocab-thinking/results.json`
- **draft-vocabの長文／実UI待ち**: 同じclosed-thinking入力で4K prefill340.60/decode33.70、16K306.00/25.98 tok/s、両方59出力・stop・最終JSON6項目正解。chunk1024/context8192・32768、temperature0＋presence1.5であり、過去のdirect CLI presence0と直結比較しない。`draft-vocab-long-20261114-171305-379061/results.json`。fresh build自体の検査は上記の通り完了したが、child環境修正後のfresh app-local fullmodel API・LUID/counter・実UIはintegration実行中で結果待ち。CPU parser/telemetry/staging計35、既存Python74＋Node396の通過だけから実UI成功を推定しない
- **UMAの既存共通回帰検査**: AMD/HIP・status/stream関連のPython74検査と、live stats・アイコン・テーマ/翻訳・resource・sampling既定値・モデル識別のNode UI6組396項目が全て通過。追加修正なし。証跡 `intel-uma-shared-regression-python.log`, `intel-uma-shared-regression-node.json`。CPU検査であり、native再build・実LUID/counter・実UIの確認はまだ含まない
- **AOTの通常設定実測**: `build-ptlh-aot` の実モデル・思考あり・INT8 KV/MTP3で84入力/1373出力、prefill91.75/decode26.13 tok/s、最終JSON6項目正解。native起動からREADYまで16.891秒で、読込時間はtok/sへ含めない。単発結果から速度向上や通常30〜40達成を主張しない。証跡 `ptlh-aot-normal/results.json`, `ptlh-aot-normal.log`。次のdraft-vocab候補は実機未検証であり、既存vocab経路の検証済み結果を流用して採用扱いにしない
- **staging初回失敗とchild環境の修正**: `staged-api-validation-initial` は通常APIの84入力/1658出力・prefill95.8/decode26.5 tok/s・最終JSON正解まで成功したが、nativeがcompiler側の `intelocl64.dll` をロードしたためisolated条件はfailedのまま保持。通常UIのlocation pointerは不変。`serve.intel` のchildだけへ `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` / `UR_LOADER_PRELOAD_FILTER=1` をsetdefaultし、明示ユーザー値と親/global環境を保持する修正は小さなstaged queueで実証した。修正後のfullmodel/APIの隔離再検証はこれからで、queue成功を配布完了としない
- **Intel UMAのメモリー表示（CPU検証段階）**: 実際に選択したLevel Zero deviceのintegrated flag・vendor/device ID・architecture・LUID/node maskをnative起動INFOへ追加。Intel用Windows samplerはLUIDが一意に一致するDXGI adapter/node 0のPDH Shared Usageだけを取得し、adapter全体・システムRAMと重複する使用量として公開する。識別不能・重複・未対応node/counterは取得不能のままとし、NVMLへfallbackしない。UMAではlegacy VRAM容量/使用量をnullに保ち、UIのRAM容量へ共有GPU分を加算しない。KV/エキスパートの専用VRAM表現・未計測の配置割合も出さない。`tests/test_intel_telemetry.py` のCPU14検査（ctypes PDH fixture・公開status・実UI描画関数のfake DOM実行）と既存options6/staging7、JS構文検査が通過。証跡 `tinystrata-work/intel-20261114/intel-uma-telemetry-cpu.log`。私有Node v26.13.1は公式ZIPのSHA256を検証して `node-runtime/` だけへ配置し、global PATH/設定を変更していない。native再build・実LUID拡張/counter・実ブラウザーでの確認は別途必要で、この変更についてGPU推論や速度測定はしていない
- **限定split-K/API出力予算**: `--dense-split-k` は既存dense layoutに加えwidth4096/rows2048のQ6だけをlossless変換し、4つのK範囲を計算→FP32加算する明示候補。scratch1 MiBを予算計上し、確保はgraph記録前、computeとjoin両方をprofileへ含める。独立9型27参照のQ6 1/2/3/4/5/7/8行、定数0.125を独立参照とする最大65536出力/1MiB境界・tail不変・上限超過拒否を通過。native/API11検査・Python3コード課題も合格。同binaryの通常設定ABAはcontrol前prefill35.39/decode26.54・1373出力、候補96.33/27.66・1658、control後101.68/27.14・1373、最終資料6項目は全正解。cold firstのprefill回帰と異なる出力長を残し、正確な改善率や通常30達成とはしない。`dense-splitk-*`, `code-quality-dense-splitk`。実験APIだけの1024出力既定を通常のDEFAULT_MAX_TOKENSへ修正、`--max-tokens`で指定可能。既存Serviceが暗黙値を残りcontextへ収め、明示client超過は拒否する。thinking/samplerは維持し、CPU4検査で検証（`tests/test_intel_options.py`）
- **再起動後に再開（2026-11-14 15:48）**: ユーザーの継続指示を受領。`powercfg /a`でS0/S1/S2/S3・休止・hybrid sleep・高速startupは全て利用不可、究極プラン有効を確認した。guard-v3もStartupから再起動しDisplay/System/ExecutionRequired・5秒heartbeat継続。従って下の「再起動後未確認」「再開まで停止」は履歴であり現状ではない。同一1373出力の再起動後baselineはprefill73.22/decode26.35 tok/s・最終JSON正解・actual IDs一致。54GPU samplesは850〜1850MHz、flags2が3件・4が51件、thermal8/256なし。`post-reboot-inference`, `post-reboot-inference-frequency.jsonl`。新しいshape限定Q6 split-K候補とAPI出力予算の修正を進める
- **ユーザー指示で一旦停止（再起動予定、2026-11-14 15:39）**: 再開手順・電源設定・直近の実測・未完了項目を `C:/Users/user/git/tinystrata-work/intel-20261114/HANDOFF-20261114-1539.md` に保存した。Modern Standby override値0は確認済みだが再起動後の反映は未確認。再開を求められるまでビルドや最適化を実行しない。検証/ビルドは全て終了、ユーザー希望のkeep-awake guardだけを残す
- **15:37時点の継続事項／電源更新**: ユーザーは、閉じたPCを吊り下げて背面吸気・側面排気を確保していると説明した。開蓋や床置きを提案し直さず、その配置のまま検証する。画面消灯も不要という後続指示を優先し、AC/DCの画面・通常/無人sleep・休止timerを0、蓋とsleep/power短押しを何もしない、hybrid sleepオフへ設定。`b12b47c0-98cb-45a0-937b-66de4bba9039`「究極のパフォーマンス」を作成・有効化し確認した。私有`power-20261114/power-guard-v3.py` はDisplay/System/ExecutionRequiredと5秒heartbeatを保持し、現ユーザーStartupの`TinyStrata Keep Awake.lnk`から再ログイン後も起動する。旧guard1/2はstopfileで終了した。`powercfg /hibernate off`はagentからは権限で拒否され、ユーザーが管理者consoleで実行後、`powercfg /a`で休止と高速startupの無効化を確認済み。ただしS0 Modern Standby自体はまだ利用可能。ユーザーはこれも無効化するよう依頼し、再起動を要する回避策を確認中。勝手に再起動しない
- **直近の実装／速度**: INT8 attentionの4-byteまとめ読込＋4出力/worker、Expertの実行行数1/2に合わせたDPASを追加。独立9型27fixtureとFP64 attention、native/API11検査通過。交互40回の4-row attentionは4K1.585→0.792ms、16K6.314→3.169ms。4K資料prefill359.89/decode35.31、16K196.41/15.21 tok/sで両方6項目正解。通常設定・思考あり84入力の資料質問は1373出力・最終JSON正解、118.56/20.76。ユーザーのスリープ解除直後は同じ実行ファイル・実際の1373出力IDも一致し100.26/27.87、GPU1400〜1900MHz、51 sample中thermal flag8/256は0、current flag4は50。究極プラン＋display要求後の同じ1373出力は101.65/27.32。プラン変更だけの加速としない。`q8-packed-loads-*`, `wake-inference`, `wake-inference-frequency.jsonl`, `ultimate-inference`
- **未採用候補**: packed ExpertのFP16 register DPASは検証済みoneMKLのFP16積に誤差0〜2.11e-7で一致したがIQ2/IQ3のgate/downで約1.5〜1.8倍遅く、私有prototypeのみで不採用。block offset明示化も実出力一致・独立fixture通過だが旧版との交互40回で速度差がなく、source変更は撤回した。`fp16-expert-micro*`, `expert-offset-micro*`。Q6 split-K prototypeはwidth4096/rows2048の4-rowだけ、SSM output .132→.071ms・full-attn output .124→.081ms程度の改善を観測。他のQKV/small/shared形状は遅いため一律には採用しない。次は該当shapeに限定して組み込み、独立FP64・実コード/API・通常設定で確認する。`q6-splitk-micro*`。まだ正式採用やIssue完了ではない
- **INT8 attention一括化／大prefillのExpert経路**: INT8の短窓をtoken別4 dispatchから窓全体4 dispatchへ変更。独立FP64は1/3/4/7/8行、異なる因果位置、255/256・511/512境界、1031末尾partial splitを検査して通過。scores/partialはmin(chunk,8)行分で予算へ加算。最初の1D版はnative/API11通過だが4K148.12/17.53、16K144.02/11.87 tok/sと前回より遅く、通常C1024も77.85/13.57で思考上限。単体40回交互では4行4K2.058→1.696ms、16K7.398→6.628、出力一致。divisionを減らした2D版の独立検査も通過し、交互単体4行4K2.113→1.679、16K7.172→6.479ms。large expertはpackedのままINT8へ固定せずT>=512で既存oneMKLへ戻す候補を追加。output scratchへ1 Expertだけcompressed復元→FP16復号→同じ領域を結果へ再利用し、重み全体の二重保持はしない。9型27標本の独立参照と新packed MKL検査は通過。実モデル4K358.63/34.10、16K201.96/12.60 tok/sで両方6項目正解。ただし通常設定・84入力の別資料質問はMTP深さ1で19.75/14.70、1797出力・約122秒で正しい最終JSONとなり、短文のprefill/通常速度は不足する。`q8-batch-quant.log`, `q8-batch-integration.log`, `q8-batch-long`, `q8-batch-normal`, `q8-attention-micro.jsonl`, `q8-attention-micro-2d.jsonl`, `packed-mkl-quant.log`, `packed-mkl-long`, `packed-mkl-normal`
- **dense選択版の比較**: 転置FP16 scratchへの復号版は独立数値検査・native/API11・Python3課題を通過したが、全denseを変換した短primeは明確な優位なし。40回交互の実行列別比較では小gate/K/Vが2〜4倍遅く、width4096のoutputもT1で遅かったため、width<=2048・rows>=2048のQ6だけへ制限した。選択版は同binary対照prefill101.61〜108.09/decode33.71〜34.31、候補85.63〜99.47/37.99〜38.17 tok/s（51入力・思考を閉じたgreedy、全6コード各10,021入力合格）。prefill回帰も残り、通常設定の持続性能へ一般化しない。4K入力258.73/26.51、16K149.65/13.52 tok/s、6項目とも正解、最低空きRAMは約9.46 GB。過去のvocab-onlyのprefill378/221より遅い条件も保持する。`dense-column-quant.log`, `dense-column-integration.log`, `code-quality-dense-column`, `dense-matrix-micro.jsonl`, `dense-selective-code/prime-validation.json`, `dense-selective-long/results.json`。INT8 attentionの検証窓をtoken別4回dispatchからまとめる候補を別buildで検証中
- **Expert配置候補の実測**: Q6_K/IQ2_S/IQ3_Sの16行word-interleaveを `--expert-xmx-layout` で明示選択。最初の全層変換はGGUFの混在形式をguardが検出して拒否した。実際にはdownの34/38/39層がIQ4_XS、MTP40層のgate/upがQ2_K・downがQ3_K。3行列とも対応する層だけを変換し、他は元のGPU経路を保持する。111 Expert tensors変換／12保持、vocab変換込みarena14,295,040,000 byte、全753のGPU可視性確認通過。単一の常駐重みと8 MiB変換stagingのみ。量子化byte保持・独立FP64の1/4行tile、native/API10テスト、Python3コード課題が通過。短prime3回は対照prefill93.0〜98.4/decode30.5〜30.8、候補101.2〜108.5/34.5〜35.1 tok/s。全6コードは各10,021入力の独立素数判定に合格。`expert-layout-quant.log`, `expert-layout-eligible-paired`, `expert-layout-eligible-integration.log`, `code-quality-expert-layout-eligible`。初回拒否は `expert-layout-paired` に保持
- **厳密確率補正の実モデル検査**: `--mtp-stochastic` はnative/API10テスト通過。通常設定・同じ64入力／最大1024出力のtarget-onlyはprefill99.63/decode16.82 tok/s、635/1163候補受理。p/q＋残差版は106.06/17.11、681/1022受理。両方思考中で上限に達し最終Cなし。受理率54.6→66.6%に比べ全体速度差は小さく、通常設定の目標達成とはしない。RNG消費は方式間で異なり、同seedの文言一致を求める方式ではない。既存post-sampler target分布を保持する厳密方式。`mtp-rejection-integration.log`, `mtp-rejection-normal/results.json`
- **MTP Q2_K/Q3_K整数カーネル**: このGGUFでは両形式はblk.40の3 Expert行列だけ。Q2_Kのmin補正を含むDP4AとQ3_Kのsigned復号を追加し、実テンソルの独立NumPy/FP64検査とnative/API10テスト通過。主モデルの量子化方式を変えた意味ではない。単体40回交互測定のT1はgate FP32約.260〜.276→Q8 DP4A約.248〜.263 ms、down約.242〜.244→.224〜.225。短prime34.66 tok/s、通常1024試行は15.44で前候補を超えず、単体改善を全体の勝利と扱わない。`q2q3-draft-quant.log`, `draft-matrix-micro.jsonl`, `q2q3-draft-code`, `q2q3-draft-normal`
- **状態メモリ／検証graph**: 明示`--device-state`ではGPU専用state/scratchだけdevice USM、logits・入力ID/位置・counts/error等CPUが読む値はsharedを保持。weightsの確保方式とは独立で、物理RAMを二重計上しない。`--verify-graph`はINT8 KVのMTP窓をdynamic token/position入力とextent別に記録（上限16 cache、128 MiB reserve）。実際のgraph使用をstderrで確認し、生成中のSTOPとprefix復帰/clean replayも含むnative/API11テスト通過。短prime3回ずつはshared/eager34.69〜35.36、device/eager34.51〜37.68、device/graph30.71〜35.94 tok/s。全9コードが独立10,021ケース合格。graphは明確な優位がなく既定に採用しない。`graph-state-integration.log`, `graph-state-native.json`, `graph-state-code`。oneMKL/XMXなしの最小構成もビルドとCPU2検査に合格（`build-bare-validation.log`）
- **dense配置候補**: `--dense-xmx-layout` はembeddingを除く対応2D Q6_Kをlossless変換し、短窓では整数XMXへ渡す。大prefillはF16 GEMMへ復号。最初のrow-major復号は独立FP64・native/API11・Python3課題に合格したが、短文prefill58.16〜60.06 tok/sへ低下（対照warm107.90〜107.94）、生成35.44〜36.18（対照34.01〜34.27）で不採用。GPU書込が連続する転置配置とGEMM指定へ変更した版を別buildで検証中。F16の値とFP32 accumulationは保持する。`dense-layout-quant.log`, `dense-layout-code`, `dense-layout-integration.log`, `code-quality-dense-layout`, `build-dense-column`
- **速度条件を明記**: 短primeと資料4K/16Kの高速値は思考を閉じたgreedy条件。通常設定の速度として30台を一般化しない。同じC99質問を正式GGUF templateの思考あり・temperature1/top_p.95/top_k20/presence1.5/penalty_last_n64/seed2026で評価すると、native INT8 KV＋vocab XMX＋MTP3は64入力・4096出力・prefill101.90/decode13.36 tok/sで上限到達、思考未完了で最終コードなし。同じ64入力ID・通常samplerの独立llama.cpp SYCL／F16 KVも4096出力・prefill66.28/decode12.20 tok/sで思考未完了。単独の移植固有エラーとは断定せず、出力上限内に最終回答がないことと実行済みCの不正を区別する。両方の失敗とRNG/KV/MTP差を保持。証跡 `merge-sort-thinking-native`, `merge-sort-thinking-independent`
- FP32 grouped-query oneMKL attention候補は独立FP64とnative/API10テストに合格し、16Kはprefill219.83/decode13.33で6項目正解。ただし初回4Kは344.58/7.82と悪化。256セル単位へ形状を揃え、未使用KV tailをzeroにする版も独立数値検査とnative/API10テストに合格したが、同binary4K反復は対照prefill283.29〜380.79/decode17.54〜26.34、候補170.30〜260.19/16.03〜16.12で明確な優位なし。既定には採用しない。`verify-mkl-long`, `verify-mkl-bucket-paired`, `verify-mkl-bucket-integration.log`
- 通常samplingのMTP改善候補としてAlgorithm1（Leviathan et al., https://arxiv.org/abs/2211.17192）のp/q受理＋正の残差分布を実装中。明示`--mtp-stochastic`、既定off。下書きには独立RNGと下書き専用履歴を使い、主履歴へは採用した実出力だけを追加。既存target-onlyモード/CUDA/HIPへ新しい方式を強制しない。支持集合の不一致・one-hot・zero確率・同一分布など計240万CPUサンプルで出力頻度と受理率が理論値の許容範囲内。実モデル検証は未完了。`speculative-distribution.log`
- 語彙出力Q6_Kのlosslessな16行word-interleaveを明示候補 `--vocab-xmx --decode-dp4a` として追加。読込時8 MiB staging内で変換し、従来headと変換済みheadを常駐併存させない。量子化byteを保持し、210→212byteのpaddingでarenaは14,058,283,008→14,062,256,128 byte（+3,973,120）。GGUF自体は変更しない。単体30回交互測定の旧DP4A/固定M4 XMXはT1 5.333295/4.730479 ms、T4 10.114851/4.786115、T8 13.861677/9.540351。単体prototypeのみheadの一時二重保持があり、実engine実装には持ち込んでいない。その後M1/2/4/8幅を選ぶ実装の独立NumPy/FP64検査・全byte保持・native/API10テスト・Python3コード課題が通過した。同binary短prime3回は対照decode30.01〜30.25、候補31.82〜34.01 tok/s。warm prefill対照98.48〜98.99、候補104.02〜106.20。候補の初回93.06も保持。4Kはprefill378.01/decode25.59、16Kは221.04/11.56 tok/s、いずれも6項目正解。短い1課題だけで持続目標達成とはしない。証跡 `vocab-interleave.jsonl`, `vocab-xmx-quant.log`, `vocab-xmx-paired`, `vocab-xmx-integration.log`, `code-quality-vocab-xmx`, `vocab-xmx-long/results.json`
- 大prefill GDN改善版は独立数値検査のほかnative/API10テストとPython3課題も通過（`prefill-gdn-integration.log`, `code-quality-prefill-gdn`）。確保方式別の同512入力3回はhost prefill85.62〜155.43/decode13.34〜27.80、shared128.74〜150.30/13.66〜17.19、device96.97〜117.13/13.23〜14.44 tok/sで全9資料回答の6項目が正解。順序・周波数の変動を含み、device確保だけで高速化したとはしない。`weight-memory-inference/results.json`
- GDNの大prefill候補: 同一processの30回交互測定でserial/tiledは51入力1.970195/0.199531 ms、128入力4.868347/0.509104、512入力19.351330/2.518278、1024入力38.574986/5.529646。7/129/1024入力の独立FP64参照で出力・状態・分割復元が通過（1024出力最大絶対差2.05e-9以下）。全batchでtiledへ統一した別buildを実推論検証中。証跡 `gdn-prefill-micro.jsonl`, `prefill-gdn-quant.log`
- Expert INT8 XMXを512入力まで拡張した候補はprefill100.24〜102.49 tok/s（再実行）、同binary oneMKL対照は105.34〜119.24だった。6項目資料回答は全て正解だが、512で速い根拠が得られずエンジンの選択範囲を9〜128へ戻した。kernelの129入力・量子化4096行の独立検証は通過。2 MiB上限チェックを保持し、失敗候補binary/生ログは `build-prefill-q8`, `prefill-q8-512/results.json`, `prefill-q8-quant.log` に残す
- 短文prefillの明示候補 `--prefill-int8-xmx --decode-dp4a` を追加。denseはoneMKL/DP4A、Expertだけ同じGGUFを復号してINT8 DPASへ渡す。初回51入力はprefill5.75/decode8.07 tok/sと悪化した記録を保持。その後同一binary・各3回の対照は従来prefill6.26〜11.80/decode6.97〜14.18、候補prefill44.98〜79.54/decode15.68〜26.89。実測中にSysmanが700 MHz＋thermal flag8を返したが全差の原因とは断定しない。全6出力は独立素数判定10,021ケース合格。流用した資料JSON判定器がraw結果にfalseを付けたのは対象違いで、`short-xmx-paired/prime-validation.json`に独立コード検査と訂正を保存した。9型27標本・17行の独立数値検査、native/API10テスト、Pythonコード3課題も合格。3課題はprefill62.33〜88.13、decode14.84〜28.04 tok/s。証跡 `short-xmx-quant.log`, `short-xmx-bench`, `short-xmx-paired`, `short-xmx-integration.log`, `code-quality-short-xmx`。512入力への拡張候補は2 MiBのQ8入力scratch上限と4096入力行境界を検査しながら別buildで評価中
- 追加検証: GDNの4分割候補は独立FP64・分割/復元検査に合格し、同一process内100回の交互測定でT1は0.087506→0.015351 ms、T4は0.206257→0.026645 ms、T8は0.358890→0.042052 ms。これはcache-hot kernel単体値であり、全体の倍率ではない。Expert INT8 DPASと組み合わせた短primeは29.91 tok/s、66/66受理。native/API10テスト通過。証跡 `gdn-micro.jsonl`, `gdn-tiled-code`, `gdn-tiled-integration.log`
- 明示的なINT8 KV候補を追加（`--kv-type int8`、Intel既定は引き続きfp32）。32要素ごとのFP16 scale・int8符号、FP32 attention accumulation。小batchは直接Q8参照、大prefillは全layer共通の1組のFP32 scratchへ展開する。全layerのFP32 KVを併存させない。非有限値/scale overflowはerror flagで検出。独立量子化境界・offset書込・FP64 attention検査、FP32 MTP0とINT8 MTP3のnative/API各10テスト通過。短prime29.83 tok/s。最新長文は4K prefill140.91/decode15.24、16K prefill115.29/decode8.89 tok/s、両方6項目正解。単発の高値と混ぜず性能変動も保持。証跡 `kv-int8-quant.log`, `kv-new-fp32-integration.log`, `kv-int8-integration.log`, `kv-int8-long/results.json`
- Cコード比較の訂正/追加: llama.cppで`n_probs=1`と実生成ID返却を有効にした再実行は390出力・22.81 tok/sだったが同じsort試験に失敗した。最初の成功を消さず両方保持する。実生成IDで最初の分岐は183であり、復号テキストを再encodeして得た50を実ID分岐として扱った説明は訂正した。`merge-first-divergence-actual.json`が更新記録。別の固定3課題（Python binary_search 5100件、merge_sort 100件、topological_sort 32 DAG＋3 cycle）はnative FP32・Q8 MTP3・独立llama全て合格。GDN更新版もnative両経路の計6課題を再確認した。証跡 `code-quality/results.json`, `code-quality-gdn/results.json`。Cの失敗と広い採用検証の未完了は維持する
- **未解決のコード品質差を保持する**: 同じ66入力ID・同じGGUF・greedy／思考を閉じたC99 merge_sort要求で、Intel因果Q8・MTP3・MTP7と純FP32因果版の生成コードは独立sort試験に失敗した（配列の書き戻しが不正、trial4 n116）。llama.cpp b12360 SYCL／F16 KV／全層GPU／同じ66入力IDは425出力・20.53 tok/sで132 sort cases＋負長no-op試験に合格した。Intel純FP32版は387出力・9.86 tok/sでも失敗したためQ8/MTP固有とは断定しない。最初の分布は近いが同じではなく、共通prefixの数値経路と実際の生成IDを追加比較中。token列の違いだけで原因や全体の劣化を断定せず、この実行可能コードの不合格を隠さない。速度だけで候補を採用しない。証跡 `merge-sort-comparison`, `merge-sort-independent`, `merge-first-divergence.json`, `merge-divergence-independent`
- MTP7までの8行verify候補を追加（既定0、chunk>=max(4,depth+1)）。独立FP64に対する3/4/7/8行と64行Q8入力境界、native/API10テストが通過した（`mtp-eight-quant.log`, `mtp-eight-integration.log`）。primeは89出力IDが因果FP32と一致したが、7候補は75/91受理・23.35 tok/sでMTP3より遅く、深くすれば速いとは扱わない。Expertを2行ずつ処理する候補は同primeで27.46 tok/sだが広い品質採用検証は未完了
- 小verify batchの並列KV走査候補はnative/API10テストと4K/16Kの6項目JSON照合を通過。今回の観測は4K prefill175.41 / decode11.35、16K prefill196.93 / decode9.83 tok/s（MTP3）。短primeは20.58と下振れし、単発の27台を持続性能の保証にしない。証跡 `mtp-small-attn-long/results.json`, `mtp-small-attn-quant.log`, `mtp-small-attn-integration.log`
- upstream再確認: Strata local/remote HEADは `13eedf552417a6a791a7d13c6597e24300ee3a09` で一致した。現行INTEL_ARC.mdにもnative Windows engineなし・setupのWindows SYCL拒否が残る。上流にQwen3.6 Windows経路が既にあると仮定しない
- Qwen MTP初期移植: 明示 `--mtp 1..3`（既定0、chunk>=4）で同じGGUFのblk.40を使う。promptは前位置のmain正規化hiddenでdraft KVをwarmし、主モデルの因果windowを全件計算、target rowから既存samplerで順に選んで不一致/terminalで終了する。破棄行でRNGやpenalty履歴を進めない。部分acceptはGDN/conv開始状態を戻して保持行の保存入力を再実行、受理draftのMTP KVはmain hidden seedで次回修復する。取消時にmain hidden seedも復元する
- IQ2/3の先読み再試験は、失敗版の動的GRF間接参照を除き、固定領域のコード/sign展開へ変更した。独立FP64 batch/grouped検査が通過し、MTP3の同じprime生成89出力は27.10 tok/s、4K資料はprefill245.42 / decode13.37 tok/s・51候補中48受理。前grouped版の4Kはprefill204.76 / decode12.79で全6項目正解、grouped版API/native10テストも通過（`mtp-group-integration.log`）。直接IQ版の品質拡張と、512以上の位置で短いverify batchを並列KV走査にする経路は継続検証中
- 51入力のprime関数生成はFP32因果参照と89出力IDが一致、66/66 drafts受理。FP32 verify版15.08、4行Q8整数verify版22.84、同Expertの重み読み込みを共有する版25.43 accepted decode tok/s。採否を短いコード例だけで決めず30〜40目標未達を維持する。FP32版・Q8 batch版はそれぞれnative/API10テスト通過（`mtp-integration.log`, `mtp-batch-integration.log`）。Q8 batch/groupedの3/4行は独立FP64参照で確認（`mtp-batch-quant.log`, `mtp-group-quant.log`）。grouped版のAPI/長文反復・性能profilingは継続中
- 重みmetadataのブロック先読み候補は採否を分ける。実GGUF＋同じGPU上の40回×3計測ではQ6_KのQKV投影が約0.30〜0.39→0.15 ms、語彙投影が約6.64→5.03 msに短縮。一方IQ2_S Expertは約0.116→0.187 ms、IQ3_Sは約0.229→0.263 msに悪化したため、IQ2/3のレジスター間接参照候補だけ撤回しQ6_K先読みを残す。単体kernel時間であり、その比率を生成tok/sの上昇と同一視しない。元の回答テストは両方で通過。証跡 `micro-control.jsonl`, `micro-iq.jsonl`, `dp4a-matched`, `iq-block-512`
- 速度変動の調査ではLevel Zero Sysmanの実行中周波数にcurrent-limit（bit2）、thermal-limit（bit3）が現れ、一時100 MHzも記録された。これは個々の低下の原因をすべて特定した意味ではない。周波数domain0はGPU、範囲100〜2500 MHz、heartbeatは約15秒のまま継続。WindowsのSysman温度sensor列挙は0件で実温度は未取得。安全制限を変更していない。SDKの `ze_bool_t` はuint8であり、初期の補助properties構造体を修正した。周波数state構造体にはboolがなく、記録したstate/flagのレイアウトは変更していない
- 2026-11-14の実機進捗: oneMKLの有界FP16行列作業領域＋Q8入力ESIMD DP4A候補は512入力のwarm replayでprefill194.98 / decode16.28 tok/s。重みGGUFは不変、KV・GDN状態はFP32。Q8入力は明示候補であり、元のFP32入力と同じ演算と説明しない。全9型27標本の独立参照と通常native/API10テストは通過したが、CUDA/HIP同等の広範な品質採用検証は未完了
- FP32 oneMKL Attention候補（QK/PVの乗算もFP32、compute_mode standard）は独立FP64参照とnative/API10テストを通過。chunk1024では4096入力249.98 / decode10.78、16384入力199.04 / decode7.74 tok/s、両方とも資料の6項目JSONが正解。chunk512の4096入力は137.94 tok/sで旧経路より遅かった。最速値を全バッチ・全入力へ一般化しない。証跡 `blas-attention-long/results.json`, `blas-attention-quant.log`, `blas-attention-integration.log`。生成は30〜40 tok/s目標未達でありIssue未完了
- 追加の性能目標: ユーザーは参照実測20〜30 tok/sに専用実装で約10 tok/s上積みする30〜40 tok/s級を希望する。prefill 100 tok/sは最低線であり最終目標ではない。Issue記載のQwen＋MTP 42.5 tok/s参考値も保持する。[358H/B390のWindows実測記事](https://jishuzhan.net/article/2109547317837979650)はQwen3.6 UD-Q3_K_M／Q8 KV／SYCLで生成21〜24 tok/s・prefill約97 tok/sと報告。[Micro Perceptronの32GB実測](https://microperceptron.com/os/benchmarks/inference-32gb)は同機Qwen3.6 NVFP4＋MTPで42.5 tok/s・4.12 tokens/stepの単発画面値。モデル・KV・MTP・OS差を残しつつ、現行低速実装の採用を正当化しない
- ユーザー指定で `gufo-org/gufo` を `C:/Users/user/git/gufo` にclone、参照revision `e8f2fe0e066faa658972b835ee53eaf8bcd23585`。16要素単位の復号・scale factoringをIntelのdecode候補へ移植し、MITの出典とlicenseを保持する。Gufo自体は変更していない
- ユーザー指定の `ryugyosoft/onwx-dash` はGPU＋XDNA2 NPUのprefill分担方式（参照revision `63302253e785539eebe133302f8dd3c339008f85`）。IntelでAMD XRTを動かす想定はせず、同作者の `onw` (`3fc5ec770d0bad1dc891686036389eb6370b4268`) とOpenVINOを参照してIntel NPUへ分担する候補を実装した。OpenVINO 2026.4.1は私有venvだけに導入し、OSドライバーを変えない。全モデルをNPU用に複製せず、実行時重みの投影グラフ＋有界USM作業領域を使う
- NPU候補の内部MatMul結果はFP16に丸められる。外側の出力tensorがF32でも内部FP32精度とは言わない。最初の数値検査はこの点を想定せず失敗（0.660730033647→0.66064453125）。丸め項を明示した独立FP64参照で全9型27標本のGPU→NPU→GPU経路を確認した。回答品質の採用検証とは別に扱う
- 2026-11-14のユーザー追加指示: 参照は[Strata Coder ticket #2412](https://github.com/Niko1221/Strata/issues/2412)。全Expert 23.42 GiB常駐、MTPありの短コード28.6 tok/s／約2K入力のコードレビュー21.7 tok/sが出ており、現在の因果Qwen約6〜10 tok/sは性能面で受け入れ不可。prefillも3桁（最低限100 tok/s以上）が必要。動作するPoCだけでIssue完了とせず、実用速度と品質の両方を満たすまで実装・検証を継続する。CoderとQwenのモデル/MTP差は条件として残すが、差を理由に現在の低速実装を採用しない
- 補助的な電源対応や途中質問で主目的を失わない。1:48:57にIssue未完了のまま最終応答を出してターンを終了したことが、その後作業が進まなかった直接原因。1:20:09〜6:55:46のKernel-Power 507詳細は `SleepEntered=false`, `SleepDurationInUs=0`, `DripsTransitions=0`。この停滞をOSの実スリープと断定した説明は訂正済み。私有証跡 `power-20261114/repeat-sleep-events.xml` とローカル実行記録で照合した
- [issue #11](https://github.com/kbc20f11/TinyStrata/issues/11)の初期PoCを専用ブランチ `feature/intel-uma-windows` に追加。実機はCore Ultra X7 358H / Arc B390（PCI b082、SYCL実アーキテクチャはPTL-H / Xe3）/ 32 GiB LPDDR5X-7467 / Windows 11 Home 26200 / driver 32.0.101.9175。Bという製品名からXe2へ分類しない
- CUDA/HIP既定とnative実装は変更せず、明示的なSYCLビルドと `python -m serve.intel` だけを追加した。標準Qwen3.6の因果テキスト経路が動く初期実装であり、Issue全体の完了や正式Intel対応とは扱わない。MTP・画像・Tiel・Clover・Gemma・LFMは未対応。KVは明示したFP32であり、CUDA/HIPの採用済みFP8/INT8設定を変更した意味ではない
- 全753テンソルを圧縮状態の単一USM領域へ置き、metadataのmmapを閉じてから確保する。host/sharedはファイルから直接読み、deviceは8 MiBの一時stagingのみ。CPU常駐重みとGPU Expertキャッシュを二重に保持しない。物理RAM・commit・対象WDDM予算を制限にし、共有GPU容量をRAMへ足さない。4 GiBのOS余裕を保持する
- host/shared/device各方式で64 MiBと4 GiB超のアドレス・可視性検査を各6回、実GGUF全体の読み込み・GPUの標本バイト照合・解放を確認した。USM方式名だけから転送ゼロや最速と断定しない。転送カウンターとプロセス/RAM/WDDMを別々に記録する
- RMS・行列積・GDNは倍精度CPU参照と照合。GGUF内の9型・27テンソル標本の量子化復号と行列積を独立NumPy/gguf参照でも確認した。数値検査成功だけで全入力の回答品質同等とは言わない。分割入力と状態復元も検査した
- 実nativeの継続・編集・不正要求・STOP復帰・seed再実行、通常HTTP処理の日本語資料計算（6400円）、SSE、get_weather(Tokyo)の構造化ツール呼出を確認した。最初のAPI起動はtokenizer抽出関数が内部で `/tokenizer` を付ける点を取り違えて失敗し、呼出先を修正した。画像要求は未対応として拒否する
- ユーザーの指摘を受け、電源設定とKernel-Powerログを調査。通常の自動sleepは0だったが、実効LIDACTIONはAC/DCとも1（sleep）、UNATTENDSLEEPは120秒。LidとIdle TimeoutのModern Standby記録が繰り返し存在した。初期の測定値はsleep影響を除外できないため `pre-power-fix` に保存し、速度採用の根拠にしない
- ACのBest performance overlayを実効GUIDで有効化し、ふた・無人sleep・通常sleep・自動休止を0へ設定。画面消灯はユーザーの希望どおりAC300/DC180秒を維持する。表示上の基本プラン名Balancedと実効performance overlayは区別する。最新Windows固有の不具合やユーザーの設定ミスと断定しない
- nativeのloading/inference中にSystemRequiredとExecutionRequiredを保持する。DisplayRequiredは取らず消灯を許可する。画面消灯に伴うModern Standby開始イベントだけで実処理停止と判定しない。01:20:09の開始イベント後、resumeイベントがない間も実機の10テストと続くコード生成が完走した。電池動作や手動sleepへのOS仕様上の制限は残る
- [実装・再現手順・未対応範囲](intel-windows.md)。私有証跡は `tinystrata-work/intel-20261114`、電源変更前後と一時的な実行継続要求は `tinystrata-work/power-20261114`。ドライバーを置換せず、oneAPI/MSVCの開発ツールを導入した。CUDA/HIP実機による再推論は今回行っていない

## Qwenの公式既定値とUIの設定表示

- ユーザーの追加指示で、Qwenの新規・省略時の既定を公式の思考あり一般用途（temperature 1.0 / top_p 0.95 / top_k 20 / min_p 0 / presence_penalty 1.5 / repetition_penalty 1）へ変更した。以前の「0.7を維持」はこの指示で更新する。Tiel・Gemma・LFM・CloverとKV/MTPは変えない。非思考や精密コーディングへの自動切替を実装した意味ではない
- 元の同一663入力・同一ツールをモデルID追加なしで再送すると、FP8の温度1.0は410出力・3.3秒で正しいQwen紹介となり、不要なツールもなかった。INT8の1.0は1756出力・18.6秒で一般的なAI紹介。0.7のINT8によるGoogle誤回答も含め、[全条件と限界](qwen-ui-model-identity-20261112.md)を保持する
- 以前のUIは1.0/.95/20を既定の目印として省略していたため、表示1.0でもサーバー0.7が適用された。数値の目印を初回移行だけにし、model/customの保存状態で区別する。モデル既定の表示は実APIの `/props` から取得し、スライダーを操作した1.0は明示送信する。reset・遅延応答・モデル切替・取得失敗後の再試行でも不一致を作らない
- 保存済みAPI設定の0.7は過去の自動値と明示選択を判別できないため全利用者へ強制移行しない。このPCのQwenプロファイルは今回の指示で1.0へ明示更新し、他の5保存プロファイルのバイト一致を確認した。ブラウザーの既存の明示カスタム値も保持する。過去ベンチと固定v1のsamplingは測定条件として保持し、現在の既定値へ書き換えない
- 実モデルの実UIで、Qwen既定表示1.00・通常の自己紹介556出力/約5.4秒、サーバー既定0.7に対する手動1.0の明示送信、LFMの0.1/1.0/50への切替とQwen復帰、取得失敗時の—表示と再試行を確認した。4812入力の訂正付き資料も約6.4秒・1回の生成で8項目すべてが独立正解と一致し、外部検索・ツール再生成はなかった。ライト・ダークの画像を目視確認した
- FP8の限定的な数値検査も実施。129行の共通履歴、第一行FP32、独立llama F16の一部行、現行CUDAと同じ保存FP8のCPU FP64比較（521/663行）は完了した。カーネル演算の検査成功と回答品質を混同せず、FP8破損の証明や全入力での同等性とは扱わない。証跡は `TinyStrata-work/qwen-fp8-numerics-20261113`、既定値・UI・ローカル設定バックアップは `TinyStrata-work/qwen-official-chat-20261113` に保管する

## Qwenの自己紹介質問とUIの実行モデル情報

- UI既定で `あなたはどのようなモデルですか` を送ると、Qwenは最初の5,974トークン／51.1秒で自己同定の長考と無意味なPython printを行い、最終回答まで計6,677トークン／約59.2秒を要した。永続停止ではない。近い別文面は約1.2秒で終了した
- MTP0でも同じ長考・ツール呼出が再現した。INT8では4.9秒で終了したが「Google製」と誤回答したため、MTP停止やINT8への既定変更で解決したと扱わない。採用済みFP8・MTP3は維持する。再現時の実効温度は0.7だったが、2026-11-13の追加試験とユーザー指示によりQwenの既定を公式の思考あり一般用途の温度1.0へ変更する
- [公式llama.cppとの比較](qwen-ui-model-identity-20261112.md)を実施した。同GGUF・元UIのsystem/user/toolsを公式b12318へ渡すと681入力／417出力／4.8秒で正常終了。ツールwrapper差を除いた完全同一663入力IDでも260出力／2.9秒で正常終了した。ただしRNG・sampler順・KV形式等が異なり、Tinyの特定kernel破損やモデル固有の限界とは断定しない
- UIだけに、検証済み `/health` が返す実行モデルIDを事実として1行追加した。名前・開発元・能力を推測しない。質問文の特別扱い、回答固定、ループ検出による強制打ち切りは追加しない。API既定やnative経路にも注入しない
- モデル情報は各requestの `body.model` と同時にsnapshot化し、計数・圧縮・旧API fallback・ツール再実行・手動KV圧縮で統一する。古いチャットのモデル名や、外部API切替に遅れたピッカーからidentityを作らない。bootstrapのstrataや不正なhealthでは省略する
- 実UIの同質問は修正後576出力／4.8秒で正常終了し、再送でも本文・思考が一致した（2回目はprefix cache再利用）。通常のPython二乗和333383335000と、Qwen→Gemma後のモデル情報・最終回答も確認した。Blackwell実機での確認であり、AMDを再試験した意味ではない。根本の推論器間の差が解消したとは報告しない
- 比較・リクエスト・SSE・スクリーンショット・設定保持記録は `TinyStrata-work/qwen-ui-identity-20261112` に保管する。公式llama配布物は私有フォルダーのみへ置き、グローバルCUDAやドライバーを変更していない

## Blackwell 24／16／12／8 GB再測定

- 2026-11-12、RTX PRO 4000 Blackwell SFF 24GB／driver597.30で現行5モデル×4容量の20起動を各1回測定した。実装ソースは `811bd9d`。生成は3課題の実生成384トークン÷総decode時間で計算し、除外warmup・4K／16K入力・対応4モデルの画像も含めて全条件が正常完了・graceful unloadした
- KVはQwen/Tiel/LFMがFP8、Gemma/CloverがINT8。Cloverは因果D0＋GPU Vision。全モデルのchunkをautoに揃え、保存されていたCloverの固定2048だけ私有設定で上書きした。容量別contextはQwen/Gemma/Clover/Tielが256K／256K／128K／64K、LFMが128K／128K／128K／64K
- 容量制限は実CUDA割り当て・全域touch・要求前後のNVML確認で行った。今回の予約量は「観測した総VRAM－16／12／8 GiB」であり、過去の固定8／12／16 GiB予約と厳密に同じ量ではない。24GB条件に予約はない。物理的な小容量GPUの帯域・演算器を再現した測定とは扱わない
- 各条件1回のmanaged-default容量別variantとして[新しい結果](../benchmark/results/rtx-pro-4000-blackwell-sff-20261112.md)とJSONを保存した。過去の4起動・KV直接比較表は測定日と条件を保持し、今回の数値で置き換えない。8GB Clover INT8は今回1/1完走であり、以前の3/4成功を解消した長期安定性の証明とはしない
- Tielの24GB入力処理がQwenより遅いのは、少なくとも既存のIQ3_XXS down（type18）向けMMQ精度保護によるfallbackを含む。Tielの主層37個はtype18で、反復入力で出力差を検出した既存guardがmain MMQを拒否する。Qwenの同名UD量子化とは各tensorの型が異なる。今回この保護や推論経路は変更していない
- 全20条件の実生成数・入力4096／16384・終了状態を別スクリプトでも照合し、保存済み全6プロファイルのハッシュ一致と予約の解放を確認した。実測・生ログ・独立照合・native再利用の現行ソース一致記録は `TinyStrata-work/blackwell-remeasure-20261112` に保管する
- [issue #1への返信](https://github.com/kbc20f11/TinyStrata/issues/1#issuecomment-6824069269)も投稿した。元のRX9070XTで、verified=1／64KiB／runtime=compiled71726391の短い再現試験が成功した報告を確認。8K全量確保済みでKV増減や長期利用を証明していない。issueは開いたままで、#10の映像出力制限と4MiB回復策は維持する

## モデル管理：未取得Cloverと共有画像ファイル

- CloverとQwenは画像用の `mmproj-F16.gguf` を共有する。Clover本体が未取得でも `disk_bytes` に共有分が含まれるため、これだけで削除ボタンを表示しない
- 未導入モデルの削除表示は `deletable_bytes > 0` または `can_delete=true`（0バイトの所有 `.part` も含む）で判定する。導入済みモデルのactive/busy等の削除ロックは維持する。hardlinkは解放量0でも所有名を削除できるので `reclaimable_bytes` を表示判定に使わない
- 共有ファイルだけならダウンロードのみ、所有する中断ファイルがあれば削除可能。私有の実ModelSwitcher/API/UIをPlaywrightで操作し、共有のみ→ローカルHTTP転送→中断→所有 `.part` の削除→共有のみ、Qwen本体と共有ファイルの保持を確認した。ライト・ダーク・狭い画面のスクリーンショットも目視確認済み。GPU推論や実GGUFの再取得は行っていない
- Nodeのモデル管理回帰139チェックとPythonのモデル管理22テスト（1件環境依存skip）が通過。Python fixtureは生成後のcatalogでも実native探索を隔離する。初回は既存CUDAバイナリのcapability probeでタイムアウトしたため、テスト全期間にパッチを保持して再確認した。製品のbackend処理は変更していない
- 私有のUI試験・画像・ライフサイクル記録は `TinyStrata-work/ui-clover-shared-20261112` に保管する

## UIインストーラーの入口とモデル取得

- 新規導入・通常更新は同じ `powershell -NoProfile -ExecutionPolicy Bypass -Command "irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install-ui.ps1' | iex"` を使う。`-Update` は通常不要で互換用に残す。`irm URL` だけでは取得した文字列を実行しないため `| iex` は必要
- UI版は初回も更新時もモデルを自動取得しない。ユーザーがモデル管理でダウンロードし、ピッカーで選択する。API専用 `install.ps1` の明示的なモデル導入経路は維持する
- remote/archive入口は最新commitを一度解決し、installer本体とソースを同じrevisionに固定する。archiveはディレクトリごと置換し、同revisionでも内容の欠損・余計な旧ファイルを検出して修復する。Git作業ツリーは自動置換しない
- source/native指紋のソートは `StringComparer.Ordinal` に固定する。`Sort-Object -CaseSensitive` はWindows PowerShell5.1とpwsh7で順序が異なり、同じ入力でも再ビルドを誘発した
- native入力・ビルド方針・実行ファイルのハッシュが一致する検証済みgenerationだけ再利用する。旧stampなしバイナリを自動で承認しない。今回の私有E2Eキャッシュは過去の実ビルド・実機検証記録と169 nativeファイルの一致を別途確認したもの
- 更新はinstallation guardを取ってから所有サービスを確認し、受付済み処理・検索/ツール子プロセス・ダウンロード中は停止しない。新しいAPIは更新中の新規生成/モデル変更を拒否し、statusとunloadだけ継続する。旧APIには新しい受付制御がないため、互換経路は複数回のidle確認とgraceful unloadを使う
- UI/APIの所有確認はコマンド・作成時刻・source/work・ポート・親子関係を使う。Windows venvでは実際の待受PIDが起動PIDの子になり、argv[0]もbase Pythonへ置き換わる。uv3.12の実環境で旧判定による誤失敗を再現し、検証済みredirector子プロセスを受け入れるよう修正した。失敗した環境からの通常再実行も成功した
- デスクトップショートカットはUI起動前に作り、`-NoStart`と起動失敗でも残す。実KnownFolder・日本語/OneDriveパス・私有Pythonを使い、モデルを固定しない。起動失敗は具体的理由・今回のstderr/終了コード・ログ先を示す
- 同一vendorの管理CUDAプロファイルも、現行generationの標準siblingと管理Vision/DLL参照へ揃える。モデル・tokenizer・sampler・KVと明示的なcustom exeは保持し、参照だけでは設定を書き換えない
- 起動元と現在モデルの認証・接続設定が異なる場合は、認証を変更せず停止前に拒否する。認証情報をsnapshotやエラーへ出さない。準備失敗のrollbackと、publish後の起動失敗を区別する。後者に旧source/venvへの完全な巻き戻しを保証していない
- 私有の新規環境でモデル0件・Qwen未取得・NoStart/ショートカット作成、通常UI・最初のLFM選択・会話保存・再実行でのSQLite論理内容とモデル選択維持を確認した。モデルは検証済み実GGUFのキャッシュfixtureを使い、巨大ファイルのネットワーク再取得は行っていない
- モデル0件ではダウンロード案内と選択placeholderを表示する。最初の非Qwenモデルが一つだけある時も、DOMの自動選択を実モデル切替と誤認させない
- 私有UI試験のLFMは数量×単価の問題で4800円と誤回答した。失敗を保持し、金額14400円を明記した別資料の抽出は正しく返した。これはインストーラーの内容照合成功として混同しない
- `.lnk` 経由の自動起動はツールの自動承認で拒否された。実リンクの作成・参照先・アイコンと正式ランチャーの直接起動は確認し、クリック起動を確認済みとは扱わない
- 調査・初期失敗・再試行・CPU検証・実UI記録は `TinyStrata-work/installer-unified-entry-20261112` に保管する
- 通常インストール先で公開URLの標準コマンドを `-Update` なしで実行し、archive置換・UI/API復帰・Gemma loaded状態・実Desktopの `.lnk` 再作成まで確認した。同じコマンドの再実行も終了コード0（約23秒）、モデル転送・native再ビルドなし。現状UI更新の確認ではサービスを再起動するため、無再起動とは説明しない
- 更新前後で全6保存プロファイルの内容、検索キーとUI設定のバイト、SQLiteの全論理内容（22チャット）・integrity_check=okが一致した。API/UI3000/18080をPlaywrightで開き、Gemma選択・入力可能・横はみ出しなしも確認した
- 本番経路ではさらにStore版Pythonの実イメージとAppExecutionAliasの違いを検出した。inspector自身の実イメージも照合候補にし、別の `python.exe` は拒否する。`FixedModelDownloader.running` は真偽値のpropertyであり関数ではない。実managerを使うhealthテストを追加し、補助fixtureのmethod形式と混同しない
- 初回公開試験のイメージ照合拒否、次のダウンロード状態型の不具合・起動待ち失敗も記録に残す。後者はローカル修正と所有APIの復旧後に公開修正版を再試験し、保存設定・履歴の一致まで確認した

## KV形式の採用条件

ユーザーはモデル別の比較結果を踏まえて採否を決めている。GemmaのFP8が実験的だからといって、他モデルもINT8へ統一したり、全モデルを対象に方針を再質問したりしない。

| モデル | 採用済みの既定・扱い |
| --- | --- |
| Qwen3.6 | 対応GPUではネイティブFP8、非対応ではINT8。既存の比較結果に基づく選定を維持 |
| Gemma 4 | **INT8が既定。FP8は品質面の問題があり、明示指定時だけの実験的オプション** |
| LFM2.5 | 対応GPUではネイティブFP8、非対応ではINT8。既存の比較結果に基づく選定を維持 |
| Clover | READMEに記載された新規管理プロファイルは因果D0・INT8。ネイティブ共通経路の `auto` や既存ローカル設定とは区別する |
| Tiel Coder | Qwen系の採用済み経路。対応GPUではネイティブFP8、非対応ではINT8 |
| GPT-OSS（対応終了） | 採用時はINT8が既定。過去の長文品質比較もINT8で固定。再追加は専用の受け入れ基準に従う |

Gemmaの `auto` はGPUに応じてFP8かINT8を選ぶ意味ではなく、常にINT8に解決される。前回「GemmaはFP8」と説明したのは説明側の誤りで、旧版・新版の調査記録はいずれも実際はINT8だった。

根拠:

- [日本語README](../README.ja.md): モデル別の自動KV表、QwenのFP8/INT8直接比較、Gemmaの「INT8（既定）／ネイティブFP8（実験的）」、LFMのFP8既定のベンチ表
- [Gemma実装](../src/gemma4/gemma4.cpp): `kAuto` を `kQ8_0` へ解決
- [対応終了前のGPT-OSS実装](https://github.com/kbc20f11/TinyStrata/blob/d93a48ff792a25343778dba57c96fe8eda08bd97/src/gemma4/gptoss.cpp): `kAuto` を `kQ8_0` へ解決
- [前回のNVIDIA再検証](../benchmark/results/cuda-revalidation-20261112.json)

保存済みプロファイルをベンチのために書き換えない。明示した形式と実際の形式を両方記録し、`auto` を実測形式として報告しない。Cloverの管理用既定と、ユーザーの既存ローカル設定も混同しない。

## 思考強度

- UIの初期値は、思考の切替だけのモデルはオン、程度を選べるモデルはmedium
- GPT-OSSの正式な段階はlow / medium / high。`xhigh` はTinyStrataではhighの別名で、独立した上位段階ではない
- 実際のインストール済みテンプレートでもhighとxhighの入力トークンが一致することを2026-11-12にCPU検証した
- [OpenAI公式モデル仕様](https://developers.openai.com/api/docs/models/gpt-oss-20b)、[Harmony形式](https://developers.openai.com/cookbook/articles/openai-harmony) を確認済み
- highにしただけで品質が改善すると断定しない。回答の必要事項、生成打ち切り、ループ、処理時間を実モデルで比較する

## NVIDIA再検証と継続調査

- 意図的な精度低下、選択されたExpertの省略、近似採用を速度のために追加しない
- GPUごとに必要な経路を分け、CUDAの修正をHIPへ無条件に適用しない
- 前回、Gemmaの生成は12GB相当で改善したが、4K入力処理は約7～10%（約0.1～0.17秒）遅い結果が2回出た
- 同一実行ファイルで8枠／32枠だけを変えた比較では、この4K入力低下は再現していない。32枠化やFP8が原因と決めつけない
- GPT-OSSの長文では、出力破損の修正後も必要条件の脱落が残る。正常終了・正常な日本語と、資料に対する正確さを別に判定する
- Gemmaの4K対照試験は2026-11-12に完了。同一962 Expert・全配置ID・MMQ実行形状・処理行数を揃えたINT8比較で、旧1836.8→新1834.1ms、再実行は旧1814.3→新1753.7msだった。安定した7～10%のコード回帰は再現せず、現行経路を維持する。計測用コードの影響があるので、この数値を新たな高速化率として宣伝しない
- 過去の低下の厳密な原因は未特定。キャッシュ・実行履歴を含む条件依存の観測として保存し、温度やFP8が原因と断定しない
- GPT-OSSの追加調査は2026-11-12のユーザー指示で終了。速度・品質・保守負担を総合して対応を終了し、未検証のHadamard候補などは取り込まない
- 過去の失敗記録を残し、後の成功や都合のよい1ケースだけで置き換えない

## GPT-OSS対応終了と再提案

- 取り込み後の最適化・不具合調査・検証に、ユーザー側でほぼ丸二日を要した。投稿者側の作業時間は不明であり、比率や時間差を捏造しない
- 軽量用途はLFMとの重複が大きく、Qwenなど既存モデルに対する十分な利点を確立できなかった。全条件で必ず遅い、モデル自体が本質的に不正確、といった主張には広げない
- INT8・温度1の長文では条件・数量の脱落が残った。highは8K上限で最終回答がなく、16K上限では9,195トークン・約123秒で終了したが作業前提の脱落が残った。温度0の反復と、温度1の確認の長期化・上限到達を区別する
- 同じ重みを使った独立したllama.cpp比較ではmedium/highとも必要事項を含む回答が得られた。TinyStrata固有の数値経路・統合の未解決事項があり、モデルの限界と断定しない
- モデル重み、保存済みプロファイル、チャット履歴は削除しない。共有Gemma実装や他モデルの採用済み高速化を無差別に巻き戻さない
- 今後のGPT-OSS再追加PRに限り、[専用の受け入れ基準](gpt-oss-reintroduction.md)を満たす実装と再現可能な比較結果が揃うまで受け入れない。他モデルを含むPR全般への新たな制限ではない
- [取り込み元PR #14への返信](https://github.com/kbc20f11/TinyStrata/pull/14#issuecomment-6811015376)は投稿済み。PRはマージ済みなので再オープンできない
- [issue #13の不備一覧](https://github.com/kbc20f11/TinyStrata/issues/13#issuecomment-6811071913)は、後から修正した初期の不足・未解決事項・未検証範囲を分けて記録した
- 現行5モデルのnative経路にMXFP4は不要。GPT専用12コンパイル単位と、Gemma/LFMの未使用type39特殊化を撤去した。共有Gemmaの画像入力・Unicodeパス・32組処理は保持し、Qwenなど別モデルの形式や汎用GGMLの型定義は削らない
- 撤去版はCUDAのGemma/LFMを再ビルドし、HIP gfx1200の変更関連8ファイルもコンパイルした。HIPはコンパイル確認であり、今回Radeon実機での再推論を行った意味ではない。インストール時間の短縮率は未測定
- 撤去版の通常インストール先・UI3000/API18080で、モデル5件の表示とGemma→LFM→Gemma切替をPlaywrightで操作し、モデル管理のスクリーンショットも目視確認した。Gemmaの画像A=赤い正方形/B=青い円/C=緑の三角形、短文資料と5,360トークンの訂正文書の8項目を原資料と照合した。LFMも同じ短文8項目に一致した
- LFMは最初の768トークン上限では思考のみで打ち切り、4,096上限では1,511トークン・約9.48秒で最終回答を返した。これを無かったことにしたり、旧版との比較なしに今回の撤去による回帰と決めつけたりしない。最初の長文fixtureも想定4Kに対し実際は3,480トークンだったため、別記録の5,360トークン資料で確認し直した
- GPTの3 API形式・切替・ダウンロードはロード前に拒否。通常UIの確認後も検索キーのバイトとSQLite履歴の論理内容は一致し、各モデル設定に追加変更はなかった。SQLiteファイルのバイトは起動時のcheckpoint等で変化し得るため、論理一致と区別する
- Python441テスト（環境依存25スキップ）、Node UI11本、PowerShell installer入口112チェックが通過。最初に見つかった古いcoreへのUI追加更新の依存不具合は修正して再確認した。インストーラー全体の再ダウンロード・全再ビルドを行った意味ではない
- 私的な実行記録・失敗と再試行・SHA・スクリーンショットは `TinyStrata-work/gptoss-quality-gemma-prefill-20261112` 内の `retired-native-build`、`hip-retired-native-compile`、`retirement-*`、`normal-ui-*` に保管した

## 品質判定の注意

- 自動チェッカーの失敗と、原資料に反する事実誤りを同一視しない。実際の最終回答・原資料・質問の項目定義を照合してから報告する
- 担当者の「山本→Yamamoto」のようなローマ字化は、原表記の不一致であり、それだけで別人への変更とは言えない。原表記を守る要件と本人の同一性を分ける
- 「今回の不足調達費」と「資料にない作業費」は別の項目。発注不要の費用0と、未記載の作業費nullが混ざる曖昧なテスト契約で、モデルの捏造と決めつけない
- 元の自動判定・回答・入力は保持し、独立した人手の照合結果や契約上の曖昧さを別記する。生成結果を見てから正解を都合よく変えない
- 出力上限で最終回答がない場合、思考内の下書きを最終回答扱いにしない。同じ文章の反復、確認が長引いた状態、回答内容の誤りも区別する
