# TinyStrata

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語** · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

**TinyStrata は、30B 級の MoE モデル向けに開発している Windows 専用・API 専用の高速推論エンジンです。** VRAM が **8–24 GB** の一般的なグラフィックボードを搭載し、システム RAM が **32 GB 以下**の PC を主な対象としています。

## TinyStrata を開発する理由

PC パーツの価格が上昇するなか、ハイエンド PC を組むことは、多くの人にとってますます難しくなっています。

そこで TinyStrata は、RTX 3060 クラス以上のグラフィックボードを搭載した PC で、AI をローカルで高速に動かすために作られました。30B 級の比較的小さな MoE モデルを、できるだけ速く動かすことに注力しています。

主な対象は、**VRAM が 8–24 GB、システム RAM が 32 GB 以下**の PC です。この範囲は、グラフィックボードを搭載した PC で最も一般的な構成に当たると考えています。

TinyStrata は Strata を基盤に、限られた VRAM でも 30B 級の MoE モデルを高速に動かせるよう、GPU メモリの効率的な活用に取り組んでいます。これにより、長い会話や文書も手元の PC で扱える環境を目指しています。

開発は Windows 上でのローカル推論と API 提供に集中しています。

ハードウェアに応じて推論・メモリ設定を自動選択するため、複雑な設定や手動調整をせずに API を使い始められます。

## UI を同梱しない理由

LLM 向けの API に対応した UI やエージェントハーネスは、すでに数多くあります。TinyStrata も一般的な API 形式に対応しているため、好みのものを接続して使えます。接続設定が面倒に感じる場合は、AI エージェントに設定を任せることもできます。UI は既存のソフトウェアに任せることで、推論エンジンと API の開発に注力しています。

## 対象モデル

一般的なグラフィックボードを搭載した PC での速度・省メモリ性・回答品質のバランスを重視し、[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) を既定のモデルとしています。比較的小さな MoE モデルに注力し、**Gemma 4 26B-A4B-it と LFM2.5-24B-A2B にも対応しています**。

**Qwen3.6-35B-A3B UD-IQ3_XXS** は、固定の [Unsloth MTP GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf)を使います。ダウンロードは MTP の重みを含む言語モデル **14.1 GB** と、ビジョンプロジェクタ（`mmproj-F16.gguf`）**0.9 GB** の 2 ファイル、合計約 **15.0 GB** です。画像エンコーダの GPU 常駐、空き VRAM に合わせたエキスパートキャッシュ、システム RAM を使った CPU でのエキスパート計算、入力時の重みのストリーミング、容量を自動選択する最大 **256K のコンテキスト**、MTP 3 に対応しています。モデルが選んだエキスパートは省略しません。KV は **FP32・INT8・FP8** に対応し、自動設定では GPU とコンパイル済みカーネルが対応していればネイティブ FP8、それ以外では INT8 を選びます。MTP は greedy 生成と温度付きサンプリングの両方で利用できます。

**Gemma 4 26B-A4B-it QAT Q4_0** は、[Google 公式の本体・ビジョンプロジェクタ](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)と、対応する [Unsloth の QAT Q4_0 MTP アシスタント](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/7b92b5b28818151e8669af2e45e88d6086f490dd/mtp-gemma-4-26B-A4B-it.gguf)を使います。ダウンロードは本体 14.439 GB、プロジェクタ 1.195 GB、アシスタント 0.252 GB の 3 ファイル、合計 **15.886 GB** です。画像エンコーダの GPU 常駐、空き VRAM に合わせたエキスパートキャッシュ、システム RAM を使った CPU でのエキスパート計算、MTP 3 に対応し、共通の API から利用できます。モデルが選んだエキスパートは省略しません。KV は **FP16・INT8** に対応し、現在の自動選択は **FP16** です。画像入力・MTP 3・RAM 併用推論・起動中のモデル切り替えは実モデルで動作確認済みです。

LFM は公式の [LiquidAI Q4_K_M GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF)（**14.53 GB**）を使い、KV は **FP16**、最大コンテキストは **128K** です。MTP・Vision はありません。空き VRAM に合わせたエキスパートキャッシュ、CPU でのエキスパート計算、入力時の重みのストリーミング、CUDA Graph を利用し、モデルが選んだエキスパートは省略しません。既定値は `temperature=0.1`、`top_k=50`、`repetition_penalty=1.05` で、思考は常に有効です。設定は Qwen・Gemma・LFM で別々に保持します。LFM エンジンは検証中であり、llama.cpp や CPU・GPU の配置が異なる場合と同じトークン列になることは保証しません。

<details>
<summary>追加モデルの実行ファイルがない場合の手動ビルド</summary>

CUDA を導入した x64 Visual Studio 開発者ターミナルで、このリポジトリから実行してください。

```powershell
cmake -S . -B ..\TinyStrata-work\build -DTINYSTRATA_BACKEND=CUDA
cmake --build ..\TinyStrata-work\build --config Release --target tinystrata-gemma4 tinystrata-lfm2 --parallel 4
```

Gemma は `run-api.bat --model-family gemma4`（既定ポート 18082）、LFM は `run-api.bat --model-family lfm2`（既定ポート 18081）で起動できます。

</details>

## インストール方法

64 ビット PowerShell で次のコマンドを実行すると、TinyStrata をインストールし、`http://127.0.0.1:18080/v1` で API を起動できます。

```powershell
irm 'https://raw.githubusercontent.com/kbc20f11/TinyStrata/refs/heads/main/install.ps1' | iex
```

Windows を再起動したあとに再び起動するには、次のコマンドを実行するか、同じ `run-api.bat` をダブルクリックしてください。

```powershell
& "$env:LOCALAPPDATA\TinyStrata\run-api.bat" --port 18080
```

または、AI エージェントに [TinyStrata の GitHub リポジトリ](https://github.com/kbc20f11/TinyStrata)のリンクを渡し、導入を任せることもできます。

### 起動したままモデルを切り替える

インストーラーは既定で Qwen を起動します。同じ API サーバーで Gemma を使うには、まずダウンロードを開始し、進捗を確認します。

```powershell
$api = 'http://127.0.0.1:18080'
Invoke-RestMethod "$api/v1/model/download" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/download?model=gemma-4-26b-a4b-it"
```

進捗確認を繰り返して `state` が `ready` になったら、明示的に切り替えます。使用中でないモデルのダウンロードは準備だけを行い、自動では切り替わりません。

```powershell
Invoke-RestMethod "$api/v1/model/switch" -Method Post -ContentType 'application/json' -Body '{"model":"gemma-4-26b-a4b-it"}'
Invoke-RestMethod "$api/v1/model/switch"
```

初めから Gemma で起動する場合は `run-api.bat --model-family gemma4` を実行します。既定の API アドレスは `http://127.0.0.1:18082/v1` です。LFM を使う場合は、上のダウンロード・進捗確認・切り替えリクエストの `gemma-4-26b-a4b-it` を `lfm2.5-24b-a2b` に置き換えるか、`run-api.bat --model-family lfm2`（既定の API アドレス `http://127.0.0.1:18081/v1`）で起動してください。

`GET /v1/model/switch` で `state` が `ready` になるまで確認してください。Qwen に戻す場合は、同じ切り替えリクエストに `{"model":"qwen3.6-35b-a3b"}` を指定します。ダウンロード・切り替えの開始時は `202`、生成中の切り替え要求は `409` を返すので、生成終了後に再試行してください。切り替え中の推論リクエストは `503` になります。ポートと認証設定はそのままです。モデルの選択は今回の起動中だけ有効で、再起動時はランチャーで選んだモデルを使用します。

## 実測速度 — Qwen3.6

次の測定は、2026-10-19 に Windows、RTX PRO 4000 Blackwell SFF Edition（24 GB）、Ryzen 9 7945HX3D、固定の Unsloth MTP UD-IQ3_XXS モデルで行いました。画像エンコーダを GPU に常駐させた通常の API 起動で、テキスト生成の速度を測定しています。

思考有効・温度 0.7 などの既定サンプリング、MTP 3、512 トークンのプロンプトチャンクを使用し、コンテキスト容量とエキスパート配置は自動選択です。FP8 欄はこの GPU で自動選択された経路、INT8 欄は KV 形式だけを INT8 に変更した比較です。ハイブリッド CUDA Graph は既定の無効設定です。

**8 GB、12 GB、16 GB の GPU を搭載した実機での計測は、今後行う予定です。** 下表のメモリを減らした行は、同じ 24 GB GPU 上で空き容量を制限して測定した結果であり、物理的な 8/12/16 GB GPU のベンチマークではありません。

| VRAM 条件 | コンテキスト容量 | INT8 生成 | ネイティブ FP8 生成 | 備考 |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 118.40 tok/s | 131.82 tok/s | VRAM にすべての重みを常駐 |
| VRAM を約 16 GB に制限 | 128K | 95.09 tok/s | 99.39 tok/s | GPU と CPU でエキスパートを計算 |
| VRAM を約 12 GB に制限 | 64K | 85.03 tok/s | 85.59 tok/s | GPU と CPU でエキスパートを計算 |
| VRAM を約 8 GB に制限 | 64K | 66.30 tok/s | 63.85 tok/s | GPU と CPU でエキスパートを計算 |

## 実測速度 — Gemma 4

2026-10-19 に同じ Windows PC（システム RAM 64 GB）で、Gemma 4 26B-A4B-it QAT Q4_0 と対応する QAT MTP アシスタントを使って計測しました。画像エンコーダを GPU に常駐させ、テキスト API 入力に対し、MTP 3・CUDA Graph・既定の思考とサンプリング（`temperature=1.0`、`top_p=0.95`、`top_k=64`）を有効にしています。コンテキスト容量・入力チャンク（1,120 トークン）・エキスパート配置は自動選択です。

| VRAM 条件 | コンテキスト容量 | FP16 生成 | INT8 生成 | 備考 |
| --- | ---: | ---: | ---: | --- |
| 24 GB | 256K | 105.56 tok/s | 118.62 tok/s | FP16：GPU と CPU で計算、INT8：全エキスパートが GPU 常駐 |
| VRAM を約 16 GB に制限 | 128K | 88.46 tok/s | 86.67 tok/s | GPU と CPU でエキスパートを計算 |
| VRAM を約 12 GB に制限 | 64K | 68.16 tok/s | 76.47 tok/s | GPU と CPU でエキスパートを計算 |
| VRAM を約 8 GB に制限 | 64K | 48.22 tok/s | 48.97 tok/s | 生成時のエキスパート計算は CPU |

同じ 3 種類の入力文を各 1 回使い、毎回、思考を含む 128 トークンを生成しました。速度は総生成トークン数を総デコード時間で割った値で、読み込み・入力処理は含みません。64K〜256K はキャッシュ容量であり、入力の長さではありません。VRAM を制限した行は、同じ 24 GB GPU 上で別途 8・12・16 GiB を確保した条件で、物理的な 16/12/8 GB GPU の測定ではありません。

## 実測速度 — LFM2.5（実験対応）

同じ Windows PC（システム RAM 64 GB）で、公式 LFM2.5-24B-A2B Q4_K_M、FP16 KV、MTP・Vision なし、CUDA Graph 有効で計測しました。現在の API の既定値（思考有効、`temperature=0.1`、`top_k=50`、`repetition_penalty=1.05`）を使い、コンテキスト容量・入力チャンク（全条件で 256 トークン）・エキスパート配置は自動選択しています。

| VRAM 条件 | コンテキスト容量 | FP16 生成 | 備考 |
| --- | ---: | ---: | --- |
| 24 GB | 128K | 151.71 tok/s | VRAM にすべての重みを常駐 |
| VRAM を約 16 GB に制限 | 128K | 111.71 tok/s | GPU と CPU でエキスパートを計算 |
| VRAM を約 12 GB に制限 | 128K | 70.80 tok/s | GPU と CPU でエキスパートを計算 |
| VRAM を約 8 GB に制限 | 64K | 51.20 tok/s | GPU と CPU でエキスパートを計算 |

Qwen の測定と同じ 3 種類の入力文をそれぞれ 2 回使い、毎回、思考を含む 128 トークンを生成しました。速度は総生成トークン数を総デコード時間で割った値で、読み込み・入力処理は含みません。VRAM を制限した行も同じ 24 GB GPU での計測であり、物理的な 8/12/16 GB GPU のベンチマークではありません。Qwen の表とはモデル・サンプリング設定が異なります。

## どのプロジェクトを使うべきか

| 目的とハードウェア | プロジェクト |
| --- | --- |
| 一般的なグラフィックボードを搭載した、**VRAM が 8–24 GB、システム RAM が 32 GB 以下**の **Windows** PC で、Qwen3.6 / Gemma 4 / LFM2.5 を API 経由で使いたい。 | TinyStrata（**Windows 専用**、開発中） |
| **RTX 3060 クラス以上の GPU** と **64 GB 以上のシステム RAM** の両方を備えている。 | [Strata（上流プロジェクト）](https://github.com/Niko1221/Strata) をお勧めします。 |
| **チャット UI が同梱されたもの**を使いたい。 | [Strata（上流プロジェクト）](https://github.com/Niko1221/Strata) をお勧めします。 |

## メモリの分担 — Qwen3.6

MoE モデルには、トークンごとに選ばれる小さなニューラルネットワークである**エキスパート**（Experts）が多数含まれています。TinyStrata は、できるだけ多くのエキスパートを GPU メモリ（VRAM）に保持します。生成中は、選択されたエキスパートがキャッシュされていない場合、システム RAM を使って CPU で計算できます。大きな入力バッチを処理するときは、代わりに RAM から一時的な GPU バッファへエキスパートの重みをストリーミングできます。モデルが選択したエキスパートは計算から除外されません。

ダウンロードするモデルファイルは合計約 **15.0 GB** です。言語モデル（MTP の重みを含む）が **14.1 GB**、ビジョンプロジェクタ（`mmproj-F16.gguf`）が **0.9 GB** です。言語モデル全体を VRAM に収める必要はありません。GPU メモリには、ビジョンエンコーダと画像処理バッファ、会話の **KV キャッシュ**、モデルのその他の重み、一時計算バッファの領域も必要です。

![TinyStrata が GPU の VRAM とシステム RAM でメモリを分担する仕組み](assets/memory-sharing.png)

これは既定の GPU 画像処理モードでの配置の概念図であり、各領域のサイズが固定されているわけではありません。モデルを読み込むたびに、TinyStrata は言語モデルを読み込む前に GPU のビジョンエンコーダを起動してウォームアップし、その重みと作業バッファを常駐させます。その後、KV キャッシュ、その他の重み、作業バッファを確保し、残りの空き VRAM からエキスパートキャッシュの容量を決めます。Windows では CUDA の空きメモリと NVIDIA ドライバの専用 VRAM の空き容量を確認し、両方取得できた場合は小さい方を使います。

使用されるエキスパートが変わると、キャッシュの内容も変わります。キャッシュ容量は、モデルを再読み込みするときに再計算されます。あとから起動したほかのアプリケーションが VRAM を使用する可能性はあります。

エキスパート配置時の余裕領域は、通常 768 MiB を目標とします。本家 Strata と同様に、本体の全エキスパートがあと少しで GPU に収まる場合は、高速化用の Q8 重みの複製を解放し、Attention のスコア用バッファがあれば縮小し、余裕領域を 256 MiB まで減らすことがあります。MTP のエキスパートは CPU でも処理できます。`--vram-reserve-mib N` で余裕領域を明示的に固定できます。これは配置時の目標であり、その後の空き容量を保証するものではありません。

通常、GPU に常駐するエキスパートが多いほど、CPU の処理は少なくなります。ハイブリッドモードでは、VRAM にキャッシュされたエキスパートの元になるコピーもシステム RAM に保持するため、VRAM を増やしても RAM 使用量が同じバイト数だけ減るわけではありません。OS やほかのアプリケーションが使う RAM も残してください。**RAM 容量**が多ければモデルデータを保持できますが、速度は CPU、RAM の帯域幅、GPU にも左右されます。Windows の「共有 GPU メモリ」は、TinyStrata が意図的に CPU でエキスパートを計算する仕組みとは異なります。

KV キャッシュは VRAM に保持され、このオフロードの仕組みではシステム RAM がエキスパートのデータを保持します。そのため、長い会話には、KV キャッシュに必要な十分な GPU メモリが引き続き必要です。

## 自動設定 — Qwen3.6

新しい構成では、KV/Attention とコンテキスト容量が自動で選択されます。8 GB、12 GB、16 GB 向けの設定を自分で選ぶ必要はありません。

RTX 3060 では INT8 が選択されます。対応する RTX 40/50 シリーズ環境では、ネイティブ FP8 を選択できます。エンジンは、GPU の世代だけでなく、コンパイル済みカーネルと GPU の機能も確認します。ネイティブ FP8 が利用できない場合、自動モードは INT8 を使用します。起動時に速度や品質のベンチマークを実行することなく、対応する処理経路を選択します。**ネイティブ FP8 Attention は引き続き実験的な機能です。**

入力チャンクは、空き VRAM、コンテキスト容量、エキスパートが GPU に収まるかに基づき、ネイティブエンジンが起動時に 512・1024・2048・4096 トークンから選びます。大きなチャンクでは、入力処理中に RAM から GPU へ重みを転送する回数を減らせます。`run-api.bat --chunk N` でサイズを固定し、`run-api.bat --chunk auto` でネイティブの自動選択に戻せます。

コンテキスト容量は、入力と生成される出力を合わせて予約する領域の最大値です。容量を大きくすると、短い会話でも KV メモリの使用量が増え、エキスパートに使える VRAM が減ります。また、容量が同じだからといって、すべての会話の長さで同じ速度になるわけではありません。長い文書を実際に処理するには、より多くの計算が必要です。規模の目安として、MTP レイヤーを含む INT8 KV だけで、**64K では約 0.71 GiB、128K では約 1.42 GiB、256K では約 2.84 GiB** を使用します。その他の重みや作業バッファは別途必要です。この実装では、ネイティブ FP8 の KV 保存容量は INT8 とほぼ同じです。利点はキャッシュ容量の大幅な削減ではなく、計算経路の違いにあります。

空き VRAM とは、GPU の箱に記載された容量ではなく、読み込み時点で利用可能なメモリのことです。Windows でのコンテキスト自動選択では、CUDA と NVIDIA ドライバが報告する空きメモリを確認し、両方取得できた場合は小さい方を使用します。そのため、ほかのアプリケーションによって、より小さいコンテキストが選ばれることがあります。再読み込みによって容量が減った場合、API は収まらなくなった入力を黙って切り詰めず、拒否します。

GGUF の重みの量子化と KV の精度は別のものです。選択されている UD-IQ3_XXS モデルは重みが圧縮されています。INT8 や FP8 は、会話のキャッシュと、その Attention 処理経路を表しています。KV の精度を下げると回答が変わる場合があるため、これらのモードは FP32 KV と同じ出力を保証しません。この実装では、見えているすべてのコンテキスト位置と、モデルが選択したすべてのエキスパートを保持します。

新しいランチャー構成では MTP 3 を使用します。MTP は数トークン先を提案し、メインモデルで検証します。生成を高速化できる場合がありますが、その効果は提案がどれだけ受理されるかによって変わります。検証はメインモデルのデコード動作を保つことを意図しています。ただし、数値計算の実行経路の違いによってトークンが変わる場合があるため、MTP を有効にした場合と無効にした場合で同じ出力になるとは限りません。

MTP は greedy 生成と温度付きサンプリングの両方に対応しており、反復・頻度・出現の各ペナルティも利用できます。サンプリング時は、メインモデルが検証の各行からトークンを選び、予測トークン列のうち先頭から一致した部分だけを採用します。

明示的な設定は自動選択より優先され、既存の手動設定も引き続き有効です。たとえば、`run-api.bat --kv-type int8` は INT8 に固定し、`run-api.bat --kv-type auto --max-context auto --expert-vram auto` は KV、コンテキスト、エキスパート配置を自動設定に戻します。

## API サーバー

**TinyStrata は API 専用です。** OpenAI 互換の Chat Completions（`/v1/chat/completions`）と Responses（`/v1/responses`）API、および Anthropic 互換の Messages API（`/v1/messages`）を提供し、ストリーミング生成にも対応しています。API クライアントや外部のフロントエンドを接続してください。TinyStrata にチャット UI は同梱されていません。

Qwen は既定で思考が有効で、`temperature=0.7`、`top_p=0.95`、`top_k=20`、`min_p=0.0`、`presence_penalty=1.5`、`repetition_penalty=1.0`、思考を含む最大出力は **32,768 トークン**です。リクエストごとの設定が優先されます。

API は、デフォルトで `http://127.0.0.1:18080` で待ち受けます。別のポートを使うには、インストール先の TinyStrata フォルダで `run-api.bat --port 18081` を実行してください。

Qwen では `run-api.bat --vision gpu` で GPU（新規構成の既定）、`--vision cpu` で CPU による画像処理を選べます。`--vision no` は画像を無効にし、`--vision-tokens N` は画像トークン数の上限（既定 4096）を指定します。選択は保存され、次回以降は省略しても維持されます。

API はモデルファイルをインストールする前でも起動できます。Qwen を選択している場合、`POST /v1/model/download` に空の JSON ボディ（`{}`）を指定すると、固定の [Unsloth Qwen3.6-35B-A3B MTP UD-IQ3_XXS 言語モデル](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf)（MTP の重みを含む 14.1 GB）と、その[ビジョンプロジェクタ](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/blob/5bc3e238d916f48a861bac2f8a1990a0e9b7e98d/mmproj-F16.gguf)（0.9 GB）、合計約 **15.0 GB** をダウンロードします。両方のファイルの SHA-256 チェックサムを検証し、トークナイザーを抽出して読み込みます。`GET /v1/model/download` で進捗を確認でき、中断したダウンロードは再開できます。ダウンロードを繰り返し要求した場合は、実行中のジョブを再利用します。`POST /v1/model/download/cancel` で転送や検証をキャンセルできます。ダウンロード完了は推論の準備完了を意味しません。ダウンロード状態が `ready` になり、`/health` が `loaded: true`（画像が有効なら `images: true` も）を返すまで待ってください。

インストーラーはエンジンの実行ファイルと GPU ランタイムライブラリを準備し、既定で Qwen とそのビジョンプロジェクタをダウンロードします。既存の構成では、明示的に変更するまで、選択済みのローカルモデルを使用し続けます。

## 謝辞

TinyStrata は [Strata](https://github.com/Niko1221/Strata) から独立したフォークであり、その実装を広く利用しています。Strata を開発してきた [Niko](https://github.com/Niko1221) と Strata コミュニティに感謝します。

TinyStrata に取り入れた KV キャッシュ、attention、量子化エキスパートのコードを開発・公開している [Aaron Flin](https://github.com/aflin) と [MoMoE プロジェクト](https://github.com/aflin/momoe) にも感謝します。

## ライセンス

TinyStrata は MIT ライセンスで公開されています。詳細は [LICENSE](LICENSE) を参照してください。

Copyright (c) 2026 Niko1221 and the Strata contributors.

利用している外部コードの出典とライセンス情報は [NOTICE](NOTICE) にまとめています。GGML は [MIT ライセンス](third_party/ggml/LICENSE) で公開されています。
