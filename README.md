# comfy-chat

日本語の自然言語を Qwen3.5-9B uncensored（llama-server）で Stable Diffusion プロンプトに変換し、ComfyUI で画像を生成する Web アプリ。

## 機能

- **日本語チャット入力** — 会話履歴を保持し、「もっと明るく」「着物姿にして」などの差分指示に対応
- **モデル別プロンプト変換** — Pony Diffusion / Illustrious / SDXL / Flux / Z-image-Turbo / Anima のプロンプト文法をチェックポイント名で自動切り替え
- **プロンプト確認パネル** — 生成前に LLM が作成した positive / negative タグを編集できる。タグ解説チップで各タグの日本語説明を即時表示（未訳タグは LLM で補完）
- **Danbooru タグ補完** — 確認パネルの positive / negative 欄で英語タグ名または日本語で 2 文字以上入力するとドロップダウンで候補表示。↑↓Enter/Esc/クリックで確定
- **バッチ生成** — 同じプロンプトで 1 / 2 / 4 枚を seed を変えて並列生成し、グリッド表示からベストを選択
- **CLIP Skip 自動制御** — Pony / Illustrious 系は `-2`、SDXL / Turbo 系は `-1` に自動設定
- **LoRA 自動選択** — 登録済み LoRA を LLM が自然言語の指示に基づいて自動選択・適用
- **LoRA 互換性管理** — LoRA をベースモデル別（Pony / SDXL / Flux / Illustrious / 汎用）に登録し、非互換 LoRA を自動除外
- **多衣装 LoRA 対応** — 衣装別の活性化トークンを description に記述し、LLM が文脈に応じて衣装を選択
- **サーバー側トリガーワード付加** — LLM に依存せず、選択された LoRA のトリガーワードをサーバーが確実に付加
- **サーバー側衣装タグ補完** — LLM が衣装タグを一部省略した場合もサーバーが全タグを補完
- **Hires fix** — `UltimateSDUpscale` + RealESRGAN 4x+ Anime 6B による 512×512 タイル分割再サンプリング（denoise 0.45）、出力は 2x（2048×2048）
- **ADetailer** — 顔（`face_yolov8n.pt`）と手（`hand_yolov8n.pt`）を YOLO で検出し、各領域を個別に再 inpaint
- **img2img** — 参照画像をドラッグ&ドロップしてアップロード、変化量スライダーで忠実度を調整
- **インペイント** — Canvas ブラシでマスクを描き、塗った範囲だけを再生成。「確定して生成」ボタンで前回と同じプロンプトのまま LLM 翻訳なしで即時再生成（指・目など違和感のある部分の修正に最適）
- **WebSocket 進捗バー** — ComfyUI の生成ステップをリアルタイムで表示
- **サンプラー・スケジューラ選択** — euler / dpmpp_2m 等のサンプラーと karras / simple 等のスケジューラをヘッダーから選択
- **ネガティブプリセット** — 「顔崩れ防止」「品質向上」「手崩れ防止」チェックボックスでネガティブプロンプトを補完
- **生成履歴ギャラリー** — 最大 100 件の生成画像・設定を localStorage に保存し、サムネイルクリックで設定を復元
- **Civitai LoRA 検索** — civitai.com / civitai.red から LoRA を検索・ダウンロード・登録（トリガーワード自動取得）
- **アスペクト比クイック設定** — 1:1 / 3:4 / 4:3 / 9:16 / 16:9 ボタンで解像度を即時変更
- **画像ダウンロード** — 生成画像をタイムスタンプ付きファイル名で保存
- **お気に入り** — チャット入力・モデル・解像度・Steps・CFG などの設定セットを名前付きで保存し復元。チャット欄に入力中の場合はメッセージを上書きしない
- **フレーズサジェスト** — 構図/ライティング/雰囲気/場所/スタイル 5カテゴリのフレーズチップをクリックして追記、最近の入力も再利用可能
- **AI 画像レビュー** — 生成画像を Vision LLM（Qwen3-VL-8B）で自動評価。指・顔・構図の問題点を日本語でフィードバックし、修正用タグ（positive / negative）を提案。チャット形式でレビューに反論・補足し、誤指摘の訂正も可能
- **ControlNet（編集タブ）** — 参照画像でポーズ・ラインアートを制御。openpose（DWPreprocessor で骨格抽出）と canny（エッジ抽出）に対応。強度スライダーで適用量を調整
- **IP-Adapter（編集タブ）** — 参照画像のスタイル・雰囲気を生成画像に転写（PLUS high strength モード）。img2img より自然な転写が可能
- **ウォーターマーク除去（動画編集タブ）** — 動画から固定位置のウォーターマークを除去して MP4 を出力。2 方式を選択可能：**標準**（cv2 TELEA、高速・数秒）・**AI**（ComfyUI SDXL インペイント、高品質・数十秒）。プレビュー画像をドラッグして除去範囲を視覚的に指定でき、音声保持オプション・繰り返しインペイント対応
- **動画生成（動画生成タブ）** — 参照画像 + テキストプロンプトから Wan2.2 I2V A14B（2段カスケード）でアニメーション WebP を生成。フレーム数・FPS・解像度・Steps・CFG・seed を調整可能。アニメスタイル LoRA（`wan2.2_i2v_animestyle_v2_low`）を ON/OFF 選択。生成結果をブラウザで即時再生 + ダウンロード

## 依存サービス

| サービス | ポート | 用途 |
|---------|--------|------|
| llama-server (`mymodel-9b-unc`) | 11434 | 日本語 → プロンプト変換 |
| ComfyUI | 8188 | 画像生成 |
| llama-server (`mymodel-vision`) | 11435 | AI 画像レビュー（Vision LLM、オンデマンド起動） |

ComfyUI の Python venv（`~/infra/comfyui/venv/`）を使用するため、追加インストール不要。

Vision LLM（port 11435）は AI レビューボタンを押したときに `~/infra/start-llama-vision.sh` で自動起動する。9B モデルを一時停止して起動するため、レビュー中は画像生成が待機状態になる。

### Wan2.2 I2V 動画生成に必要なモデル

```
~/infra/comfyui/models/
├── unet/HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf   # Stage 1（粗生成）
├── unet/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf     # Stage 2（精細化）
├── vae/wan_2.1_vae.safetensors                             # 16ch VAE（48ch は不可）
├── text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors   # テキストエンコーダー
└── loras/wan2.2_i2v_animestyle_v2_low.safetensors          # アニメスタイル LoRA
```

### ComfyUI カスタムノード

```
~/infra/comfyui/custom_nodes/
├── ComfyUI-Impact-Pack           # FaceDetailer, DetailerForEach 等（ADetailer 使用時に必要）
├── ComfyUI-Impact-Subpack        # UltralyticsDetectorProvider（ADetailer 使用時に必要）
├── ComfyUI_UltimateSDUpscale     # UltimateSDUpscale（タイル分割 Hires fix）
├── comfyui_controlnet_aux        # DWPreprocessor（ControlNet openpose 使用時に必要）
└── ComfyUI_IPAdapter_plus        # IPAdapterUnifiedLoader, IPAdapterAdvanced（IP-Adapter 使用時に必要）
```

```
~/infra/comfyui/models/ultralytics/bbox/
├── face_yolov8n.pt
└── hand_yolov8n.pt
```

### Z-image-Turbo 使用時に必要なモデル

```
~/infra/comfyui/models/
├── unet/z_image_turbo_bf16.safetensors   # UNETLoader で読み込み（FP8 推論）
├── clip/qwen_3_4b.safetensors            # Lumina2 用 CLIP テキストエンコーダー
└── vae/ae.safetensors                    # AE VAE
```

### Anima-base 使用時に必要なモデル

```
~/infra/comfyui/models/
├── unet/anima-base-v1.0.safetensors      # UNETLoader で読み込み
├── clip/qwen_3_06b_base.safetensors      # Qwen Vision 用 CLIP（type=qwen_image）
└── vae/qwen_image_vae.safetensors        # Qwen Image VAE
```

nArnima Turbo 等のマージ済みチェックポイントは `CheckpointLoaderSimple` で読み込み、VAE のみ上記 `qwen_image_vae.safetensors` を使用。

### ControlNet 使用時に必要なモデル

```
~/infra/comfyui/models/controlnet/
├── openpose-sdxl.safetensors             # OpenPose SDXL（ポーズ制御）
└── canny-sdxl.safetensors                # Canny SDXL（エッジ制御）
```

### IP-Adapter 使用時に必要なモデル

```
~/infra/comfyui/models/ipadapter/
└── ip-adapter-plus_sdxl_vit-h.safetensors   # IPAdapterUnifiedLoader が自動認識
```

## 起動

```bash
~/projects/comfy-chat/start.sh
```

llama-server（mymodel-9b-unc）と ComfyUI が未起動の場合、自動で起動を試みる。

起動後: http://localhost:9000

## 画面構成

UI は 5 つのタブに分かれている。

| タブ | 内容 |
|------|------|
| 🎨 生成 | 画像生成チャット（メイン機能） |
| 🧩 LoRA 管理 | LoRA の登録・編集・削除・Civitai 検索 |
| 📷 履歴 | 過去の生成画像ギャラリー |
| 🎬 動画生成 | Wan2.2 I2V による動画生成 |
| 🎞️ 動画編集 | 動画ウォーターマーク除去 |

## 使い方

### 画像生成（テキスト → 画像）

1. 「🎨 生成」タブを開く
2. ヘッダーのセレクターでチェックポイントモデル・解像度・Steps・CFG・サンプラー・スケジューラを設定
3. 必要に応じて **Hires fix**（高解像度化）・**ADetail**（顔・手修正）をチェック
4. チャット欄に日本語で内容を入力し「生成」ボタン（または Ctrl+Enter）
5. **確認パネル** が表示され、LLM が生成した positive / negative タグを確認・編集できる
   - タグ解説チップで各タグの日本語訳を即時確認（`tags.json` ローカルルックアップ、未訳タグのみ LLM 補完）
   - positive / negative 欄に英語または日本語で入力すると Danbooru タグ補完ドロップダウンが出る
   - LoRA のチェックボックスで使用 LoRA を変更可能
   - 「この内容で生成」ボタンで ComfyUI に送信
6. 進捗バーで生成状況を確認しながら待機、完了後に右パネルへ画像が表示される

| チェックボックス | 効果 | 処理時間増加 |
|----------------|------|-------------|
| Hires fix | 生成後に 2x 高解像度化 | +50〜100% |
| ADetail | 顔・手を個別に再描画 | +30〜60% |

### バッチ生成（複数バリエーション）

1. 入力欄の「バッチ」セレクタで枚数（1 / 2 / 4）を選択
2. 通常通り生成を実行
3. 2 枚以上の場合、生成結果がグリッドで表示される
   - サムネイルをクリックすると大きく表示、Seed・プロンプト情報も更新される
   - ダウンロード・コピーは選択中の画像に対して動作する

> バッチの 1 枚目はユーザーが指定した seed（-1 はランダム）、2 枚目以降は新たにランダム生成される。

### img2img（参照画像から生成）

1. 入力欄上部の「📎 参照画像」エリアに画像をドラッグ&ドロップ（またはクリックして選択）
2. **変化量** スライダーで参照画像への忠実度を調整（低い値 = 元画像に近い、高い値 = 大きく変化）
3. テキスト入力で変更指示を書いて生成

### インペイント（部分再生成）

1. img2img と同様に参照画像をアップロード
2. 「✏️ マスク」ボタンをクリックしてマスクエディタを開く
3. 再生成したい部分をブラシで塗る（消しゴムモード・ブラシサイズ調整可）

**方法 A：ワンクリック即時再生成（前回生成済み画像の修正に最適）**

4. 「確定して生成」ボタンをクリック — 前回と同じプロンプトで LLM 翻訳なし・確認パネルなしで即座に生成が始まる

**方法 B：テキスト指示を変えて再生成**

4. 「確定」でマスクを適用してモーダルを閉じる（ボタンが緑色に変わる）
5. チャット欄にテキストで指示を書いて生成 — マスク範囲のみが再描画される

> 方法 A は `lastPositive`（前回の生成プロンプト）が必要なため、セッション開始直後や外部画像からの初回インペイントには使用不可。その場合は方法 B を使用する。

### ネガティブプリセット

生成パネル下部のチェックボックスで追加ネガティブプロンプトを選択できる。

| プリセット | 効果 |
|-----------|------|
| 顔崩れ防止 | `bad face, asymmetrical face, ...` を追加 |
| 品質向上 | `blurry, low quality, jpeg artifacts, ...` を追加 |
| 手崩れ防止 | `bad hands, extra fingers, ...` を追加 |

確認パネルでネガティブプロンプトを直接編集することもできる。

### 生成履歴

「📷 履歴」タブに最新 100 件の生成結果がサムネイルグリッドで表示される。サムネイルをクリックすると「🎨 生成」タブへ移動し、そのときのプロンプト・チェックポイント・全パラメータが復元される。

### お気に入り

- 「⭐」ボタン → 名前を入力して「★ 保存」で現在の入力・設定を記録
- 保存済みの項目をクリックするとチャット入力・モデル・全パラメータが復元される
- 設定はブラウザの localStorage に保存されるためリロード後も維持される

### フレーズサジェスト

- 入力欄上部の構図・ライティング・雰囲気・場所・スタイル タブからフレーズチップをクリック
- 選んだフレーズが「、フレーズ」の形でテキストエリアに追記される
- 「最近」タブには過去の生成入力が記録され、クリックでテキストエリアに再現できる

### AI 画像レビュー

1. 画像を生成する
2. 右パネルの「🔍 AI レビュー」ボタンをクリック（Vision モデルが自動起動、約60〜90秒）
3. **スコア（1〜10）**・**問題リスト**・**総合コメント** が日本語で表示される
4. **positive+ / negative+** に修正タグ案がある場合は「↓ 入力欄に反映」ボタンで次回生成の指示として入力欄にセット
5. レビュー内容に誤りがある場合は下部の入力欄に反論・補足を入力して送信 — Vision LLM が評価を更新する

> Vision レビュー中は 9B LLM が停止するため、チャット入力からの新規生成はレビュー完了後に行うこと。

### 動画生成（Wan2.2 I2V）

1. 「🎬 動画生成」タブを開く
2. 参照画像をドラッグ&ドロップまたはクリックしてアップロード（PNG / JPEG）
3. **プロンプト**に英語でモーション説明を入力（例: `tomopi smiling, waving hand, room background`）
4. 必要に応じて **ネガティブプロンプト**・**詳細オプション**（フレーム数・FPS・解像度・Steps・CFG・seed）を調整
5. 「🎬 動画を生成」ボタンをクリック（処理時間: 最大 20 分程度）
6. 完了後にアニメーション WebP がブラウザ内で自動再生される
7. 「⬇️ ダウンロード」リンクで `.webp` ファイルを保存

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| アニメスタイル LoRA | ON | `wan2.2_i2v_animestyle_v2_low` を適用してアニメ風に最適化 |
| フレーム数 | 81 | 約5秒 @ 16fps |
| 幅 × 高さ | 576 × 1024 | 縦型（9:16）。横型にする場合は 1024 × 576 |
| CFG | 3.5 | Wan2.2 推奨値（画像生成より低め） |

> **要件**: ComfyUI が起動済みであること（`~/infra/start-comfyui.sh`）。Wan2.2 I2V A14B GGUF モデルが `~/infra/comfyui/models/unet/` に配置済みであること。

### ウォーターマーク除去（動画編集）

1. 「🎞️ 動画編集」タブを開く
2. 動画ファイル（MP4 / MOV / WebM、最大 200 MB）を選択
3. **「フレーム確認」** をクリックして最初のフレームをプレビュー表示
4. プレビュー画像の上でウォーターマーク部分を**ドラッグして選択**  
   → X1 / Y1 / X2 / Y2 座標フィールドに自動反映される（手動入力も可）
5. **処理方式**を選択する

| 方式 | 特徴 | 所要時間 |
|------|------|---------|
| 標準 (TELEA) | cv2.inpaint による高速補間 | 数秒 |
| AI (ComfyUI) | SDXL インペインティング（NoobAI-XL）。シーングループを検出して代表フレームのみ推論し全フレームにパッチ適用 | 数十秒〜 |

6. **「音声を保持する」** チェックボックスで音声の有無を選択
7. **「ウォーターマーク除去」** をクリックして処理開始
8. 完了後に動画プレビューとダウンロードリンクが表示される
9. 違和感が残る場合は **「🔄 再インペイント」** をクリックして結果動画を入力として再処理  
   （AI モードは毎回異なるシードで推論するため、複数回試すと別の補完結果が得られる）

> **AI モード**は ComfyUI が起動している必要がある（`~/infra/start-comfyui.sh`）。`sdxl_vae.safetensors` が `~/infra/comfyui/models/vae/` に必要。

### LoRA の登録

「🧩 LoRA 管理」タブで登録・編集・削除ができる。

**手動登録（ローカルファイル）**

1. `~/infra/comfyui/models/loras/` に `.safetensors` を配置
2. LoRA 管理タブで「ファイルから追加」からファイルを選択
3. 説明・トリガーワード・強度・ベースモデル種別を入力して登録

**Civitai から検索・ダウンロード**

1. LoRA 管理タブ下部の「Civitai 検索」欄にキーワードを入力
2. 検索サイト（civitai.com / civitai.red）と件数（10〜100件）を選択して「検索」
3. 結果カードのサムネイルをクリックすると civitai.com のモデルページを確認できる
4. 「⬇ DL & 登録」ボタンでダウンロード → `loras.json` に自動登録（トリガーワードは `trainedWords` から自動取得）

> **注**: civitai.red は NSFW コンテンツを含む。API キーが必要な場合は環境変数 `CIVITAI_API_KEY` を設定する。

登録した LoRA は LLM がユーザーの指示を解析して自動選択する。非互換なベースモデルの LoRA は生成時に自動除外され、LLM への選択肢にも現れない。

#### 多衣装 LoRA の description 記述形式

衣装ごとに異なる活性化トークンを持つ LoRA は、description に以下の形式で記述する。LLM がユーザーの指示から衣装を判断し、対応するトークンと関連タグを positive に出力する。

```
キャラ説明...[衣装名A]: token=TOKEN_A, tags=tag1, tag2; [衣装名B]: token=TOKEN_B, tags=tag3, tag4
```

例（urakaze-08.safetensors）:
```
艦これ浦風(青髪・青目)。[制服]: token=urakazeKC, tags=school uniform, serafuku, ...; [秋着物]: token=urakazeautumnKC, tags=kimono, yukata, ...
```

#### force_tags フィールド

成人向け状況 LoRA など、使用時に必ずプロンプトに含める必要があるタグを `force_tags` フィールドで指定する。LLM の出力に関係なくサーバーが必ず付加する。

```json
{
  "filename": "pussy_juice_tail.safetensors",
  "force_tags": "rating:explicit",
  ...
}
```

#### LoRA ベースモデル種別

| 種別 | 対象 |
|------|------|
| `pony` | Pony Diffusion V6 XL など Pony ベースのモデル |
| `sdxl` | SDXL Base 1.0 など標準 SDXL ベースのモデル |
| `flux` | Flux.1 ベースのモデル |
| `illustrious` | Illustrious XL / NoobAI XL ベースのモデル（`sdxl` LoRA とも互換） |
| `anima` | Anima-base / nArnima Turbo など Anima ベースのモデル |
| `any` | ベースモデルを問わず使用できる LoRA |

### プロンプト変換の仕組み

| チェックポイント | 変換スタイル | CLIP Skip |
|----------------|-------------|-----------|
| `*pony*` | Danbooru タグ形式（`score_9, source_anime, ...`） | -2 |
| `*illustrious*` / `*noobai*` | Danbooru タグ形式（`masterpiece, best quality, newest, absurdres, highres, ...`） | -2 |
| `*flux*` | 自然言語の長文記述 | -1 |
| `Z-image-Turbo` | SDXL タグ形式（Steps=6, CFG=1.5 推奨） | -1 |
| `Anima-base` / `*anima*` / `*narni*` | Danbooru タグ＋自然言語混在形式（anime style）。Steps=20、CFG=3.5、sampler=euler、scheduler=simple に自動設定。Hires fix 強制無効 | -1 |
| その他 | SDXL 向けタグ＋フレーズ混在形式（`masterpiece, best quality, ...`） | -1 |

#### Pony モデルの positive タグ構成（サーバー側で組み立て）

```
[force_tags]           rating:explicit など LoRA 必須タグ
[trigger_words]        LoRA の外見基本タグ（常時付加）
[PONY_QUALITY_PREFIX]  score_9, score_8_up, ..., detailed eyes（品質タグ）
[LLM 出力]             衣装・活性化トークン・状況タグ
```

LLM が品質タグを省略した場合もサーバーが先頭に補完する。タグの重複は dedup 処理で除去される。

## ファイル構成

```
comfy-chat/
├── app.py            # aiohttp.web サーバー（ポート 9000）
├── comfy_utils.py    # LLM / ComfyUI API 連携、ワークフロー構築
├── system_prompt.py  # モデル別システムプロンプト定数
├── wan_utils.py      # Wan2.2 I2V 動画生成ユーティリティ
├── wm_utils.py       # ウォーターマーク除去（cv2 TELEA）
├── wm_comfy.py       # ウォーターマーク除去（ComfyUI AI インペイント）
├── loras.json        # LoRA レジストリ（自動生成・更新）
├── static/
│   ├── index.html    # シングルファイル Web UI（5タブ構成）
│   └── tags.json     # Danbooru タグ補完データ（6,947件・日本語訳付き）
├── tools/
│   ├── build_tags_json.py      # tags.json 生成スクリプト（WD14 CSV → JSON 変換）
│   ├── danbooru-jp.csv         # 人力日本語翻訳（427件）
│   └── danbooru-machine-jp.csv # 機械翻訳（~100k件）
└── start.sh          # サービス起動チェック → venv → Python 実行
```

## ComfyUI ワークフロー構成

### 通常チェックポイント（Pony / Illustrious / SDXL / Flux）

| ノード番号 | クラス | 役割 | 条件 |
|-----------|--------|------|------|
| 1 | CheckpointLoaderSimple | モデルロード | 常時 |
| 20 | CLIPSetLastLayer | CLIP Skip 制御 | 常時 |
| 3 | VAELoader (`sdxl_vae.safetensors`) | VAE ロード | 常時 |
| 2 | EmptyLatentImage | テキスト生成用 latent | txt2img 時 |
| 30 | LoadImage | 参照画像ロード | img2img / インペイント時 |
| 35 | ImageScale | 参照画像を width×height にリサイズ | img2img / インペイント時 |
| 31 | VAEEncode | 参照画像を latent 化 | img2img（マスクなし）時 |
| 32 | LoadImage | マスク画像ロード | インペイント時 |
| 36 | ImageScale | マスク画像を width×height にリサイズ | インペイント時 |
| 33 | ImageToMask | グレースケール → マスクテンソル変換 | インペイント時 |
| 34 | VAEEncodeForInpaint | マスク付き latent 化（grow_mask_by=6） | インペイント時 |
| 100〜 | LoraLoader | LoRA チェーン（直列） | LoRA 使用時 |
| 4〜7 | CLIPTextEncode × 2 + KSampler + VAEDecode | 基本生成パイプライン | 常時 |
| 8 | SaveImage | 保存（`comfy_chat/auto`） | 常時（入力元が変わる） |
| 9 | UpscaleModelLoader | アップスケールモデルロード | Hires fix ON 時 |
| 10 | UltimateSDUpscale | タイル分割 2x 再サンプリング（512×512 タイル） | Hires fix ON 時 |
| 200〜201 | UltralyticsDetectorProvider + FaceDetailer | 顔修正 | ADetail ON 時 |
| 202〜204 | UltralyticsDetectorProvider + BboxDetectorSEGS + DetailerForEach | 手修正 | ADetail ON 時 |

### Z-image-Turbo（UNETLoader 方式）

CheckpointLoaderSimple の代わりに以下のノードを使用する。LoRA チェーン・img2img・インペイント・Hires fix・ADetail は通常と共通。

| ノード番号 | クラス | 役割 |
|-----------|--------|------|
| 1 | UNETLoader (`z_image_turbo_bf16.safetensors`, `weight_dtype=fp8_e4m3fn`) | UNET ロード（FP8 推論） |
| 22 | ModelSamplingAuraFlow (`shift=3.0`) | AuraFlow サンプリングスケール適用 |
| 21 | CLIPLoader (`qwen_3_4b.safetensors`, `type=lumina2`) | Lumina2 用 CLIP ロード |
| 3 | VAELoader (`ae.safetensors`) | AE VAE ロード |

### Anima-base（UNET/CLIP/VAE 分割ロード方式）

CheckpointLoaderSimple の代わりに以下のノードを使用する。LoRA チェーン・img2img・インペイントは通常と共通。Hires fix は強制無効。

| ノード番号 | クラス | 役割 |
|-----------|--------|------|
| 1 | UNETLoader (`anima-base-v1.0.safetensors`) | UNET ロード |
| 2 | CLIPLoader (`qwen_3_06b_base.safetensors`, `type=qwen_image`) | Qwen Vision 用 CLIP ロード |
| 3 | VAELoader (`qwen_image_vae.safetensors`) | Qwen Image VAE ロード |

nArnima Turbo 等のマージ済みチェックポイント（名前に `anima` / `narni` を含む）は CheckpointLoaderSimple でロードし、VAE のみ `qwen_image_vae.safetensors` に差し替える。

### IP-Adapter（500番台ノード）

`ref_image` が指定され、かつ Anima モデル以外のときに追加される。LoRA チェーンの後に model_src を差し替える形で挿入される。

| ノード番号 | クラス | 役割 | 条件 |
|-----------|--------|------|------|
| 500 | IPAdapterUnifiedLoader (`preset=PLUS (high strength)`) | IP-Adapter + モデルロード | ref_image あり・非 Anima |
| 501 | LoadImage | スタイル参照画像ロード | 同上 |
| 502 | IPAdapterAdvanced (`weight_type=style transfer`) | スタイル転写適用 | 同上 |

### ControlNet（300番台ノード）

`controlnet_image` が指定され、かつ Anima モデル以外のときに追加される。KSampler の positive/negative conditioning を差し替える形で挿入される。

| ノード番号 | クラス | 役割 | 条件 |
|-----------|--------|------|------|
| 300 | ControlNetLoader | ControlNet モデルロード | controlnet_image あり・非 Anima |
| 301 | LoadImage | ControlNet 参照画像ロード | 同上 |
| 302 | DWPreprocessor（openpose）/ Canny（canny） | 骨格またはエッジ抽出 | 同上 |
| 303 | ControlNetApplyAdvanced (`start_percent=0.0, end_percent=1.0`) | conditioning への適用 | 同上 |

## API エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| `GET` | `/` | Web UI |
| `GET` | `/static/tags.json` | Danbooru タグ補完データ（6,947件・日本語翻訳付き） |
| `GET` | `/api/checkpoints` | 利用可能チェックポイント一覧 |
| `POST` | `/api/translate` | LLM 変換 + LoRA 後処理のみ実行（ComfyUI 未送信）、確認パネル用 |
| `POST` | `/api/generate` | 確認済みタグで ComfyUI に送信して画像生成 |
| `POST` | `/api/explain-tags` | positive タグを LLM で日本語解説（`tags.json` 未収録タグの補完用） |
| `GET` | `/api/image` | ComfyUI `/view` へのプロキシ |
| `GET` | `/api/health` | LLM・ComfyUI の死活確認 |
| `POST` | `/api/upload` | 参照画像・マスク画像を ComfyUI にアップロード |
| `POST` | `/api/review` | Vision LLM で生成画像をレビュー（`image_url`, `positive`, `user_message`, `review_history` を受け取り JSON 評価を返す） |
| `GET` | `/api/loras` | 登録済み LoRA 一覧 |
| `POST` | `/api/loras` | LoRA 登録 |
| `PATCH` | `/api/loras/{filename}` | LoRA の description / strength / base_model を更新 |
| `DELETE` | `/api/loras/{filename}` | LoRA 削除 |
| `GET` | `/api/lora-files` | `models/loras/` のファイル一覧 |
| `GET` | `/api/negative-presets` | ネガティブプリセット一覧 |
| `GET` | `/api/civitai/search` | Civitai LoRA 検索（`query`, `limit`, `domain`, `tag` パラメータ） |
| `GET` | `/api/civitai/model` | Civitai 単一モデル情報取得（`id`, `domain` パラメータ） |
| `POST` | `/api/civitai/download` | Civitai LoRA ダウンロード＆自動登録 |
| `POST` | `/api/wan-i2v` | 参照画像 + プロンプトから Wan2.2 I2V で動画生成（アニメーション WebP を返す） |
| `POST` | `/api/wm/preview` | 動画の最初のフレームを JPEG で返す（除去範囲確認用） |
| `POST` | `/api/wm/process` | ウォーターマーク除去処理（`method=telea\|comfy`, `x1/y1/x2/y2`, `keep_audio`） |

### POST /api/translate リクエスト形式

LLM 翻訳と LoRA 後処理のみを行い、確認パネル用データを返す。ComfyUI へは送信しない。

```json
{
  "message": "青空の下で猫が昼寝している、アニメ風",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "checkpoint": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"
}
```

レスポンス: `positive`, `negative`, `loras`, `available_loras`, `explanation`

### POST /api/generate リクエスト形式

```json
{
  "message": "青空の下で猫が昼寝している、アニメ風",
  "confirmed_positive": "score_9, cat, ...",
  "confirmed_negative": "low quality, ...",
  "loras": [{"name": "anime_style.safetensors", "strength": 0.75}],
  "history": [],
  "checkpoint": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
  "width": 1024,
  "height": 1024,
  "steps": 25,
  "cfg": 7.0,
  "sampler": "euler_ancestral",
  "scheduler": "karras",
  "seed": -1,
  "batch": 1,
  "hires_fix": false,
  "adetail": false,
  "init_image": null,
  "mask_image": null,
  "denoise": 0.75,
  "ref_image": null,
  "ip_adapter_strength": 0.5,
  "controlnet_image": null,
  "controlnet_type": "openpose",
  "controlnet_strength": 0.8,
  "negative_presets": [],
  "client_id": "uuid-for-websocket-progress"
}
```

- `confirmed_positive` / `confirmed_negative` を指定すると LLM 翻訳と LoRA 後処理をスキップし、そのタグをそのまま使用する（確認パネル経由の生成）
- `confirmed_positive` を省略すると `message` から LLM 変換を実行する
- `seed: -1` でランダムシード。レスポンスの `seed` に実際の値が含まれる
- `batch`: 1〜4 枚。2 枚以上の場合、レスポンスの `images` 配列に各画像の `url` と `seed` が含まれる
- `init_image` / `mask_image` は `/api/upload` が返すファイル名を指定する
- `ref_image` は IP-Adapter のスタイル参照画像（`/api/upload` が返すファイル名）。Anima モデル時は無視される
- `controlnet_image` は ControlNet の参照画像（`/api/upload` が返すファイル名）。Anima モデル時は無視される
- `controlnet_type` は `"openpose"` または `"canny"`
- `client_id` を指定すると `ws://localhost:8188/ws?clientId=<uuid>` 経由で生成進捗をリアルタイム受信できる

### POST /api/loras リクエスト形式

```json
{
  "filename": "anime_style_v2.safetensors",
  "description": "アニメ・マンガ風のイラストスタイルにする",
  "trigger_words": "anime style",
  "force_tags": "",
  "default_strength": 0.75,
  "base_model": "pony"
}
```

### GET /api/civitai/search クエリパラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `query` | 必須 | 検索キーワード（英語推奨） |
| `limit` | 10 | 取得件数（最大 100） |
| `domain` | `civitai.com` | `civitai.com` または `civitai.red` |
| `tag` | — | `character` / `style` / `clothing` / `poses` 等でフィルタ |

### POST /api/civitai/download リクエスト形式

```json
{
  "model_id": 12345,
  "version_id": 67890,
  "domain": "civitai.com"
}
```

ダウンロード完了後、`~/infra/comfyui/models/loras/` に `.safetensors` を保存し、`loras.json` にトリガーワード・ベースモデル種別を自動登録する。

## 生成画像の保存先

```
~/infra/comfyui/output/comfy_chat/auto_*.png
```

pony-auto の出力（`output/pony/`）とは別フォルダに分離されている。

## トラブルシューティング

**LLM インジケーターが赤い**
```bash
curl http://localhost:11434/v1/models   # mymodel-9b-unc が表示されるか確認
~/infra/start-llama-9b-unc.sh           # 未起動なら手動起動
```

**ComfyUI インジケーターが赤い**
```bash
curl http://localhost:8188/system_stats
~/infra/start-comfyui.sh
tail -f /tmp/comfyui.log
```

**ADetailer で "Cannot import" エラー**
```bash
# Impact-Pack と Impact-Subpack が両方あるか確認
ls ~/infra/comfyui/custom_nodes/ComfyUI-Impact-Pack
ls ~/infra/comfyui/custom_nodes/ComfyUI-Impact-Subpack
# YOLO モデルがあるか確認
ls ~/infra/comfyui/models/ultralytics/bbox/
# ultralytics がインストールされているか確認
source ~/infra/comfyui/venv/bin/activate && python -c "import ultralytics"
```

**Civitai ダウンロードが 401 / 403 で失敗する**
```bash
export CIVITAI_API_KEY="your_api_key_here"
~/projects/comfy-chat/start.sh
```

**Z-image-Turbo で "model not found" エラー**
```bash
ls ~/infra/comfyui/models/unet/z_image_turbo_bf16.safetensors
ls ~/infra/comfyui/models/clip/qwen_3_4b.safetensors
ls ~/infra/comfyui/models/vae/ae.safetensors
```

**Anima-base で "model not found" エラー**
```bash
ls ~/infra/comfyui/models/unet/anima-base-v1.0.safetensors
ls ~/infra/comfyui/models/clip/qwen_3_06b_base.safetensors
ls ~/infra/comfyui/models/vae/qwen_image_vae.safetensors
```

**ControlNet で "DWPreprocessor not found" エラー**
```bash
ls ~/infra/comfyui/custom_nodes/comfyui_controlnet_aux
# なければインストール
cd ~/infra/comfyui/custom_nodes
git clone --depth=1 https://github.com/Fannovel16/comfyui_controlnet_aux.git
```

**IP-Adapter で "IPAdapterUnifiedLoader not found" エラー**
```bash
ls ~/infra/comfyui/custom_nodes/ComfyUI_IPAdapter_plus
ls ~/infra/comfyui/models/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors
```

**画像生成がタイムアウトする**
ComfyUI のログで VRAM 不足や別ジョブの競合を確認する。llama-server と ComfyUI の同時使用で VRAM はピーク約 8〜10GB（9B モデル + Pony XL の場合）。ADetail ON ではさらに追加される。バッチ生成は ComfyUI のキューで順次処理されるため、枚数分の時間がかかる。
