# comfy-chat

日本語の自然言語を Qwen3.5-9B uncensored（llama-server）で Stable Diffusion プロンプトに変換し、ComfyUI で画像を生成する Web アプリ。

## 機能

- **日本語チャット入力** — 会話履歴を保持し、「もっと明るく」「着物姿にして」などの差分指示に対応
- **モデル別プロンプト変換** — Pony Diffusion / SDXL / Flux のプロンプト文法をチェックポイント名で自動切り替え
- **CLIP Skip 2** — Pony / AutismMix 系モデルで自動適用（`CLIPSetLastLayer stop_at_clip_layer=-2`）
- **LoRA 自動選択** — 登録済み LoRA を LLM が自然言語の指示に基づいて自動選択・適用
- **LoRA 互換性管理** — LoRA をベースモデル別（Pony / SDXL / Flux / 汎用）に登録し、非互換 LoRA を自動除外
- **多衣装 LoRA 対応** — 衣装別の活性化トークンを description に記述し、LLM が文脈に応じて衣装を選択
- **サーバー側トリガーワード付加** — LLM に依存せず、選択された LoRA のトリガーワードをサーバーが確実に付加
- **サーバー側衣装タグ補完** — LLM が衣装タグを一部省略した場合もサーバーが全タグを補完
- **Hires fix** — RealESRGAN 4x+ Anime 6B で 4x アップスケール後 2x に縮小して再サンプリング（denoise 0.45）
- **ADetailer** — 顔（`face_yolov8n.pt`）と手（`hand_yolov8n.pt`）を YOLO で検出し、各領域を個別に再 inpaint
- **img2img** — 参照画像をドラッグ&ドロップしてアップロード、変化量スライダーで忠実度を調整
- **インペイント** — Canvas ブラシでマスクを描き、塗った範囲だけを再生成
- **WebSocket 進捗バー** — ComfyUI の生成ステップをリアルタイムで表示
- **ネガティブプリセット** — 「顔崩れ防止」「品質向上」「手崩れ防止」チェックボックスでネガティブプロンプトを補完
- **生成履歴ギャラリー** — 最大 100 件の生成画像・設定を localStorage に保存し、サムネイルクリックで設定を復元
- **Civitai LoRA 検索** — civitai.com / civitai.red から LoRA を検索・ダウンロード・登録（トリガーワード自動取得）
- **アスペクト比クイック設定** — 1:1 / 3:4 / 4:3 / 9:16 / 16:9 ボタンで解像度を即時変更
- **画像ダウンロード** — 生成画像をタイムスタンプ付きファイル名で保存
- **お気に入り** — チャット入力・モデル・解像度・Steps・CFG などの設定セットを名前付きで保存し復元
- **フレーズサジェスト** — 構図/ライティング/雰囲気/場所/スタイル 5カテゴリのフレーズチップをクリックして追記、最近の入力も再利用可能

## 依存サービス

| サービス | ポート | 用途 |
|---------|--------|------|
| llama-server (`mymodel-9b-unc`) | 11434 | 日本語 → プロンプト変換 |
| ComfyUI | 8188 | 画像生成 |

ComfyUI の Python venv（`~/infra/comfyui/venv/`）を使用するため、追加インストール不要。

### ComfyUI カスタムノード（ADetailer 使用時に必要）

```
~/infra/comfyui/custom_nodes/
├── ComfyUI-Impact-Pack      # FaceDetailer, DetailerForEach 等
└── ComfyUI-Impact-Subpack   # UltralyticsDetectorProvider
```

```
~/infra/comfyui/models/ultralytics/bbox/
├── face_yolov8n.pt
└── hand_yolov8n.pt
```

## 起動

```bash
~/projects/comfy-chat/start.sh
```

llama-server（mymodel-9b-unc）と ComfyUI が未起動の場合、自動で起動を試みる。

起動後: http://localhost:9000

## 画面構成

UI は 3 つのタブに分かれている。

| タブ | 内容 |
|------|------|
| 🎨 生成 | 画像生成チャット（メイン機能） |
| 🧩 LoRA 管理 | LoRA の登録・編集・削除・Civitai 検索 |
| 📷 履歴 | 過去の生成画像ギャラリー |

## 使い方

### 画像生成（テキスト → 画像）

1. 「🎨 生成」タブを開く
2. ヘッダーのセレクターでチェックポイントモデル・解像度・Steps・CFG を設定（アスペクト比ボタンで素早く変更可）
3. 必要に応じて **Hires fix**（高解像度化）・**ADetail**（顔・手修正）をチェック
4. チャット欄に日本語で内容を入力し「生成」ボタン（または Ctrl+Enter）
5. 進捗バーで生成状況を確認しながら待機、完了後に右パネルへ画像が表示される
6. 使用プロンプト・Seed・LoRA が確認できる。「💾 保存」ボタンで画像をダウンロード

| チェックボックス | 効果 | 処理時間増加 |
|----------------|------|-------------|
| Hires fix | 生成後に 2x 高解像度化 | +50〜100% |
| ADetail | 顔・手を個別に再描画 | +30〜60% |

### img2img（参照画像から生成）

1. 入力欄上部の「📎 参照画像」エリアに画像をドラッグ&ドロップ（またはクリックして選択）
2. **変化量** スライダーで参照画像への忠実度を調整（低い値 = 元画像に近い、高い値 = 大きく変化）
3. テキスト入力で変更指示を書いて生成

### インペイント（部分再生成）

1. img2img と同様に参照画像をアップロード
2. 「✏️ マスク」ボタンをクリックしてマスクエディタを開く
3. 再生成したい部分をブラシで塗る（消しゴムモード・ブラシサイズ調整可）
4. 「確定」でマスクを適用（ボタンが緑色に変わる）
5. テキストで指示を書いて生成 — マスク範囲のみが再描画される

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
| `any` | ベースモデルを問わず使用できる LoRA |

### プロンプト変換の仕組み

| チェックポイント | 変換スタイル |
|----------------|-------------|
| `*pony*` | Danbooru タグ形式（`score_9, source_anime, ...`） |
| `*flux*` | 自然言語の長文記述 |
| その他 | SDXL 向けタグ＋フレーズ混在形式（`masterpiece, best quality, ...`） |

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
├── pony_utils.py     # LLM / ComfyUI API 連携、ワークフロー構築
├── system_prompt.py  # モデル別システムプロンプト定数
├── loras.json        # LoRA レジストリ（自動生成・更新）
├── static/
│   └── index.html    # シングルファイル Web UI（3タブ構成）
└── start.sh          # サービス起動チェック → venv → Python 実行
```

## ComfyUI ワークフロー構成

| ノード番号 | クラス | 役割 | 条件 |
|-----------|--------|------|------|
| 1 | CheckpointLoaderSimple | モデルロード | 常時 |
| 20 | CLIPSetLastLayer | CLIP Skip 2 | 常時 |
| 2 | EmptyLatentImage | テキスト生成用 latent | txt2img 時 |
| 3〜7 | VAELoader → KSampler → VAEDecode | 基本生成パイプライン | 常時 |
| 8 | SaveImage | 保存 | 常時（入力元が変わる） |
| 30 | LoadImage | 参照画像ロード | img2img / インペイント時 |
| 31 | VAEEncode | 参照画像を latent 化 | img2img（マスクなし）時 |
| 32 | LoadImage | マスク画像ロード | インペイント時 |
| 33 | ImageToMask | グレースケール → マスクテンソル変換 | インペイント時 |
| 34 | VAEEncodeForInpaint | マスク付き latent 化（grow_mask_by=6） | インペイント時 |
| 100〜 | LoraLoader | LoRA チェーン | LoRA 使用時 |
| 9〜14 | Upscale → KSampler → VAEDecode | Hires fix | Hires fix ON 時 |
| 200〜201 | UltralyticsDetectorProvider + FaceDetailer | 顔修正 | ADetail ON 時 |
| 202〜204 | UltralyticsDetectorProvider + BboxDetectorSEGS + DetailerForEach | 手修正 | ADetail ON 時 |

## API エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| `GET` | `/` | Web UI |
| `GET` | `/api/checkpoints` | 利用可能チェックポイント一覧 |
| `POST` | `/api/generate` | 画像生成（LLM 変換 → ComfyUI 送信） |
| `GET` | `/api/image` | ComfyUI `/view` へのプロキシ |
| `GET` | `/api/health` | LLM・ComfyUI の死活確認 |
| `POST` | `/api/upload` | 参照画像・マスク画像を ComfyUI にアップロード |
| `GET` | `/api/loras` | 登録済み LoRA 一覧 |
| `POST` | `/api/loras` | LoRA 登録 |
| `DELETE` | `/api/loras/{filename}` | LoRA 削除 |
| `GET` | `/api/lora-files` | `models/loras/` のファイル一覧 |
| `GET` | `/api/negative-presets` | ネガティブプリセット一覧 |
| `GET` | `/api/civitai/search` | Civitai LoRA 検索（`query`, `limit`, `domain` パラメータ） |
| `POST` | `/api/civitai/download` | Civitai LoRA ダウンロード＆自動登録 |

### POST /api/generate リクエスト形式

```json
{
  "message": "青空の下で猫が昼寝している、アニメ風",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "checkpoint": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
  "width": 1024,
  "height": 1024,
  "steps": 25,
  "cfg": 7.0,
  "seed": -1,
  "hires_fix": false,
  "adetail": false,
  "init_image": null,
  "mask_image": null,
  "denoise": 0.75,
  "negative_presets": [],
  "client_id": "uuid-for-websocket-progress"
}
```

`seed: -1` でランダムシード。レスポンスに使用された実際の seed 値が含まれる。`init_image` / `mask_image` は `/api/upload` が返すファイル名を指定する。`mask_image` を指定した場合は img2img ではなくインペイントワークフローが使用される。`denoise` は `init_image` 指定時のみ有効。`client_id` を指定すると `ws://localhost:8188/ws?clientId=<uuid>` 経由で生成進捗をリアルタイム受信できる。

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

**画像生成がタイムアウトする**
ComfyUI のログで VRAM 不足や別ジョブの競合を確認する。llama-server と ComfyUI の同時使用で VRAM はピーク約 8〜10GB（9B モデル + Pony XL の場合）。ADetail ON ではさらに追加される。
