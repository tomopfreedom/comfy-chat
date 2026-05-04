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

## 使い方

### 画像生成

1. ヘッダーのセレクターでチェックポイントモデル・解像度・Steps・CFG を設定
2. 必要に応じて **Hires fix**（高解像度化）・**ADetail**（顔・手修正）をチェック
3. チャット欄に日本語で内容を入力し「生成」ボタン（または Ctrl+Enter）
4. 右パネルに画像が表示され、使用されたプロンプト・Seed・LoRA が確認できる

| チェックボックス | 効果 | 処理時間増加 |
|----------------|------|-------------|
| Hires fix | 生成後に 2x 高解像度化 | +50〜100% |
| ADetail | 顔・手を個別に再描画 | +30〜60% |

### LoRA の登録

1. 「🧩 LoRA 管理」ボタンをクリック
2. `~/infra/comfyui/models/loras/` に配置した `.safetensors` ファイルを選択
3. 説明・トリガーワード・強度・ベースモデル種別を入力して登録

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
| `flux` | Flux.1 ベースのモデル（現在 UI には非表示、将来拡張用） |
| `any` | ベースモデルを問わず使用できる LoRA |

### プロンプト変換の仕組み

| チェックポイント | 変換スタイル |
|----------------|-------------|
| `*pony*` | Danbooru タグ形式（`score_9, source_anime, ...`） |
| `*flux*` | 自然言語の長文記述 |
| その他 | SDXL 向けタグ＋フレーズ混在形式（`masterpiece, best quality, ...`） |

#### Pony モデルの positive タグ構成（サーバー側で組み立て）

```
[force_tags]       rating:explicit など LoRA 必須タグ
[trigger_words]    LoRA の外見基本タグ（常時付加）
[PONY_QUALITY_PREFIX]  score_9, score_8_up, ..., detailed eyes（品質タグ）
[LLM 出力]         衣装・活性化トークン・状況タグ
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
│   └── index.html    # シングルファイル Web UI
└── start.sh          # サービス起動チェック → venv → Python 実行
```

## ComfyUI ワークフロー構成

| ノード番号 | クラス | 役割 | 条件 |
|-----------|--------|------|------|
| 1 | CheckpointLoaderSimple | モデルロード | 常時 |
| 20 | CLIPSetLastLayer | CLIP Skip 2 | 常時 |
| 2-7 | EmptyLatentImage → VAEDecode | 基本生成パイプライン | 常時 |
| 8 | SaveImage | 保存 | 常時（入力元が変わる） |
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
| `GET` | `/api/loras` | 登録済み LoRA 一覧 |
| `POST` | `/api/loras` | LoRA 登録 |
| `DELETE` | `/api/loras/{filename}` | LoRA 削除 |
| `GET` | `/api/lora-files` | `models/loras/` のファイル一覧 |

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
  "adetail": false
}
```

`seed: -1` でランダムシード。レスポンスに使用された実際の seed 値が含まれる。

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

**画像生成がタイムアウトする**
ComfyUI のログで VRAM 不足や別ジョブの競合を確認する。llama-server と ComfyUI の同時使用で VRAM はピーク約 8〜10GB（9B モデル + Pony XL の場合）。ADetail ON ではさらに追加される。
