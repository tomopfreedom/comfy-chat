# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 起動

```bash
~/projects/comfy-chat/start.sh
```

依存サービス（llama-server `mymodel-9b-unc` と ComfyUI）が未起動の場合は自動起動を試みる。
起動後: http://localhost:9000

直接 Python で実行する場合（venv は手動でアクティベートする必要あり）:

```bash
source ~/infra/comfyui/venv/bin/activate
python ~/projects/comfy-chat/app.py --port 9000
```

依存パッケージ（aiohttp 等）は ComfyUI の venv に同梱されており、追加インストール不要。

## アーキテクチャ

### Python バックエンド（3ファイル構成）

```
app.py          HTTP ハンドラ（aiohttp.web）、LoRA レジストリ管理、positive の後処理
comfy_utils.py   LLM 呼び出し（translate_prompt）、ComfyUI ワークフロー構築と送信
system_prompt.py  モデル別システムプロンプト定数（PONY / SDXL / FLUX）
```

**リクエストの流れ（POST /api/generate）**:
1. `app.py`: チェックポイント名で `_filter_loras_for_model()` → 互換 LoRA のみを選別
2. `pony_utils.translate_prompt()`: 会話履歴 + LoRA リストをシステムプロンプトに注入 → LLM へ送信 → JSON 抽出
3. `app.py`: LLM が選んだ LoRA をホワイトリスト検証 → `trigger_words` / `force_tags` / 衣装タグをサーバー側で positive に付加 → dedup
4. `pony_utils.submit_image_async()`: `_build_workflow()` でノードグラフ構築 → `/prompt` 送信 → ポーリング（3秒間隔、300秒タイムアウト）

### ComfyUI ワークフロー ノード番号体系

| 番号帯 | 用途 |
|-------|------|
| 1-8   | 基本パイプライン（CheckpointLoader, CLIPSkip, KSampler, VAEDecode, SaveImage） |
| 9-14  | Hires fix（UpscaleModel → ImageScale → VAEEncode → KSampler → VAEDecode） |
| 20    | CLIPSetLastLayer（stop_at_clip_layer=-2、常時適用） |
| 30-36 | img2img / インペイント（LoadImage, ImageScale, VAEEncode / VAEEncodeForInpaint） |
| 100+  | LoRA チェーン（LoRA 数に応じて動的に生成） |
| 200-204 | ADetailer（FaceDetailer, BboxDetectorSEGS, DetailerForEach） |

`save_image_src` は adetail→204、hires_fix→14、通常→7 と事前決定し、SaveImage（ノード8）の入力として渡す。ポーリングは常に outputs["8"]["images"] を監視する。

### img2img / インペイント の latent 切り替え

`_build_workflow()` 内で生成モードに応じて latent_src を事前計算し、KSampler に渡す:

```
mask_image あり → node 34 (VAEEncodeForInpaint) — インペイント
init_image のみ → node 31 (VAEEncode)          — img2img
なし            → node 2  (EmptyLatentImage)    — txt2img
```

node 35 (ImageScale): init_image を width×height にリサイズ（VRAM対策・出力サイズ統一）
node 36 (ImageScale): mask_image を width×height にリサイズ（インペイント時のみ）

マスクは「赤=マスク範囲、黒=そのまま」で PNG エンコードし `/api/upload` で ComfyUI にアップロード。
ComfyUI側は `ImageToMask(channel="red")` で解釈する。

### LoRA パイプライン（サーバー側後処理）

LLM は LoRA の `filename` を選ぶが、活性化トークン（LoRA 固有の任意文字列）は LLM の語彙にないため誤りやすい。そのため:

- **`trigger_words`**: キャラ基本外見タグ。毎回 positive の先頭に無条件付加
- **`force_tags`**: 必須タグ（例: `rating:explicit`）。LLM 出力に関わらず付加
- **`costume_map`**: `description` の `token=X, tags=Y,Z` を正規表現でパース。positive 内のタグと照合して最多一致の衣装を選択し、そのトークン + 全タグを付加

Pony モデル時は `PONY_QUALITY_PREFIX` を最後に先頭へ付加し、全体を `_dedup_tags()` で重複除去する。

### フロントエンド

`static/index.html` のシングルファイル構成（バンドルツール不使用）。

- **localStorage キー**: `comfy_chat_favorites`（お気に入り）、`comfy_chat_recent_inputs`（最近の入力）
- **img2img 状態**: `uploadedImageName`（アップロード済みファイル名）、`maskCanvas`（ネイティブ解像度の赤/黒マスク）、`uploadedMaskName`
- **マスク描画**: `maskCanvas` はネイティブ画像サイズ、表示用 `display-canvas` はビューポートサイズにスケール。ペイント座標を `maskCanvas.width / dCanvas.width` で逆変換して書き込む
- **フレーズサジェスト**: `PHRASES` 配列（5カテゴリ）+ `RECENT_KEY` で最近の入力15件を保持

### LLM 特有の処理

- `_strip_think()`: Qwen3 系モデルが出力する `<think>...</think>` タグを除去してから JSON 抽出
- 会話履歴は直近 3 往復分（6メッセージ）のみ送信（`history[-6:]`）
- `chat_template_kwargs: {"enable_thinking": false}` で Qwen3 の思考モードを無効化

### loras.json スキーマ

```json
{
  "filename": "xxx.safetensors",
  "description": "... [衣装名]: token=TOKEN, tags=tag1,tag2; ...",
  "trigger_words": "tag1, tag2",
  "force_tags": "rating:explicit",
  "default_strength": 0.75,
  "base_model": "pony"   // pony | sdxl | flux | any
}
```

`force_tags` フィールドは任意。多衣装 LoRA は `description` に `token=`, `tags=` を `;` 区切りで記述する。

## 依存サービス

| サービス | ポート | 確認コマンド |
|---------|--------|-------------|
| llama-server (`mymodel-9b-unc`) | 11434 | `curl http://localhost:11434/v1/models` |
| ComfyUI | 8188 | `curl http://localhost:8188/system_stats` |

ComfyUI カスタムノード（ADetailer に必要）:
- `~/infra/comfyui/custom_nodes/ComfyUI-Impact-Pack`
- `~/infra/comfyui/custom_nodes/ComfyUI-Impact-Subpack`
- `~/infra/comfyui/models/ultralytics/bbox/face_yolov8n.pt`
- `~/infra/comfyui/models/ultralytics/bbox/hand_yolov8n.pt`

## チェックポイントのモデル種別判定

`_ckpt_type(ckpt_name)` はチェックポイント名（小文字）に `pony` / `flux` が含まれるかで判定する。
これにより LoRA フィルタリング・システムプロンプト選択・品質タグ付加が自動切り替わる。
