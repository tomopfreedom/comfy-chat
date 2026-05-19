## comfy-chat 概要

日本語自然言語 → Stable Diffusion プロンプト変換 + 画像生成 Web アプリ。

**生成フロー:**
1. ユーザーが日本語で描写を入力（例: 「赤い髪の少女が海辺に立っている」）
2. LLM（Qwen3.5-9B 無検閲版）が英語の positive / negative prompt に翻訳し、最適な LoRA を自動選択
3. サーバー側で `trigger_words` / `force_tags` / 衣装タグを付加・重複除去
4. ComfyUI の `/prompt` エンドポイントにワークフローを送信 → 完成までポーリング
5. 生成結果をブラウザに表示

---

## 実装済み機能

### AI 画像レビュー機能 (Vision LLM)
**概要**: 生成した画像を Vision LLM（Qwen3-VL-8B-Instruct）で解析し、指の数・解剖学的エラー・
構図の崩れ・プロンプトとの整合性を自動評価する。

**実装内容**:
- 生成完了後に「🔍 AI レビュー」ボタンを表示
- 9B LLM を一時停止して Vision LLM（port 11435）を自動起動
- スコア(1-10)・問題リスト・総合コメント・修正タグ提案（positive/negative）を日本語で表示
- 修正提案が特定された場合は「↓ 入力欄に反映」ボタンでチャット入力欄に内容をセット
- レビュー結果に対してチャット形式で反論・補足が可能（誤指摘の訂正など）
- 評価モデル: `Qwen3-VL-8B-Instruct-Q4_K_M.gguf`（`start-llama-vision.sh` で port 11435）

### UltimateSDUpscale（タイル分割 Hires fix）
**概要**: hires_fix 時の再サンプリングを `ComfyUI_UltimateSDUpscale` による 512px タイル分割処理に置き換え。旧実装（6ノード）を 2ノードに集約し、VRAM ピークを削減しながら出力品質を向上させる。

**実装内容**:
- 旧: `UpscaleModelLoader → ImageUpscaleWithModel → ImageScale → VAEEncode → KSampler → VAEDecode`（6ノード）
- 新: `UpscaleModelLoader → UltimateSDUpscale`（2ノード）
- タイル 512×512・`upscale_by=2.0`・`mode_type="Linear"` で動作
- VRAM ピーク約 9.5GB（旧比 −1.5GB）、1024×1024 入力 → 2048×2048 出力
- Extension: `ComfyUI_UltimateSDUpscale`（サブモジュール `ultimate-upscale-for-automatic1111` を含む）

### タグ補完（Danbooru タグ英語・日本語入力対応）
**概要**: 確認パネルの `confirmed_positive` / `confirmed_negative` 編集中に Danbooru タグ候補をドロップダウン表示。英語入力（前方一致優先）・日本語入力（部分一致）の両方に対応。

**実装内容**:
- WD14 tagger の `selected_tags.csv`（一般タグ 6,947 件）を `static/tags.json` に変換して配信
- 日本語翻訳は `boorutan/booru-japanese-tag` の人力翻訳（427件）＋機械翻訳（~100k件）をマージ（カバレッジ 99.7%）
- 2文字以上の入力でドロップダウン表示、↑↓Enter/Esc/クリックで操作
- 確認パネルのタグ解説チップも同データでローカルルックアップ（即時表示）し、未訳タグのみ LLM で補完

### マスクモーダルからのワンクリック再生成（インペイント即時生成）
**概要**: マスク編集モーダルに「確定して生成」ボタンを追加。
comfy-chat で生成済みの画像の違和感のある部分（指・目など）をマスクして、
同じプロンプトのまま LLM 翻訳なし・確認パネルなしで即時インペイント再生成できる。

**実装内容**:
- 「確定して生成」ボタン: マスクアップロード → モーダルを閉じる → `generateFromMask()` 呼び出し
- `generateFromMask()`: `lastPositive` / `lastNegative` を `confirmed_positive` / `confirmed_negative` として `/api/generate` に直接 POST（LLM 翻訳スキップ）
- `lastPositive` がない場合（初回 / 外部画像）はアラートを出してガード。その場合は既存のチャット入力フローでマスクが自動適用される
- マスクアップロード処理を `uploadMask()` に共通化（`mask-apply-btn` も同関数を使用）
- マスク外は元画像保持、マスク内のみ再生成（`VAEEncodeForInpaint` + `grow_mask_by: 6`）

---

# comfy-chat 追加機能候補（未実装）

## 優先度: 中

### 2. 未登録 LoRA の自動検出
**概要**: ComfyUI の `models/loras` フォルダにあるが、まだ `loras.json` に登録されていないファイルを検出し、登録を促す。

### 3. Flux 系列の導入（次世代高精度生成）
**概要**: 構造把握・文字描写に優れた Flux モデルへの対応。
**実装方針**:
- `system_prompt.py` の `FLUX_SYSTEM_PROMPT`（文章形式）を有効化。
- DiT アーキテクチャ用の新規ワークフロー構築（UnetLoader ではなく専用ノードを使用）。
- GGUF 版 Flux による 16GB VRAM での運用最適化。

### 7. Anima 対応（Cosmos-Predict2 系アニメモデル）

**概要**: NVIDIA Cosmos-Predict2-2B ベースのアニメ特化モデル「Anima」への対応。SDXL 系とは全く異なるアーキテクチャで、複数キャラクター描写・背景込みシーンの品質が Illustrious 系を上回ると報告されている。**Anima-Turbo（公式高速版、近日予定）のリリース後に再評価する。**

**調査日**: 2026-05-15

**Anima の特徴**:
- **ベース**: NVIDIA Cosmos-Predict2-2B（SDXL 系ではない）
- **強み**: 複数キャラクター同時描写・背景の奥行き/大気感・プロンプト追従性（自然言語＋Danbooru タグ両対応）
- **弱点**: 推論速度（BF16対応GPUでも約3倍遅い）・LoRAエコシステム未成熟・アーティストスタイルドリフト（RoPE構造的問題）・衣装色指定の忠実度が低い
- **速度改善策**: `--fp16-unet` + TorchCompileModel で最大75%改善。Anima-Turbo 待ち
- **ライセンス**: 生成画像の商用利用は可、モデル本体の商用サービス組み込みは不可

**comfy-chat への実装方針**:
- `_build_workflow()` を全面書き換え（Flux対応と同規模）
  - `CheckpointLoader`（ノード1）→ `UNETLoader` + `CLIPLoader` + `VAELoader` に分離
  - `CLIPSkip`（ノード20）不要
  - `KSampler` → `SamplerCustomAdvanced` 等 Cosmos 系サンプラーに変更
  - 推奨サンプラー: `er_sde`（シャープ）/ `euler_a`（柔らか）、ステップ数 30〜50、CFG 4〜6
- `system_prompt.py` に Anima 向けプロンプト定数を追加（タグ＋自然言語ハイブリッド）
- `_ckpt_type()` に `anima` 判定を追加し LoRA フィルタ・システムプロンプトを切り替え
- LoRA は Anima 互換品に差し替えが必要（WAI-Anima 等が候補）

**ファイル配置**:
```
anima-base-v1.0.safetensors  (3.89GB, bf16)  → ComfyUI/models/diffusion_models/
qwen_3_06b_base.safetensors  (1.11GB)        → ComfyUI/models/text_encoders/
qwen_image_vae.safetensors   (242MB)         → ComfyUI/models/vae/
```

**ダウンロード先**: `circlestone-labs/Anima`（HuggingFace）/ CivitAI [Anima Official]

**再評価トリガー**:
- Anima-Turbo リリースで速度差が 1.5倍以内に縮まったとき
- Illustrious 互換 LoRA の Anima 移植が進んだとき

---

## 優先度: 高

### 6. ComfyUI キュー状況の可視化
**概要**: 現在の ComfyUI キュー長（待機件数）をリアルタイム表示する。ComfyUI の `/queue` エンドポイントをポーリングするだけで実装でき、生成中のユーザー体験を即座に改善できる。

## 優先度: 低

### 4. ControlNet 対応
**概要**: 参照画像でポーズ・ラインアートなどの構図を制御。
**実装方針**:
- `_build_workflow()` に ControlNet ノード群（300番台）を追加。
- UI にコントロールタイプ選択（openpose / canny / depth / lineart）と強度スライダーを追加。
- E5（AdvancedControlNet）とセットで実装することを推奨（強度制御なしは過剰適用しやすい）。

### 5. 会話コンテキスト管理（タブ/スレッド）
**概要**: セッションをタブに分割し、タブごとに独立した会話履歴を保持する。フロントエンドの大幅改修が必要。

---

# ComfyUI カスタムノード（Extension）による拡張候補

現在インストール済み: `ComfyUI-Impact-Pack`（ADetailer）のみ。
`_build_workflow()` にノードを追加するだけで統合できる。

## 優先度: 高

### E3. WD14 Tagger ノード（画像 → タグ自動抽出）
**Extension**: `ComfyUI-WD14-Tagger`
**概要**: img2img 時に参照画像から Danbooru タグを自動生成し、LLM のコンテキストに注入する。
**実装方針**:
- `handle_upload` 後にタグ抽出を非同期実行 → `translate_prompt()` のシステムプロンプトに参照タグとして追加。
- 既存の `static/tags.json`（6,947件）と同じ語彙を共有するため整合性が高い。タグ補完機能との相乗効果も大きい。

## 優先度: 中

### E2. IP-Adapter（参照画像スタイル転写）
**Extension**: `ComfyUI_IPAdapter_plus`
**概要**: アップロードした参照画像のスタイル・キャラクター・雰囲気を生成画像に転写する。img2img より自然で柔軟な転写が可能。
**実装方針**:
- UI に「スタイル参照画像」アップロード欄と強度スライダーを追加。
- `_build_workflow()` に IP-Adapter ノード群（500番台）を追加し、KSampler の conditioning に接続。
- `IPAdapterModelLoader` + `IPAdapterApply` の 2 ノード構成。

### E4. Inpaint Crop & Stitch（高品質インペイント）
**Extension**: `ComfyUI-Inpaint-CropAndStitch`
**概要**: マスク領域を切り抜いて高解像度でインペイントし、元画像に貼り戻す。全体解像度で処理する現行実装より顔・手の細部が大幅向上。ADetailer との相乗効果が高い。
**実装方針**:
- インペイントパス（node 34: `VAEEncodeForInpaint`）を本 Extension のノード群に置き換え。

### E6. SUPIR（AI 画像復元・超解像）
**Extension**: `ComfyUI-SUPIR`
**概要**: 生成済み画像を AI で復元・高精細化する後処理機能。
**実装方針**:
- 生成完了後に「✨ 画質向上」ボタンを追加し、SUPIR ワークフローを別途実行。

## 優先度: 低

### E5. AdvancedControlNet（ControlNet 強度制御）
**Extension**: `ComfyUI-Advanced-ControlNet`
**概要**: ControlNet の条件付け強度をサンプリングステップごとに制御。#4 の ControlNet 実装後に必要になる（単独では意味をなさない）。
**実装方針**:
- 標準 `ControlNetApply` の代替として使用。ステップ範囲指定（start/end %）で過剰適用を防止。
