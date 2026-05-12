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

---

# comfy-chat 追加機能候補（未実装）

## 優先度: 高

### 1. タグ補完 + WAI キャラ選択パネル
**概要**: `confirmed_positive` / `confirmed_negative` 編集中に Danbooru/e621 タグ候補をドロップダウン表示。折りたたみ式の WAI キャラクター選択パネルからキャラ名をプロンプトに注入。
**実装方針**: `danbooru_e621_merged.csv` と `wai-characters.csv` を `static/` に配置し、`/static` ルートで配信。フロントエンドのみの変更で完結。

## 優先度: 中

### 2. 未登録 LoRA の自動検出
**概要**: ComfyUI の `models/loras` フォルダにあるが、まだ `loras.json` に登録されていないファイルを検出し、登録を促す。

### 3. Flux 系列の導入（次世代高精度生成）
**概要**: 構造把握・文字描写に優れた Flux モデルへの対応。
**実装方針**:
- `system_prompt.py` の `FLUX_SYSTEM_PROMPT`（文章形式）を有効化。
- DiT アーキテクチャ用の新規ワークフロー構築（UnetLoader ではなく専用ノードを使用）。
- GGUF 版 Flux による 16GB VRAM での運用最適化。

### 4. ControlNet 対応
**概要**: 参照画像でポーズ・ラインアートなどの構図を制御。
**実装方針**:
- `_build_workflow()` に ControlNet ノード群（300番台）を追加。
- UI にコントロールタイプ選択（openpose / canny / depth / lineart）と強度スライダーを追加。

## 優先度: 低

### 5. 会話コンテキスト管理（タブ/スレッド）
**概要**: セッションをタブに分割し、タブごとに独立した会話履歴を保持する。

### 6. ComfyUI キュー状況の可視化
**概要**: 現在の ComfyUI キュー長（待機件数）をリアルタイム表示する。
