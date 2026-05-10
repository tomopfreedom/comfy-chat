## comfy-chat 概要

日本語自然言語 → Stable Diffusion プロンプト変換 + 画像生成 Web アプリ。

**生成フロー:**
1. ユーザーが日本語で描写を入力（例: 「赤い髪の少女が海辺に立っている」）
2. LLM（Qwen3.5-9B 無検閲版）が英語の positive / negative prompt に翻訳し、最適な LoRA を自動選択
3. サーバー側で `trigger_words` / `force_tags` / 衣装タグを付加・重複除去
4. ComfyUI の `/prompt` エンドポイントにワークフローを送信 → 完成までポーリング
5. 生成結果をブラウザに表示

---

# comfy-chat 追加機能候補（未実装）

## 優先度: 高（生成能力・体験の飛躍的向上）

### 1. Flux 系列の導入（次世代高精度生成）
**概要**: 構造把握・文字描写に優れた Flux モデルへの対応。
**実装方針**:
- `system_prompt.py` の `FLUX_SYSTEM_PROMPT`（文章形式）を有効化。
- DiT アーキテクチャ用の新規ワークフロー構築（UnetLoader ではなく専用ノードを使用）。
- GGUF 版 Flux による 16GB VRAM での運用最適化。

### 2. バッチ生成（複数 seed バリエーション）✅ 実装済み
**概要**: 同じプロンプトで N 枚（seed を変えて）まとめて生成し、ベストを選ぶ。
**実装済み内容**:
- UI にバッチ数セレクタ（1/2/4 枚）を追加。
- `handle_generate` 内で `asyncio.gather()` を使い N 回の `submit_image_async` を並列実行。
- グリッド表示でサムネイル選択 → 拡大表示の UI フロー。

## 優先度: 中（利便性・こだわり機能）

### 4. AI 画像レビュー機能 (Vision LLM)
**概要**: 生成した画像を Vision LLM で解析し、違和感（指の数、解剖学的エラー、構図の崩れ）を自動評価する。
**使用モデル**: `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`
**実装方針**:
- `llama-server` で Vision モデルを起動し、API 経由で画像を送信。
- 生成完了後に「AI レビュー」ボタンを表示し、修正ポイントを日本語でフィードバック。
- フィードバックを次回の生成指示（チャット履歴）に自動反映させる連携。

### 5. ControlNet 対応
**概要**: 参照画像でポーズ・ラインアートなどの構図を制御。
**実装方針**:
- `_build_workflow()` に ControlNet ノード群（300番台）を追加。
- UI にコントロールタイプ選択（openpose / canny / depth / lineart）と強度スライダーを追加。

### 6. 未登録 LoRA の自動検出
**概要**: ComfyUI の `models/loras` フォルダにあるが、まだ `loras.json` に登録されていないファイルを検出し、登録を促す。

### 7. 会話コンテキスト管理（タブ/スレッド）
**概要**: セッションをタブに分割し、タブごとに独立した会話履歴を保持する。

## 優先度: 低（システム補助）

### 8. ComfyUI キュー状況の可視化
**概要**: 現在の ComfyUI キュー長（待機件数）をリアルタイム表示する。

---

# comfy-chat 機能追加計画: タグ補完 + WAI キャラ選択パネル

## Context

comfy-chat は「日本語テキスト → LLM → タグ生成」でComfyUIの画像生成を行うWebアプリ。
現状はタグを手で追記・修正する際の補助機能がない。
WAI-illustrious-SDXL への移行を検討しているため、Danbooruタグ補完と
5149キャラのクイック選択機能を追加して使い勝力を向上させる。

**追加機能2本:**
- **機能A**: Danbooru/e621 タグ補完（confirmed_positive / negative 編集中に候補ドロップダウン）
- **機能B**: WAI キャラクター選択パネル（折りたたみ式、クリックしてキャラ名をプロンプトに注入）

## 変更ファイル一覧

| ファイル | 変更種別 |
|---------|---------|
| `app.py` | 1行追加（line 493 直後） |
| `static/index.html` | CSS ~150行 + HTML ~15行 + JS ~220行 追加、既存 1118行を3行に変更 |
| `static/danbooru_e621_merged.csv` | 新規配置（.gitignore 対象） |
| `static/wai-characters.csv` | 新規配置（.gitignore 対象） |
| `.gitignore` | 末尾に2行追加 |

## ステップ 1: データファイルの準備

```bash
# danbooru/e621 タグ補完CSV（a1111-tagcomplete 配布版）
wget -O ~/projects/comfy-chat/static/danbooru_e621_merged.csv \
  "https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/raw/main/tags/danbooru_e621_merged.csv"

# WAI キャラ一覧CSV（flagrantia/character_select_stand_alone_app）
wget -O ~/projects/comfy-chat/static/wai-characters.csv \
  "https://huggingface.co/datasets/flagrantia/character_select_stand_alone_app/resolve/main/data/wai_characters.csv"
```

**CSV形式の確認:**
- `danbooru_e621_merged.csv`: `name,score,count,aliases` 4列ヘッダーなし
  - score: 0=General, 1=Artist, 3=Copyright, 4=Character, 5=Meta, 7=e621-General
- `wai-characters.csv`: `name,md5hash` 2列ヘッダーなし、5149行

## ステップ 2: .gitignore に追記

```
static/danbooru_e621_merged.csv
static/wai-characters.csv
```

## ステップ 3: app.py に静的ルートを追加

**変更箇所:** `app.py` line 493（`add_get("/api/health", ...)` の直後の空行）

```python
    app.router.add_get("/api/health",            handle_health)
    app.router.add_static("/static", STATIC_DIR)   # ← この1行を追加
```

これにより `fetch("/static/danbooru_e621_merged.csv")` 等が解決する。

## 設計上の注意点

- **キャラ名の注入タイミング**: `/api/translate` レスポンス受信後、`addConfirmMsg` 呼び出し前（line 1118）。ユーザーが確認パネルで編集可能なため後から修正可能。
- **LoRA後処理との関係**: `_apply_lora_postprocess()` はバックエンドで `confirmed_positive` に対して `_dedup_tags()` のみ実行するため、キャラ名の重複は自動排除される。
- **CSV の遅延ロード**: 両CSVは初回使用時に1度だけ fetch する（`charLoaded` / `tagLoaded` フラグ）。
- **イベント委任**: `.confirm-pos-ta` / `.confirm-neg-ta` は動的生成要素のため `document.addEventListener` による委任で実装する。`esc()` 関数は line 1087 に既存のため再定義不要。
- **wai-characters.csv の実際のカラム確認**: ダウンロード後、先頭数行で `name,md5` の2カラム構成を確認すること。フォーマットが異なる場合はパースロジックを調整。
