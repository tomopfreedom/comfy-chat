## comfy-chat 概要

日本語自然言語 → Stable Diffusion プロンプト変換 + 画像生成 Web アプリ。

**生成フロー:**
1. ユーザーが日本語で描写を入力（例: 「赤い髪の少女が海辺に立っている」）
2. LLM（Qwen3.5-9B 無検閲版）が英語の positive / negative prompt に翻訳し、最適な LoRA を自動選択
3. サーバー側で `trigger_words` / `force_tags` / 衣装タグを付加・重複除去
4. ComfyUI の `/prompt` エンドポイントにワークフローを送信 → 完成までポーリング
5. 生成結果をブラウザに表示

**搭載済み機能:** img2img / インペイント（赤/黒マスク描画）/ Hi-res fix / ADetailer / seed 固定 / タグ解説（LLM）

> **設計上の注意**: LLM → サーバー後処理 → ComfyUI 送信というパイプラインが核心部。`trigger_words` / 衣装 costume_map / dedup を LLM 語彙外で完結させる設計がボトルネック回避の鍵であり、新機能追加時もこのパイプラインを壊さないことが重要。

---

# comfy-chat 追加機能候補

## 優先度: 高

### 1. 生成履歴ギャラリー

**概要**: 過去の生成画像をサムネイル一覧で見返せるサイドパネル。

**実装方針**:
- `localStorage` に `{seed, positive, negative, image_url, checkpoint, loras, timestamp}` を保存（最大 100 件）
- プレビューパネル右側にギャラリーペインを追加（トグル表示）
- サムネイルクリックで当時のプロンプト・設定を入力欄に復元
- 「お気に入り」ピン留め機能（`comfy_chat_favorites` キーと統合）

**影響範囲**: `static/index.html` のみ。バックエンド変更なし。

---

### 2. WebSocket 進捗表示

**概要**: 画像生成中に KSampler のステップ進捗（例: `15 / 25`）をリアルタイム表示。

**実装方針**:
- ComfyUI は `ws://localhost:8188/ws?clientId=<uuid>` で進捗メッセージを配信
  - メッセージタイプ: `progress` `{"value": N, "max": M}`、`execution_cached`、`executed`
- `/api/generate` の呼び出し前にフロントエンドが WebSocket 接続を開き、`prompt_id` と紐付け
- `pony_utils.submit_image_async()` は `prompt_id` を返すだけに変更し、ポーリングをフロントエンドの WS 受信に置き換える
- 既存のポーリングロジック（`/history/{prompt_id}`）はフォールバックとして残せる

**影響範囲**: `app.py`（`handle_generate` の戻り値に `prompt_id` 追加）、`pony_utils.py`（`submit_image_async` の戻り値変更）、`static/index.html`（WebSocket 接続追加）。

---

### 3. バッチ生成（複数 seed バリエーション）

**概要**: 同じプロンプトで N 枚（seed を変えて）まとめて生成し、ベストを選ぶ。

**実装方針**:
- UI にバッチ数セレクタ（1/2/4 枚）を追加
- `handle_generate` 内で `asyncio.gather()` を使い N 回の `submit_image_async` を並列実行
- プレビューパネルに N 枚のサムネイルグリッドを表示し、クリックで拡大
- 各画像のシードは結果パネルに個別表示 → 気に入ったシードを固定再生成に使える

**影響範囲**: `app.py`（`handle_generate`）、`static/index.html`（UI 追加）。

---

### 4. プロンプトお気に入り・テンプレート

**概要**: 頻繁に使うプロンプト構成を名前付きで保存し、ワンクリックで呼び出す。

**背景**: 同じ描写パターンを繰り返す際に毎回入力し直す必要がある。`comfy_chat_favorites` キーはすでに `localStorage` に存在するが、入力テキスト全体を保存する UI が未実装。

**実装方針**:
- 入力欄の横に「★ お気に入り登録」ボタンを追加 → 名前入力ダイアログで保存
- サイドパネル（または #1 のギャラリーパネルと統合）にテンプレート一覧を表示
- クリックでテキストエリアに内容を展開 → 編集して送信できる
- `localStorage` キー `comfy_chat_favorites` に `{name, text, timestamp}[]` 形式で保存

**影響範囲**: `static/index.html` のみ。

---

## 優先度: 中

### 5. 設定プリセット保存

**概要**: width / height / steps / cfg / sampler / scheduler / hires_fix / adetail の組み合わせを名前付きで保存・呼び出す。

**実装方針**:
- `localStorage` キー `comfy_chat_presets` に `{name, params}[]` を保存
- ヘッダーに「プリセット」ドロップダウン + 保存ボタンを追加
- デフォルトプリセット例: 「高品質」(steps=30, hires+adetail)、「高速」(steps=15)、「ポートレート」(768×1152)

**影響範囲**: `static/index.html` のみ。

---

### 6. アスペクト比クイック設定ボタン

**概要**: よく使うアスペクト比をワンクリックで幅・高さに反映。

**実装方針**:
- ヘッダーにボタン群を追加: 1:1 (1024×1024) / 3:4 (768×1024) / 4:3 (1024×768) / 9:16 (768×1152) / 16:9 (1152×768)
- 現在の width / height 入力を連動更新するだけ

**影響範囲**: `static/index.html` のみ。

---

### 7. ControlNet 対応

**概要**: 参照画像でポーズ・ラインアートなどの構図を制御。

**実装方針**:
- 前提: ComfyUI に `ComfyUI-Advanced-ControlNet` カスタムノードと ControlNet モデルの導入が必要
- `/api/upload` による参照画像アップロードはすでに実装済み
- `_build_workflow()` に ControlNet ノード群（300番台）を追加
  - `ControlNetLoader`、`ControlNetApplyAdvanced`
- UI にコントロールタイプ選択（openpose / canny / depth / lineart）と強度スライダーを追加

**影響範囲**: `pony_utils.py`（`_build_workflow()`）、`app.py`（パラメータ追加）、`static/index.html`（UI 追加）。ComfyUI 側のノードインストールも必要。

---

### 8. LoRA 自動登録

**概要**: ComfyUI にインストール済みの新着 LoRA を検出し、`loras.json` への追加を提案する。

**背景**: 現在 `loras.json` は手動編集のみ。新しい LoRA をインストールするたびに JSON を直接書き換える必要があり運用負荷が高い。

**実装方針**:
- `GET /api/models/loras`（ComfyUI）でインストール済み LoRA ファイル名一覧を取得
- `loras.json` と突き合わせて未登録のものを抽出
- `app.py` に `GET /api/loras/detect` エンドポイントを追加してフロントエンドに差分を返す
- UI に「未登録 LoRA が N 件見つかりました」バナーを表示 → クリックで雛形付き登録ダイアログを開く

**影響範囲**: `app.py`（エンドポイント追加）、`static/index.html`（検出 UI 追加）。

---

### 9. 画像ダウンロード・共有ボタン

**概要**: 生成画像をワンクリックで保存、およびシード + プロンプトをクリップボードにコピーする。

**背景**: 現状はブラウザの右クリック保存のみ。シードと入力テキストを一緒に記録する手段がない。

**実装方針**:
- 画像プレビュー下に「Download」ボタンを追加（`<a href="..." download="...">` で実装）
- 「シード + プロンプトをコピー」ボタンで `seed: XXXX\npositive: ...\nnegative: ...` をクリップボードに書き込む
- ファイル名は `comfy_YYYYMMDD_HHMMSS_seedXXXX.png` 形式

**影響範囲**: `static/index.html` のみ。

---

### 10. ネガティブプロンプトプリセット

**概要**: 「顔崩れ防止」「品質向上」などのネガティブタグセットをユーザーが ON/OFF 切り替えできる。

**背景**: 現在は LLM が返す negative prompt を固定で使用。ユーザーが品質制御タグを追加・除外したい場合の手段がない。

**実装方針**:
- プリセット例: 「顔崩れ防止」(`bad face, crooked nose, ...`)、「品質向上」(`low quality, worst quality, ...`)、「手崩れ防止」(`bad hands, extra fingers, ...`)
- `static/index.html` にプリセット checkbox 群を追加
- 選択済みプリセットを `/api/generate` リクエストの `negative_presets` フィールドで送信
- `app.py` でサーバー側 negative に付加し `_dedup_tags()` で重複除去

**影響範囲**: `app.py`（`handle_generate` のパラメータ追加）、`static/index.html`（checkbox UI 追加）。

---

## 優先度: 低

### 11. ネガティブプロンプト直接編集 ✅ 実装済み

**概要**: 生成前確認パネルでネガティブプロンプトもユーザーが直接編集できるようにする。

**実装済み**: `.confirm-neg-ta` textarea が確認パネルに既に存在し、`confirmed_negative` フィールドとして `/api/generate` に送信済み。

---

### 12. プロンプト履歴検索

**概要**: `comfy_chat_recent_inputs`（最近の入力 15 件）を検索できる入力補完。

**実装方針**:
- テキストエリアの `input` イベントで履歴をインクリメンタル検索
- マッチした候補をドロップダウンで表示し、クリックで補完

**影響範囲**: `static/index.html` のみ。

---

### 13. 会話コンテキスト管理（タブ/スレッド）

**概要**: セッションをタブに分割し、タブごとに独立した会話履歴を保持する。

**背景**: 現在は直近 3 往復（6 メッセージ）のみ送信される。「この画像をベースに背景だけ変えて」のような継続的編集フローでは文脈が失われやすい。

**実装方針**:
- `localStorage` にタブ別会話履歴 `{tab_id, name, history[]}` を保存
- ヘッダーにタブバーを追加。「+ 新しいセッション」でタブを追加、タブ間を切り替え
- `/api/generate` はステートレスなため、フロントエンドがアクティブタブの履歴全体を送信するだけで対応可能（バックエンド変更なし）
- タブを閉じると履歴を削除（確認ダイアログ付き）

**影響範囲**: `static/index.html` のみ。

---

### 14. ComfyUI キュー状況の可視化

**概要**: 現在の ComfyUI キュー長をリアルタイム表示し、「キューが詰まっている」状態を把握しやすくする。

**背景**: comfy-chat 外で ComfyUI を操作しているときにキューが積み上がり、生成が遅延しても原因が分かりにくい。

**実装方針**:
- ComfyUI の `GET /queue` から `{"queue_running": [...], "queue_pending": [...]}` を取得
- `app.py` に `GET /api/queue_status` エンドポイントを追加してプロキシ（CORS 回避）
- 生成ボタン付近にキューバッジ（例: 「待機: 3 件」）を表示
- 5 秒ごとのポーリングで更新（#2 の WebSocket が実装済みであれば `status` イベントから流用可能）

**影響範囲**: `app.py`（エンドポイント追加）、`static/index.html`（バッジ UI 追加）。
