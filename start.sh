#!/bin/bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── 秘密情報の読み込み ────────────────────────────────────────────
if [[ -f ~/infra/secrets.env ]]; then
  source ~/infra/secrets.env
fi
PORT=9000
LLAMA_URL="http://localhost:11434"
COMFY_URL="http://localhost:8188"

get_llama_model() {
  curl -fsS "$LLAMA_URL/v1/models" 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || true
}

# ── llama-server: mymodel-9b-unc を確認 ──────────────────────────
current_model=$(get_llama_model)

if [[ "$current_model" != "mymodel-9b-unc" ]]; then
  echo "[comfy-chat] llama-server を uncensored モデルに切り替えます..."
  ~/infra/start-llama-9b-unc.sh
  echo "[comfy-chat] モデルのロードを待機中..."
  for i in $(seq 1 24); do
    loaded=$(get_llama_model)
    [[ "$loaded" == "mymodel-9b-unc" ]] && break
    sleep 5
  done
  if [[ "$loaded" != "mymodel-9b-unc" ]]; then
    echo "[comfy-chat] llama-server の準備に失敗しました。/tmp/llama-server.log を確認してください"
    exit 1
  fi
  echo "[comfy-chat] llama-server 準備完了 (mymodel-9b-unc)"
else
  echo "[comfy-chat] llama-server 確認済み (mymodel-9b-unc)"
fi

# ── ComfyUI: 未起動なら起動 ──────────────────────────────────────
if ! curl -fsS "$COMFY_URL/system_stats" > /dev/null 2>&1; then
  echo "[comfy-chat] ComfyUI を起動します..."
  ~/infra/start-comfyui.sh
  echo "[comfy-chat] ComfyUI の起動を待機中..."
  for i in $(seq 1 12); do
    curl -fsS "$COMFY_URL/system_stats" > /dev/null 2>&1 && break
    sleep 5
  done
  if ! curl -fsS "$COMFY_URL/system_stats" > /dev/null 2>&1; then
    echo "[comfy-chat] ComfyUI の準備に失敗しました。/tmp/comfyui.log を確認してください"
    exit 1
  fi
  echo "[comfy-chat] ComfyUI 準備完了"
else
  echo "[comfy-chat] ComfyUI 確認済み"
fi

# ── 既存プロセスを停止 ───────────────────────────────────────────
pkill -f "python.*app\.py.*--port $PORT" 2>/dev/null && sleep 1 || true

echo "[comfy-chat] Web アプリを起動します: http://localhost:$PORT"
source ~/infra/comfyui/venv/bin/activate
exec python "$APP_DIR/app.py" --port $PORT
