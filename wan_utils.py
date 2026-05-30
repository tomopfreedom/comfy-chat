"""Wan2.2 I2V A14B カスケード動画生成ユーティリティ。

画像生成 (comfy_utils.py) とは分離した動画生成専用モジュール。
"""

import asyncio
import time
import uuid
from typing import Optional

import aiohttp

from comfy_utils import COMFY_BASE, POLL_INTERVAL

# ──── モデル定数 ────────────────────────────────────────────────────

WAN_MODEL_A14B_LOW  = "LowNoise/Wan2.2-I2V-A14B-LowNoise-Q5_K_M.gguf"
WAN_MODEL_A14B_HIGH = "HighNoise/Wan2.2-I2V-A14B-HighNoise-Q5_K_M.gguf"
# 後方互換エイリアス
WAN_MODEL_A14B = WAN_MODEL_A14B_LOW

WAN_ANIME_LORA = "wan2.2_i2v_animestyle_v2_low.safetensors"


# ──── ワークフロービルダー ──────────────────────────────────────────

def _build_wan_a14b_workflow(
        positive: str,
        negative: str,
        seed: int,
        start_image: str,
        width: int = 576,
        height: int = 1024,
        frames: int = 81,
        fps: int = 16,
        steps: int = 20,
        cfg: float = 3.5,
        sampler_name: str = "euler",
        scheduler: str = "simple",
        unet_name: str = WAN_MODEL_A14B_LOW,
        unet_high_name: str = WAN_MODEL_A14B_HIGH,
        vae_name: str = "wan_2.1_vae.safetensors",
        clip_name: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        lora_name: Optional[str] = WAN_ANIME_LORA,
        lora_strength: float = 1.0,
        filename_prefix: str = "wan_i2v/wan") -> dict:
    """Wan2.2 I2V-A14B 2段カスケードワークフロー。

    A14B は HighNoise → LowNoise の2ステージカスケード構造:
      - HighNoise (Stage 1): 粗い16ch動画latentを生成 (steps 0→high_end_step)
      - LowNoise  (Stage 2): HighNoise出力を受け取り精細化 (steps high_end_step→end)
      → ComfyUI の WAN21.concat_cond() が KSampler 実行時に自動的に
        [noise 16ch + image 16ch + mask 4ch = 36ch] に結合する

    重要: Wan2.2 5B とは異なり A14B は Wan2.1 VAE (16ch) を使用。
          wan2.2_vae.safetensors (48ch) を使うと 36ch エラーが発生する。

    ノード番号:
      1    UnetLoaderGGUF       — LowNoise GGUF (精細化ステージ)
      1H   UnetLoaderGGUF       — HighNoise GGUF (粗生成ステージ)
      1HL  LoraLoaderModelOnly  — HighNoise に LoRA 適用 (lora_name 指定時)
      1L   LoraLoaderModelOnly  — LowNoise に LoRA 適用 (lora_name 指定時)
      2    CLIPLoader           — UMT5-XXL テキストエンコーダー (type=wan)
      3    VAELoader            — Wan2.1 VAE (16ch)
      4H   ModelSamplingSD3     — HighNoise サンプリングシフト (shift=8.0)
      4L   ModelSamplingSD3     — LowNoise サンプリングシフト (shift=8.0)
      5    LoadImage            — 参照画像
      5b   ImageScale           — 参照画像を width×height にリサイズ
      7    CLIPTextEncode       — positive テキスト条件付け
      8    CLIPTextEncode       — negative テキスト条件付け
      9    WanImageToVideo      — 画像を conditioning に注入 → COND×2 + LATENT(16ch) 出力
      10   KSamplerAdvanced     — Stage 1: HighNoise (start=0, end=high_end_step)
      11   KSamplerAdvanced     — Stage 2: LowNoise (start=high_end_step, end=steps)
      12   VAEDecode            — latent → 画像フレーム列
      13   SaveAnimatedWEBP     — アニメーション WebP 保存
    """
    # カスケード分割点: HighNoise が担当するステップ数 (総ステップの約半分)
    high_end_step = steps // 2

    low_model_src  = "1L"  if lora_name else "1"
    high_model_src = "1HL" if lora_name else "1H"

    workflow = {
        # ─── モデルローダー ───
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": unet_name},          # LowNoise
        },
        "1H": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": unet_high_name},     # HighNoise
        },
        # ─── テキスト/VAEエンコーダー ───
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip_name, "type": "wan"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},            # Wan2.1 VAE (16ch)
        },
        # ─── サンプリングシフト (A14B カスケードに必須) ───
        "4H": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": [high_model_src, 0], "shift": 8.0},
        },
        "4L": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": [low_model_src, 0], "shift": 8.0},
        },
        # ─── 参照画像 ───
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": start_image, "upload": "image"},
        },
        "5b": {
            "class_type": "ImageScale",
            "inputs": {
                "image":          ["5", 0],
                "width":          width,
                "height":         height,
                "upscale_method": "lanczos",
                "crop":           "center",
            },
        },
        # ─── テキスト条件付け ───
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": positive},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": negative},
        },
        # ─── I2V conditioning ───
        # WanImageToVideo: start_image を VAEエンコードして concat_latent_image として
        # conditioning に設定する。KSampler 実行時に WAN21.concat_cond() が
        # [noise 16ch + image 16ch + mask 4ch] を自動的に結合し 36ch 入力を生成。
        # 出力: [CONDITIONING(pos), CONDITIONING(neg), LATENT(16ch zeros)]
        "9": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive":    ["7", 0],
                "negative":    ["8", 0],
                "vae":         ["3", 0],
                "start_image": ["5b", 0],
                "width":       width,
                "height":      height,
                "length":      frames,
                "batch_size":  1,
            },
        },
        # ─── Stage 1: HighNoise KSampler (粗生成) ───
        # return_with_leftover_noise="enable": 最終ノイズをそのまま次ステージへ渡す
        "10": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model":                    ["4H", 0],
                "add_noise":               "enable",
                "noise_seed":               seed,
                "steps":                    steps,
                "cfg":                      cfg,
                "sampler_name":             sampler_name,
                "scheduler":               scheduler,
                "positive":                 ["9", 0],
                "negative":                 ["9", 1],
                "latent_image":             ["9", 2],   # 16ch zeros
                "start_at_step":            0,
                "end_at_step":              high_end_step,
                "return_with_leftover_noise": "enable",
            },
        },
        # ─── Stage 2: LowNoise KSampler (精細化) ───
        # latent_image に HighNoise の出力を受け取る
        # concat_cond() が HighNoise latent + 参照画像 + mask = 36ch を自動構築
        "11": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model":                    ["4L", 0],
                "add_noise":               "disable",
                "noise_seed":               seed,
                "steps":                    steps,
                "cfg":                      cfg,
                "sampler_name":             sampler_name,
                "scheduler":               scheduler,
                "positive":                 ["9", 0],   # 同じ conditioning (concat_latent_image 入り)
                "negative":                 ["9", 1],
                "latent_image":             ["10", 0],  # HighNoise 出力 latent
                "start_at_step":            high_end_step,
                "end_at_step":              10000,
                "return_with_leftover_noise": "disable",
            },
        },
        # ─── デコード & 保存 ───
        "12": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
        },
        "13": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images":          ["12", 0],
                "filename_prefix": filename_prefix,
                "fps":             fps,
                "lossless":        False,
                "quality":         90,
                "method":          "default",
            },
        },
    }

    if lora_name:
        # LoRA は両ステージのモデルに同じものを適用する
        workflow["1HL"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model":          ["1H", 0],
                "lora_name":      lora_name,
                "strength_model": lora_strength,
            },
        }
        workflow["1L"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model":          ["1", 0],
                "lora_name":      lora_name,
                "strength_model": lora_strength,
            },
        }

    return workflow


# ──── 送信（即返し）& 完了チェック ────────────────────────────────────

async def queue_wan_i2v_async(
        positive: str,
        negative: str,
        seed: int,
        start_image: str,
        session: aiohttp.ClientSession,
        width: int = 576,
        height: int = 1024,
        frames: int = 81,
        fps: int = 16,
        steps: int = 20,
        cfg: float = 5.0,
        sampler_name: str = "uni_pc",
        scheduler: str = "simple",
        unet_name: str = WAN_MODEL_A14B_LOW,
        lora_name: Optional[str] = None,
        lora_strength: float = 1.0,
        filename_prefix: str = "wan_i2v/wan") -> str:
    """ワークフローをキューに入れて prompt_id を即返す（完了は待たない）。"""
    client_id = str(uuid.uuid4())
    workflow = _build_wan_a14b_workflow(
        positive=positive, negative=negative, seed=seed,
        start_image=start_image, width=width, height=height,
        frames=frames, fps=fps, steps=steps, cfg=cfg,
        sampler_name=sampler_name, scheduler=scheduler,
        unet_name=unet_name, lora_name=lora_name,
        lora_strength=lora_strength, filename_prefix=filename_prefix,
    )
    async with session.post(
        f"{COMFY_BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json(content_type=None)

    if "prompt_id" not in data:
        err = data.get("error", {})
        msg = err.get("details") or err.get("message", "validation error")
        raise ValueError(f"Wan I2V ワークフロー検証エラー: {msg}")
    return data["prompt_id"]


async def check_wan_i2v_result(
        prompt_id: str,
        session: aiohttp.ClientSession) -> Optional[dict]:
    """prompt_id の完了を一度だけ確認する。完了なら結果 dict、未完了なら None を返す。"""
    try:
        async with session.get(
            f"{COMFY_BASE}/history/{prompt_id}",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            hist = await resp.json(content_type=None)
    except Exception:
        return None

    if prompt_id not in hist:
        return None
    entry = hist[prompt_id]
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        msgs = status.get("messages", [])
        raise RuntimeError(f"Wan I2V ComfyUI エラー: {msgs}")
    outputs = entry.get("outputs", {})
    if "13" in outputs:
        imgs = outputs["13"].get("images", [])
        if imgs:
            img = imgs[0]
            return {
                "filename": img["filename"],
                "subfolder": img.get("subfolder", "wan_i2v"),
                "type":     img.get("type", "output"),
            }
    return None


# ──── 送信 & ポーリング（後方互換） ─────────────────────────────────────

async def submit_wan_i2v_async(
        positive: str,
        negative: str,
        seed: int,
        start_image: str,
        session: aiohttp.ClientSession,
        width: int = 576,
        height: int = 1024,
        frames: int = 81,
        fps: int = 16,
        steps: int = 20,
        cfg: float = 5.0,
        sampler_name: str = "uni_pc",
        scheduler: str = "simple",
        unet_name: str = WAN_MODEL_A14B_LOW,
        lora_name: Optional[str] = None,
        lora_strength: float = 1.0,
        client_id: Optional[str] = None,
        filename_prefix: str = "wan_i2v/wan") -> Optional[dict]:
    """Wan2.2 I2V-A14B カスケードワークフローを ComfyUI に送信してポーリングする。

    lora_name に WAN_ANIME_LORA 等を指定するとアニメ LoRA を両ステージに適用。
    戻り値: {"filename": "wan_xxxxx.webp", "subfolder": "wan_i2v", "type": "output"}
    または None（タイムアウト）
    """
    if client_id is None:
        client_id = str(uuid.uuid4())

    workflow = _build_wan_a14b_workflow(
        positive=positive,
        negative=negative,
        seed=seed,
        start_image=start_image,
        width=width,
        height=height,
        frames=frames,
        fps=fps,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        unet_name=unet_name,
        lora_name=lora_name,
        lora_strength=lora_strength,
        filename_prefix=filename_prefix,
    )

    async with session.post(
        f"{COMFY_BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json(content_type=None)

    if "prompt_id" not in data:
        err = data.get("error", {})
        msg = err.get("details") or err.get("message", "validation error")
        raise ValueError(f"Wan I2V ワークフロー検証エラー: {msg}")

    prompt_id = data["prompt_id"]

    # ポーリング: SaveAnimatedWEBP (ノード "13") の完了を検出
    # A14B カスケードは2段。llama-server 同時起動時はVRAMオフロードで1ステップ約90秒かかるため40分に設定
    POLL_TIMEOUT_WAN = 2400  # 40 分
    deadline = time.monotonic() + POLL_TIMEOUT_WAN
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            async with session.get(
                f"{COMFY_BASE}/history/{prompt_id}",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                hist = await resp.json(content_type=None)
        except Exception:
            continue
        if prompt_id not in hist:
            continue
        entry = hist[prompt_id]
        status = entry.get("status", {})
        if status.get("status_str") == "error":
            msgs = status.get("messages", [])
            raise RuntimeError(f"Wan I2V ComfyUI エラー: {msgs}")
        outputs = entry.get("outputs", {})
        if "13" in outputs:
            imgs = outputs["13"].get("images", [])
            if imgs:
                img = imgs[0]
                return {
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", "wan_i2v"),
                    "type": img.get("type", "output"),
                }
    return None
