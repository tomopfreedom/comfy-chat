#!/usr/bin/env python3
"""ComfyUI Chat - 日本語自然言語 → 画像生成 Web アプリ"""

import argparse
import asyncio
import json
import os
import pathlib
import random
import re
import subprocess
import time
import urllib.parse
import uuid

import aiohttp
from aiohttp import web

from comfy_utils import (
    COMFY_BASE, LLM_MODEL, PONY_QUALITY_PREFIX, ILLUSTRIOUS_QUALITY_PREFIX,
    ANIMA_QUALITY_PREFIX,
    translate_prompt, explain_tags, review_image, submit_image_async, get_checkpoints,
)
from wan_utils import (
    submit_wan_i2v_async,
    queue_wan_i2v_async,
    check_wan_i2v_result,
    WAN_MODEL_A14B, WAN_MODEL_A14B_LOW, WAN_ANIME_LORA,
)
from wm_utils import (
    get_first_frame_jpeg, remove_watermark_video, _validate_video,
)
from wm_comfy import remove_watermark_video_comfy

# ──── llama-server 管理 ────────────────────────────────────────────────────
LLAMA_HEALTH_URL  = "http://localhost:11434/health"
LLAMA_START_SCRIPT = os.path.expanduser("~/infra/start-llama-9b-unc.sh")

def _stop_llama_server() -> None:
    """llama-server を停止して VRAM を解放する。"""
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    time.sleep(2)

async def _restart_llama_server(session: aiohttp.ClientSession) -> None:
    """llama-server を再起動し、/health が ok になるまで最大90秒待機する。"""
    subprocess.Popen([LLAMA_START_SCRIPT])
    deadline = asyncio.get_event_loop().time() + 90
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(3)
        try:
            async with session.get(
                LLAMA_HEALTH_URL, timeout=aiohttp.ClientTimeout(total=3)
            ) as r:
                data = await r.json(content_type=None)
                if data.get("status") == "ok":
                    return
        except Exception:
            pass

# ──── ComfyUI ログ進捗パーサー ────────────────────────────────────────────────
COMFYUI_LOG = "/tmp/comfyui.log"

def _parse_comfyui_progress() -> dict:
    """ComfyUI ログの末尾をパースして Wan2.2 I2V の進捗を返す。

    Returns:
        {stage, step, total_steps, percent, eta_seconds, sec_per_step}
        stage: 1=HighNoise処理中, 2=LowNoise処理中, 0=不明
    """
    import re
    try:
        with open(COMFYUI_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 5000))
            tail = f.read().decode("utf-8", errors="ignore")

        # "Requested to load WAN21" の出現回数でステージを判定
        stage = tail.count("Requested to load WAN21")

        # "70%|███ | 7/10 [23:01<09:51, 197.14s/it]" 形式をパース
        matches = re.findall(
            r'(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s+\[[\d:]+<([\d:]+),\s*([\d.]+)s/it\]',
            tail,
        )
        if matches:
            pct, cur, total, eta_str, spit = matches[-1]
            parts = eta_str.split(":")
            if len(parts) == 2:
                eta_sec = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                eta_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                eta_sec = 0
            return {
                "stage":        max(1, stage),
                "step":         int(cur),
                "total_steps":  int(total),
                "percent":      int(pct),
                "eta_seconds":  eta_sec,
                "sec_per_step": float(spit),
            }
    except Exception:
        pass
    return {"stage": 0, "step": 0, "total_steps": 0, "percent": 0, "eta_seconds": 0, "sec_per_step": 0}


# ──── ComfyUI 管理 ─────────────────────────────────────────────────────────
COMFYUI_START_SCRIPT = os.path.expanduser("~/infra/start-comfyui.sh")

def _stop_comfyui() -> None:
    """ComfyUI を停止する。"""
    subprocess.run(["pkill", "-f", "python main.py"], check=False)
    time.sleep(2)

async def _restart_comfyui(session: aiohttp.ClientSession) -> None:
    """ComfyUI を再起動し、/system_stats が応答するまで最大120秒待機する。"""
    subprocess.Popen(["bash", COMFYUI_START_SCRIPT])
    deadline = asyncio.get_event_loop().time() + 120
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(3)
        try:
            async with session.get(
                f"{COMFY_BASE}/system_stats", timeout=aiohttp.ClientTimeout(total=3)
            ) as r:
                await r.read()
                return
        except Exception:
            pass

STATIC_DIR = pathlib.Path(__file__).parent / "static"
LORA_REGISTRY_FILE = pathlib.Path(__file__).parent / "loras.json"

NEGATIVE_PRESETS = {
    "顔崩れ防止": "bad face, crooked nose, asymmetrical eyes, misaligned eyes, uneven eyes, bad teeth, malformed mouth, ugly face",
    "品質向上":   "low quality, worst quality, normal quality, jpeg artifacts, blurry, pixelated, grainy, noisy, out of focus",
    "手崩れ防止": "bad hands, extra fingers, missing fingers, fused fingers, malformed hands, poorly drawn hands, mutated hands, deformed hands",
}

# ──── Civitai ─────────────────────────────────────────────────────
CIVITAI_API = "https://civitai.com/api/v1"
CIVITAI_UA  = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")
LORA_DIR = pathlib.Path.home() / "infra/comfyui/models/loras"

_BASE_MODEL_MAP = {
    "pony": "pony", "sdxl 1.0": "sdxl", "sdxl turbo": "sdxl",
    "sdxl lightning": "sdxl", "sdxl": "sdxl",
    "flux.1 d": "flux", "flux.1 s": "flux", "flux": "flux",
    "illustrious": "illustrious", "illustrious xl": "illustrious",
    "noobai xl": "illustrious",
}


def _civitai_map_base(bm: str) -> str:
    return _BASE_MODEL_MAP.get(bm.lower().strip(), "any")


def _civitai_headers() -> dict:
    h = {"User-Agent": CIVITAI_UA}
    if CIVITAI_API_KEY:
        h["Authorization"] = f"Bearer {CIVITAI_API_KEY}"
    return h


# ──── LoRA レジストリ ─────────────────────────────────────────────

def _load_registry() -> list:
    if LORA_REGISTRY_FILE.exists():
        return json.loads(LORA_REGISTRY_FILE.read_text(encoding="utf-8"))
    return []


def _save_registry(registry: list) -> None:
    LORA_REGISTRY_FILE.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _parse_costume_map(description: str) -> dict:
    """description の 'token=X, tags=Y,Z,...' を {token: tags_str} に変換する。"""
    result = {}
    for m in re.finditer(r'token=([^,;]+),\s*tags=([^;]+)', description):
        result[m.group(1).strip()] = m.group(2).strip()
    return result


def _dedup_tags(prompt: str) -> str:
    seen: set = set()
    result: list = []
    for tag in prompt.split(","):
        key = tag.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(tag.strip())
    return ", ".join(result)


def _ckpt_type(ckpt_name: str) -> str:
    """チェックポイント名からベースモデル種別を返す。"""
    name = ckpt_name.lower()
    if "anima" in name or "narni" in name:
        return "anima"
    if "pony" in name:
        return "pony"
    if "flux" in name:
        return "flux"
    if "illustrious" in name or "noobai" in name:
        return "illustrious"
    if "z-image" in name:
        return "lumina2"
    return "sdxl"


def _filter_loras_for_model(registry: list, ckpt_name: str) -> list:
    """選択中チェックポイントと互換性のある LoRA のみ返す。
    base_model が 'any' または一致するモデル種別の LoRA を通過させる。
    illustrious チェックポイントは sdxl LoRA とも互換（同一 SDXL ベース）。
    """
    ckpt = _ckpt_type(ckpt_name)
    if ckpt == "illustrious":
        allowed = ("illustrious", "sdxl", "any")
    elif ckpt == "anima":
        allowed = ("anima", "any")
    else:
        allowed = (ckpt, "any")
    return [e for e in registry if e.get("base_model", "any") in allowed]


def _apply_lora_postprocess(positive: str, selected_loras: list, registry_map: dict, checkpoint: str) -> str:
    """LoRA のトリガーワード・force_tags・衣装タグ・品質タグを付加し dedup して返す。"""
    for lora in selected_loras:
        entry = registry_map.get(lora.get("name", ""))
        if entry:
            trigger = entry.get("trigger_words", "").strip()
            if trigger:
                positive = trigger + ", " + positive

    for lora in selected_loras:
        entry = registry_map.get(lora.get("name", ""))
        if entry:
            force = entry.get("force_tags", "").strip()
            if force:
                positive = force + ", " + positive

    for lora in selected_loras:
        entry = registry_map.get(lora.get("name", ""))
        if not entry or "token=" not in entry.get("description", ""):
            continue
        costume_map = _parse_costume_map(entry["description"])
        if not costume_map:
            continue

        known_lower = {t.lower() for t in costume_map.keys()}
        tag_list = [t.strip() for t in positive.split(",") if t.strip()]
        tag_list = [t for t in tag_list if t.lower() not in known_lower]

        positive_set = {t.lower() for t in tag_list}
        best_token = list(costume_map.keys())[0]
        best_count = 0
        for token, tags in costume_map.items():
            count = sum(1 for t in tags.split(",") if t.strip().lower() in positive_set)
            if count > best_count:
                best_count = count
                best_token = token

        tag_list.append(best_token)
        tag_list.extend(t.strip() for t in costume_map[best_token].split(","))
        positive = ", ".join(tag_list)

    ckpt_kind = _ckpt_type(checkpoint)
    if ckpt_kind == "pony":
        positive = PONY_QUALITY_PREFIX + ", " + positive
    elif ckpt_kind == "illustrious":
        positive = ILLUSTRIOUS_QUALITY_PREFIX + ", " + positive
    elif ckpt_kind == "anima":
        positive = ANIMA_QUALITY_PREFIX + ", " + positive

    return _dedup_tags(positive)


# ──── ハンドラ ────────────────────────────────────────────────────

async def handle_index(request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def handle_checkpoints(request):
    checkpoints = await get_checkpoints(request.app["session"])
    # Z-image-Turbo (split) を仮想的に追加
    if "Z-image-Turbo" not in checkpoints:
        checkpoints.append("Z-image-Turbo")
    # Anima-base (UNET/CLIP/VAE split) を仮想的に追加
    if "Anima-base" not in checkpoints:
        checkpoints.append("Anima-base")
    return web.json_response({"checkpoints": checkpoints})


async def handle_loras_get(request):
    checkpoint = request.query.get("checkpoint")
    registry = request.app["lora_registry"]
    if checkpoint:
        registry = _filter_loras_for_model(registry, checkpoint)
    return web.json_response(registry)


async def handle_loras_post(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    filename    = body.get("filename", "").strip()
    description = body.get("description", "").strip()
    if not filename or not description:
        return web.json_response(
            {"ok": False, "error": "filename と description は必須です"}, status=400
        )

    registry = request.app["lora_registry"]
    if any(e["filename"] == filename for e in registry):
        return web.json_response({"ok": False, "error": "既に登録済みです"}, status=400)

    base_model = body.get("base_model", "any")
    if base_model not in ("pony", "sdxl", "flux", "illustrious", "anima", "any"):
        base_model = "any"

    entry = {
        "filename":         filename,
        "description":      description,
        "trigger_words":    body.get("trigger_words", ""),
        "default_strength": float(body.get("default_strength", 0.75)),
        "base_model":       base_model,
    }
    registry.append(entry)
    _save_registry(registry)
    return web.json_response({"ok": True, "entry": entry})


async def handle_loras_patch(request):
    filename = request.match_info["filename"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    registry = request.app["lora_registry"]
    entry = next((e for e in registry if e["filename"] == filename), None)
    if entry is None:
        return web.json_response({"ok": False, "error": "見つかりません"}, status=404)

    if "description" in body:
        desc = body["description"].strip()
        if desc:
            entry["description"] = desc

    if "default_strength" in body:
        try:
            strength = float(body["default_strength"])
        except (TypeError, ValueError):
            return web.json_response({"ok": False, "error": "無効な値です"}, status=400)
        if not 0.1 <= strength <= 2.0:
            return web.json_response(
                {"ok": False, "error": "強度は 0.1〜2.0 の範囲で指定してください"}, status=400
            )
        entry["default_strength"] = round(strength, 2)

    if "base_model" in body:
        bm = body["base_model"]
        if bm not in ("pony", "sdxl", "flux", "illustrious", "anima", "any"):
            return web.json_response({"ok": False, "error": "無効な base_model です"}, status=400)
        entry["base_model"] = bm

    _save_registry(registry)
    return web.json_response({"ok": True, "entry": entry})


async def handle_loras_delete(request):
    filename = request.match_info["filename"]
    registry = request.app["lora_registry"]
    new_registry = [e for e in registry if e["filename"] != filename]
    if len(new_registry) == len(registry):
        return web.json_response({"ok": False, "error": "見つかりません"}, status=404)
    request.app["lora_registry"] = new_registry
    _save_registry(new_registry)
    return web.json_response({"ok": True})


async def handle_lora_files(request):
    """ComfyUI の models/loras フォルダにあるファイル一覧を返す。"""
    try:
        async with request.app["session"].get(
            f"{COMFY_BASE}/api/models/loras",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json(content_type=None)
        files = data if isinstance(data, list) else []
    except Exception:
        files = []
    return web.json_response({"files": files})


async def handle_explain_tags(request):
    """ポジティブタグを LLM で日本語解説してチップ用データを返す。"""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    positive = body.get("positive", "").strip()
    if not positive:
        return web.json_response({"ok": True, "tags": []})

    try:
        tags = await explain_tags(positive, request.app["session"])
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})

    return web.json_response({"ok": True, "tags": tags})


async def handle_translate(request):
    """LLM 翻訳 + LoRA 後処理のみ実行し、ユーザー確認用タグを返す。ComfyUI へは送信しない。"""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    japanese_text = body.get("message", "").strip()
    if not japanese_text:
        return web.json_response({"ok": False, "error": "メッセージが空です"}, status=400)

    history    = body.get("history", [])
    checkpoint = body.get("checkpoint", "ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    session    = request.app["session"]
    registry   = _filter_loras_for_model(request.app["lora_registry"], checkpoint)

    try:
        prompt_data = await translate_prompt(
            japanese_text, history, checkpoint, session, registry
        )
    except Exception as e:
        return web.json_response({"ok": False, "error": f"LLM エラー: {e}"})

    positive = prompt_data.get("positive", "")
    negative = prompt_data.get("negative", "")
    valid_names = {e["filename"] for e in registry}
    selected_loras = [
        l for l in prompt_data.get("loras", [])
        if isinstance(l, dict) and l.get("name") in valid_names
    ]
    registry_map = {e["filename"]: e for e in registry}
    positive = _apply_lora_postprocess(positive, selected_loras, registry_map, checkpoint)

    return web.json_response({
        "ok":              True,
        "positive":        positive,
        "negative":        negative,
        "loras":           selected_loras,
        "available_loras": registry,   # 互換LoRA全リスト（確認パネルのチェックボックスで利用）
        "explanation":     prompt_data.get("explanation", ""),
    })


async def handle_generate(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    japanese_text = body.get("message", "").strip()
    if not japanese_text:
        return web.json_response({"ok": False, "error": "メッセージが空です"}, status=400)

    history    = body.get("history", [])
    checkpoint = body.get("checkpoint", "ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    width  = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    steps  = int(body.get("steps", 25))
    cfg    = float(body.get("cfg", 7.0))
    seed      = int(body.get("seed", -1))
    batch     = max(1, min(int(body.get("batch", 1)), 4))
    hires_fix = bool(body.get("hires_fix", False))
    adetail   = bool(body.get("adetail", False))
    init_image       = body.get("init_image") or None
    mask_image       = body.get("mask_image") or None
    denoise_strength = float(body.get("denoise", 0.75))
    ref_image           = body.get("ref_image") or None
    try:
        ip_adapter_strength = float(body.get("ip_adapter_strength", 0.5))
    except (ValueError, TypeError):
        ip_adapter_strength = 0.5
    controlnet_image    = body.get("controlnet_image") or None
    controlnet_type     = body.get("controlnet_type", "openpose")
    try:
        controlnet_strength = float(body.get("controlnet_strength", 0.8))
    except (ValueError, TypeError):
        controlnet_strength = 0.8
    sampler_name = body.get("sampler", "euler_ancestral")
    scheduler    = body.get("scheduler", "karras")
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    session  = request.app["session"]
    # 選択中チェックポイントと互換性のある LoRA のみ LLM に渡す
    registry = _filter_loras_for_model(request.app["lora_registry"], checkpoint)

    client_id  = body.get("client_id") or None
    negative_presets = body.get("negative_presets", [])
    confirmed_positive = body.get("confirmed_positive")
    confirmed_negative = body.get("confirmed_negative")
    direct_tags = bool(body.get("direct_tags", False))
    valid_names = {e["filename"] for e in registry}

    if direct_tags:
        # タグ直接入力モード: LLM翻訳をスキップし、LoRAのtrigger_words/force_tagsのみ付加する
        positive = _dedup_tags(str(confirmed_positive or japanese_text).strip())
        negative = str(confirmed_negative or "")
        selected_loras = [
            l for l in body.get("loras", [])
            if isinstance(l, dict) and l.get("name") in valid_names
        ]
        registry_map = {e["filename"]: e for e in registry}
        positive = _apply_lora_postprocess(positive, selected_loras, registry_map, checkpoint)
    elif confirmed_positive is not None:
        # ユーザーが確認・編集済みのタグを受け取り、LoRA後処理を再適用する。
        # 再適用しないと確認画面でLoRAを変更してもtrigger/force_tags/衣装タグが反映されない。
        positive = _dedup_tags(str(confirmed_positive))
        negative = str(confirmed_negative or "")
        selected_loras = [
            l for l in body.get("loras", [])
            if isinstance(l, dict) and l.get("name") in valid_names
        ]
        registry_map = {e["filename"]: e for e in registry}
        positive = _apply_lora_postprocess(positive, selected_loras, registry_map, checkpoint)
    else:
        try:
            prompt_data = await translate_prompt(
                japanese_text, history, checkpoint, session, registry
            )
        except Exception as e:
            return web.json_response({"ok": False, "error": f"LLM エラー: {e}"})

        positive = prompt_data.get("positive", "")
        negative = prompt_data.get("negative", "")
        selected_loras = [
            l for l in prompt_data.get("loras", [])
            if isinstance(l, dict) and l.get("name") in valid_names
        ]
        registry_map = {e["filename"]: e for e in registry}
        positive = _apply_lora_postprocess(positive, selected_loras, registry_map, checkpoint)

    if negative_presets:
        preset_tags = ", ".join(
            NEGATIVE_PRESETS[p] for p in negative_presets if p in NEGATIVE_PRESETS
        )
        if preset_tags:
            negative = _dedup_tags(negative + ", " + preset_tags) if negative else preset_tags

    seeds = [seed] + [random.randint(0, 2**32 - 1) for _ in range(batch - 1)]
    tasks = [
        submit_image_async(
            positive, negative, s, width, height, steps, cfg,
            checkpoint, selected_loras, session,
            hires_fix=hires_fix, adetail=adetail,
            init_image=init_image, denoise_strength=denoise_strength,
            mask_image=mask_image,
            ref_image=ref_image,
            ip_adapter_strength=ip_adapter_strength,
            controlnet_image=controlnet_image,
            controlnet_type=controlnet_type,
            controlnet_strength=controlnet_strength,
            sampler_name=sampler_name, scheduler=scheduler,
            client_id=client_id if i == 0 else str(uuid.uuid4()),
        )
        for i, s in enumerate(seeds)
    ]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"ComfyUI エラー: {e}"})

    images = []
    for s, result in zip(seeds, results):
        if isinstance(result, Exception):
            continue
        if result is None:
            continue
        fn = result.get("filename", "")
        sf = result.get("subfolder", "")
        ty = result.get("type", "output")
        images.append({"url": f"/api/image?filename={fn}&subfolder={sf}&type={ty}", "seed": s})

    if not images:
        first_err = next((r for r in results if isinstance(r, Exception)), None)
        msg = str(first_err) if first_err else "画像生成タイムアウト（300秒経過）"
        return web.json_response({"ok": False, "error": f"ComfyUI エラー: {msg}"})

    return web.json_response({
        "ok":        True,
        "positive":  positive,
        "negative":  negative,
        "seed":      images[0]["seed"],
        "image_url": images[0]["url"],
        "images":    images,
        "loras":     selected_loras,
    })



async def _upload_image_to_comfy(session: aiohttp.ClientSession, image_path: str) -> str:
    """ローカルパスの画像を ComfyUI の /upload/image に送信し、ファイル名を返す。

    image_path が絶対パスの場合はバイナリを読み込んでアップロードする。
    相対ファイル名の場合は ComfyUI の input/ にすでにあるとみなしてそのまま返す。
    """
    if os.path.isabs(image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        filename = os.path.basename(image_path)
        # asyncio.to_thread で同期 I/O をオフロード
        data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: open(image_path, "rb").read()
        )
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename, content_type="image/png")
        async with session.post(
            f"{COMFY_BASE}/upload/image",
            data=form,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            result = await resp.json(content_type=None)
        return result.get("name", filename)
    return image_path


async def handle_wan_i2v(request):
    """Wan2.2 I2V A14B カスケード 動画生成エンドポイント。

    リクエスト JSON:
      init_image    str   — 参照画像。以下のいずれかを指定:
                            - ComfyUI input/ のファイル名（例: "tomopi_test.png"）
                            - サーバー上の絶対パス（例: "/home/.../batch_00001_.png"）
      message       str   — 動画の内容を説明するプロンプト
      use_anime_lora bool — アニメ LoRA を適用するか（default: True）
      lora_strength float — アニメ LoRA の強度 (default: 1.0)
      width         int   — 幅 (default: 576)
      height        int   — 高さ (default: 1024)
      frames        int   — フレーム数 (default: 81)
      steps         int   — サンプリングステップ数 (default: 20)
      cfg           float — CFG スケール (default: 3.5)
      seed          int   — シード (-1=ランダム)
      negative      str   — ネガティブプロンプト (default: "")
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    init_image = body.get("init_image", "").strip()
    if not init_image:
        return web.json_response({"ok": False, "error": "init_image が必要です"}, status=400)

    # A14B 2段カスケード（HighNoise+LowNoise）固定
    unet_name      = WAN_MODEL_A14B_LOW
    use_anime_lora = body.get("use_anime_lora", True)
    lora_strength  = float(body.get("lora_strength", 1.0))
    lora_name      = WAN_ANIME_LORA if use_anime_lora else None

    positive = body.get("message", "").strip() or "anime girl, natural motion, high quality"
    negative = body.get("negative", "")
    width    = int(body.get("width",  576))
    height   = int(body.get("height", 1024))
    frames   = int(body.get("frames", 81))
    fps      = int(body.get("fps",    16))
    steps    = int(body.get("steps",  20))
    cfg      = float(body.get("cfg",  3.5))   # A14B 公式推奨値
    seed     = int(body.get("seed",  -1))
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    session = request.app["session"]

    # VRAM を確保するために llama-server を停止
    _stop_llama_server()

    try:
        # 絶対パスの場合は ComfyUI にアップロードしてファイル名を取得
        comfy_image_name = await _upload_image_to_comfy(session, init_image)
        prompt_id = await queue_wan_i2v_async(
            positive=positive,
            negative=negative,
            seed=seed,
            start_image=comfy_image_name,
            session=session,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            unet_name=unet_name,
            lora_name=lora_name,
            lora_strength=lora_strength,
        )
    except Exception as e:
        asyncio.create_task(_restart_llama_server(session))
        return web.json_response({"ok": False, "error": f"Wan I2V エラー: {e}"})

    # prompt_id を返してフロントエンドにポーリングさせる
    return web.json_response({
        "ok":        True,
        "prompt_id": prompt_id,
        "positive":  positive,
        "seed":      seed,
    })


async def handle_wan_i2v_progress(request):
    """Wan I2V の進捗確認エンドポイント。完了時は llama-server を再起動する。

    Response:
      running: {status:"running", stage, step, total_steps, percent, eta_seconds}
      done:    {status:"done", image_url, filename}
      error:   {status:"error", error}
    """
    prompt_id = request.match_info["prompt_id"]
    session   = request.app["session"]
    completed = request.app.setdefault("wan_completed", set())

    # 完了済みキャッシュ（重複再起動防止）
    if prompt_id in completed:
        # 既完了: history から結果を取得して返す
        try:
            result = await check_wan_i2v_result(prompt_id, session)
        except RuntimeError as e:
            return web.json_response({"status": "error", "error": str(e)})
        if result:
            fn = result["filename"]
            sf = result.get("subfolder", "wan_i2v")
            ty = result.get("type", "output")
            return web.json_response({
                "status":    "done",
                "image_url": f"/api/image?filename={fn}&subfolder={sf}&type={ty}",
                "filename":  fn,
            })

    # 完了チェック
    try:
        result = await check_wan_i2v_result(prompt_id, session)
    except RuntimeError as e:
        asyncio.create_task(_restart_llama_server(session))
        return web.json_response({"status": "error", "error": str(e)})

    if result:
        completed.add(prompt_id)
        asyncio.create_task(_restart_llama_server(session))
        fn = result["filename"]
        sf = result.get("subfolder", "wan_i2v")
        ty = result.get("type", "output")
        return web.json_response({
            "status":    "done",
            "image_url": f"/api/image?filename={fn}&subfolder={sf}&type={ty}",
            "filename":  fn,
        })

    # 実行中: ログから進捗をパース
    prog = _parse_comfyui_progress()
    return web.json_response({"status": "running", **prog})


async def handle_upload(request):
    """参照画像を ComfyUI の /upload/image に転送し、ファイル名を返す。"""
    try:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "image":
            return web.json_response(
                {"ok": False, "error": "image フィールドがありません"}, status=400
            )
        data = await field.read()
        filename = field.filename or "upload.png"

        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename,
                       content_type=field.headers.get("Content-Type", "image/png"))
        async with request.app["session"].post(
            f"{COMFY_BASE}/upload/image",
            data=form,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            result = await resp.json(content_type=None)

        uploaded_name = result.get("name", filename)
        return web.json_response({"ok": True, "filename": uploaded_name})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def handle_image(request):
    """ComfyUI /view エンドポイントへのプロキシ。CORS 回避と URL 集約のため。"""
    q = request.rel_url.query
    filename  = q.get("filename", "")
    subfolder = q.get("subfolder", "")
    type_     = q.get("type", "output")

    try:
        async with request.app["session"].get(
            f"{COMFY_BASE}/view",
            params={"filename": filename, "subfolder": subfolder, "type": type_},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            content = await resp.read()
            ct = resp.headers.get("Content-Type", "image/png").split(";")[0]
        return web.Response(body=content, content_type=ct)
    except Exception as e:
        return web.Response(status=502, text=f"ComfyUI proxy error: {e}")


async def handle_review(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

    image_url = (body.get("image_url") or "").strip()
    if not image_url.startswith("/api/image"):
        return web.json_response({"ok": False, "error": "image_url が不正です"}, status=400)

    positive       = body.get("positive") or ""
    user_message   = body.get("user_message") or ""
    review_history = body.get("review_history") or None
    session        = request.app["session"]

    try:
        result = await review_image(
            image_url, positive, session,
            user_message=user_message, review_history=review_history,
        )
        if result.get("positive_fix"):
            existing = {t.strip().lower() for t in positive.split(",") if t.strip()}
            new_tags = [t for t in result["positive_fix"].split(",")
                        if t.strip() and t.strip().lower() not in existing]
            result["positive_fix"] = ", ".join(t.strip() for t in new_tags)
        return web.json_response({"ok": True, **result})
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)})


async def handle_health(request):
    session = request.app["session"]
    results = {"llm": False, "comfyui": False}

    try:
        async with session.get(
            "http://localhost:11434/v1/models",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            data = await resp.json(content_type=None)
        models = [m["id"] for m in data.get("data", [])]
        results["llm"] = LLM_MODEL in models
    except Exception:
        pass

    try:
        async with session.get(
            f"{COMFY_BASE}/system_stats",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            await resp.read()
        results["comfyui"] = True
    except Exception:
        pass

    return web.json_response(results)


async def handle_llm_stop(request):
    """llama-server を停止する（UI ボタン用）。"""
    _stop_llama_server()
    return web.json_response({"ok": True, "message": "llama-server を停止しました"})


async def handle_llm_start(request):
    """llama-server を起動する（UI ボタン用）。バックグラウンドで起動して即レスポンス。"""
    session = request.app["session"]
    asyncio.create_task(_restart_llama_server(session))
    return web.json_response({"ok": True, "message": "llama-server を起動中..."})


async def handle_comfyui_stop(request):
    """ComfyUI を停止する（UI ボタン用）。"""
    _stop_comfyui()
    return web.json_response({"ok": True, "message": "ComfyUI を停止しました"})


async def handle_comfyui_start(request):
    """ComfyUI を起動する（UI ボタン用）。バックグラウンドで起動して即レスポンス。"""
    session = request.app["session"]
    asyncio.create_task(_restart_comfyui(session))
    return web.json_response({"ok": True, "message": "ComfyUI を起動中..."})


async def handle_negative_presets(request: web.Request) -> web.Response:
    return web.json_response(list(NEGATIVE_PRESETS.keys()))


# ──── Civitai ハンドラ ────────────────────────────────────────────

_CIVITAI_TAGS = {"character", "style", "concept", "clothing", "poses", "background"}

async def handle_civitai_search(request: web.Request) -> web.Response:
    query  = request.rel_url.query.get("query", "").strip()
    limit  = min(int(request.rel_url.query.get("limit", 10)), 100)
    domain = request.rel_url.query.get("domain", "civitai.com")
    tag    = request.rel_url.query.get("tag", "").strip().lower()
    if domain not in ("civitai.com", "civitai.red"):
        domain = "civitai.com"
    if tag not in _CIVITAI_TAGS:
        tag = ""
    if not query:
        return web.json_response({"ok": False, "error": "query パラメータが必要です"}, status=400)

    base = f"https://{domain}/api/v1"
    api_params: dict = {"type": "LORA", "query": query, "sort": "Highest Rated", "limit": limit}
    if tag:
        api_params["tag"] = tag
    params = urllib.parse.urlencode(api_params)
    url = f"{base}/models?{params}"
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                url, headers=_civitai_headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return web.json_response(
                        {"ok": False, "error": f"Civitai API エラー: {resp.status}"}, status=502)
                data = await resp.json(content_type=None)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"ネットワークエラー: {e}"}, status=502)

    results = [_civitai_model_to_dict(m) for m in data.get("items", [])]
    return web.json_response({"ok": True, "results": results})


def _civitai_model_to_dict(m: dict) -> dict:
    ver    = (m.get("modelVersions") or [{}])[0]
    images = ver.get("images", [])
    return {
        "id":           m["id"],
        "name":         m.get("name", ""),
        "rating":       round(m.get("stats", {}).get("rating", 0), 1),
        "downloads":    m.get("stats", {}).get("downloadCount", 0),
        "base_model":   _civitai_map_base(ver.get("baseModel", "")),
        "trigger_words": ", ".join(ver.get("trainedWords", [])),
        "version_id":   ver.get("id"),
        "version_name": ver.get("name", ""),
        "preview_url":  images[0]["url"] if images else None,
        "model_url":    f"https://civitai.com/models/{m['id']}",
    }


async def handle_civitai_model(request: web.Request) -> web.Response:
    model_id = request.rel_url.query.get("id", "").strip()
    domain   = request.rel_url.query.get("domain", "civitai.com")
    if domain not in ("civitai.com", "civitai.red"):
        domain = "civitai.com"
    if not model_id.isdigit():
        return web.json_response({"ok": False, "error": "id パラメータ（数字）が必要です"}, status=400)

    url = f"https://{domain}/api/v1/models/{model_id}"
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                url, headers=_civitai_headers(),
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return web.json_response(
                        {"ok": False, "error": f"Civitai API エラー: {resp.status}"}, status=502)
                m = await resp.json(content_type=None)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"ネットワークエラー: {e}"}, status=502)

    return web.json_response({"ok": True, "result": _civitai_model_to_dict(m)})


async def handle_civitai_download(request: web.Request) -> web.Response:
    body       = await request.json()
    model_id   = body.get("model_id")
    version_id = body.get("version_id")
    domain     = body.get("domain", "civitai.com")
    if domain not in ("civitai.com", "civitai.red"):
        domain = "civitai.com"
    base = f"https://{domain}/api/v1"
    if not model_id:
        return web.json_response({"ok": False, "error": "model_id が必要です"}, status=400)

    hdrs = _civitai_headers()
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                f"{base}/models/{model_id}", headers=hdrs,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return web.json_response(
                        {"ok": False, "error": f"モデル情報取得失敗: {resp.status}"}, status=502)
                model = await resp.json(content_type=None)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=502)

    versions = model.get("modelVersions") or []
    if not versions:
        return web.json_response({"ok": False, "error": "バージョンが見つかりません"}, status=404)

    if version_id:
        ver = next((v for v in versions if v["id"] == version_id), None)
        if ver is None:
            return web.json_response(
                {"ok": False, "error": f"version_id {version_id} が見つかりません"}, status=404)
    else:
        ver = versions[0]

    sf_files = [f for f in ver.get("files", []) if f.get("name", "").endswith(".safetensors")]
    if not sf_files:
        return web.json_response(
            {"ok": False, "error": ".safetensors ファイルが見つかりません"}, status=404)
    target   = max(sf_files, key=lambda f: f.get("sizeKB", 0))
    # パストラバーサル対策: 外部APIのファイル名からbasename のみ抽出し拡張子・保存先を検証する
    filename = pathlib.Path(target["name"]).name
    if not filename.endswith(".safetensors"):
        return web.json_response({"ok": False, "error": "不正なファイル名（.safetensors 以外）"}, status=400)
    dl_url   = target["downloadUrl"]
    dest     = LORA_DIR / filename
    if not dest.resolve().is_relative_to(LORA_DIR.resolve()):
        return web.json_response({"ok": False, "error": "パストラバーサル検出"}, status=400)

    # API キーをクエリパラメータとして付与（リダイレクト後も有効、ヘッダーは中継で落ちる）
    if CIVITAI_API_KEY:
        sep = "&" if "?" in dl_url else "?"
        dl_url = f"{dl_url}{sep}token={CIVITAI_API_KEY}"

    skip = dest.exists()
    if not skip:
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    dl_url, headers=hdrs,
                    timeout=aiohttp.ClientTimeout(total=600),
                ) as resp:
                    if resp.status != 200:
                        return web.json_response(
                            {"ok": False, "error": f"ダウンロード失敗: {resp.status}"}, status=502)
                    ct = resp.headers.get("Content-Type", "")
                    if "text/html" in ct:
                        return web.json_response(
                            {"ok": False,
                             "error": "Civitai が HTML を返しました。CIVITAI_API_KEY が未設定か無効です。"
                                      " export CIVITAI_API_KEY=<your_token> を設定して再起動してください。"},
                            status=502)
                    with open(dest, "wb") as fp:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            fp.write(chunk)
        except Exception as e:
            if dest.exists():
                dest.unlink()
            return web.json_response({"ok": False, "error": f"ダウンロードエラー: {e}"}, status=500)

    trigger_words = ", ".join(ver.get("trainedWords", []))
    base_model    = _civitai_map_base(ver.get("baseModel", ""))
    desc_raw      = re.sub(r"<[^>]+>", " ",
                           (model.get("description") or model.get("name") or filename))
    description   = (desc_raw[:200] + "…") if len(desc_raw) > 200 else desc_raw.strip() or filename

    registry = _load_registry()
    existing = next((e for e in registry if e["filename"] == filename), None)
    if existing:
        reg_note = "既に comfy-chat に登録済みです"
    else:
        registry.append({
            "filename":         filename,
            "description":      description,
            "trigger_words":    trigger_words,
            "default_strength": 0.75,
            "base_model":       base_model,
        })
        _save_registry(registry)
        request.app["lora_registry"] = registry
        reg_note = "trigger_words は Civitai 公開情報から自動取得。LoRA管理画面で確認・編集を推奨。"

    return web.json_response({
        "ok":           True,
        "filename":     filename,
        "base_model":   base_model,
        "trigger_words": trigger_words,
        "skip":         skip,
        "note":         reg_note,
    })


# ──── ウォーターマーク除去 ────────────────────────────────────────

async def handle_wm_preview(request: web.Request) -> web.Response:
    """POST /api/wm/preview — 動画の最初のフレームを JPEG で返す。"""
    reader = await request.multipart()
    data = b""
    filename = "upload.mp4"
    async for field in reader:
        if field.name == "video":
            filename = field.filename or filename
            data = await field.read()
            break

    if not data:
        return web.Response(status=400, text="video フィールドが見つかりません")
    try:
        _validate_video(filename, data)
    except ValueError as e:
        return web.Response(status=400, text=str(e))

    loop = asyncio.get_event_loop()
    try:
        jpeg = await loop.run_in_executor(None, get_first_frame_jpeg, data)
    except Exception as e:
        return web.Response(status=500, text=f"フレーム取得エラー: {e}")

    return web.Response(body=jpeg, content_type="image/jpeg")


async def handle_wm_process(request: web.Request) -> web.Response:
    """POST /api/wm/process — ウォーターマークを除去した MP4 を返す。"""
    reader = await request.multipart()
    data = b""
    filename = "upload.mp4"
    params: dict[str, str] = {}
    async for field in reader:
        if field.name == "video":
            filename = field.filename or filename
            data = await field.read()
        else:
            params[field.name] = (await field.read()).decode()

    if not data:
        return web.Response(status=400, text="video フィールドが見つかりません")
    try:
        _validate_video(filename, data)
    except ValueError as e:
        return web.Response(status=400, text=str(e))

    # 数値パラメータのバリデーション
    try:
        x1 = int(params.get("x1", "1140"))
        y1 = int(params.get("y1", "560"))
        x2 = int(params.get("x2", "1245"))
        y2 = int(params.get("y2", "665"))
    except ValueError:
        return web.Response(status=400, text="座標は整数で指定してください")

    keep_audio = params.get("keep_audio", "true").lower() not in ("false", "0", "")
    method     = params.get("method", "telea").lower()

    if method == "comfy":
        # AI インペインティング（ComfyUI）
        ckpt_name = params.get("ckpt_name", "NoobAI-XL-v1.1.safetensors")
        session = request.app["session"]
        try:
            mp4 = await remove_watermark_video_comfy(
                data, x1, y1, x2, y2, keep_audio, session, ckpt_name
            )
        except Exception as e:
            return web.Response(status=500, text=f"AI 処理エラー: {e}")
    else:
        # 標準インペインティング（cv2 TELEA）
        loop = asyncio.get_event_loop()
        try:
            mp4 = await loop.run_in_executor(
                None,
                lambda: remove_watermark_video(data, x1, y1, x2, y2, keep_audio),
            )
        except Exception as e:
            return web.Response(status=500, text=f"処理エラー: {e}")

    return web.Response(
        body=mp4,
        content_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="clean.mp4"'},
    )


# ──── アプリ起動 ──────────────────────────────────────────────────

async def on_startup(app):
    app["session"]       = aiohttp.ClientSession()
    app["lora_registry"] = _load_registry()


async def on_cleanup(app):
    await app["session"].close()


def main():
    ap = argparse.ArgumentParser(description="ComfyUI Chat サーバー")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_get("/",                      handle_index)
    app.router.add_static("/static",             STATIC_DIR)
    app.router.add_get("/api/checkpoints",       handle_checkpoints)
    app.router.add_get("/api/loras",             handle_loras_get)
    app.router.add_post("/api/loras",            handle_loras_post)
    app.router.add_patch("/api/loras/{filename}",  handle_loras_patch)
    app.router.add_delete("/api/loras/{filename}", handle_loras_delete)
    app.router.add_get("/api/lora-files",        handle_lora_files)
    app.router.add_post("/api/explain-tags",       handle_explain_tags)
    app.router.add_post("/api/translate",         handle_translate)
    app.router.add_post("/api/generate",         handle_generate)
    app.router.add_post("/api/wan-i2v",                    handle_wan_i2v)
    app.router.add_get("/api/wan-i2v/progress/{prompt_id}", handle_wan_i2v_progress)
    app.router.add_post("/api/upload",           handle_upload)
    app.router.add_get("/api/image",             handle_image)
    app.router.add_post("/api/review",           handle_review)
    app.router.add_get("/api/health",            handle_health)
    app.router.add_post("/api/llm/stop",          handle_llm_stop)
    app.router.add_post("/api/llm/start",         handle_llm_start)
    app.router.add_post("/api/comfyui/stop",      handle_comfyui_stop)
    app.router.add_post("/api/comfyui/start",     handle_comfyui_start)
    app.router.add_get("/api/negative-presets",  handle_negative_presets)
    app.router.add_get("/api/civitai/search",    handle_civitai_search)
    app.router.add_get("/api/civitai/model",     handle_civitai_model)
    app.router.add_post("/api/civitai/download", handle_civitai_download)
    app.router.add_post("/api/wm/preview",       handle_wm_preview)
    app.router.add_post("/api/wm/process",       handle_wm_process)

    print(f"ComfyUI Chat → http://localhost:{args.port}")
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
