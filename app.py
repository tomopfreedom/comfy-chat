#!/usr/bin/env python3
"""ComfyUI Chat - 日本語自然言語 → 画像生成 Web アプリ"""

import argparse
import json
import pathlib
import random
import re

import aiohttp
from aiohttp import web

from pony_utils import (
    COMFY_BASE, LLM_MODEL, PONY_QUALITY_PREFIX, ILLUSTRIOUS_QUALITY_PREFIX,
    translate_prompt, explain_tags, submit_image_async, get_checkpoints,
)

STATIC_DIR = pathlib.Path(__file__).parent / "static"
LORA_REGISTRY_FILE = pathlib.Path(__file__).parent / "loras.json"

NEGATIVE_PRESETS = {
    "顔崩れ防止": "bad face, crooked nose, asymmetrical eyes, misaligned eyes, uneven eyes, bad teeth, malformed mouth, ugly face",
    "品質向上":   "low quality, worst quality, normal quality, jpeg artifacts, blurry, pixelated, grainy, noisy, out of focus",
    "手崩れ防止": "bad hands, extra fingers, missing fingers, fused fingers, malformed hands, poorly drawn hands, mutated hands, deformed hands",
}


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
    if "pony" in name:
        return "pony"
    if "flux" in name:
        return "flux"
    if "illustrious" in name or "noobai" in name:
        return "illustrious"
    return "sdxl"


def _filter_loras_for_model(registry: list, ckpt_name: str) -> list:
    """選択中チェックポイントと互換性のある LoRA のみ返す。
    base_model が 'any' または一致するモデル種別の LoRA を通過させる。
    illustrious チェックポイントは sdxl LoRA とも互換（同一 SDXL ベース）。
    """
    ckpt = _ckpt_type(ckpt_name)
    allowed = (ckpt, "any") if ckpt != "illustrious" else ("illustrious", "sdxl", "any")
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

    return _dedup_tags(positive)


# ──── ハンドラ ────────────────────────────────────────────────────

async def handle_index(request):
    return web.FileResponse(STATIC_DIR / "index.html")


async def handle_checkpoints(request):
    checkpoints = await get_checkpoints(request.app["session"])
    return web.json_response({"checkpoints": checkpoints})


async def handle_loras_get(request):
    return web.json_response(request.app["lora_registry"])


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
    if base_model not in ("pony", "sdxl", "flux", "illustrious", "any"):
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
        if bm not in ("pony", "sdxl", "flux", "illustrious", "any"):
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
    hires_fix = bool(body.get("hires_fix", False))
    adetail   = bool(body.get("adetail", False))
    init_image       = body.get("init_image") or None
    mask_image       = body.get("mask_image") or None
    denoise_strength = float(body.get("denoise", 0.75))
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
    valid_names = {e["filename"] for e in registry}

    if confirmed_positive is not None:
        # ユーザーが確認・編集済みのタグをそのまま使用（LLM・LoRA後処理を再実行しない）
        positive = _dedup_tags(str(confirmed_positive))
        negative = str(confirmed_negative or "")
        selected_loras = [
            l for l in body.get("loras", [])
            if isinstance(l, dict) and l.get("name") in valid_names
        ]
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

    try:
        img_info = await submit_image_async(
            positive, negative, seed, width, height, steps, cfg,
            checkpoint, selected_loras, session,
            hires_fix=hires_fix, adetail=adetail,
            init_image=init_image, denoise_strength=denoise_strength,
            mask_image=mask_image,
            sampler_name=sampler_name, scheduler=scheduler,
            client_id=client_id,
        )
    except Exception as e:
        return web.json_response({"ok": False, "error": f"ComfyUI エラー: {e}"})

    if img_info is None:
        return web.json_response({"ok": False, "error": "画像生成タイムアウト（300秒経過）"})

    filename  = img_info.get("filename", "")
    subfolder = img_info.get("subfolder", "")
    type_     = img_info.get("type", "output")
    image_url = f"/api/image?filename={filename}&subfolder={subfolder}&type={type_}"

    return web.json_response({
        "ok":        True,
        "positive":  positive,
        "negative":  negative,
        "seed":      seed,
        "image_url": image_url,
        "loras":     selected_loras,
    })


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

    url = f"{COMFY_BASE}/view?filename={filename}&subfolder={subfolder}&type={type_}"
    try:
        async with request.app["session"].get(
            url, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            content = await resp.read()
            ct = resp.headers.get("Content-Type", "image/png").split(";")[0]
        return web.Response(body=content, content_type=ct)
    except Exception as e:
        return web.Response(status=502, text=f"ComfyUI proxy error: {e}")


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


async def handle_negative_presets(request: web.Request) -> web.Response:
    return web.json_response(list(NEGATIVE_PRESETS.keys()))


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
    app.router.add_get("/api/checkpoints",       handle_checkpoints)
    app.router.add_get("/api/loras",             handle_loras_get)
    app.router.add_post("/api/loras",            handle_loras_post)
    app.router.add_patch("/api/loras/{filename}",  handle_loras_patch)
    app.router.add_delete("/api/loras/{filename}", handle_loras_delete)
    app.router.add_get("/api/lora-files",        handle_lora_files)
    app.router.add_post("/api/explain-tags",       handle_explain_tags)
    app.router.add_post("/api/translate",         handle_translate)
    app.router.add_post("/api/generate",         handle_generate)
    app.router.add_post("/api/upload",           handle_upload)
    app.router.add_get("/api/image",             handle_image)
    app.router.add_get("/api/health",            handle_health)
    app.router.add_get("/api/negative-presets",  handle_negative_presets)

    print(f"ComfyUI Chat → http://localhost:{args.port}")
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
