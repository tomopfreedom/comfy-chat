"""ComfyUI / llama-server 連携ユーティリティ。
pony_auto.py から必要な関数をコピーし、async 化・パラメータ化した版。
"""

import asyncio
import json
import re
import time
import uuid
from typing import Optional

import aiohttp

from system_prompt import PONY_SYSTEM_PROMPT, SDXL_SYSTEM_PROMPT, FLUX_SYSTEM_PROMPT, ILLUSTRIOUS_SYSTEM_PROMPT

LLAMA_URL = "http://localhost:11434/v1/chat/completions"
COMFY_BASE = "http://localhost:8188"
LLM_MODEL = "mymodel-9b-unc"

POLL_INTERVAL = 3

# Pony モデルで必須の品質タグ。LLM が無視した場合もサーバー側で保証する。
PONY_QUALITY_PREFIX = (
    "score_9, score_8_up, score_7_up, masterpiece, 1girl, source_anime, "
    "(anime style:1.2), (clear lines:1.2), (simple aesthetic:1.1), "
    "bright lighting, colorful, (depth of field:1.3), detailed eyes"
)
ILLUSTRIOUS_QUALITY_PREFIX = "masterpiece, best quality, newest, absurdres, highres"
POLL_TIMEOUT = 300


# ──── テキスト処理 ───────────────────────────────────────────────

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict:
    text = _strip_think(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise ValueError(f"JSON not found in LLM response: {text[:200]}")


def _select_system_prompt(ckpt_name: str) -> str:
    name = ckpt_name.lower()
    if "pony" in name:
        return PONY_SYSTEM_PROMPT
    elif "flux" in name:
        return FLUX_SYSTEM_PROMPT
    elif "illustrious" in name or "noobai" in name:
        return ILLUSTRIOUS_SYSTEM_PROMPT
    return SDXL_SYSTEM_PROMPT


def _inject_loras(base_prompt: str, registry: list) -> str:
    """登録済み LoRA をシステムプロンプトに埋め込み、LLM に選択させる。
    トリガーワードはサーバーが positive の先頭に自動付加するが、
    LLM がトリガーワードの内容を知らないとキャラ描写が矛盾するため、
    内容を明示したうえでキャラ外見タグの追加を禁止する。"""
    if not registry:
        return base_prompt
    lines = ["\n\nAvailable LoRAs — select 0 or more that match the user's description:"]
    for entry in registry:
        strength = entry.get("default_strength", 0.75)
        trigger = entry.get("trigger_words", "").strip()
        lines.append(f'- {entry["filename"]}: {entry["description"]} (strength: {strength})')
        if trigger:
            lines.append(
                f'  NOTE: The server will AUTO-PREPEND these base appearance tags: [{trigger}]. '
                f'Do NOT duplicate or contradict them.'
            )
        desc = entry.get("description", "")
        if "token=" in desc:
            lines.append(
                f'  COSTUME SELECTION: The description above lists multiple costumes with their '
                f'activation tokens (token=...) and tags. Based on the user\'s request, select '
                f'the appropriate costume token and include it PLUS its listed tags directly in '
                f'your positive output. Default to the first listed costume if unspecified.'
            )
    lines += [
        '\nAdd a "loras" key to your JSON with selected LoRAs:',
        '{"positive": "...", "negative": "...", "loras": [...], "explanation": "..."}',
        'If no LoRAs match, use "loras": []',
    ]
    return base_prompt + "\n".join(lines)


# ──── ComfyUI ワークフロー ────────────────────────────────────────

def _build_workflow(positive: str, negative: str, seed: int,
                    width: int, height: int, steps: int, cfg: float,
                    ckpt_name: str = "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
                    loras: list = None,
                    hires_fix: bool = False,
                    upscale_model: str = "RealESRGAN_x4plus_anime_6B.pth",
                    adetail: bool = False,
                    init_image: Optional[str] = None,
                    denoise_strength: float = 0.75,
                    mask_image: Optional[str] = None,
                    sampler_name: str = "euler_ancestral",
                    scheduler: str = "karras") -> dict:
    if loras is None:
        loras = []

    # CLIP Skip 判定: Pony, NoobAI, Illustrious 系は -2、それ以外は -1
    name_l = ckpt_name.lower()
    clip_skip = -2 if ("pony" in name_l or "noob" in name_l or "illustrious" in name_l) else -1

    # ワークフロー辞書の初期化
    workflow = {}

    # 1. ロード部分の構成
    if ckpt_name == "Z-image-Turbo":
        workflow["1"] = {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "z_image_turbo_bf16.safetensors", "weight_dtype": "fp8_e4m3fn"}
        }
        workflow["22"] = {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["1", 0], "shift": 3.0}
        }
        workflow["21"] = {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "lumina2"}
        }
        workflow["3"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ae.safetensors"}
        }
        model_src_base = ["22", 0]
        clip_src_base  = ["21", 0]
    else:
        workflow["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name}
        }
        workflow["20"] = {
            "class_type": "CLIPSetLastLayer",
            "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip}
        }
        workflow["3"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "sdxl_vae.safetensors"}
        }
        model_src_base = ["1", 0]
        clip_src_base  = ["20", 0]

    # 2. LoRA ノードを 100 番台で直列に連結
    prev_model, prev_clip = model_src_base, clip_src_base
    for i, lora in enumerate(loras):
        node_id = str(100 + i)
        strength = float(lora.get("strength", 0.75))
        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": prev_model,
                "clip":  prev_clip,
                "lora_name":      lora["name"],
                "strength_model": strength,
                "strength_clip":  strength,
            },
        }
        prev_model = [node_id, 0]
        prev_clip  = [node_id, 1]
    
    model_src = prev_model
    clip_src  = prev_clip

    # 3. Latent 入力の決定 (txt2img / img2img / Inpaint)
    if init_image:
        workflow["30"] = {
            "class_type": "LoadImage",
            "inputs": {"image": init_image, "upload": "image"},
        }
        workflow["35"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image":          ["30", 0],
                "upscale_method": "lanczos",
                "width":           width,
                "height":          height,
                "crop":            "disabled",
            },
        }
        if mask_image:
            workflow["32"] = {
                "class_type": "LoadImage",
                "inputs": {"image": mask_image, "upload": "image"},
            }
            workflow["36"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image":          ["32", 0],
                    "upscale_method": "lanczos",
                    "width":           width,
                    "height":          height,
                    "crop":            "disabled",
                },
            }
            workflow["33"] = {
                "class_type": "ImageToMask",
                "inputs": {"image": ["36", 0], "channel": "red"},
            }
            workflow["34"] = {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels":       ["35", 0],
                    "vae":          ["3", 0],
                    "mask":         ["33", 0],
                    "grow_mask_by": 6,
                },
            }
            latent_src = ["34", 0]
        else:
            workflow["31"] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["35", 0], "vae": ["3", 0]},
            }
            latent_src = ["31", 0]
    else:
        workflow["2"] = {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
        latent_src = ["2", 0]

    base_denoise = denoise_strength if init_image else 1.0

    # 4. メインの生成ノード群
    workflow.update({
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": positive, "clip": clip_src}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": clip_src}},
        "6": {"class_type": "KSampler",
              "inputs": {
                  "seed": seed, "steps": steps, "cfg": cfg,
                  "sampler_name": sampler_name, "scheduler": scheduler,
                  "denoise": base_denoise, "model": model_src,
                  "positive": ["4", 0], "negative": ["5", 0],
                  "latent_image": latent_src,
              }},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
    })

    # 5. Hires fix
    if hires_fix:
        hires_steps = max(10, steps // 2)
        workflow["9"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": upscale_model},
        }
        workflow["10"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {"upscale_model": ["9", 0], "image": ["7", 0]},
        }
        workflow["11"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "lanczos",
                "width":  width * 2,
                "height": height * 2,
                "crop": "disabled",
            },
        }
        workflow["12"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["11", 0], "vae": ["3", 0]},
        }
        workflow["13"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": hires_steps, "cfg": cfg,
                "sampler_name": sampler_name, "scheduler": scheduler,
                "denoise": 0.45, "model": model_src,
                "positive": ["4", 0], "negative": ["5", 0],
                "latent_image": ["12", 0],
            },
        }
        workflow["14"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["13", 0], "vae": ["3", 0]},
        }

    # 6. ADetailer
    if adetail:
        adetail_steps = max(10, steps // 2)
        base_image_src = ["14", 0] if hires_fix else ["7", 0]
        workflow["200"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/face_yolov8n.pt"},
        }
        workflow["201"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": base_image_src, "model": model_src, "clip": clip_src, "vae": ["3", 0],
                "guide_size": 384, "guide_size_for": True, "max_size": 1024,
                "seed": seed, "steps": adetail_steps, "cfg": cfg,
                "sampler_name": sampler_name, "scheduler": scheduler,
                "positive": ["4", 0], "negative": ["5", 0],
                "denoise": 0.45, "feather": 5, "noise_mask": True, "force_inpaint": True,
                "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
                "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False",
                "drop_size": 10, "bbox_detector": ["200", 0], "wildcard": "", "cycle": 1,
            },
        }
        workflow["202"] = {
            "class_type": "UltralyticsDetectorProvider",
            "inputs": {"model_name": "bbox/hand_yolov8n.pt"},
        }
        workflow["203"] = {
            "class_type": "BboxDetectorSEGS",
            "inputs": {
                "bbox_detector": ["202", 0], "image": ["201", 0],
                "threshold": 0.3, "dilation": 10, "crop_factor": 3.0, "drop_size": 10, "labels": "hand",
            },
        }
        workflow["204"] = {
            "class_type": "DetailerForEach",
            "inputs": {
                "image": ["201", 0], "segs": ["203", 0], "model": model_src, "clip": clip_src, "vae": ["3", 0],
                "guide_size": 384, "guide_size_for": True, "max_size": 1024,
                "seed": seed, "steps": adetail_steps, "cfg": cfg,
                "sampler_name": sampler_name, "scheduler": scheduler,
                "positive": ["4", 0], "negative": ["5", 0],
                "denoise": 0.45, "feather": 5, "noise_mask": True, "force_inpaint": True,
                "wildcard": "", "cycle": 1,
            },
        }

    # 出力先決定
    if adetail:
        save_image_src = ["204", 0]
    elif hires_fix:
        save_image_src = ["14", 0]
    else:
        save_image_src = ["7", 0]

    workflow["8"] = {
        "class_type": "SaveImage",
        "inputs": {"images": save_image_src, "filename_prefix": "comfy_chat/auto"}
    }

    return workflow


# ──── 非同期 API 呼び出し ─────────────────────────────────────────

async def get_checkpoints(session: aiohttp.ClientSession) -> list:
    try:
        async with session.get(
            f"{COMFY_BASE}/api/models/checkpoints",
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            data = await resp.json(content_type=None)
        return data if isinstance(data, list) else []
    except Exception:
        return []


async def translate_prompt(japanese_text: str, history: list,
                            ckpt_name: str,
                            session: aiohttp.ClientSession,
                            lora_registry: list = None) -> dict:
    base_prompt = _select_system_prompt(ckpt_name)
    system_prompt = _inject_loras(base_prompt, lora_registry or [])
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])  # 最新3往復分のみ（トークン節約）
    messages.append({"role": "user", "content": japanese_text})

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(
        LLAMA_URL,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        data = await resp.json(content_type=None)
    raw = data["choices"][0]["message"]["content"]
    return _extract_json(raw)


_EXPLAIN_TAGS_SYS = (
    "Stable Diffusion プロンプトタグを日本語で解説します。\n"
    'Output ONLY valid JSON: {"tags": [{"en": "tag", "jp": "日本語"}, ...]}\n'
    "品質タグ（score_9, score_8_up, score_7_up, masterpiece, best quality, newest, absurdres, "
    "highres, source_anime, source_pony, rating:explicit, 1girl, 1boy 等）は除外してください。\n"
    "各タグを簡潔な日本語（2〜10文字）で説明してください。"
)


async def explain_tags(positive: str, session: aiohttp.ClientSession) -> list:
    """ポジティブタグを日本語解説付きリストで返す。品質・汎用タグは除外する。"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _EXPLAIN_TAGS_SYS},
            {"role": "user",   "content": f"タグ: {positive}"},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(
        LLAMA_URL, json=payload,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        data = await resp.json(content_type=None)
    raw = data["choices"][0]["message"]["content"]
    return _extract_json(raw).get("tags", [])


async def submit_image_async(positive: str, negative: str, seed: int,
                              width: int, height: int, steps: int, cfg: float,
                              ckpt_name: str,
                              loras: list,
                              session: aiohttp.ClientSession,
                              hires_fix: bool = False,
                              adetail: bool = False,
                              init_image: Optional[str] = None,
                              denoise_strength: float = 0.75,
                              mask_image: Optional[str] = None,
                              sampler_name: str = "euler_ancestral",
                              scheduler: str = "karras",
                              client_id: Optional[str] = None) -> Optional[dict]:
    if client_id is None:
        client_id = str(uuid.uuid4())
    workflow = _build_workflow(positive, negative, seed, width, height, steps, cfg,
                               ckpt_name, loras, hires_fix=hires_fix, adetail=adetail,
                               init_image=init_image, denoise_strength=denoise_strength,
                               mask_image=mask_image,
                               sampler_name=sampler_name, scheduler=scheduler)

    async with session.post(
        f"{COMFY_BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json(content_type=None)

    if "prompt_id" not in data:
        # ワークフロー検証エラー: node_errors から詳細を取り出す
        err = data.get("error", {})
        node_errors = data.get("node_errors", {})
        msg = err.get("details") or err.get("message", "validation error")
        for ne in node_errors.values():
            for e in ne.get("errors", []):
                detail = e.get("details", "")
                if detail:
                    msg = detail
                    break
            break
        raise ValueError(f"ワークフロー検証エラー: {msg}")

    prompt_id = data["prompt_id"]

    deadline = time.monotonic() + POLL_TIMEOUT
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
            for msg in status.get("messages", []):
                if msg[0] == "execution_error" and isinstance(msg[1], dict):
                    exc = msg[1].get("exception_message", "不明")
                    node_type = msg[1].get("node_type", "")
                    raise ValueError(f"実行エラー ({node_type}): {exc}")
            raise ValueError("ComfyUI実行エラー（詳細不明）")
        images = entry.get("outputs", {}).get("8", {}).get("images", [])
        if images:
            return images[0]  # {"filename": "...", "subfolder": "...", "type": "output"}
        return None  # 完了したが画像なし（予期しない）

    return None  # タイムアウト
