"""Irodori-TTS + WAN I2V + ffmpeg による音声付き動画生成パイプライン。

2種類のパイプラインを提供:

[voiced-wan] init_image 手動指定版:
  1. voice_text → Irodori-TTS → voice.wav
  2. wav 秒数 × fps → 最近傍 4n+1 フレーム数
  3. message → WAN I2V 動画生成（フレーム数指定）
  4. ffmpeg: video.webp + voice.wav → output.mp4

[auto-voiced-video] フルオート版:
  1. user_message → LLM → シナリオ(scene_prompt + dialogue)
  2. 画像生成(ComfyUI) ‖ TTS生成(Irodori-TTS)  ← 並列
  3. 音声長 → フレーム数 → WAN I2V（生成画像を init_image として使用）
  4. ffmpeg: video.webp + voice.wav → output.mp4
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Optional

import aiohttp

IRODORI_DIR = Path.home() / "infra/irodori-tts"
VOICED_OUTPUT_DIR = Path.home() / "infra/comfyui/output/voiced_videos"

_WAN_FRAMES_MIN = 17
_WAN_FRAMES_MAX = 81

_DIALOGUE_SYS = (
    "あなたはアニメキャラクターのセリフ生成アシスタントです。\n"
    "動画シーンの説明から、ともぴちゃん（明るく元気な女の子）がその場面で言いそうな"
    "自然なセリフを1〜2文（30文字以内）で生成してください。\n"
    '必ず JSON のみで出力: {"dialogue": "セリフ"}'
)


def snap_to_wan_frames(duration: float, fps: int = 16) -> int:
    """秒数から最近傍の WAN フレーム数（4n+1 形式）を計算する。"""
    raw = round(duration * fps)
    n = max(4, round((raw - 1) / 4))
    frames = n * 4 + 1
    return min(frames, _WAN_FRAMES_MAX)


def get_wav_duration(wav_path: str) -> float:
    """WAV ファイルの再生時間（秒）を返す。"""
    with wave.open(wav_path, "r") as wf:
        return wf.getnframes() / wf.getframerate()


def generate_voice(text: str, output_wav: str) -> bool:
    """Irodori-TTS で音声を生成する。失敗時は False を返す。"""
    if not IRODORI_DIR.exists():
        return False

    python_bin = IRODORI_DIR / ".venv/bin/python"
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
    script = (
        f'import sys; sys.path.insert(0, "{IRODORI_DIR}"); '
        f'from irodori_tts import IrodoriTTS; '
        f'tts = IrodoriTTS(); '
        f'tts.generate(text="{safe_text}", output="{output_wav}"); '
        f'print("TTS OK")'
    )

    if python_bin.exists():
        cmd = [str(python_bin), "-c", script]
    else:
        cmd = ["uv", "run", "--directory", str(IRODORI_DIR), "python", "-c", script]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return result.returncode == 0 and os.path.exists(output_wav)


def compose_mp4(video_webp: str, voice_wav: str, output_mp4: str) -> bool:
    """ffmpeg で WebP アニメーション + WAV を MP4 に合成する。

    -shortest で音声長に動画を揃える。WAN フレーム数が音声長から逆算されているため
    通常は音声と動画の長さがほぼ一致し、わずかな端数のみ切り捨てられる。
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_webp,
        "-i", voice_wav,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        output_mp4,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


async def generate_dialogue(scene: str, session: aiohttp.ClientSession) -> str:
    """シーン説明から LLM でセリフを自動生成する。"""
    from comfy_utils import LLAMA_URL, LLM_MODEL
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _DIALOGUE_SYS},
            {"role": "user", "content": f"シーン: {scene}"},
        ],
        "temperature": 0.8,
        "max_tokens": 100,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with session.post(
            LLAMA_URL, json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json(content_type=None)
        raw = data["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        parsed = json.loads(re.search(r"\{.*?\}", raw, re.DOTALL).group())
        return parsed.get("dialogue", "")
    except Exception:
        return ""


async def run_voiced_wan_pipeline(
    *,
    scene: str,
    voice_text: Optional[str],
    init_image: str,
    session: aiohttp.ClientSession,
    fps: int = 16,
    width: int = 576,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 3.5,
    seed: int = -1,
    use_anime_lora: bool = True,
    job_id: str,
) -> dict:
    """音声付き WAN 動画の完全パイプラインを実行する。

    Returns:
        dict: mp4_path, voice_text, frames, duration, seed
    Raises:
        RuntimeError: いずれかのステップが失敗した場合
    """
    import random
    from wan_utils import queue_wan_i2v_async, check_wan_i2v_result
    from wan_utils import WAN_MODEL_A14B_LOW, WAN_ANIME_LORA

    VOICED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "voice.wav")

        # ── Step 1: セリフが未指定なら LLM で自動生成 ────────────────
        if not voice_text:
            voice_text = await generate_dialogue(scene, session)
        if not voice_text:
            raise RuntimeError("セリフの生成に失敗しました")

        # ── Step 2: TTS → WAV ────────────────────────────────────────
        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(None, generate_voice, voice_text, wav_path)
        if not ok:
            raise RuntimeError(f"Irodori-TTS の音声生成に失敗しました: {voice_text!r}")

        # ── Step 3: 音声長 → フレーム数（タイミング合わせの核心）──────
        duration = get_wav_duration(wav_path)
        frames = snap_to_wan_frames(duration, fps)

        # ── Step 4: WAN I2V 動画生成 ─────────────────────────────────
        lora_name = WAN_ANIME_LORA if use_anime_lora else None
        prompt_id = await queue_wan_i2v_async(
            positive=scene,
            negative="",
            seed=seed,
            start_image=init_image,
            session=session,
            width=width,
            height=height,
            frames=frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            unet_name=WAN_MODEL_A14B_LOW,
            lora_name=lora_name,
            lora_strength=1.0,
        )

        # ── Step 5: WAN 完了待ち（最大 600 秒）──────────────────────
        timeout, interval, elapsed = 600, 5, 0
        result = None
        while elapsed < timeout:
            result = await check_wan_i2v_result(prompt_id, session)
            if result:
                break
            await asyncio.sleep(interval)
            elapsed += interval
        if not result:
            raise RuntimeError(f"WAN 動画生成タイムアウト (prompt_id={prompt_id})")

        fn = result["filename"]
        sf = result.get("subfolder", "wan_i2v")
        webp_path = str(Path.home() / "infra/comfyui/output" / sf / fn)
        if not os.path.exists(webp_path):
            raise RuntimeError(f"WebP ファイルが見つかりません: {webp_path}")

        # ── Step 6: ffmpeg で MP4 合成 ─────────────────────────────
        mp4_path = str(VOICED_OUTPUT_DIR / f"{job_id}.mp4")
        ok = await loop.run_in_executor(None, compose_mp4, webp_path, wav_path, mp4_path)
        if not ok:
            raise RuntimeError("ffmpeg による MP4 合成に失敗しました")

    return {
        "mp4_path": mp4_path,
        "voice_text": voice_text,
        "frames": frames,
        "duration": round(duration, 2),
        "seed": seed,
    }


# ──── フルオート版パイプライン (AnimaBase v6 LoRA + LightX2V) ────────

# LightX2V 入力解像度（横長固定）
_AUTO_WIDTH  = 832
_AUTO_HEIGHT = 480

# ともぴちゃん固定フィールド（AnimaTool 構造化プロンプト用）
_ANIMA_UNET     = "anima-base-v1.0.safetensors"
_ANIMA_LORAS    = [{"name": "tomopi_lora_v6_epoch20.safetensors", "weight": 0.8}]
_ANIMA_CHAR     = "tomopi"
_ANIMA_APPEAR   = "dark blue hair, gradient hair, cyan tips, orange eyes, large breasts"
_ANIMA_QUALITY  = "masterpiece, best quality, newest, highres, safe"
_ANIMA_BASE_URL = "http://localhost:8188"

_SCENARIO_SYS = (
    "あなたはアニメ動画のシナリオ生成AIです。\n"
    "ユーザーの指示から、ともぴちゃん（明るく元気な女の子）の短い動画シナリオを生成します。\n"
    "必ず以下のJSON形式のみで出力してください:\n"
    "{\n"
    '  "tags": "Danbooru英語タグ（動作・表情・構図・服装、カンマ区切り）",\n'
    '  "environment": "背景と光の状況（英語）",\n'
    '  "dialogue": "ともぴちゃんのセリフ（日本語、30文字以内）"\n'
    "}\n"
    "tags には動作・表情・服装・カメラ構図を含めてください。"
    "キャラクター固有の外観情報（髪色・目の色など）は不要です（自動付加されます）。"
)


async def generate_scenario(user_message: str, session: aiohttp.ClientSession) -> dict:
    """ユーザーメッセージから LLM でシナリオ（Anima構造化フィールド＋セリフ）を生成する。

    Returns:
        dict: tags (str), environment (str), dialogue (str)
    """
    from comfy_utils import LLAMA_URL, LLM_MODEL
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SCENARIO_SYS},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.8,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(
        LLAMA_URL, json=payload,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json(content_type=None)
    raw = data["choices"][0]["message"]["content"]
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    parsed = json.loads(re.search(r"\{.*?\}", raw, re.DOTALL).group())
    return {
        "tags":        parsed.get("tags", ""),
        "environment": parsed.get("environment", ""),
        "dialogue":    parsed.get("dialogue", ""),
    }


async def _generate_anima_image(
    tags: str,
    environment: str,
    seed: int,
    session: aiohttp.ClientSession,
    steps: int = 25,
    cfg: float = 4.5,
) -> str:
    """AnimaTool の /anima/generate を呼び出して画像のローカルパスを返す。

    Returns:
        str: 生成画像のローカルファイルパス
    Raises:
        RuntimeError: 生成失敗時
    """
    payload = {
        "unet_name":            _ANIMA_UNET,
        "loras":                _ANIMA_LORAS,
        "quality_meta_year_safe": _ANIMA_QUALITY,
        "character":            _ANIMA_CHAR,
        "appearance":           _ANIMA_APPEAR,
        "tags":                 tags,
        "environment":          environment,
        "width":                _AUTO_WIDTH,
        "height":               _AUTO_HEIGHT,
        "steps":                steps,
        "cfg":                  cfg,
        "seed":                 seed,
        "filename_prefix":      "auto_voiced/anima_frame",
    }
    async with session.post(
        f"{_ANIMA_BASE_URL}/anima/generate",
        json=payload,
        timeout=aiohttp.ClientTimeout(total=300),
    ) as resp:
        result = await resp.json(content_type=None)

    if "error" in result:
        raise RuntimeError(f"AnimaTool エラー: {result['error']}")

    images = result.get("images", [])
    if not images:
        raise RuntimeError("AnimaTool が画像を返しませんでした")

    file_path = images[0].get("file_path", "")
    if not file_path or not os.path.isfile(file_path):
        raise RuntimeError(f"AnimaTool の出力画像が見つかりません: {file_path!r}")

    return file_path


def compose_mp4_with_audio(video_mp4: str, voice_wav: str, output_mp4: str) -> bool:
    """ffmpeg で LightX2V 出力 MP4 に WAV 音声を付加する。

    映像は再エンコードせず copy するため高速。
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_mp4,
        "-i", voice_wav,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        output_mp4,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


async def run_auto_voiced_video_pipeline(
    *,
    user_message: str,
    session: aiohttp.ClientSession,
    fps: int = 16,
    img_steps: int = 25,
    img_cfg: float = 4.5,
    seed: int = -1,
    job_id: str,
) -> dict:
    """シナリオ生成 → 画像生成(Anima)＋TTS並列 → LightX2V動画 → MP4 の完全自動パイプライン。

    Returns:
        dict: mp4_path, tags, environment, dialogue, frames, duration, seed
    Raises:
        RuntimeError: いずれかのステップが失敗した場合
    """
    import random
    from lightx2v_utils import run_lightx2v_i2v

    VOICED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # ── Step 1: シナリオ生成（LLM）────────────────────────────────
    # llama-server が必要なのでここより前に停止してはいけない
    scenario = await generate_scenario(user_message, session)
    tags        = scenario["tags"]
    environment = scenario["environment"]
    dialogue    = scenario["dialogue"]
    if not dialogue:
        raise RuntimeError("シナリオ生成でセリフが空でした")

    # シナリオ確定後に llama-server を停止（画像生成＋LightX2V で VRAM を確保）
    import subprocess as _sp
    _sp.run(["systemctl", "--user", "stop", "llama-server"], check=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "voice.wav")
        loop = asyncio.get_event_loop()

        # ── Step 2: 画像生成(Anima) ‖ TTS 並列実行 ──────────────
        img_task = _generate_anima_image(
            tags=tags,
            environment=environment,
            seed=seed,
            session=session,
            steps=img_steps,
            cfg=img_cfg,
        )
        tts_task = loop.run_in_executor(None, generate_voice, dialogue, wav_path)

        img_path, tts_ok = await asyncio.gather(img_task, tts_task)

        if not tts_ok:
            raise RuntimeError(f"Irodori-TTS の音声生成に失敗しました: {dialogue!r}")

        # ── Step 3: 音声長 → フレーム数 ────────────────────────────
        duration = get_wav_duration(wav_path)
        frames = snap_to_wan_frames(duration, fps)

        # ── Step 4: LightX2V で動画生成（832×480 横長）────────────
        # LightX2V は MP4 を直接出力するため WebP→MP4 変換不要
        lx2v_out = str(VOICED_OUTPUT_DIR / f"{job_id}_raw.mp4")
        lx2v_result = await run_lightx2v_i2v(
            positive=f"{tags}, {environment}",
            negative="",
            seed=seed,
            image_path=img_path,
            frames=frames,
            fps=fps,
            out_path=lx2v_out,
        )

        # ── Step 5: 音声付加（映像 copy、音声 aac）───────────────
        mp4_path = str(VOICED_OUTPUT_DIR / f"{job_id}.mp4")
        ok = await loop.run_in_executor(
            None, compose_mp4_with_audio, lx2v_result["path"], wav_path, mp4_path
        )
        if not ok:
            raise RuntimeError("ffmpeg による音声付加に失敗しました")

    return {
        "mp4_path":    mp4_path,
        "tags":        tags,
        "environment": environment,
        "dialogue":    dialogue,
        "frames":      frames,
        "duration":    round(duration, 2),
        "seed":        seed,
    }
