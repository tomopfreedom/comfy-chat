"""ComfyUI AI インペインティングを使ったウォーターマーク除去（高品質版）

通常の cv2.inpaint (TELEA) より自然な仕上がりを得られる。
シーングループ検出により代表フレームのみを ComfyUI で処理し、
パッチを同グループ内の全フレームに適用することで処理時間を最小化する。
"""

import asyncio
import os
import random
import tempfile
import time
from fractions import Fraction

import aiohttp
import av
import cv2
import numpy as np

COMFY_BASE = "http://localhost:8188"
POLL_INTERVAL = 3.0   # 秒
POLL_TIMEOUT  = 300   # 秒

DEFAULT_CKPT     = "NoobAI-XL-v1.1.safetensors"
INPAINT_STEPS    = 15
INPAINT_CFG      = 5.0
INPAINT_DENOISE  = 0.55

# 平均絶対差がこの値を超えるとシーン切り替えと判定
SCENE_DIFF_THRESHOLD = 20.0


# ──── ComfyUI ユーティリティ ─────────────────────────────────────


async def _upload_bytes_to_comfy(
    session: aiohttp.ClientSession,
    data: bytes,
    filename: str,
    content_type: str = "image/png",
) -> str:
    """バイト列を ComfyUI /upload/image に送信し、返却されたファイル名を返す。"""
    form = aiohttp.FormData()
    form.add_field("image", data, filename=filename, content_type=content_type)
    async with session.post(
        f"{COMFY_BASE}/upload/image",
        data=form,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        result = await resp.json(content_type=None)
    return result.get("name", filename)


async def _download_comfy_result(
    session: aiohttp.ClientSession,
    info: dict,
) -> np.ndarray:
    """ComfyUI 出力画像をダウンロードして BGR ndarray で返す。"""
    params = {
        "filename": info["filename"],
        "subfolder": info.get("subfolder", ""),
        "type": info.get("type", "output"),
    }
    async with session.get(
        f"{COMFY_BASE}/view",
        params=params,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as resp:
        raw = await resp.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("ComfyUI 結果画像のデコードに失敗しました")
    return img


# ──── 画像処理ユーティリティ ─────────────────────────────────────


def _make_mask_png(
    height: int, width: int,
    x1: int, y1: int, x2: int, y2: int,
) -> bytes:
    """ひし形マスク PNG を生成する（赤=インペイント範囲、黒=保持）。"""
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    rx = max((x2 - x1) / 2.0, 1.0)
    ry = max((y2 - y1) / 2.0, 1.0)
    Y_idx, X_idx = np.ogrid[y1:y2, x1:x2]
    diamond = (np.abs(X_idx - cx) / rx + np.abs(Y_idx - cy) / ry) <= 1.0
    mask[y1:y2, x1:x2][diamond] = [0, 0, 255]  # BGR → 赤
    ok, buf = cv2.imencode(".png", mask)
    if not ok:
        raise RuntimeError("マスク PNG エンコードに失敗しました")
    return buf.tobytes()


def _frame_to_png(frame_bgr: np.ndarray) -> bytes:
    """BGR ndarray を PNG バイト列に変換する。"""
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        raise RuntimeError("フレーム PNG エンコードに失敗しました")
    return buf.tobytes()


def _detect_scene_groups(
    frames: list,
    y1: int, y2: int,
    x1: int, x2: int,
    height: int, width: int,
) -> list:
    """ウォーターマーク周辺領域の比較でシーングループを検出する。

    Returns:
        list[list[int]]: グループごとのフレームインデックスリスト
    """
    if not frames:
        return []

    # ウォーターマーク外側を比較領域とする（ロゴ自体の差異を除外）
    pad = 40
    sy = slice(max(0, y1 - pad), min(height, y2 + pad))
    sx = slice(max(0, x1 - pad), min(width, x2 + pad))

    groups = [[0]]
    ref = frames[0][sy, sx].astype(np.float32)

    for i in range(1, len(frames)):
        curr = frames[i][sy, sx].astype(np.float32)
        diff = float(np.mean(np.abs(curr - ref)))
        if diff > SCENE_DIFF_THRESHOLD:
            groups.append([i])
            ref = curr
        else:
            groups[-1].append(i)

    return groups


# ──── ComfyUI ワークフロー ────────────────────────────────────────


def _build_inpaint_workflow(
    init_image: str,
    mask_image: str,
    width: int,
    height: int,
    ckpt_name: str,
    seed: int,
) -> dict:
    """SDXL インペインティングワークフロー JSON を構築する。

    既存 comfy_utils._build_workflow と同じノード番号規則を使用:
      1  = CheckpointLoaderSimple
      3  = VAELoader (sdxl_vae.safetensors)
      20 = CLIPSetLastLayer
      4  = CLIPTextEncode (positive)
      5p = CLIPTextEncode (negative)
      30 = LoadImage (init_image)
      32 = LoadImage (mask_image)
      35 = ImageScale (init)
      36 = ImageScale (mask)
      33 = ImageToMask
      34 = VAEEncodeForInpaint
      6  = KSampler
      7  = VAEDecode
      8  = SaveImage  ← ポーリングは outputs["8"] を監視
    """
    # NoobAI / Pony / Illustrious 系は CLIP Skip -2
    name_l = ckpt_name.lower()
    clip_skip = -2 if ("pony" in name_l or "noob" in name_l or "illustrious" in name_l) else -1

    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt_name},
        },
        "20": {
            "class_type": "CLIPSetLastLayer",
            "inputs": {"clip": ["1", 1], "stop_at_clip_layer": clip_skip},
        },
        # 外部 VAE（sdxl_vae.safetensors）をロード — ビルトイン VAE の NaN 問題を回避
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "sdxl_vae.safetensors"},
        },
        # ポジティブ: 背景を自然に補完するよう誘導
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["20", 0],
                "text": "high quality background, clean, seamless, no watermark, no logo, no symbol",
            },
        },
        # ネガティブ: スパークル / 記号類を抑制
        "5p": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["20", 0],
                "text": "watermark, logo, text, sparkle, glitter, star, glow, symbol, "
                        "low quality, worst quality",
            },
        },
        # init_image をロード
        "30": {
            "class_type": "LoadImage",
            "inputs": {"image": init_image, "upload": "image"},
        },
        # mask_image をロード
        "32": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image, "upload": "image"},
        },
        # フレームをターゲットサイズにリサイズ
        "35": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["30", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        },
        # マスクをターゲットサイズにリサイズ
        "36": {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["32", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        },
        # 赤チャンネルをマスクとして解釈
        "33": {
            "class_type": "ImageToMask",
            "inputs": {"image": ["36", 0], "channel": "red"},
        },
        # インペイント用 VAE エンコード
        "34": {
            "class_type": "VAEEncodeForInpaint",
            "inputs": {
                "pixels": ["35", 0],
                "vae": ["3", 0],
                "mask": ["33", 0],
                "grow_mask_by": 6,
            },
        },
        # KSampler
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5p", 0],
                "latent_image": ["34", 0],
                "seed": seed,
                "steps": INPAINT_STEPS,
                "cfg": INPAINT_CFG,
                "sampler_name": "euler_ancestral",
                "scheduler": "karras",
                "denoise": INPAINT_DENOISE,
            },
        },
        # VAE デコード
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["3", 0]},
        },
        # SaveImage: ノード番号を 8 に固定（ポーリングは outputs["8"] を監視）
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": "wm_clean/frame",
            },
        },
    }


async def _inpaint_frame_comfy(
    session: aiohttp.ClientSession,
    frame_bgr: np.ndarray,
    mask_png: bytes,
    width: int,
    height: int,
    ckpt_name: str,
    group_idx: int,
) -> np.ndarray:
    """1フレームを ComfyUI でインペインティングして結果 BGR ndarray を返す。"""
    import uuid

    seed = random.randint(0, 2**32 - 1)
    frame_name = f"wm_frame_{group_idx:04d}.png"
    mask_name  = f"wm_mask_{group_idx:04d}.png"

    # フレーム・マスクをアップロード（並列）
    loop = asyncio.get_event_loop()
    frame_png = await loop.run_in_executor(None, _frame_to_png, frame_bgr)
    comfy_frame, comfy_mask = await asyncio.gather(
        _upload_bytes_to_comfy(session, frame_png, frame_name),
        _upload_bytes_to_comfy(session, mask_png,  mask_name),
    )

    # ワークフローを構築して送信
    client_id = str(uuid.uuid4())
    workflow = _build_inpaint_workflow(
        init_image=comfy_frame,
        mask_image=comfy_mask,
        width=width,
        height=height,
        ckpt_name=ckpt_name,
        seed=seed,
    )

    async with session.post(
        f"{COMFY_BASE}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        data = await resp.json(content_type=None)

    if "prompt_id" not in data:
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
        raise ValueError(f"ComfyUI ワークフロー検証エラー: {msg}")

    prompt_id = data["prompt_id"]

    # ポーリング（3秒間隔、最大 300 秒）
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
        status = hist[prompt_id].get("status", {})
        if status.get("status_str") == "error":
            for msg_entry in status.get("messages", []):
                if msg_entry[0] == "execution_error" and isinstance(msg_entry[1], dict):
                    exc = msg_entry[1].get("exception_message", "不明")
                    node_type = msg_entry[1].get("node_type", "")
                    raise ValueError(f"ComfyUI 実行エラー ({node_type}): {exc}")
            raise ValueError("ComfyUI インペイント実行エラー（詳細不明）")
        images = hist[prompt_id].get("outputs", {}).get("8", {}).get("images", [])
        if images:
            return await _download_comfy_result(session, images[0])
        if hist[prompt_id].get("status", {}).get("completed"):
            return None  # 完了したが画像なし

    raise TimeoutError(f"ComfyUI インペイント タイムアウト (group {group_idx})")


# ──── 動画再合成（同期） ──────────────────────────────────────────


def _reconstruct_video(
    frames: list,
    fps: float,
    width: int,
    height: int,
    tmp_in: str,
    keep_audio: bool,
    tmp_out: str,
) -> bytes:
    """処理済みフレームリストから MP4 バイト列を生成する（同期関数）。"""
    in_container = out_container = None
    try:
        in_container  = av.open(tmp_in, "r")
        out_container = av.open(tmp_out, "w")

        fps_frac = Fraction(fps).limit_denominator(1001)
        v_out = out_container.add_stream("libx264", rate=fps_frac)
        v_out.width   = width
        v_out.height  = height
        v_out.pix_fmt = "yuv420p"
        v_out.options = {"crf": "18", "preset": "fast"}

        audio_pairs: list = []
        if keep_audio:
            for a_in in in_container.streams:
                if a_in.type == "audio":
                    a_out = out_container.add_stream(template=a_in)
                    audio_pairs.append((a_in, a_out))

        # 映像フレームをエンコード
        for bgr in frames:
            rgb = bgr[:, :, ::-1]
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in v_out.encode(av_frame):
                out_container.mux(pkt)
        for pkt in v_out.encode():
            out_container.mux(pkt)

        # 音声パケットをコピー
        for a_in, a_out in audio_pairs:
            in_container.seek(0)
            for pkt in in_container.demux(a_in):
                if pkt.dts is None:
                    continue
                pkt.stream = a_out
                out_container.mux(pkt)

        in_container.close()
        in_container = None
        out_container.close()
        out_container = None

        with open(tmp_out, "rb") as f:
            return f.read()

    finally:
        if in_container:
            in_container.close()
        if out_container:
            out_container.close()


# ──── メインエントリーポイント ────────────────────────────────────


async def remove_watermark_video_comfy(
    video_bytes: bytes,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    keep_audio: bool,
    session: aiohttp.ClientSession,
    ckpt_name: str = DEFAULT_CKPT,
) -> bytes:
    """ComfyUI AI インペインティングでウォーターマークを除去した MP4 バイト列を返す。

    Args:
        video_bytes: 入力動画のバイト列
        x1, y1, x2, y2: ウォーターマークの矩形領域（ピクセル座標）
        keep_audio: True の場合、元の音声ストリームを保持する
        session: aiohttp.ClientSession（app["session"] を渡す）
        ckpt_name: ComfyUI チェックポイント名

    処理フロー:
        1. 全フレームを cv2 で抽出
        2. ウォーターマーク周辺の差分でシーングループを検出
        3. グループごとに代表フレームを ComfyUI でインペイント
        4. インペイント済みパッチを同グループの全フレームに適用
        5. PyAV で映像 + 音声を再合成して返す
    """
    loop = asyncio.get_event_loop()
    tmp_in = tmp_out = None

    try:
        # ── 1. 動画をテンポラリファイルに書き出してフレームを抽出 ──
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_in = f.name

        cap = cv2.VideoCapture(tmp_in)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 座標をフレーム範囲内にクランプ
        _x1 = max(0, min(x1, width  - 1))
        _y1 = max(0, min(y1, height - 1))
        _x2 = max(_x1 + 1, min(x2, width))
        _y2 = max(_y1 + 1, min(y2, height))

        frames: list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError("動画からフレームを読み取れませんでした")

        # ── 2. ひし形マスク PNG を生成 ──────────────────────────────
        mask_png = await loop.run_in_executor(
            None, _make_mask_png, height, width, _x1, _y1, _x2, _y2
        )

        # ── 3. シーングループを検出 ─────────────────────────────────
        groups = await loop.run_in_executor(
            None, _detect_scene_groups, frames, _y1, _y2, _x1, _x2, height, width
        )

        # ── 4. 各グループの代表フレームを ComfyUI でインペイント ────
        cleaned = [f.copy() for f in frames]

        for group_idx, group in enumerate(groups):
            rep_idx   = group[0]
            rep_frame = frames[rep_idx]

            result_frame = await _inpaint_frame_comfy(
                session=session,
                frame_bgr=rep_frame,
                mask_png=mask_png,
                width=width,
                height=height,
                ckpt_name=ckpt_name,
                group_idx=group_idx,
            )

            if result_frame is None:
                # インペイント失敗 → スキップ（元フレームのまま）
                continue

            # サイズが異なる場合はリサイズ
            if result_frame.shape[:2] != (height, width):
                result_frame = cv2.resize(
                    result_frame, (width, height), interpolation=cv2.INTER_LANCZOS4
                )

            # パッチ（ウォーターマーク領域）を切り出してグループ全フレームに適用
            patch = result_frame[_y1:_y2, _x1:_x2]
            for idx in group:
                cleaned[idx][_y1:_y2, _x1:_x2] = patch

        # ── 5. PyAV で動画を再合成 ──────────────────────────────────
        fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        mp4_bytes = await loop.run_in_executor(
            None,
            lambda: _reconstruct_video(cleaned, fps, width, height, tmp_in, keep_audio, tmp_out),
        )

        return mp4_bytes

    finally:
        for p in (tmp_in, tmp_out):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
