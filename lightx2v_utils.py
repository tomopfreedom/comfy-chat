"""LightX2V NVFP4 I2V ラッパー。

Wan2.1-I2V-14B-480P + NVFP4 4ステップ蒸留モデルを使って
画像から横長動画（832×480）を生成する。ComfyUI 不要。

モデルは初回呼び出し時にロードされモジュールレベルでキャッシュされる。
2回目以降はロード済みモデルを再利用するため高速。
"""

import asyncio
import os
import sys
import threading
import uuid
from typing import Optional

# ──── モデルパス定数 ─────────────────────────────────────────────────────────

LIGHTX2V_BASE_MODEL = os.path.expanduser("~/infra/models/Wan2.1-I2V-14B-480P")
LIGHTX2V_NVFP4_CKPT = os.path.expanduser(
    "~/infra/models/nvfp4/wan2.1_i2v_480p_nvfp4_lightx2v_4step.safetensors"
)
LIGHTX2V_OUTPUT_DIR = os.path.expanduser("~/infra/comfyui/output/lightx2v")

# LightX2V のインストールパスを sys.path に追加
_LIGHTX2V_DIR = os.path.expanduser("~/infra/LightX2V")
if _LIGHTX2V_DIR not in sys.path:
    sys.path.insert(0, _LIGHTX2V_DIR)

# ──── モデルキャッシュ ───────────────────────────────────────────────────────

_pipeline = None        # LightX2VPipeline のシングルトン
_pipeline_frames = None  # 現在のジェネレーターのフレーム数
_pipeline_lock = threading.Lock()  # 2重ロード・concurrent generate 防止


def is_available() -> bool:
    """LightX2V と必要なモデルファイルが揃っているか確認する。"""
    try:
        import lightx2v  # noqa: F401
    except ImportError:
        return False
    if not os.path.isdir(LIGHTX2V_BASE_MODEL):
        return False
    if not os.path.isfile(LIGHTX2V_NVFP4_CKPT):
        return False
    return True


def _load_pipeline(frames: int = 81):
    """LightX2VPipeline をロードしてキャッシュする。
    フレーム数が変わった場合はジェネレーターのみ再作成する。
    Lock で2重ロードと concurrent generate/create_generator を防止する。
    """
    global _pipeline, _pipeline_frames
    from lightx2v import LightX2VPipeline

    with _pipeline_lock:
        if _pipeline is None:
            pipe = LightX2VPipeline(
                model_path=LIGHTX2V_BASE_MODEL,
                model_cls="wan2.1_distill",  # 4ステップ蒸留モデル
                task="i2v",
            )
            pipe.enable_offload(
                cpu_offload=True,
                offload_granularity="block",
                text_encoder_offload=True,
                image_encoder_offload=False,
                vae_offload=False,
            )
            pipe.enable_quantize(
                dit_quantized=True,
                dit_quantized_ckpt=LIGHTX2V_NVFP4_CKPT,
                quant_scheme="nvfp4",
            )
            _pipeline = pipe

        if _pipeline_frames != frames:
            _pipeline.create_generator(
                attn_mode="sage_attn2",
                infer_steps=4,       # NVFP4 4ステップ蒸留
                height=480,
                width=832,
                num_frames=frames,
                guidance_scale=1.0,  # Wan2.1 蒸留モデル推奨値
                sample_shift=5.0,
            )
            _pipeline_frames = frames

        return _pipeline


def _generate_sync(
        positive: str,
        negative: str,
        seed: int,
        image_path: str,
        frames: int,
        out_path: str,
) -> None:
    """同期的に動画を生成してファイルに出力する（asyncio.to_thread で呼ぶ）。"""
    pipe = _load_pipeline(frames)
    pipe.generate(
        seed=seed,
        image_path=image_path,
        prompt=positive,
        negative_prompt=negative,
        save_result_path=out_path,
    )


async def run_lightx2v_i2v(
        positive: str,
        negative: str,
        seed: int,
        image_path: str,
        width: int = 832,    # 互換性のために受け取るが 832 固定
        height: int = 480,   # 互換性のために受け取るが 480 固定
        frames: int = 81,
        fps: int = 16,
        out_path: Optional[str] = None,
) -> dict:
    """LightX2V NVFP4 で画像から横長動画（832×480）を生成する。

    Args:
        positive:   ポジティブプロンプト
        negative:   ネガティブプロンプト
        seed:       乱数シード
        image_path: 参照画像のローカルパス（832×480 横長画像を推奨）
        width:      常に 832 固定（引数は wan_utils との互換性のために受け取る）
        height:     常に 480 固定（引数は wan_utils との互換性のために受け取る）
        frames:     フレーム数（デフォルト 81 ≈ 5秒 @ 16fps）
        fps:        FPS（出力ファイル名管理用・実際の fps は LightX2V が制御）
        out_path:   出力 MP4 パス（省略時は LIGHTX2V_OUTPUT_DIR に自動生成）

    Returns:
        {"filename": str, "subfolder": "lightx2v", "type": "output", "path": str}

    Raises:
        RuntimeError: LightX2V または必要なモデルが見つからない場合
        FileNotFoundError: 参照画像が見つからない場合
    """
    if not is_available():
        missing = []
        try:
            import lightx2v  # noqa: F401
        except ImportError:
            missing.append("lightx2v パッケージ（~/infra/setup_lightx2v.sh を実行してください）")
        if not os.path.isdir(LIGHTX2V_BASE_MODEL):
            missing.append(f"ベースモデル: {LIGHTX2V_BASE_MODEL}")
        if not os.path.isfile(LIGHTX2V_NVFP4_CKPT):
            missing.append(f"NVFP4 チェックポイント: {LIGHTX2V_NVFP4_CKPT}")
        raise RuntimeError("LightX2V の準備が整っていません:\n" + "\n".join(f"  - {m}" for m in missing))

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"参照画像が見つかりません: {image_path}")

    os.makedirs(LIGHTX2V_OUTPUT_DIR, exist_ok=True)
    if out_path is None:
        uid = str(uuid.uuid4())[:8]
        out_path = os.path.join(LIGHTX2V_OUTPUT_DIR, f"lightx2v_{uid}.mp4")

    # 同期処理をスレッドで実行（aiohttp イベントループをブロックしない）
    await asyncio.to_thread(
        _generate_sync,
        positive=positive,
        negative=negative,
        seed=seed,
        image_path=image_path,
        frames=frames,
        out_path=out_path,
    )

    return {
        "filename":  os.path.basename(out_path),
        "subfolder": "lightx2v",
        "type":      "output",
        "path":      out_path,
    }
