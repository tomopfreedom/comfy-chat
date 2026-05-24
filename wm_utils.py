"""ウォーターマーク除去ユーティリティ

固定マスク + cv2.inpaint(TELEA) でフレームを補間し、
PyAV で映像/音声を再合成して MP4 バイト列を返す。
"""

import os
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

# アップロード動画の最大サイズ (200 MB)
MAX_VIDEO_BYTES = 200 * 1024 * 1024

# 許可する動画拡張子
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".webm"}


def _validate_video(filename: str, data: bytes) -> None:
    """ファイル名・サイズのバリデーション。問題があれば ValueError を送出する。"""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTS:
        raise ValueError(f"未対応の形式: {ext}（許可: {', '.join(ALLOWED_VIDEO_EXTS)}）")
    if len(data) > MAX_VIDEO_BYTES:
        mb = len(data) / 1024 / 1024
        raise ValueError(f"ファイルサイズが上限 (200 MB) を超えています: {mb:.1f} MB")


def get_first_frame_jpeg(video_bytes: bytes) -> bytes:
    """動画の最初のフレームを JPEG バイト列で返す。"""
    tmp_in = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_in = f.name

        cap = cv2.VideoCapture(tmp_in)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("フレームの読み取りに失敗しました")

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("JPEG エンコードに失敗しました")
        return buf.tobytes()
    finally:
        if tmp_in and os.path.exists(tmp_in):
            os.unlink(tmp_in)


def _structured_fill(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """行・列方向の線形補間でマスク領域を自然に充填する。

    TELEA/NS は大きなマスクや複雑な背景（机の縁＋椅子フレームの交差等）で
    ぼやけを生じやすい。この関数は各マスクピクセルについて
    「行の左右両端で非マスクの最近傍ピクセル」と
    「列の上下両端で非マスクの最近傍ピクセル」から 2 次線形補間し、
    それらを距離加重で合算することで輪郭のシャープネスを保つ。
    """
    result = frame.copy().astype(np.float32)
    h, w = mask.shape
    rows, cols = np.where(mask > 0)

    for y, x in zip(rows, cols):
        row = frame[y].astype(np.float32)
        col = frame[:, x].astype(np.float32)
        row_mask = mask[y]
        col_mask = mask[:, x]

        # 行方向: 左・右の最近傍非マスクピクセル
        left_xs  = np.where((np.arange(w) < x) & (row_mask == 0))[0]
        right_xs = np.where((np.arange(w) > x) & (row_mask == 0))[0]
        # 列方向: 上・下の最近傍非マスクピクセル
        top_ys   = np.where((np.arange(h) < y) & (col_mask == 0))[0]
        bot_ys   = np.where((np.arange(h) > y) & (col_mask == 0))[0]

        samples, weights = [], []

        if left_xs.size and right_xs.size:
            lx, rx = left_xs[-1], right_xs[0]
            t = (x - lx) / max(rx - lx, 1)
            samples.append((1 - t) * row[lx] + t * row[rx])
            weights.append(1.0 / max(rx - lx, 1))

        if top_ys.size and bot_ys.size:
            ty, by = top_ys[-1], bot_ys[0]
            t = (y - ty) / max(by - ty, 1)
            samples.append((1 - t) * col[ty] + t * col[by])
            weights.append(1.0 / max(by - ty, 1))

        if samples:
            w_sum = sum(weights)
            result[y, x] = sum(s * wt / w_sum for s, wt in zip(samples, weights))

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_watermark_video(
    video_bytes: bytes,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    keep_audio: bool = True,
    inpaint_radius: int = 15,
) -> bytes:
    """固定マスク領域をインペインティングで除去した動画の MP4 バイト列を返す。

    Args:
        video_bytes: 入力動画のバイト列
        x1, y1, x2, y2: ウォーターマークの矩形領域 (ピクセル座標)
        keep_audio: True の場合、元の音声ストリームを保持する
        inpaint_radius: TELEA 初期処理の半径 (px)。_structured_fill がメイン処理
    """
    tmp_in = tmp_out = None
    in_container = out_container = None
    try:
        # ── 入力をテンポラリファイルへ書き出す ──────────────────────
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_in = f.name

        fd, tmp_out = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

        # ── cv2 でフレームをインペイント ─────────────────────────────
        cap = cv2.VideoCapture(tmp_in)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 座標をフレーム範囲内にクランプ
        _x1 = max(0, min(x1, width  - 1))
        _y1 = max(0, min(y1, height - 1))
        _x2 = max(_x1 + 1, min(x2, width))
        _y2 = max(_y1 + 1, min(y2, height))

        # ひし形マスクを生成（スパークル形状に合わせて角を残す）
        # 矩形全塗りより余分な背景ピクセルが少なく、inpaint の仕上がりが自然になる
        mask = np.zeros((height, width), dtype=np.uint8)
        cx = (_x1 + _x2) / 2.0
        cy = (_y1 + _y2) / 2.0
        rx = (_x2 - _x1) / 2.0
        ry = (_y2 - _y1) / 2.0
        Y_idx, X_idx = np.ogrid[_y1:_y2, _x1:_x2]
        diamond = (np.abs(X_idx - cx) / rx + np.abs(Y_idx - cy) / ry) <= 1.0
        mask[_y1:_y2, _x1:_x2][diamond] = 255

        processed: list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            clean = cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA)
            processed.append(clean)
        cap.release()

        if not processed:
            raise RuntimeError("動画からフレームを読み取れませんでした")

        # ── PyAV で映像+音声を再合成 ──────────────────────────────────
        in_container  = av.open(tmp_in, "r")
        out_container = av.open(tmp_out, "w")

        # 映像ストリーム設定（PyAV は rate に float を受け付けないため Fraction に変換）
        fps_frac = Fraction(fps).limit_denominator(1001)
        v_out = out_container.add_stream("libx264", rate=fps_frac)
        v_out.width   = width
        v_out.height  = height
        v_out.pix_fmt = "yuv420p"
        v_out.options = {"crf": "18", "preset": "fast"}

        # 音声ストリーム設定（コピー）
        audio_pairs: list = []
        if keep_audio:
            for a_in in in_container.streams:
                if a_in.type == "audio":
                    a_out = out_container.add_stream(template=a_in)
                    audio_pairs.append((a_in, a_out))

        # 映像フレームをエンコード
        for bgr in processed:
            rgb = bgr[:, :, ::-1]  # BGR → RGB
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in v_out.encode(av_frame):
                out_container.mux(pkt)
        # フラッシュ
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
        for p in (tmp_in, tmp_out):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
