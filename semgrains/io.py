from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError

EXTENSIONS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Source:
    name: str
    rgb: np.ndarray
    gray: np.ndarray
    mode: str
    dtype: str
    frames: int


def normalize(a, percentiles=(1, 99)):
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        raise ValueError("Изображение содержит NaN или бесконечные значения")
    lo, hi = np.percentile(a, percentiles)
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32) if hi > lo else np.zeros_like(a)


def load_image(source, name=None):

    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)
    try:
        with Image.open(source) as im:
            mode, frames = im.mode, getattr(im, "n_frames", 1)
            a = np.asarray(im)
            dtype = str(a.dtype)
            if im.mode in ("P", "PA", "CMYK", "YCbCr", "HSV", "RGBA", "RGBa"):

                rgba = im.convert("RGBA")
                bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
                rgb = np.asarray(Image.alpha_composite(bg, rgba).convert("RGB"))
                g = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255
            elif a.ndim == 2:
                if a.dtype == np.uint8:
                    g = a.astype(np.float32) / 255
                else:

                    g = normalize(a, (0, 100))
                rgb = np.repeat(np.uint8(np.clip(g, 0, 1) * 255)[..., None], 3, axis=2)
            else:
                rgb = np.asarray(im.convert("RGB"))
                g = np.asarray(im.convert("L"), dtype=np.float32) / 255
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Не удалось прочитать изображение: {exc}") from exc
    if min(g.shape) < 16:
        raise ValueError("Изображение слишком маленькое (минимум 16 × 16)")
    return Source(name or (Path(source).name if isinstance(source, (str, Path)) else "image"),
                  rgb.copy(), g.copy(), mode, dtype, frames)


def detect_crop(gray):

    h, w = gray.shape
    dark_fraction = (gray < 0.13).mean(axis=1)
    for y in range(int(h * .6), h - 3):
        if dark_fraction[y:y + 3].mean() > .72 and dark_fraction[max(0, y-4):y].mean() < .4:
            return y
    return h


def propose_scale_bar(gray, crop):

    if crop >= gray.shape[0] - 3:
        return None
    best = 0

    for row in gray[crop:, :max(20, gray.shape[1] // 4)]:
        binary = row > .8
        d = np.diff(np.r_[False, binary, False].astype(np.int8))
        for start, stop in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if 5 <= stop - start <= gray.shape[1] * .20:
                best = max(best, stop - start)
    return int(best) if best else None


def image_paths(folder):
    return sorted((p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                  key=lambda p: p.name.casefold())
