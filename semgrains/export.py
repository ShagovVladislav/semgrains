from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage import segmentation
import tifffile
from .measurements import measure_grains


def safe_stem(name):

    return re.sub(r'[^\w.()+-]', '_', Path(name).name)


def color_labels(labels):
    ids = labels.astype(np.uint64)
    colors = np.stack(((ids*73 + 41) % 211 + 44, (ids*131 + 23) % 211 + 44,
                       (ids*193 + 17) % 211 + 44), axis=-1).astype(np.uint8)
    colors[labels == 0] = 0
    return colors


def overlay(rgb, labels):
    out = rgb[:labels.shape[0]].copy()
    boundaries = segmentation.find_boundaries(labels, mode="inner")
    border_ids = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    border_objects = np.isin(labels, border_ids[border_ids > 0])
    out[boundaries & ~border_objects] = (30, 255, 100)
    out[boundaries & border_objects] = (255, 110, 50)
    return out


def stage_arrays(result, labels=None):
    lab = result.labels if labels is None else labels
    return [result.source.rgb, np.uint8(result.preprocessed*255), np.uint8(result.edge*255),
            color_labels(result.raw), color_labels(lab), overlay(result.source.rgb, lab)]


STAGE_NAMES = ["Original", "Preprocessed", "Edge map", "Raw segmentation", "Postprocessed segmentation", "Overlay on original"]


def save_panel(result, path, labels=None):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8))
    for ax, a, title in zip(axes.ravel(), stage_arrays(result, labels), STAGE_NAMES):
        ax.imshow(a, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{result.source.name} | {result.method} | D={result.diameter:.1f} px | {result.seconds:.3f} s", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def export_result(result, folder, labels=None, operations=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    corrected = labels is not None
    labels = result.labels if labels is None else labels
    objects = measure_grains(labels, result.source.gray[:result.crop], result.edge, result.config)
    objects.to_csv(folder / "objects.csv", index=False, encoding="utf-8-sig")
    objects[objects.included].to_csv(folder / "grains.csv", index=False, encoding="utf-8-sig")
    objects[~objects.included].to_csv(folder / "excluded.csv", index=False, encoding="utf-8-sig")
    tifffile.imwrite(folder / "labels.tif", labels.astype(np.uint32), photometric="minisblack", compression="zlib")
    if not corrected:
        tifffile.imwrite(folder / "raw_labels.tif", result.raw.astype(np.uint32), photometric="minisblack", compression="zlib")
    for i, (a, name) in enumerate(zip(stage_arrays(result, labels), STAGE_NAMES), 1):
        Image.fromarray(a).save(folder / f"{i:02d}_{name.lower().replace(' ', '_')}.png")
    Image.fromarray(np.uint8(result.disagreement)*255).save(folder / "disagreement.png")
    save_panel(result, folder / "stages.png", labels)
    metadata = {"source": result.source.name, "source_shape": list(result.source.gray.shape),
                "source_pixels_sha256": sha256(result.source.rgb.tobytes()).hexdigest(),
                "method": result.method, "corrected": corrected, "config": result.config.to_dict(),
                "crop_bottom": result.crop, "estimated_diameter_px": result.diameter,
                "seconds": result.seconds, "diagnostics": result.diagnostics,
                "count_included": int(objects.included.sum()), "count_excluded": int((~objects.included).sum()),
                "coordinate_system": "x right, y down; origin at top-left of original image; crop removes bottom only",
                "scale_confirmed": result.config.unit != "px", "operations": operations or []}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def zip_folder(folder):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(folder).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder).as_posix())
    return buf.getvalue()
