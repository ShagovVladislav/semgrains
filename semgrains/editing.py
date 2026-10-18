import numpy as np
from scipy import ndimage as ndi
from skimage import draw, segmentation


def apply_edit(labels, operation):
    out = labels.copy()
    kind = operation["type"]
    ids = sorted(set(int(i) for i in operation.get("ids", []) if int(i) > 0))
    existing = set(np.unique(out))
    if any(i not in existing for i in ids):
        raise ValueError("Выбранный объект отсутствует в текущей маске")
    if kind == "exclude":
        if not ids:
            raise ValueError("Выберите хотя бы один объект")
        out[np.isin(out, ids)] = 0
    elif kind == "merge":
        if len(ids) < 2:
            raise ValueError("Выберите минимум два соседних зерна")
        mask = np.isin(out, ids)
        if ndi.label(mask)[1] != 1:
            raise ValueError("Объединять можно только соприкасающиеся объекты")
        out[mask] = ids[0]
    elif kind == "split":
        if len(ids) != 1:
            raise ValueError("Для разделения выберите одно зерно")
        points = np.asarray(operation.get("points", []), dtype=float)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
            raise ValueError("Нарисуйте линию через выбранное зерно от края до края")
        if not np.isfinite(points).all():
            raise ValueError("Некорректные координаты линии")
        mask = out == ids[0]
        barrier = np.zeros_like(mask)
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            rr, cc = draw.line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
            valid = (rr >= 0) & (cc >= 0) & (rr < out.shape[0]) & (cc < out.shape[1])
            barrier[rr[valid], cc[valid]] = True
        components, n = ndi.label(mask & ~ndi.binary_dilation(barrier))
        sizes = np.bincount(components.ravel())
        for i in range(1, n+1):
            if sizes[i] < 3:
                components[components == i] = 0
        components, _, _ = segmentation.relabel_sequential(components)
        if components.max() < 2:
            raise ValueError("Линия не разделила зерно: проведите её от одного края до другого")
        filled = segmentation.watershed(np.zeros_like(mask, dtype=float), components, mask=mask)
        next_id = int(out.max()) + 1
        for i in range(1, int(filled.max())+1):
            out[filled == i] = ids[0] if i == 1 else next_id + i - 2
    else:
        raise ValueError(f"Неизвестная операция: {kind}")
    return out.astype(np.int32)


def replay_edits(labels, operations):
    for operation in operations:
        labels = apply_edit(labels, operation)
    return labels


def polygon_mask(shape, polygons, ignored=None):
    labels = np.zeros(shape, dtype=np.int32)
    for i, points in enumerate(polygons, 1):
        xy = np.asarray(points, dtype=float)
        rr, cc = draw.polygon(xy[:, 1], xy[:, 0], shape=shape)
        labels[rr, cc] = i
    ignore = np.zeros(shape, dtype=bool)
    for points in ignored or []:
        xy = np.asarray(points, dtype=float)
        rr, cc = draw.polygon(xy[:, 1], xy[:, 0], shape=shape)
        ignore[rr, cc] = True
    return labels, ignore
