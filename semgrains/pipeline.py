from dataclasses import dataclass
from time import perf_counter
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import filters, measure, morphology, segmentation
from .config import Config
from .io import Source, detect_crop, normalize
METHODS = ('dark', 'gradient', 'ensemble')

@dataclass
class Result:
    source: Source
    method: str
    config: Config
    crop: int
    diameter: float
    preprocessed: np.ndarray
    edge: np.ndarray
    raw: np.ndarray
    labels: np.ndarray
    disagreement: np.ndarray
    seconds: float
    diagnostics: dict

def _disk_kernel(radius):
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def estimate_diameter(gray):
    factor = min(1.0, 400 / max(gray.shape))
    if factor < 1:
        small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        return estimate_diameter(small) / factor
    u8 = np.uint8(normalize(gray) * 255)
    u8 = cv2.GaussianBlur(u8, (0, 0), 0.65)
    radius = max(2, min(gray.shape) / 80)
    dark = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, _disk_kernel(radius)).astype(float)
    if dark.max() < 2:
        return float(max(6, min(gray.shape) / 25))
    threshold = filters.threshold_otsu(dark)
    dist = ndi.distance_transform_edt(dark <= threshold)
    peaks = morphology.h_maxima(ndi.gaussian_filter(dist, 0.6), 0.7).astype(bool)
    distances = dist[peaks]
    distances = distances[(distances > 1) & (distances < min(gray.shape) / 8)]
    estimate = float(np.median(distances) * 3.0) if len(distances) else min(gray.shape) / 25
    return float(np.clip(estimate, 6, min(gray.shape) / 5))

def preprocess(gray, diameter, clip):
    u8 = np.uint8(normalize(gray) * 255)
    denoised = cv2.bilateralFilter(u8, 5, 20, max(1.2, diameter / 12))
    tile_count = max(2, min(16, int(min(gray.shape) / max(diameter * 3, 16))))
    enhanced = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_count, tile_count)).apply(denoised)
    return normalize(enhanced)

def valley_map(gray, diameter):
    sigmas = (max(0.6, diameter * 0.03), max(1.0, diameter * 0.06), max(1.5, diameter * 0.1))
    return normalize(filters.sato(gray, sigmas=sigmas, black_ridges=True))

def boundary_maps(pre, diameter, original=None, valley_weight=0, valleys=None):
    u8 = np.uint8(pre * 255)
    blackhat = cv2.morphologyEx(u8, cv2.MORPH_BLACKHAT, _disk_kernel(diameter * 0.23))
    dark = normalize(ndi.gaussian_filter(blackhat.astype(np.float32), 0.5), (1, 99.5))
    gradients = [filters.scharr(ndi.gaussian_filter(pre, s)) for s in (max(0.45, diameter / 30), max(0.9, diameter / 15), max(1.4, diameter / 9))]
    gradient = normalize(0.55 * gradients[0] + 0.3 * gradients[1] + 0.15 * gradients[2], (1, 99.5))
    if valley_weight and original is not None and (np.ptp(original) > 1e-06):
        if valleys is None:
            valleys = valley_map(original, diameter)
        dark = (1 - valley_weight) * dark + valley_weight * valleys
        gradient = (1 - valley_weight) * gradient + valley_weight * valleys
    return (dark, gradient)

def markers_from_edges(edge, diameter, prominence):
    if np.ptp(edge) < 1e-06:
        return np.zeros_like(edge, dtype=np.int32)
    threshold = filters.threshold_otsu(edge)
    interior = edge < threshold
    dist = ndi.gaussian_filter(ndi.distance_transform_edt(interior), 0.55)
    maxima = morphology.h_maxima(dist, max(0.25, diameter * prominence)).astype(bool)
    maxima &= dist > 0.8
    markers, _ = ndi.label(maxima)
    if not markers.any():
        p = np.unravel_index(np.argmax(dist), dist.shape)
        markers[p] = 1
    return markers.astype(np.int32)

def _marker_centers(markers):
    return np.array([p.centroid for p in measure.regionprops(markers)], dtype=float).reshape(-1, 2)

def marker_watershed(edge, markers, mask=None):
    if not markers.any():
        return np.zeros_like(markers, dtype=np.int32)
    relief = np.asarray(edge, dtype=np.float32).copy()
    relief[markers > 0] = float(relief.min()) - 1.0
    return segmentation.watershed(relief, markers, mask=mask).astype(np.int32)

def ensemble_markers(a, b, edge, diameter):
    centers = list(_marker_centers(a))
    for p in _marker_centers(b):
        if not centers:
            centers.append(p)
            continue
        nearest = cKDTree(centers).query(p)
        q = np.asarray(centers[nearest[1]])
        points = np.round(np.linspace(p, q, max(3, int(nearest[0]) + 1))).astype(int)
        ridge = edge[points[:, 0], points[:, 1]].max()
        if nearest[0] > max(2.0, diameter * 0.32) or (nearest[0] > 2 and ridge > 0.65):
            centers.append(p)
    markers = np.zeros_like(a)
    for i, p in enumerate(centers, 1):
        markers[tuple(np.round(p).astype(int))] = i
    return markers

def adjacency_strength(labels, edge):
    pairs, values = ([], [])
    for left, right, e1, e2 in ((labels[:, :-1], labels[:, 1:], edge[:, :-1], edge[:, 1:]), (labels[:-1, :], labels[1:, :], edge[:-1, :], edge[1:, :])):
        valid = (left != right) & (left > 0) & (right > 0)
        if valid.any():
            pairs.append(np.sort(np.stack((left[valid], right[valid]), axis=1), axis=1))
            values.append(np.maximum(e1[valid], e2[valid]))
    if not pairs:
        return []
    pairs, values = (np.concatenate(pairs), np.concatenate(values))
    unique, inv = np.unique(pairs, axis=0, return_inverse=True)
    sums, counts = (np.bincount(inv, weights=values), np.bincount(inv))
    return [(int(a), int(b), float(s / n), int(n)) for (a, b), s, n in zip(unique, sums, counts)]

def adaptive_postprocess(labels, edge, diameter, config, alternate_markers=None):
    labels = labels.copy()
    parent = np.arange(int(labels.max()) + 1)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    merges, splits = (0, 0)
    for a, b, strength, length in sorted(adjacency_strength(labels, edge), key=lambda t: t[2]):
        if strength < config.merge_threshold and length >= 2:
            ra, rb = (root(a), root(b))
            if ra != rb:
                parent[rb] = ra
                merges += 1
    labels = np.array([root(i) for i in range(len(parent))], dtype=np.int32)[labels]
    labels, _, _ = segmentation.relabel_sequential(labels)
    if alternate_markers is not None:
        next_id = int(labels.max()) + 1
        for region in measure.regionprops(labels):
            sl = region.slice
            mask = labels[sl] == region.label
            seeds = np.where(mask, alternate_markers[sl], 0)
            ids = np.unique(seeds[seeds > 0])
            if len(ids) < 2:
                continue
            proposal = marker_watershed(edge[sl], seeds, mask=mask)
            adj = adjacency_strength(proposal, edge[sl])
            stable_interiors = all((ndi.distance_transform_edt(np.pad(proposal == identifier, 1)).max() >= max(1.5, diameter * 0.1) for identifier in ids))
            if stable_interiors and adj and all((s >= config.split_threshold and length >= 2 for _, _, s, length in adj)):
                local = labels[sl]
                for identifier in ids:
                    local[proposal == identifier] = next_id
                    next_id += 1
                splits += len(ids) - 1
    removed = 0
    areas = np.bincount(labels.ravel())
    for region in measure.regionprops(labels):
        if region.area >= max(2, diameter ** 2 * 0.04):
            continue
        y0, x0, y1, x1 = region.bbox
        sl = (slice(max(0, y0 - 1), min(labels.shape[0], y1 + 1)), slice(max(0, x0 - 1), min(labels.shape[1], x1 + 1)))
        mask = labels[sl] == region.label
        boundary = segmentation.find_boundaries(mask, mode='inner')
        strength = float(edge[sl][boundary].mean()) if boundary.any() else 0
        ring = ndi.binary_dilation(mask) & ~mask
        neighbors = np.unique(labels[sl][ring])
        neighbors = neighbors[neighbors > 0]
        if not len(neighbors):
            continue
        radius = ndi.distance_transform_edt(np.pad(mask, 1)).max()
        tiny_weak = region.area < max(2, diameter ** 2 * 0.006) and strength < 0.15
        enclosed_speck = len(neighbors) == 1 and areas[neighbors[0]] >= 8 * region.area and (radius < max(1.5, diameter * 0.08)) and (strength < config.split_threshold)
        if tiny_weak or enclosed_speck:
            contacts = [p for p in adjacency_strength(labels[sl], edge[sl]) if region.label in p[:2] and areas[p[1] if p[0] == region.label else p[0]] >= max(2, diameter ** 2 * 0.04)]
            if not contacts:
                continue
            a, b, _, _ = min(contacts, key=lambda p: (p[2], -p[3]))
            target = b if a == region.label else a
            labels[sl][mask] = target
            areas[target] += int(region.area)
            areas[region.label] = 0
            removed += 1
    labels, _, _ = segmentation.relabel_sequential(labels)
    return (labels.astype(np.int32), {'merged': merges, 'split': splits, 'artifacts_removed': removed})

def segment_all(source: Source, config=Config()):
    start = perf_counter()
    h = source.gray.shape[0]
    crop = config.crop_bottom if config.crop_bottom is not None else detect_crop(source.gray)
    if not 8 <= crop <= h:
        raise ValueError(f'Некорректная граница рабочей области: {crop}, высота {h}')
    roi = source.gray[:crop]
    diameter = config.diameter_px or estimate_diameter(roi)
    pre = preprocess(roi, diameter, config.clahe_clip)
    valleys = valley_map(roi, diameter) if config.valley_weight and np.ptp(roi) > 1e-06 else None
    dark, gradient = boundary_maps(pre, diameter, roi, config.valley_weight, valleys)
    valley_markers = markers_from_edges(valleys, diameter, config.marker_prominence) if valleys is not None else None
    maps = {'dark': dark, 'gradient': gradient}
    markers, raw = ({}, {})
    method_times = {}
    shared_seconds = perf_counter() - start
    if np.std(roi) < 1e-05:
        dark[:] = gradient[:] = 0
    for name, edge in maps.items():
        t = perf_counter()
        markers[name] = markers_from_edges(edge, diameter, config.marker_prominence)
        if valley_markers is not None:
            markers[name] = ensemble_markers(markers[name], valley_markers, edge, diameter)
        raw[name] = marker_watershed(edge, markers[name])
        method_times[name] = perf_counter() - t
    t = perf_counter()
    bd = segmentation.find_boundaries(raw['dark'])
    bg = segmentation.find_boundaries(raw['gradient'])
    agreement = bd & ndi.binary_dilation(bg) | bg & ndi.binary_dilation(bd)
    disagreement = (bd | bg) & ~agreement
    maps['ensemble'] = np.clip(0.55 * dark + 0.45 * gradient + 0.12 * ndi.gaussian_filter(agreement.astype(float), 0.6), 0, 1)
    markers['ensemble'] = ensemble_markers(markers['dark'], markers['gradient'], maps['ensemble'], diameter)
    raw['ensemble'] = marker_watershed(maps['ensemble'], markers['ensemble'])
    method_times['ensemble'] = perf_counter() - t + method_times['dark'] + method_times['gradient']
    results = {}
    for name in METHODS:
        t = perf_counter()
        if config.adaptive:
            alternate = markers_from_edges(maps['ensemble'], diameter, config.marker_prominence * 0.7) if name == 'ensemble' else None
            post, diagnostics = adaptive_postprocess(raw[name], maps[name], diameter, config, alternate)
        else:
            post, diagnostics = (raw[name].copy(), {'merged': 0, 'split': 0, 'artifacts_removed': 0})
        diagnostics.update({'raw_count': int(raw[name].max()), 'post_count': int(post.max()), 'disagreement_fraction': float(disagreement.mean()), 'low_contrast': bool(np.std(roi) < 0.015), 'timing_note': 'preprocessing + required segmentation branches + postprocessing; no export or measurements', 'frames': source.frames})
        results[name] = Result(source, name, config, crop, diameter, pre, maps[name], raw[name], post, disagreement, shared_seconds + method_times[name] + perf_counter() - t, diagnostics)
    return results
