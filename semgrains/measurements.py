import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation

COLUMNS = ["id", "included", "exclusion_reason", "area_px2", "perimeter_px", "centroid_x_px", "centroid_y_px",
           "major_axis_px", "minor_axis_px", "eccentricity", "equivalent_diameter_px", "feret_max_px",
           "boundary_gradient_mean", "boundary_gradient_cv", "boundary_edge_mean", "boundary_edge_cv", "boundary_contrast_mean", "boundary_uniformity",
           "area", "perimeter", "major_axis", "minor_axis", "equivalent_diameter", "feret_max", "unit", "area_unit"]


def measure_grains(labels, original_gray, edge, config):
    records = []


    source_gradient = filters.scharr(original_gray)
    border_ids = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    h, w = labels.shape
    for r in measure.regionprops(labels):

        y0, x0, y1, x1 = r.bbox
        sl = (slice(max(0, y0-2), min(h, y1+2)), slice(max(0, x0-2), min(w, x1+2)))
        mask = labels[sl] == r.label
        contour = segmentation.find_boundaries(mask, mode="inner")
        vals = source_gradient[sl][contour]
        mean = float(vals.mean()) if vals.size else 0
        cv = float(vals.std() / max(mean, 1e-6)) if vals.size else 0
        edge_vals = edge[sl][contour]
        edge_mean = float(edge_vals.mean()) if edge_vals.size else 0
        edge_cv = float(edge_vals.std() / max(edge_mean, 1e-6)) if edge_vals.size else 0
        inside = mask & ~ndi.binary_erosion(mask, iterations=2)
        outside = ndi.binary_dilation(mask, iterations=2) & ~mask
        g = original_gray[sl]
        contrast = float(g[inside].mean() - g[outside].mean()) if inside.any() and outside.any() else 0
        s = config.spacing
        record = {"id": int(r.label), "included": r.label not in border_ids,
                  "exclusion_reason": "image_border" if r.label in border_ids else "",
                  "area_px2": float(r.area), "perimeter_px": float(r.perimeter_crofton),
                  "centroid_x_px": float(r.centroid[1]), "centroid_y_px": float(r.centroid[0]),
                  "major_axis_px": float(r.axis_major_length), "minor_axis_px": float(r.axis_minor_length),
                  "eccentricity": float(r.eccentricity), "equivalent_diameter_px": float(r.equivalent_diameter_area),
                  "feret_max_px": float(r.feret_diameter_max), "boundary_gradient_mean": mean,
                  "boundary_gradient_cv": cv, "boundary_edge_mean": edge_mean, "boundary_edge_cv": edge_cv, "boundary_contrast_mean": contrast,
                  "boundary_uniformity": 1/(1+cv), "area": float(r.area)*s*s,
                  "perimeter": float(r.perimeter_crofton)*s, "major_axis": float(r.axis_major_length)*s,
                  "minor_axis": float(r.axis_minor_length)*s, "equivalent_diameter": float(r.equivalent_diameter_area)*s,
                  "feret_max": float(r.feret_diameter_max)*s, "unit": config.unit, "area_unit": config.unit + "^2"}
        records.append(record)
    return pd.DataFrame.from_records(records, columns=COLUMNS).astype({"id": "int64", "included": "bool"})
