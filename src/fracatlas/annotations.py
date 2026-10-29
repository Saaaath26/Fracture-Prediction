"""Explicit COCO fracture annotation conversions. Never anatomical masks."""

import cv2
import numpy as np


def polygon_points(flat, width, height):
    a = np.asarray(flat, dtype=float)
    if a.ndim != 1 or a.size < 6 or a.size % 2 or not np.isfinite(a).all():
        raise ValueError("Invalid polygon")
    p = a.reshape(-1, 2)
    if (
        not np.isfinite([width, height]).all()
        or width <= 0
        or height <= 0
        or (p < -1e-6).any()
        or (p[:, 0] > width + 1e-6).any()
        or (p[:, 1] > height + 1e-6).any()
    ):
        raise ValueError("Polygon outside image bounds")
    # Snap only floating-point boundary noise, never materially out-of-bounds labels.
    p = np.clip(p, [0, 0], [width, height])
    if cv2.contourArea(p.astype(np.float32)) <= 0:
        raise ValueError("Degenerate polygon")
    return p


def yolo_box(box, width, height):
    x, y, w, h = map(float, box)
    if (
        not np.isfinite([x, y, w, h, width, height]).all()
        or min(w, h, width, height) <= 0
    ):
        raise ValueError("Invalid box")
    if min(x, y) < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
        raise ValueError("Box outside image bounds")
    return [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]


def _supported(annotation):
    if annotation.get("category_id") != 1 or annotation.get("iscrowd", 0):
        raise ValueError("Unsupported category or crowd annotation")


def yolo_lines(annotations, width, height, task):
    if task not in {"detect", "segment"}:
        raise ValueError("Unknown annotation task")
    lines = []
    for a in annotations:
        _supported(a)
        if task == "detect":
            values = yolo_box(a["bbox"], width, height)
        else:
            polygons = a["segmentation"]
            if not isinstance(polygons, list) or len(polygons) != 1:
                raise ValueError(
                    "Multipart/RLE instance requires explicit conversion policy"
                )
            values = (
                polygon_points(polygons[0], width, height) / [width, height]
            ).ravel()
        lines.append("0 " + " ".join(f"{v:.9f}" for v in values))
    return lines


def fracture_mask(annotations, width, height):
    out = np.zeros((height, width), np.uint8)
    for a in annotations:
        _supported(a)
        if not isinstance(a["segmentation"], list):
            raise ValueError("RLE masks not supported")
        for poly in a["segmentation"]:
            points = polygon_points(poly, width, height)
            cv2.fillPoly(out, [np.rint(points).astype(np.int32)], 1)
    return out
