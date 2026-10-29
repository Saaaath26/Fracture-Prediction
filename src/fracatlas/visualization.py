"""Consistent overlays in original radiograph coordinates."""

import cv2
import numpy as np


def overlay(image, annotations=(), contours=(), predictions=None):
    a = np.asarray(image)
    out = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB) if a.ndim == 2 else a.copy()
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(
            out, (round(x), round(y)), (round(x + w), round(y + h)), (255, 90, 40), 2
        )
        for polygon in ann.get("segmentation", []):
            if not isinstance(polygon, (list, tuple)):
                continue
            cv2.polylines(
                out,
                [np.rint(np.array(polygon).reshape(-1, 2)).astype(np.int32)],
                True,
                (255, 90, 40),
                2,
            )
    for c in contours:
        color = (50, 220, 120) if c.contour_type == "bone_candidate" else (60, 180, 255)
        cv2.polylines(
            out,
            [np.asarray(c.points, np.int32)],
            True,
            color if c.valid else (180, 180, 180),
            2,
        )
    if predictions:
        for box in predictions.get("boxes_xyxy", []):
            x1, y1, x2, y2 = map(round, box)
            cv2.rectangle(out, (x1, y1), (x2, y2), (200, 60, 240), 2)
        for poly in predictions.get("polygons", []):
            cv2.polylines(
                out, [np.rint(poly).astype(np.int32)], True, (200, 60, 240), 2
            )
    return out
