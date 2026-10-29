"""Geometry in pixel coordinates; physical measurements require verified spacing."""

import cv2
import numpy as np
from .contours import Contour


def features(contour: Contour, spacing_mm=None, image_shape=None):
    if not contour.valid:
        raise ValueError("Invalid/truncated contour cannot yield anatomical features")
    p = np.asarray(contour.points, np.float32).reshape(-1, 2)
    if len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Invalid contour points")
    area = float(cv2.contourArea(p))
    perimeter = float(cv2.arcLength(p, True))
    if area <= 0 or perimeter <= 0:
        raise ValueError("Degenerate contour")
    hull = float(cv2.contourArea(cv2.convexHull(p)))
    m = cv2.moments(p)
    _, (width, height), angle = cv2.minAreaRect(p)
    if m["m00"] == 0:
        raise ValueError("Degenerate contour moments")
    major, minor = max(width, height), min(width, height)
    orientation = (angle + (90 if height > width else 0)) % 180
    result = dict(
        area_px2=area,
        perimeter_px=perimeter,
        centroid_x_px=m["m10"] / m["m00"],
        centroid_y_px=m["m01"] / m["m00"],
        circularity=4 * np.pi * area / perimeter**2,
        solidity=area / hull,
        major_axis_px=major,
        minor_axis_px=minor,
        aspect_ratio=major / minor if minor else None,
        orientation_deg=orientation,
        turning_angle_std_rad=_turning_std(p),
    )
    if image_shape is not None:
        height_px, width_px = image_shape[:2]
        if min(height_px, width_px) <= 0:
            raise ValueError("Invalid image shape")
        result["centroid_x_norm"] = result["centroid_x_px"] / float(width_px)
        result["centroid_y_norm"] = result["centroid_y_px"] / float(height_px)
    if spacing_mm is not None:
        sx, sy = map(float, spacing_mm)
        if not np.isfinite([sx, sy]).all() or min(sx, sy) <= 0:
            raise ValueError("Invalid pixel spacing")
        scaled = p * [sx, sy]
        result.update(
            area_mm2=area * sx * sy,
            perimeter_mm=float(cv2.arcLength(scaled.astype(np.float32), True)),
        )
    return result


def _turning_std(points):
    """Scale-free spread of turning angles. Zero for a regular polygon, not a clinical curvature."""

    closed = np.vstack([points, points[:1]])
    edges = np.diff(closed, axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    turns = np.diff(np.unwrap(np.r_[angles, angles[0]]))
    turns = (turns + np.pi) % (2 * np.pi) - np.pi
    return float(np.std(turns))


def fracture_relationship(contour: Contour, fracture_mask):
    if not contour.valid:
        raise ValueError("Invalid contour")
    mask = np.asarray(fracture_mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected binary fracture mask")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return dict(
            fracture_present=False,
            centroid_distance_to_boundary_px=None,
            overlap_fraction=None,
        )
    p = np.asarray(contour.points, np.float32).reshape(-1, 2)
    if (
        (p < 0).any()
        or (p[:, 0] >= mask.shape[1]).any()
        or (p[:, 1] >= mask.shape[0]).any()
    ):
        raise ValueError("Contour and mask coordinate systems disagree")
    anatomical = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(anatomical, [np.rint(p).astype(np.int32)], 1)
    return dict(
        fracture_present=True,
        centroid_distance_to_boundary_px=float(
            cv2.pointPolygonTest(p, (float(xs.mean()), float(ys.mean())), True)
        ),
        signed_distance_convention="positive inside anatomical candidate",
        overlap_fraction=float(anatomical[ys, xs].mean()),
    )
