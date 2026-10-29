"""Classical candidate contours. These are hypotheses, not anatomical ground truth."""

from dataclasses import dataclass, field
import cv2
import numpy as np
from .preprocessing import preprocess
from .schemas import CANDIDATE_CONTOURS, BONE_CANDIDATE


@dataclass
class Contour:
    points: np.ndarray
    contour_type: str
    valid: bool
    warnings: list
    method: str = "clahe_otsu_morphology_v1"
    uncertainty: str = "uncalibrated; anatomical identity requires expert verification"
    quality: dict = field(default_factory=dict)


def extract_contours(image, contour_type="bone_candidate", min_area_fraction=0.005):
    if contour_type not in CANDIDATE_CONTOURS:
        raise ValueError("Only candidate anatomical contours can be extracted")
    if not 0 < min_area_fraction < 1:
        raise ValueError("Invalid minimum area")
    a = preprocess(image, clahe=True)
    if np.ptp(a) == 0:
        return []
    blur = cv2.GaussianBlur(a, (5, 5), 0)
    if contour_type == "bone_candidate":
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Low-intensity foreground envelope; not a bone segmentation.
        mask = (blur > max(5, float(np.percentile(blur, 20)))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = a.shape
    result = []
    for c in sorted(found, key=cv2.contourArea, reverse=True):
        p = c.reshape(-1, 2)
        area = cv2.contourArea(c)
        if area < h * w * min_area_fraction:
            continue
        border = bool(
            (p[:, 0] <= 0).any()
            or (p[:, 1] <= 0).any()
            or (p[:, 0] >= w - 1).any()
            or (p[:, 1] >= h - 1).any()
        )
        warnings = [
            "unverified anatomical identity",
            "external contour only; overlapping structures are not separated",
            "radiographic projection and positioning can change apparent shape",
        ]
        if contour_type == BONE_CANDIDATE:
            warnings.append("Otsu bright-region candidate, not a bone segmentation")
        else:
            warnings.append(
                "low-intensity foreground envelope, not a soft-tissue segmentation"
            )
        if border:
            warnings.append("touches image boundary; potentially truncated")
        if area > 0.9 * h * w:
            warnings.append("implausibly large foreground")
        valid = len(p) >= 3 and not border and area <= 0.9 * h * w
        quality = dict(
            status="candidate" if valid else "abstain",
            identity_verified=False,
            calibrated_confidence=None,
            area_fraction=float(area / (h * w)),
            touches_border=border,
            external_contour_only=True,
            separates_overlapping_bones=False,
        )
        result.append(Contour(p, contour_type, valid, warnings, quality=quality))
    return result
