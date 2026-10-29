"""Versioned JSON exports; NaN is rejected rather than silently serialized."""

import json
from pathlib import Path
import numpy as np
from .schemas import ANATOMICAL_CONTOURS


def _default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    text = json.dumps(data, indent=2, allow_nan=False, default=_default)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


def feature_record(image_id, features, provenance, contour_type="bone_candidate"):
    if contour_type not in ANATOMICAL_CONTOURS:
        raise ValueError("Fracture masks cannot be exported as anatomical contours")
    if not provenance.get("method") or not provenance.get("image_sha256"):
        raise ValueError("Method and source hash are required")
    return dict(
        schema_version="1.1",
        record_type="anatomical_contour_candidate",
        image_id=image_id,
        modality="radiograph",
        contour_type=contour_type,
        fracture_mask_equivalent=False,
        external_rgb_limb=False,
        coordinate_convention="xy_pixels_origin_top_left",
        features=features,
        provenance=provenance,
        units="pixels unless explicitly suffixed _mm; norm fields are fractions of the annotation frame",
        clinically_validated=False,
        rgb_transfer={"status": "experimental_unvalidated"},
    )
