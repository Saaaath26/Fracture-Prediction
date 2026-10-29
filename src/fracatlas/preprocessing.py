"""Deterministic radiograph preprocessing, preserving source data."""

import io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps


def read_gray(path):
    """Stored pixels only. Annotation workflows use ``aligned_gray``."""

    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def interpret_image(blob, expected_size=None):
    """Decode stored pixels and decide whether EXIF orientation matches annotations.

    EXIF is applied only when the stored size disagrees with the annotation size
    and the oriented size matches that annotation size exactly. A display tag is
    not applied when stored pixels already match the annotation frame.
    Truncated-image loading is never enabled.
    """

    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        raw = image.size
        orientation = int(image.getexif().get(274, 1) or 1)
        gray = np.asarray(image.convert("L")).copy()
        aligned = raw
        applied = False
        if expected_size is not None and raw != tuple(expected_size):
            oriented = ImageOps.exif_transpose(image)
            if oriented.size != tuple(expected_size):
                raise ValueError(
                    "COCO dimension mismatch: "
                    f"stored {raw}, exif_oriented {oriented.size}, annotation {tuple(expected_size)}"
                )
            aligned = oriented.size
            applied = True
    return gray, _policy(raw, aligned, orientation, applied)


def aligned_gray(path, expected_size=None):
    """Grayscale in the annotation coordinate frame."""

    blob = Path(path).read_bytes()
    gray, policy = interpret_image(blob, expected_size)
    if not policy["exif_applied"]:
        return gray, policy
    with Image.open(io.BytesIO(blob)) as image:
        image.load()
        view = ImageOps.exif_transpose(image).convert("L")
        return np.asarray(view).copy(), policy


def aligned_pil(path, expected_size=None):
    """RGB image in the annotation frame, for derived dataset files only."""

    with Image.open(path) as image:
        image.load()
        raw = image.size
        orientation = int(image.getexif().get(274, 1) or 1)
        applied = False
        view = image
        aligned = raw
        if expected_size is not None and raw != tuple(expected_size):
            view = ImageOps.exif_transpose(image)
            if view.size != tuple(expected_size):
                raise ValueError(
                    "COCO dimension mismatch: "
                    f"stored {raw}, exif_oriented {view.size}, annotation {tuple(expected_size)}"
                )
            applied = True
            aligned = view.size
        return view.convert("RGB").copy(), _policy(raw, aligned, orientation, applied)


def _policy(raw, aligned, orientation, applied):
    return dict(
        policy="exif_transpose_matches_annotation" if applied else "stored_pixels",
        stored_size=[raw[0], raw[1]],
        aligned_size=[aligned[0], aligned[1]],
        exif_orientation=orientation,
        exif_applied=applied,
    )


def normalize(image):
    a = np.asarray(image, dtype=np.float32)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("Expected a finite nonempty grayscale image")
    lo, hi = np.percentile(a, [1, 99])
    return np.zeros_like(a) if hi <= lo else np.clip((a - lo) / (hi - lo), 0, 1)


def preprocess(image, clahe=False):
    out = np.rint(normalize(image) * 255).astype(np.uint8)
    return cv2.createCLAHE(2.0, (8, 8)).apply(out) if clahe else out
