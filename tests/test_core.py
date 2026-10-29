import json
import numpy as np
import pytest
import yaml
from fracatlas.preprocessing import normalize, preprocess
from fracatlas.annotations import yolo_box, yolo_lines, fracture_mask, polygon_points
from fracatlas.contours import Contour, extract_contours
from fracatlas.geometry import features, fracture_relationship
from fracatlas.config import load_config
from fracatlas.serialization import write_json, feature_record
from fracatlas.evaluation import mask_metrics, classification_metrics


@pytest.fixture
def square():
    return Contour(
        np.array([[2, 2], [8, 2], [8, 8], [2, 8]]), "bone_candidate", True, []
    )


def test_normalize_constant():
    assert not normalize(np.ones((10, 10))).any()
    assert preprocess(np.arange(100).reshape(10, 10)).dtype == np.uint8


@pytest.mark.parametrize(
    "bad", [np.array([[np.nan]]), np.array([]), np.zeros((2, 2, 3))]
)
def test_bad_image(bad):
    with pytest.raises(ValueError):
        normalize(bad)


def test_annotations():
    assert yolo_box([10, 20, 20, 40], 100, 200) == [0.2, 0.2, 0.2, 0.2]
    a = {
        "category_id": 1,
        "bbox": [2, 2, 6, 6],
        "segmentation": [[2, 2, 8, 2, 8, 8, 2, 8]],
    }
    assert yolo_lines([a], 10, 10, "detect")[0].startswith("0 0.500")
    assert len(yolo_lines([a], 10, 10, "segment")[0].split()) == 9
    assert fracture_mask([a], 10, 10)[5, 5] == 1
    assert yolo_lines([], 10, 10, "segment") == []


@pytest.mark.parametrize(
    "box", [[0, 0, 0, 1], [-1, 0, 2, 2], [9, 9, 2, 2], [0, 0, float("nan"), 1]]
)
def test_bad_boxes(box):
    with pytest.raises(ValueError):
        yolo_box(box, 10, 10)


def test_bad_polygon_and_multipart():
    with pytest.raises(ValueError):
        polygon_points([0, 0, 1, 1, 2, 2], 10, 10)
    with pytest.raises(ValueError):
        yolo_lines(
            [{"category_id": 1, "segmentation": [[0, 0, 1, 0, 1, 1]] * 2}],
            10,
            10,
            "segment",
        )


def test_geometry(square):
    f = features(square, (0.5, 2))
    assert f["area_px2"] == 36 and f["perimeter_px"] == 24 and f["area_mm2"] == 36
    assert f["centroid_x_px"] == 5 and f["solidity"] == 1
    assert f["perimeter_mm"] == 30
    mask = np.zeros((12, 12), np.uint8)
    mask[5, 5] = 1
    relationship = fracture_relationship(square, mask)
    assert (
        relationship["overlap_fraction"] == 1
        and relationship["centroid_distance_to_boundary_px"] == 3
    )
    assert (
        fracture_relationship(square, np.zeros_like(mask))["overlap_fraction"] is None
    )


def test_rejected_geometry(square):
    square.valid = False
    with pytest.raises(ValueError):
        features(square)


def test_contour_types():
    a = np.zeros((100, 100), np.uint8)
    a[20:80, 30:70] = 220
    found = extract_contours(a)
    assert found and any(c.valid for c in found)
    assert all(c.contour_type == "bone_candidate" for c in found)
    assert extract_contours(np.zeros_like(a)) == []
    with pytest.raises(ValueError):
        extract_contours(a, "fracture")


def test_serialization(tmp_path, square):
    r = feature_record(
        "sample", features(square), {"method": "test", "image_sha256": "abc"}
    )
    write_json(tmp_path / "features.json", r)
    assert (
        json.loads((tmp_path / "features.json").read_text())["clinically_validated"]
        is False
    )
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"x": float("nan")})
    with pytest.raises(ValueError):
        feature_record("x", {}, {}, "fracture_mask")


def test_config_paths_and_models(tmp_path):
    for version in (11, 26):
        for task, suffix in [("detect", ""), ("segment", "-seg")]:
            data = dict(
                model=f"yolo{version}n{suffix}.pt",
                task=task,
                dataset_root="data with spaces",
                prepared_root="prepared",
                output_root="run",
            )
            p = tmp_path / "config.yaml"
            p.write_text(yaml.safe_dump(data))
            c = load_config(p)
            assert c.dataset_root == tmp_path / "data with spaces"
    data["model"] = "yolo26n.pt"
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(p)


def test_metrics():
    a = np.zeros((5, 5), np.uint8)
    assert mask_metrics(a, a)["dice"] == 1
    b = a.copy()
    b[2, 2] = 1
    assert mask_metrics(a, b)["dice"] == 0
    assert classification_metrics([0, 1], [0.1, 0.9])["auroc"] == 1
    assert classification_metrics([1, 1], [0.5, 0.7])["auroc"] is None


def test_scale_translation_and_circle(square):
    wide = Contour(
        np.array([[0, 0], [20, 0], [20, 4], [0, 4]], np.float32),
        "bone_candidate",
        True,
        [],
    )
    base = features(wide, image_shape=(30, 40))
    scaled = features(Contour(wide.points * 2, "bone_candidate", True, []))
    shifted = features(Contour(wide.points + [5, 7], "bone_candidate", True, []))
    assert scaled["area_px2"] == pytest.approx(base["area_px2"] * 4)
    assert scaled["perimeter_px"] == pytest.approx(base["perimeter_px"] * 2)
    for key in ("circularity", "solidity", "aspect_ratio", "turning_angle_std_rad"):
        assert scaled[key] == pytest.approx(base[key])
        assert shifted[key] == pytest.approx(base[key])
    assert shifted["centroid_x_px"] == pytest.approx(base["centroid_x_px"] + 5)
    assert base["orientation_deg"] == pytest.approx(0, abs=1e-3)
    assert base["centroid_x_norm"] == pytest.approx(base["centroid_x_px"] / 40)
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    circle = np.stack([40 + 30 * np.cos(angles), 50 + 30 * np.sin(angles)], 1)
    assert (
        features(Contour(circle.astype(np.float32), "bone_candidate", True, []))[
            "circularity"
        ]
        > 0.95
    )
    with pytest.raises(ValueError):
        features(
            Contour(np.array([[0, 0], [3, 1]], np.float32), "bone_candidate", True, [])
        )


def test_disconnected_and_abstention():
    image = np.zeros((80, 80), np.uint8)
    image[10:30, 10:30] = 255
    image[50:70, 50:75] = 255
    found = [c for c in extract_contours(image) if c.valid]
    assert len(found) == 2
    assert all(c.quality["identity_verified"] is False for c in found)
    border = np.zeros((40, 40), np.uint8)
    border[0:12, 8:24] = 255
    abstained = extract_contours(border)
    assert abstained and any(c.quality["status"] == "abstain" for c in abstained)
    twice = extract_contours(image)
    assert all(
        np.array_equal(a.points, b.points)
        for a, b in zip(extract_contours(image), twice)
    )


def test_polygon_roundoff_tolerance():
    p = polygon_points([0, 0, 10, 0, 10, 10.000000000000002], 10, 10)
    assert p.max() == 10
    with pytest.raises(ValueError):
        polygon_points([0, 0, 10, 0, 10, 10.01], 10, 10)
