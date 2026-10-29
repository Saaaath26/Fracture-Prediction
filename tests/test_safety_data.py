import json
import pytest
from fracatlas.data import Dataset, prepare
from fracatlas.yolo import train
from fracatlas.analysis import train_relationship, read_table
from fracatlas.neural_training import train_neural, verified_manifest


def test_audit_prepare(tiny, tmp_path):
    d = Dataset(tiny)
    a = d.audit()
    assert not a["errors"] and a["images"] == 3
    assert a["patient_leakage"].startswith("unresolved")
    out = tmp_path / "prepared with spaces"
    prepare(d, out, "segment")
    assert (out / "labels/val/1.txt").is_file()
    assert json.loads((out / "provenance.json").read_text())["task"] == "segment"
    with pytest.raises(FileExistsError):
        prepare(d, out, "segment")
    with pytest.raises(ValueError):
        prepare(d, tiny / "output", "detect")


def test_split_leakage(tiny):
    (tiny / "Utilities/Fracture Split/test.csv").write_text("image_id\n0.jpg\n")
    assert any("Split overlap" in x for x in Dataset(tiny).audit()["errors"])


def test_duplicate_pixels(tiny):
    p = tiny / "images/Fractured"
    (p / "1.jpg").write_bytes((p / "0.jpg").read_bytes())
    assert Dataset(tiny).audit()["cross_split_duplicates"]


def test_training_blocked_before_import_or_io(tmp_path):
    with pytest.raises(PermissionError):
        train(None)
    with pytest.raises(PermissionError):
        train_relationship("missing", [], "logistic", tmp_path / "run")
    with pytest.raises(PermissionError):
        train_neural("missing", "unet", tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_target_leakage_rejected():
    with pytest.raises(ValueError):
        read_table("missing", ["fracture_count"])
    with pytest.raises(ValueError):
        read_table("missing", ["overlap_fraction"])


def test_patient_leakage_table(tmp_path):
    f = tmp_path / "table.csv"
    f.write_text(
        "image_id,patient_id,split,label,area_px2\na,p,train,0,3\nb,p,val,1,4\n"
    )
    with pytest.raises(ValueError, match="leakage"):
        read_table(f, ["area_px2"])


def test_unverified_masks_rejected(tmp_path):
    f = tmp_path / "manifest.csv"
    f.write_text(
        "image,mask,patient_id,split,verified,annotation_source,contour_type\na,b,p,train,false,test,verified_bone\n"
    )
    with pytest.raises(ValueError, match="verification"):
        verified_manifest(f, "unet")


def test_identical_duplicate_paths_recorded(tiny):
    other = tiny / "images/Non_fractured"
    other.mkdir()
    (other / "0.jpg").write_bytes((tiny / "images/Fractured/0.jpg").read_bytes())
    dataset = Dataset(tiny)
    assert len(dataset.images) == 3
    assert len(dataset.audit()["duplicate_source_paths"]) == 1
    assert dataset.images["0.jpg"].parent.name == "Fractured"


def test_curated_patient_split(tiny, tmp_path):
    manifest = tmp_path / "reviewed.csv"
    manifest.write_text(
        "image_id,patient_id,split\n0.jpg,p0,train\n1.jpg,p1,val\n2.jpg,p2,test\n"
    )
    dataset = Dataset(tiny, manifest)
    assert not dataset.audit()["errors"]
    assert dataset.audit()["patient_leakage"] == "checked"
    manifest.write_text(
        "image_id,patient_id,split\n0.jpg,p0,train\n1.jpg,p0,val\n2.jpg,p2,test\n"
    )
    assert "Patient overlap across splits" in Dataset(tiny, manifest).audit()["errors"]


def test_prepared_integrity(tiny, tmp_path):
    from fracatlas.yolo import _prepared
    from fracatlas.config import Experiment

    output = tmp_path / "prepared"
    prepare(Dataset(tiny), output, "detect")
    config = Experiment("yolo11n.pt", "detect", tiny, output, tmp_path / "run")
    _prepared(config)
    (output / "labels/train/0.txt").write_text("invalid")
    with pytest.raises(ValueError, match="changed"):
        _prepared(config)


def test_export_geometry_table(tmp_path):
    from fracatlas.exports import export_table, GEOMETRY_FIELDS

    feature = tmp_path / "feature.json"
    feature.write_text(
        json.dumps(
            {
                "image_id": "a",
                "records": [
                    {
                        "contour_id": 0,
                        "contour_type": "bone_candidate",
                        "features": {k: 1 for k in GEOMETRY_FIELDS},
                    }
                ],
            }
        )
    )
    manifest = tmp_path / "review.csv"
    manifest.write_text("image_id,patient_id,split,label,contour_id\na,p,train,1,0\n")
    export_table([feature], manifest, tmp_path / "table.csv")
    assert "fracture_count" not in (tmp_path / "table.csv").read_text()
    X, y, split = read_table(tmp_path / "table.csv", ["area_px2"])
    assert X[0, 0] == 1 and y[0] == 1
