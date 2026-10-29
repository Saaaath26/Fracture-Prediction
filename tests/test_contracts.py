"""Interface, integrity and geometry-contract tests. No model fitting."""

import csv
import io
import json
import numpy as np
import pytest
import yaml
from PIL import Image, ImageOps
from fracatlas.annotations import fracture_mask
from fracatlas.cli import main
from fracatlas.config import Experiment, load_config
from fracatlas.data import Dataset, prepare
from fracatlas.devices import resolve_device
from fracatlas.integrity import propose_exclusions, write_integrity_reports
from fracatlas.neural_training import prepare_nnunet
from fracatlas.yolo import evaluate, predict


def test_device_selection_does_not_require_a_model(monkeypatch):
    import fracatlas.devices as devices

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:0") == "cuda:0"
    monkeypatch.setattr(devices, "list_compute_devices", lambda: ["cpu", "mps"])
    assert resolve_device("auto") == "mps"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_fracture_mask_rejects_other_categories():
    with pytest.raises(ValueError):
        fracture_mask(
            [{"category_id": 2, "segmentation": [[0, 0, 4, 0, 4, 4]]}], 10, 10
        )


def test_unassigned_truncation_does_not_block_prepare(tiny, tmp_path):
    folder = tiny / "images" / "Non_fractured"
    folder.mkdir()
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(buffer, format="JPEG")
    (folder / "bad.jpg").write_bytes(buffer.getvalue()[:40])
    with (tiny / "dataset.csv").open("a", newline="") as handle:
        csv.writer(handle).writerow(["bad.jpg", 0, 0, 0])
    dataset = Dataset(tiny)
    audit = dataset.audit()
    assert "bad.jpg" in audit["unusable_unassigned"]
    assert not any(item.startswith("bad.jpg") for item in audit["errors"])
    prepare(dataset, tmp_path / "prepared", "detect")


def test_in_split_truncation_blocks_prepare(tiny, tmp_path):
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(buffer, format="JPEG")
    (tiny / "images" / "Fractured" / "2.jpg").write_bytes(buffer.getvalue()[:40])
    with pytest.raises(ValueError, match="audit failed"):
        prepare(Dataset(tiny), tmp_path / "prepared", "detect")


def test_duplicate_exclusion_policy(tiny):
    fractured = tiny / "images" / "Fractured"
    (fractured / "1.jpg").write_bytes((fractured / "0.jpg").read_bytes())
    dataset = Dataset(tiny)
    rows = propose_exclusions(dataset, dataset.audit())
    assert any(
        row["image_id"] == "1.jpg" and row["action"] == "exclude_duplicate"
        for row in rows
    )
    assert all(row["image_id"] != "0.jpg" for row in rows)
    coco_path = tiny / "Annotations/COCO JSON/COCO_fracture_masks.json"
    coco = json.loads(coco_path.read_text())
    for ann in coco["annotations"]:
        if ann["image_id"] == 1:
            ann["bbox"] = [1, 1, 4, 4]
    coco_path.write_text(json.dumps(coco))
    dataset = Dataset(tiny)
    conflict = propose_exclusions(dataset, dataset.audit())
    assert {row["image_id"] for row in conflict if "conflict" in row["reason"]} == {
        "0.jpg",
        "1.jpg",
    }


def test_within_split_annotation_conflict_is_quarantined(tiny):
    fractured = tiny / "images" / "Fractured"
    (fractured / "4.jpg").write_bytes((fractured / "0.jpg").read_bytes())
    with (tiny / "dataset.csv").open("a", newline="") as handle:
        csv.writer(handle).writerow(["4.jpg", 1, 1, 0])
    with (tiny / "Utilities/Fracture Split/train.csv").open("a") as handle:
        handle.write("4.jpg\n")
    coco_path = tiny / "Annotations/COCO JSON/COCO_fracture_masks.json"
    coco = json.loads(coco_path.read_text())
    coco["images"].append(dict(id=4, file_name="4.jpg", width=20, height=20))
    coco["annotations"].append(
        dict(
            id=4,
            image_id=4,
            category_id=1,
            bbox=[1, 1, 5, 5],
            segmentation=[[1, 1, 6, 1, 6, 6, 1, 6]],
        )
    )
    coco_path.write_text(json.dumps(coco))
    rows = propose_exclusions(Dataset(tiny), Dataset(tiny).audit())
    assert {row["image_id"] for row in rows if "within_split" in row["reason"]} == {
        "0.jpg",
        "4.jpg",
    }


def test_exclusion_manifest_removes_cross_split_copy(tiny, tmp_path):
    fractured = tiny / "images" / "Fractured"
    (fractured / "1.jpg").write_bytes((fractured / "0.jpg").read_bytes())
    # Keep validation nonempty after dropping the duplicate.
    extra = tiny / "images" / "Fractured" / "3.jpg"
    Image.fromarray(np.full((20, 20), 40, np.uint8)).save(extra)
    with (tiny / "dataset.csv").open("a", newline="") as handle:
        csv.writer(handle).writerow(["3.jpg", 1, 1, 0])
    with (tiny / "Utilities/Fracture Split/valid.csv").open("a") as handle:
        handle.write("3.jpg\n")
    coco_path = tiny / "Annotations/COCO JSON/COCO_fracture_masks.json"
    coco = json.loads(coco_path.read_text())
    coco["images"].append(dict(id=3, file_name="3.jpg", width=20, height=20))
    coco["annotations"].append(
        dict(
            id=3,
            image_id=3,
            category_id=1,
            bbox=[2, 2, 8, 8],
            segmentation=[[2, 2, 10, 2, 10, 10, 2, 10]],
        )
    )
    coco_path.write_text(json.dumps(coco))
    dataset = Dataset(tiny)
    assert dataset.audit()["cross_split_duplicates"]
    manifest = tmp_path / "exclude.csv"
    manifest.write_text(
        "image_id,reason,action,group_id\n1.jpg,duplicate,exclude_duplicate,0.jpg|1.jpg\n"
    )
    repaired = Dataset(tiny, exclusion_manifest=manifest)
    audit = repaired.audit()
    assert not audit["cross_split_duplicates"]
    assert not audit["errors"]
    assert "1.jpg" not in repaired.splits["val"]


def test_exif_alignment_is_derived_and_source_is_unchanged(tiny, tmp_path):
    path = tiny / "images" / "Fractured" / "0.jpg"
    original = path.read_bytes()
    image = Image.fromarray(np.zeros((30, 10), np.uint8))
    exif = image.getexif()
    exif[274] = 8
    image.save(path, exif=exif)
    stored = path.read_bytes()
    coco_path = tiny / "Annotations/COCO JSON/COCO_fracture_masks.json"
    coco = json.loads(coco_path.read_text())
    for record in coco["images"]:
        if record["file_name"] == "0.jpg":
            record["width"], record["height"] = 30, 10
    for ann in coco["annotations"]:
        if ann["image_id"] == 0:
            ann["bbox"] = [12, 2, 6, 4]
            ann["segmentation"] = [[12, 2, 18, 2, 18, 6, 12, 6]]
    coco_path.write_text(json.dumps(coco))
    dataset = Dataset(tiny)
    audit = dataset.audit()
    assert not audit["errors"]
    assert audit["orientation_corrections"][0]["image_id"] == "0.jpg"
    prepare(dataset, tmp_path / "prepared", "detect")
    derived = Image.open(tmp_path / "prepared" / "images" / "train" / "0.png")
    assert derived.size == (30, 10)
    assert path.read_bytes() == stored
    assert path.read_bytes() != original
    assert ImageOps.exif_transpose(Image.open(path)).size == (30, 10)


def test_integrity_reports_do_not_train(tiny, tmp_path):
    inventory = write_integrity_reports(Dataset(tiny), tmp_path / "integrity")
    assert inventory["training_executed"] is False
    assert inventory["patient_identifiers"].startswith("absent")
    assert (tmp_path / "integrity" / "derived_split_manifest.csv").is_file()


def test_extract_export_roundtrip(tiny, tmp_path):
    image = np.zeros((100, 100), np.uint8)
    image[20:80, 30:70] = 220
    Image.fromarray(image).save(tiny / "images/Fractured/0.jpg")
    coco_path = tiny / "Annotations/COCO JSON/COCO_fracture_masks.json"
    coco = json.loads(coco_path.read_text())
    for record in coco["images"]:
        if record["file_name"] == "0.jpg":
            record["width"] = record["height"] = 100
    coco_path.write_text(json.dumps(coco))
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                model="yolo11n.pt",
                task="detect",
                dataset_root=str(tiny),
                prepared_root=str(tmp_path / "prepared"),
                output_root=str(tmp_path / "run"),
            )
        )
    )
    output = tmp_path / "features.json"
    main(
        [
            "extract",
            "--config",
            str(config),
            "--image-id",
            "0.jpg",
            "--output",
            str(output),
        ]
    )
    document = json.loads(output.read_text())
    assert document["records"]
    assert document["records"][0]["usable_as_fracture_presence_feature"] is False
    assert document["contour_types_are_not"]
    with pytest.raises(SystemExit):
        main(
            [
                "extract",
                "--config",
                str(config),
                "--image-id",
                "missing.jpg",
                "--output",
                str(tmp_path / "missing.json"),
            ]
        )
    manifest = tmp_path / "review.csv"
    contour_id = document["records"][0]["contour_id"]
    manifest.write_text(
        "image_id,patient_id,split,label,contour_id,hand\n"
        f"0.jpg,patient-0,train,1,{contour_id},1\n"
    )
    from fracatlas.exports import export_table
    from fracatlas.analysis import read_table

    export_table([output], manifest, tmp_path / "table.csv")
    text = (tmp_path / "table.csv").read_text()
    assert "overlap_fraction" not in text and "hand" in text
    read_table(tmp_path / "table.csv", ["area_px2"])


def test_missing_checkpoint_and_split_fail_before_backend(tiny, tmp_path):
    prepared = tmp_path / "prepared"
    prepare(Dataset(tiny), prepared, "detect")
    config = Experiment("yolo11n.pt", "detect", tiny, prepared, tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="local weights"):
        predict(
            config,
            tmp_path / "missing.pt",
            tiny / "images" / "Fractured" / "0.jpg",
            tmp_path / "prediction.json",
        )
    provenance = json.loads((prepared / "provenance.json").read_text())
    provenance["file_sha256"] = {}
    (prepared / "provenance.json").write_text(json.dumps(provenance))
    for child in (prepared / "images" / "test").iterdir():
        child.unlink()
    with pytest.raises(FileNotFoundError, match="Prepared split"):
        evaluate(config, tmp_path / "missing.pt", "test", tmp_path / "eval.json")


def test_prepare_nnunet_exports_without_training(tmp_path):
    image_a = tmp_path / "a.png"
    mask_a = tmp_path / "a-mask.png"
    image_b = tmp_path / "b.png"
    mask_b = tmp_path / "b-mask.png"
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(image_a)
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(mask_a)
    Image.fromarray(np.full((8, 8), 30, np.uint8)).save(image_b)
    Image.fromarray(np.full((8, 8), 255, np.uint8)).save(mask_b)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "image,mask,patient_id,split,verified,annotation_source,contour_type\n"
        "a.png,a-mask.png,p0,train,true,expert-review,verified_bone\n"
        "b.png,b-mask.png,p1,val,true,expert-review,verified_bone\n"
    )
    before = image_a.read_bytes()
    destination = tmp_path / "nnunet"
    prepare_nnunet(manifest, destination)
    assert image_a.read_bytes() == before
    spec = json.loads((destination / "dataset.json").read_text())
    assert "fracture" in spec["description"].lower()
    assert not list((destination).rglob("*.pt"))


def test_dashboard_status_and_config_device(tmp_path):
    from fracatlas.dashboard import default_dataset_root, model_status

    status = model_status()
    assert status["yolo11"] == "untrained_unavailable"
    assert status["yolo26"] == "untrained_unavailable"
    assert status["training_executed_by_dashboard"] is False
    assert status["clinical_validation"] is False
    assert (default_dataset_root() / "dataset.csv").is_file()
    data = dict(
        model="yolo11n.pt",
        task="detect",
        dataset_root="data",
        prepared_root="prepared",
        output_root="run",
        device="tpu",
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_config(path)


def test_prediction_record_declares_unverified_mapping(tmp_path):
    from fracatlas.yolo import prediction_record

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"not-a-real-model")
    record = prediction_record(
        "IMG.jpg", 10, 12, "detect", checkpoint, [], [], [], [], 640, "cpu"
    )
    assert record["record_type"] == "model_prediction"
    assert record["mapping_verified_against_installed_ultralytics"] is False
    assert record["training_executed_by_this_call"] is False
