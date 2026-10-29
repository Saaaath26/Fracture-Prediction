"""Lazy YOLO adapters. Training requires explicit acknowledgement and prepared data."""

from pathlib import Path
import hashlib
import json
from .devices import resolve_device
from .guards import require_training
from .serialization import write_json
from .evaluation import save_evaluation


def _prepared(config):
    path = config.prepared_root / "data.yaml"
    provenance = config.prepared_root / "provenance.json"
    if not path.is_file() or not provenance.is_file():
        raise FileNotFoundError("Run prepare first")
    p = json.loads(provenance.read_text())
    if p["task"] != config.task or Path(p["source"]) != config.dataset_root:
        raise ValueError("Prepared dataset does not match experiment")
    if p.get("split_manifest") != (
        str(config.split_manifest) if config.split_manifest else None
    ):
        raise ValueError("Prepared split manifest differs from configuration")
    if p.get("exclusion_manifest") != (
        str(config.exclusion_manifest) if config.exclusion_manifest else None
    ):
        raise ValueError("Prepared exclusion manifest differs from configuration")
    for name, digest in p.get("file_sha256", {}).items():
        candidate = (config.prepared_root / name).resolve()
        if config.prepared_root not in candidate.parents or not candidate.is_file():
            raise ValueError("Missing or unsafe prepared dataset file")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Prepared dataset changed: {name}")
    if p["audit"]["errors"]:
        raise ValueError("Prepared data failed audit")
    return path, p


def train(config, authorized=False, acknowledge_unresolved_leakage=False):
    require_training(authorized)
    config.validate()
    data, provenance = _prepared(config)
    if (
        provenance["audit"]["patient_leakage"].startswith("unresolved")
        and not acknowledge_unresolved_leakage
    ):
        raise ValueError(
            "Patient leakage unresolved; obtain IDs or explicitly acknowledge limitation"
        )
    if config.output_root.exists():
        raise FileExistsError("Choose a fresh experiment output root")
    from ultralytics import YOLO

    config.output_root.mkdir(parents=True)
    write_json(config.output_root / "experiment.json", config.serializable())
    model = YOLO(config.model, task=config.task)
    return model.train(
        data=str(data),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        seed=config.seed,
        device=resolve_device(config.device),
        workers=config.workers,
        deterministic=True,
        project=str(config.output_root),
        name="train",
        exist_ok=False,
    )


def _local_model(checkpoint, task):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("Inference/evaluation require existing local weights")
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task=task)
    if model.task != task:
        raise ValueError("Checkpoint/task mismatch")
    return model


def prediction_record(
    image_id,
    width,
    height,
    task,
    checkpoint,
    boxes_xyxy,
    scores,
    classes,
    polygons,
    imgsz,
    device,
):
    checkpoint = Path(checkpoint)
    return dict(
        schema_version="1.1",
        record_type="model_prediction",
        image_id=image_id,
        modality="radiograph",
        width=width,
        height=height,
        task=task,
        coordinate_convention="xyxy_pixels_and_polygons_in_the_source_image",
        coordinate_mapping=(
            "Ultralytics boxes.xyxy and masks.xy are consumed as original-image pixels. "
            "This adapter does not rescale them."
        ),
        mapping_verified_against_installed_ultralytics=False,
        imgsz=imgsz,
        device=device,
        checkpoint=str(checkpoint),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        boxes_xyxy=boxes_xyxy,
        scores=scores,
        classes=classes,
        polygons=polygons,
        clinically_validated=False,
        training_executed_by_this_call=False,
    )


def predict(config, checkpoint, source, output):
    from PIL import Image

    source = Path(source).resolve()
    with Image.open(source) as im:
        width, height = im.size
    device = resolve_device(config.device)
    model = _local_model(checkpoint, config.task)
    result = model.predict(
        source=str(source), imgsz=config.imgsz, device=device, verbose=False
    )[0]
    boxes = result.boxes
    if boxes is None:
        boxes_xyxy, scores, classes = [], [], []
    else:
        boxes_xyxy = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
    write_json(
        output,
        prediction_record(
            source.name,
            width,
            height,
            config.task,
            checkpoint,
            boxes_xyxy,
            scores,
            classes,
            [] if result.masks is None else [p.tolist() for p in result.masks.xy],
            config.imgsz,
            device,
        ),
    )


def evaluate(config, checkpoint, split, output):
    data, _ = _prepared(config)
    if split not in {"val", "test"}:
        raise ValueError("Use val or test")
    split_dir = config.prepared_root / "images" / split
    if not split_dir.is_dir() or not any(split_dir.iterdir()):
        raise FileNotFoundError(f"Prepared split is missing or empty: {split}")
    device = resolve_device(config.device)
    model = _local_model(checkpoint, config.task)
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=config.imgsz,
        batch=config.batch,
        device=device,
        workers=config.workers,
        project=str(Path(output).parent),
        name="yolo_eval",
    )
    save_evaluation(
        output,
        config.task,
        split,
        {k: float(v) for k, v in metrics.results_dict.items()},
        dict(checkpoint=str(Path(checkpoint).resolve()), config=config.serializable()),
    )
