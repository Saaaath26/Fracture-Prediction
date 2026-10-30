"""Strict experiment configurations; relative paths resolve beside the YAML file."""

from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from .devices import resolve_device


@dataclass(frozen=True)
class Experiment:
    model: str
    task: str
    dataset_root: Path
    prepared_root: Path
    output_root: Path
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    seed: int = 42
    device: str = "cpu"
    workers: int = 0
    patience: int = 40
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    cos_lr: bool = True
    close_mosaic: int = 15
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.25
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.2
    amp: bool = True
    deterministic: bool = False
    split_manifest: Path | None = None
    exclusion_manifest: Path | None = None

    def validate(self):
        names = {
            f"yolo{v}{s}{suffix}.pt"
            for v in (11, 26)
            for s in "nsmlx"
            for suffix in ("", "-seg")
        }
        if self.model not in names:
            raise ValueError("Use a verified YOLO11/YOLO26 checkpoint name")
        if self.task not in {"detect", "segment"}:
            raise ValueError("task must be detect or segment")
        if ("-seg" in self.model) != (self.task == "segment"):
            raise ValueError("Checkpoint/task mismatch")
        for key in ("epochs", "imgsz", "batch"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if (
            type(self.seed) is not int
            or type(self.workers) is not int
            or self.workers < 0
        ):
            raise ValueError(
                "seed/workers must be integers; workers must be nonnegative"
            )
        resolve_device(self.device)
        if self.optimizer not in {"SGD", "Adam", "AdamW", "auto"}:
            raise ValueError("optimizer must be SGD, Adam, AdamW, or auto")
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("patience must be a nonnegative integer")
        if type(self.close_mosaic) is not int or self.close_mosaic < 0:
            raise ValueError("close_mosaic must be a nonnegative integer")
        for key in (
            "lr0",
            "lrf",
            "weight_decay",
            "warmup_epochs",
            "mosaic",
            "mixup",
            "copy_paste",
            "fliplr",
            "flipud",
            "degrees",
            "translate",
            "scale",
            "hsv_h",
            "hsv_s",
            "hsv_v",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number")
            if value < 0:
                raise ValueError(f"{key} must be nonnegative")
        for key in ("mosaic", "mixup", "copy_paste", "fliplr", "flipud"):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must be between 0 and 1")
        if not isinstance(self.cos_lr, bool) or not isinstance(self.amp, bool):
            raise ValueError("cos_lr and amp must be booleans")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a boolean")
        if str(self.device).startswith("cuda"):
            pixels = self.imgsz * self.imgsz * self.batch
            limit = 640 * 640 * (8 if self.task == "segment" else 16)
            if pixels > limit:
                raise ValueError(
                    "imgsz/batch is too large for a 16GB GPU. "
                    "Use imgsz 640, batch 16 for detection, or batch 8 for segmentation."
                )
        for target in (self.prepared_root, self.output_root):
            if target == self.dataset_root or self.dataset_root in target.parents:
                raise ValueError("Outputs must be outside the original dataset")
        return self

    def serializable(self):
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }


def load_config(path):
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    for key in ("dataset_root", "prepared_root", "output_root"):
        data[key] = (path.parent / Path(data[key]).expanduser()).resolve()
    if data.get("device") is not None:
        data["device"] = str(data["device"])
    for key in (
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "mosaic",
        "mixup",
        "copy_paste",
        "fliplr",
        "flipud",
        "degrees",
        "translate",
        "scale",
        "hsv_h",
        "hsv_s",
        "hsv_v",
    ):
        if key in data and data[key] is not None:
            data[key] = float(data[key])
    for key in ("patience", "close_mosaic"):
        if key in data and data[key] is not None:
            data[key] = int(data[key])
    for key in ("split_manifest", "exclusion_manifest"):
        if data.get(key):
            data[key] = (path.parent / Path(data[key]).expanduser()).resolve()
        else:
            data[key] = None
    return Experiment(**data).validate()
