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
    for key in ("split_manifest", "exclusion_manifest"):
        if data.get(key):
            data[key] = (path.parent / Path(data[key]).expanduser()).resolve()
        else:
            data[key] = None
    return Experiment(**data).validate()
