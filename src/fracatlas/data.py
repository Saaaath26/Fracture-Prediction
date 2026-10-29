"""Read-only dataset indexing, leakage audit and isolated YOLO dataset preparation."""

import csv
import hashlib
import shutil
from collections import defaultdict, Counter
from pathlib import Path
import cv2
import numpy as np
import yaml
from .annotations import yolo_lines
from .preprocessing import aligned_pil, interpret_image
from .schemas import EXCLUSION_ACTIONS
from .serialization import write_json


def _read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


class Dataset:
    def __init__(self, root, split_manifest=None, exclusion_manifest=None):
        self.root = Path(root).resolve()
        rows = _read_csv(self.root / "dataset.csv")
        self.metadata = {r["image_id"]: r for r in rows}
        if len(rows) != len(self.metadata):
            raise ValueError("Duplicate metadata IDs")
        self.images = {}
        self.duplicate_paths = []
        for path in sorted((self.root / "images").rglob("*.jpg")):
            if path.name in self.images:
                previous = self.images[path.name]
                if previous.read_bytes() != path.read_bytes():
                    raise ValueError(
                        f"Conflicting duplicate image filename: {path.name}"
                    )
                self.duplicate_paths.append([str(previous), str(path)])
                expected = (
                    "Fractured"
                    if self.metadata[path.name]["fractured"] == "1"
                    else "Non_fractured"
                )
                if previous.parent.name == expected:
                    continue
            self.images[path.name] = path
        coco = json_load(self.root / "Annotations/COCO JSON/COCO_fracture_masks.json")
        self.coco_images = {im["id"]: im for im in coco["images"]}
        if len(self.coco_images) != len(coco["images"]):
            raise ValueError("Duplicate COCO image IDs")
        self.annotations = defaultdict(list)
        for a in coco["annotations"]:
            self.annotations[self.coco_images[a["image_id"]]["file_name"]].append(a)
        self.splits = {}
        for split, filename in [("train", "train"), ("val", "valid"), ("test", "test")]:
            self.splits[split] = [
                r["image_id"]
                for r in _read_csv(
                    self.root / f"Utilities/Fracture Split/{filename}.csv"
                )
            ]

        self.split_manifest = (
            str(Path(split_manifest).resolve()) if split_manifest else None
        )
        self.source_image_count = len(self.images)
        if split_manifest:
            selected = _read_csv(split_manifest)
            if not selected or len({r["image_id"] for r in selected}) != len(selected):
                raise ValueError("Curated manifest must contain unique image IDs")
            self.splits = {s: [] for s in ("train", "val", "test")}
            for r in selected:
                if r.get("split") not in self.splits or not r.get("patient_id"):
                    raise ValueError(
                        "Curated split requires verified patient_id and train/val/test"
                    )
                if (
                    r["image_id"] not in self.metadata
                    or r["image_id"] not in self.images
                ):
                    raise ValueError("Unknown curated image ID")
                self.metadata[r["image_id"]]["patient_id"] = r["patient_id"]
                self.splits[r["split"]].append(r["image_id"])
            names = {r["image_id"] for r in selected}
            self.images = {n: p for n, p in self.images.items() if n in names}
            self.metadata = {n: r for n, r in self.metadata.items() if n in names}
            self.coco_images = {
                i: r for i, r in self.coco_images.items() if r["file_name"] in names
            }
            self.annotations = {n: a for n, a in self.annotations.items() if n in names}

        self.coco_size = {
            row["file_name"]: (row["width"], row["height"])
            for row in self.coco_images.values()
        }
        self.exclusions = {}
        self.exclusion_manifest = (
            str(Path(exclusion_manifest).resolve()) if exclusion_manifest else None
        )
        if exclusion_manifest:
            selected = _read_csv(exclusion_manifest)
            if not selected:
                raise ValueError("Exclusion manifest is empty")
            for row in selected:
                name = row.get("image_id")
                if (
                    not name
                    or name in self.exclusions
                    or row.get("action") not in EXCLUSION_ACTIONS
                    or not row.get("reason")
                ):
                    raise ValueError("Invalid exclusion manifest row")
                if name not in self.images or name not in self.metadata:
                    raise ValueError(f"Unknown exclusion image ID: {name}")
                self.exclusions[name] = dict(row)
            for split in self.splits:
                self.splits[split] = [
                    name for name in self.splits[split] if name not in self.exclusions
                ]
        self.alignment = {}

    def audit(self):
        errors = []
        warnings = []
        unusable_unassigned = []
        orientation_corrections = []
        exif_not_applied = []
        byte_groups = defaultdict(list)
        pixel_groups = defaultdict(list)
        near_groups = defaultdict(list)
        sizes = {}
        memberships = defaultdict(list)
        for split, ids in self.splits.items():
            if not ids:
                errors.append(f"Empty split: {split}")
            for name in ids:
                memberships[name].append(split)
            if len(ids) != len(set(ids)):
                errors.append(f"Duplicate IDs within {split}")
        for name, splits in memberships.items():
            if len(splits) > 1:
                errors.append(f"Split overlap: {name}")
            if name not in self.images:
                errors.append(f"Missing split image: {name}")
        if set(self.images) != set(self.metadata):
            errors.append("Image/metadata ID mismatch")
        self.alignment = {}
        for name, path in self.images.items():
            blob = path.read_bytes()
            byte_groups[hashlib.sha256(blob).hexdigest()].append(name)
            try:
                gray, policy = interpret_image(blob, self.coco_size.get(name))
                self.alignment[name] = policy
                sizes[name] = tuple(policy["aligned_size"])
                if policy["exif_applied"]:
                    orientation_corrections.append(dict(image_id=name, **policy))
                elif policy["exif_orientation"] not in (0, 1):
                    reason = (
                        "stored pixels already match the annotation frame"
                        if name in self.coco_size
                        else "no annotation dimensions; stored pixels kept"
                    )
                    exif_not_applied.append(
                        dict(
                            image_id=name,
                            exif_orientation=policy["exif_orientation"],
                            reason=reason,
                        )
                    )
                pixel_groups[
                    hashlib.sha256(
                        str(gray.shape).encode() + gray.tobytes()
                    ).hexdigest()
                ].append(name)
                small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
                fingerprint = np.packbits(small[:, 1:] > small[:, :-1]).tobytes().hex()
                near_groups[fingerprint].append(name)
                annotations = self.annotations.get(name, [])
                if bool(int(self.metadata[name]["fractured"])) != bool(annotations):
                    errors.append(f"Fracture metadata/annotation mismatch: {name}")
                if int(self.metadata[name]["fracture_count"]) != len(annotations):
                    errors.append(f"Fracture count mismatch: {name}")
                for task in ("detect", "segment"):
                    yolo_lines(annotations, *sizes[name], task)
            except (OSError, ValueError, KeyError) as e:
                message = f"{name}: {e}"
                if isinstance(e, OSError) and name not in memberships:
                    warnings.append(message)
                    unusable_unassigned.append(name)
                else:
                    errors.append(message)
        for im in self.coco_images.values():
            name = im["file_name"]
            if name not in sizes:
                continue
            if sizes[name] != (im["width"], im["height"]):
                errors.append(f"COCO dimension mismatch: {name}")
        duplicates = lambda groups: [v for v in groups.values() if len(v) > 1]
        exact = duplicates(pixel_groups)
        cross = [
            g for g in exact if len({s for n in g for s in memberships.get(n, [])}) > 1
        ]
        if cross:
            errors.append("Decoded-pixel duplicates cross official splits")
        patient_available = all(
            self.metadata.get(n, {}).get("patient_id") for n in memberships
        )
        patients = defaultdict(set)
        if patient_available:
            for name, splits in memberships.items():
                patients[self.metadata[name]["patient_id"]].update(splits)
        patient_leaks = [p for p, s in patients.items() if len(s) > 1]
        if patient_leaks:
            errors.append("Patient overlap across splits")
        return dict(
            images=len(self.images),
            source_image_count=self.source_image_count,
            excluded_by_curated_manifest=self.source_image_count - len(self.images),
            split_manifest=self.split_manifest,
            exclusion_manifest=self.exclusion_manifest,
            exclusions=sorted(self.exclusions),
            duplicate_source_paths=self.duplicate_paths,
            metadata_rows=len(self.metadata),
            coco_images=len(self.coco_images),
            fracture_instances=sum(map(len, self.annotations.values())),
            class_counts=dict(Counter(r["fractured"] for r in self.metadata.values())),
            splits={s: len(v) for s, v in self.splits.items()},
            unassigned_images=len(set(self.images) - set(memberships)),
            errors=errors,
            warnings=warnings,
            unusable_unassigned=sorted(unusable_unassigned),
            orientation_corrections=orientation_corrections,
            exif_orientation_not_applied=exif_not_applied,
            blocking_policy=(
                "In-split corruption, unresolved geometry, cross-split exact pixel "
                "duplicates and patient overlap are blocking. Unassigned JPEG decode "
                "failures are warnings and stay out of prepared splits. EXIF rotation "
                "is applied only when it is the unique size match for the annotation frame."
            ),
            byte_duplicates=duplicates(byte_groups),
            pixel_duplicates=exact,
            cross_split_duplicates=cross,
            near_duplicate_candidates=duplicates(near_groups),
            near_duplicate_policy=(
                "Exact 64-bit dHash collisions only; review manually; not exhaustive; "
                "not removed automatically"
            ),
            patient_leakage=(
                "checked" if patient_available else "unresolved: no patient identifiers"
            ),
            patient_overlap=patient_leaks,
            anatomical_annotations="absent: COCO contains fracture regions only",
            multiscan_images=sum(
                int(r.get("multiscan", 0)) for r in self.metadata.values()
            ),
        )


def prepare(dataset, destination, task):
    destination = Path(destination).resolve()
    if destination == dataset.root or dataset.root in destination.parents:
        raise ValueError("Cannot write inside source dataset")
    if destination.exists():
        raise FileExistsError("Use a new preparation directory")
    report = dataset.audit()
    if report["errors"]:
        raise ValueError(f"Dataset audit failed: {report['errors'][:5]}")
    derived_files = {}
    for split, ids in dataset.splits.items():
        for sub in ("images", "labels"):
            (destination / sub / split).mkdir(parents=True, exist_ok=True)
        for name in ids:
            source = dataset.images[name]
            policy = dataset.alignment[name]
            width, height = policy["aligned_size"]
            lines = yolo_lines(dataset.annotations.get(name, []), width, height, task)
            if policy["exif_applied"]:
                image, _ = aligned_pil(source, dataset.coco_size.get(name))
                filename = Path(name).with_suffix(".png").name
                image.save(destination / "images" / split / filename)
                method = "derived_png_exif_transpose"
            else:
                filename = name
                # Copy bytes. Hard links are rejected because permission or write
                # changes would alter the original dataset inode.
                shutil.copy2(source, destination / "images" / split / filename)
                method = "copied_stored_bytes"
            derived_files[name] = dict(prepared_name=filename, method=method, **policy)
            (
                destination / "labels" / split / Path(name).with_suffix(".txt")
            ).write_text("\n".join(lines) + "\n" if lines else "")
    spec = dict(
        path=str(destination),
        train="images/train",
        val="images/val",
        test="images/test",
        names={0: "fractured"},
    )
    (destination / "data.yaml").write_text(yaml.safe_dump(spec))
    write_json(destination / "filename_map.json", derived_files)
    write_json(
        destination / "provenance.json",
        dict(
            source=str(dataset.root),
            task=task,
            splits=dataset.splits,
            audit=report,
            derived_files=derived_files,
            image_materialization=(
                "stored bytes copied; EXIF-aligned annotation frames saved as PNG; "
                "hardlinks not used"
            ),
            policy=(
                "reviewed_custom_split"
                if dataset.split_manifest
                else "authors_fractured_only; negatives remain unassigned"
            ),
            split_manifest=dataset.split_manifest,
            exclusion_manifest=dataset.exclusion_manifest,
            file_sha256={
                str(p.relative_to(destination)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in destination.rglob("*")
                if p.is_file() and p.name != "provenance.json"
            },
        ),
    )
    return destination / "data.yaml"


def json_load(path):
    import json

    return json.loads(Path(path).read_text())
