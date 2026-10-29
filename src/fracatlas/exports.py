"""Export reviewed geometry records to classifier CSV without inventing patient IDs."""

import csv
import json
from pathlib import Path
from .schemas import ANATOMICAL_CONTOURS, CONFOUNDER_FIELDS, GEOMETRY_EXPORT_FIELDS

GEOMETRY_FIELDS = GEOMETRY_EXPORT_FIELDS


def export_table(feature_files, reviewed_manifest, output):
    """One explicit contour per image; no automatic largest-contour anatomical assignment.

    Manifest: image_id,patient_id,split,label,contour_id. The reviewer identifies
    the anatomical candidate appropriate to the experiment before export.
    """
    documents = {}
    for path in feature_files:
        doc = json.loads(Path(path).read_text())
        if doc["image_id"] in documents:
            raise ValueError("Duplicate feature file image ID")
        documents[doc["image_id"]] = doc
    with Path(reviewed_manifest).open(newline="") as handle:
        reader = csv.DictReader(handle)
        manifest_fields = reader.fieldnames or []
        rows = list(reader)
    confounders = [name for name in CONFOUNDER_FIELDS if name in manifest_fields]
    result = []
    for row in rows:
        if (
            not row.get("patient_id")
            or row.get("split") not in {"train", "val", "test"}
            or row.get("label") not in {"0", "1"}
        ):
            raise ValueError("Verified patient ID, split and binary label required")
        if row["image_id"] not in documents:
            raise ValueError(f"No feature file for {row['image_id']}")
        candidates = documents[row["image_id"]]["records"]
        selected = [c for c in candidates if str(c["contour_id"]) == row["contour_id"]]
        if len(selected) != 1:
            raise ValueError("Select exactly one valid contour per image")
        record = selected[0]
        contour_type = record.get("contour_type", "")
        if contour_type not in ANATOMICAL_CONTOURS:
            raise ValueError(
                "Contour type must be an anatomical candidate or verified anatomy"
            )
        result.append(
            {
                **{k: row[k] for k in ("image_id", "patient_id", "split", "label")},
                "contour_type": contour_type,
                **{k: row[k] for k in confounders},
                **{k: record["features"][k] for k in GEOMETRY_FIELDS},
            }
        )
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    # Validate leakage before publishing the table.
    seen = {}
    for row in result:
        for key in ("patient_id", "image_id"):
            token = (key, row[key])
            if token in seen and (key == "image_id" or seen[token] != row["split"]):
                raise ValueError("Duplicate/leaking manifest")
            seen[token] = row["split"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        fieldnames = [
            "image_id",
            "patient_id",
            "split",
            "label",
            "contour_type",
            *confounders,
            *GEOMETRY_FIELDS,
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result)
