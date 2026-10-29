"""Derived integrity manifests. Original images and annotations are never modified."""

import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
from .serialization import write_json


def annotation_signature(annotations):
    signature = []
    for ann in annotations:
        bbox = tuple(round(float(value), 4) for value in ann.get("bbox", []))
        segmentation = ann.get("segmentation")
        if isinstance(segmentation, list):
            polygons = tuple(
                tuple(round(float(value), 4) for value in polygon)
                for polygon in segmentation
            )
        else:
            polygons = ("nonpolygon",)
        signature.append((bbox, polygons))
    return tuple(sorted(signature))


def propose_exclusions(dataset, audit):
    """Quarantine unusable files and cross-split exact duplicates.

    Identical annotations keep the lexicographically first image in its official
    split. Disagreeing annotations quarantine the whole group. Nothing here
    invents a patient identifier.
    """

    rows = []
    seen = set()

    def add(image_id, reason, action, group_id):
        if image_id in seen:
            return
        seen.add(image_id)
        rows.append(
            dict(image_id=image_id, reason=reason, action=action, group_id=group_id)
        )

    for image_id in audit.get("unusable_unassigned", []):
        add(
            image_id,
            "strict_jpeg_decode_failed_partial_recovery_rejected",
            "exclude_unusable",
            "",
        )
    for message in list(audit.get("warnings", [])) + list(audit.get("errors", [])):
        if "COCO dimension mismatch" in message:
            if message.startswith("COCO dimension mismatch:"):
                image_id = message.split(":", 1)[1].strip().split()[0]
            else:
                image_id = message.split(":", 1)[0].strip()
            add(
                image_id,
                "annotation_dimensions_not_unambiguously_recoverable",
                "exclude_pending_review",
                "",
            )
        elif "truncated" in message or "broken data stream" in message:
            image_id = message.split(":", 1)[0].strip()
            if image_id in dataset.images:
                add(
                    image_id,
                    "strict_jpeg_decode_failed_partial_recovery_rejected",
                    "exclude_unusable",
                    "",
                )
    for group in audit.get("cross_split_duplicates", []):
        names = tuple(sorted(group))
        group_id = "|".join(names)
        conflict = (
            len(
                {
                    annotation_signature(dataset.annotations.get(name, []))
                    for name in names
                }
            )
            > 1
        )
        if conflict:
            for name in names:
                add(
                    name,
                    "exact_decoded_pixel_duplicate_cross_split_annotation_conflict",
                    "exclude_pending_review",
                    group_id,
                )
        else:
            for name in names[1:]:
                add(
                    name,
                    f"exact_decoded_pixel_duplicate_of_{names[0]}",
                    "exclude_duplicate",
                    group_id,
                )
    membership = {}
    for split, image_ids in dataset.splits.items():
        for image_id in image_ids:
            membership.setdefault(image_id, set()).add(split)
    for group in audit.get("pixel_duplicates", []):
        names = tuple(sorted(group))
        present = [name for name in names if name in membership]
        if len(present) < 2:
            continue
        splits = {split for name in present for split in membership[name]}
        if len(splits) != 1:
            continue
        conflict = (
            len(
                {
                    annotation_signature(dataset.annotations.get(name, []))
                    for name in present
                }
            )
            > 1
        )
        if not conflict:
            continue
        group_id = "|".join(names)
        for name in present:
            add(
                name,
                "exact_decoded_pixel_duplicate_within_split_annotation_conflict",
                "exclude_pending_review",
                group_id,
            )
    return sorted(rows, key=lambda row: row["image_id"])


def write_exclusion_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "reason", "action", "group_id"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _relative(dataset, image_id):
    return str(dataset.images[image_id].relative_to(dataset.root))


def _pixel_stats(path):
    with Image.open(path) as image:
        image.load()
        gray = np.asarray(image.convert("L"))
        return dict(
            stored_size=[image.size[0], image.size[1]],
            byte_size=path.stat().st_size,
            pixel_sha256=hashlib.sha256(gray.tobytes()).hexdigest(),
        )


def _partial_decode(path):
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as image:
            image.load()
            return dict(
                possible=True, size=[image.size[0], image.size[1]], mode=image.mode
            )
    except (OSError, ValueError) as exc:
        return dict(possible=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def write_integrity_reports(dataset, destination):
    """Audit the dataset and write versioned derived manifests. Does not prepare or train."""

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use a new integrity output directory")
    destination.mkdir(parents=True, exist_ok=True)
    audit = dataset.audit()
    exclusions = propose_exclusions(dataset, audit)
    write_exclusion_manifest(destination / "exclusion_manifest.csv", exclusions)
    excluded = {row["image_id"] for row in exclusions}
    derived_rows = []
    for split, image_ids in dataset.splits.items():
        for image_id in image_ids:
            if image_id in excluded:
                continue
            derived_rows.append(
                dict(
                    image_id=image_id,
                    split=split,
                    patient_id_status="unresolved_no_patient_identifier",
                )
            )
    derived_path = destination / "derived_split_manifest.csv"
    with derived_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "split", "patient_id_status"]
        )
        writer.writeheader()
        writer.writerows(derived_rows)

    truncated = []
    for image_id in audit["unusable_unassigned"]:
        path = dataset.images[image_id]
        strict = next(
            (
                message
                for message in audit["warnings"]
                if message.startswith(f"{image_id}:")
            ),
            "strict decode failed",
        )
        truncated.append(
            dict(
                image_id=image_id,
                relative_path=_relative(dataset, image_id),
                strict_decode="failed",
                strict_error=strict,
                partial_decode_with_truncated_loader=_partial_decode(path),
                accepted_for_research=False,
                reason="Partial JPEG recovery would conceal a corrupt sample",
                in_official_split=False,
                fractured=dataset.metadata[image_id].get("fractured"),
                original_modified=False,
            )
        )

    duplicate_groups = []
    reported = set(tuple(sorted(group)) for group in audit["cross_split_duplicates"])
    membership = {}
    for split, image_ids in dataset.splits.items():
        for image_id in image_ids:
            membership.setdefault(image_id, []).append(split)
    within_conflicts = []
    for group in audit["pixel_duplicates"]:
        names = tuple(sorted(group))
        if names in reported:
            continue
        present = [name for name in names if name in membership]
        if (
            len(present) < 2
            or len({s for name in present for s in membership[name]}) != 1
        ):
            continue
        signatures = {
            annotation_signature(dataset.annotations.get(name, [])) for name in present
        }
        if len(signatures) > 1:
            within_conflicts.append(names)
    for group in list(audit["cross_split_duplicates"]) + within_conflicts:
        members = []
        for image_id in sorted(group):
            info = _pixel_stats(dataset.images[image_id])
            info.update(
                image_id=image_id,
                relative_path=_relative(dataset, image_id),
                splits=[
                    split for split, ids in dataset.splits.items() if image_id in ids
                ],
                fracture_count=dataset.metadata[image_id].get("fracture_count"),
                annotation_signature=annotation_signature(
                    dataset.annotations.get(image_id, [])
                ),
            )
            members.append(info)
        hashes = {member["pixel_sha256"] for member in members}
        signatures = {member["annotation_signature"] for member in members}
        duplicate_groups.append(
            dict(
                members=[
                    {k: v for k, v in member.items() if k != "annotation_signature"}
                    for member in members
                ],
                decoded_pixels_identical=len(hashes) == 1,
                byte_identical=len({member["byte_size"] for member in members}) == 1
                and len({dataset.images[m["image_id"]].read_bytes() for m in members})
                == 1,
                annotation_conflict=len(signatures) > 1,
                patient_independence="not_established",
                leakage=(
                    "cross_split_exact_decoded_pixels"
                    if tuple(sorted(group)) in reported
                    else "within_split_exact_decoded_pixels"
                ),
            )
        )

    inventory = dict(
        integrity_policy_version=1,
        dataset_root=str(dataset.root),
        original_files_policy="read_only_not_modified",
        images=audit["images"],
        metadata_rows=audit["metadata_rows"],
        official_splits=audit["splits"],
        derived_split_counts={
            split: sum(row["split"] == split for row in derived_rows)
            for split in ("train", "val", "test")
        },
        exclusions=len(exclusions),
        blocking_errors=audit["errors"],
        warning_count=len(audit["warnings"]),
        unusable_unassigned=len(audit["unusable_unassigned"]),
        orientation_corrections=audit["orientation_corrections"],
        exif_orientation_not_applied_count=len(audit["exif_orientation_not_applied"]),
        cross_split_duplicate_groups=len(audit["cross_split_duplicates"]),
        pixel_duplicate_groups=len(audit["pixel_duplicates"]),
        byte_duplicate_groups=len(audit["byte_duplicates"]),
        near_duplicate_groups=len(audit["near_duplicate_candidates"]),
        patient_leakage=audit["patient_leakage"],
        anatomical_annotations=audit["anatomical_annotations"],
        patient_identifiers="absent; derived split does not claim patient independence",
        training_executed=False,
    )
    write_json(destination / "dataset_inventory.json", inventory)
    write_json(destination / "truncated_jpegs.json", truncated)
    write_json(destination / "duplicate_groups.json", duplicate_groups)
    write_json(
        destination / "orientation_review.json",
        dict(
            applied_because_unique_size_match=audit["orientation_corrections"],
            not_applied=audit["exif_orientation_not_applied"],
            rule=(
                "Apply EXIF orientation only when stored size differs from the "
                "annotation size and the oriented size matches it exactly. "
                "Other orientation tags stay on stored pixels so annotation "
                "coordinates are not rotated away from the labeled frame."
            ),
        ),
    )
    write_json(
        destination / "near_duplicate_candidates.json",
        dict(
            policy=audit["near_duplicate_policy"],
            groups=audit["near_duplicate_candidates"],
            automatically_excluded=False,
        ),
    )
    write_json(
        destination / "audit_summary.json",
        {
            key: value
            for key, value in audit.items()
            if key != "near_duplicate_candidates"
        },
    )
    manifest_hash = hashlib.sha256(
        (destination / "exclusion_manifest.csv").read_bytes()
    ).hexdigest()
    write_json(
        destination / "manifest_checksums.json",
        dict(
            exclusion_manifest_sha256=manifest_hash,
            derived_split_manifest_sha256=hashlib.sha256(
                derived_path.read_bytes()
            ).hexdigest(),
            note="These checksums cover derived manifests only, not original dataset files.",
        ),
    )
    return inventory
