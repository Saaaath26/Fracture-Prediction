"""Contour extraction document shared by the CLI and dashboard."""

import hashlib
from . import __version__
from .annotations import fracture_mask
from .contours import extract_contours
from .geometry import features, fracture_relationship
from .preprocessing import aligned_gray
from .serialization import feature_record

PROJECTION_CAVEAT = (
    "Measurements are radiographic projections in the annotation frame. "
    "Position, view, scale and acquisition can explain apparent geometric differences. "
    "Pixel quantities are not millimeters."
)


def extract_document(
    dataset, image_id, contour_type="bone_candidate", min_area_fraction=0.005
):
    if image_id not in dataset.images:
        raise ValueError(f"Unknown image id: {image_id}")
    path = dataset.images[image_id]
    image, alignment = aligned_gray(path, dataset.coco_size.get(image_id))
    contours = extract_contours(image, contour_type, min_area_fraction)
    mask = fracture_mask(
        dataset.annotations.get(image_id, []), image.shape[1], image.shape[0]
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records = []
    for index, contour in enumerate(contours):
        if not contour.valid:
            continue
        record = feature_record(
            image_id,
            features(contour, image_shape=image.shape),
            dict(
                method=contour.method,
                image_sha256=digest,
                source=str(path),
                uncertainty=contour.uncertainty,
                warnings=contour.warnings,
                parameters={"min_area_fraction": min_area_fraction},
                software_version=__version__,
                alignment=alignment,
            ),
            contour.contour_type,
        )
        record.update(
            contour_id=index,
            points=contour.points,
            quality=contour.quality,
            fracture_relationship=fracture_relationship(contour, mask),
            fracture_relationship_role="descriptive_ground_truth_only",
            usable_as_fracture_presence_feature=False,
            relationship_source="ground_truth_fracture_annotation; not a classifier input",
        )
        records.append(record)
    document = dict(
        schema_version="1.1",
        record_type="contour_extraction",
        image_id=image_id,
        modality="radiograph",
        dataset_root=str(dataset.root),
        image_sha256=digest,
        alignment=alignment,
        coordinate_convention="xy_pixels_origin_top_left_annotation_frame",
        image_shape=list(image.shape),
        measurement_frame="radiographic_projection_pixels",
        physical_calibration=False,
        projection_caveat=PROJECTION_CAVEAT,
        contour_types_are_not=[
            "fracture_region_mask",
            "verified_bone_segmentation",
            "external_rgb_limb_silhouette",
        ],
        software_version=__version__,
        clinically_validated=False,
        records=records,
        rejected=[
            dict(
                contour_id=index,
                contour_type=contour.contour_type,
                warnings=contour.warnings,
                quality=contour.quality,
            )
            for index, contour in enumerate(contours)
            if not contour.valid
        ],
    )
    return document, image, contours
