"""Shared names for records that must not be treated as interchangeable."""

GEOMETRY_EXPORT_FIELDS = (
    "area_px2",
    "perimeter_px",
    "circularity",
    "solidity",
    "aspect_ratio",
    "major_axis_px",
    "minor_axis_px",
)

SCALE_FREE_FIELDS = ("circularity", "solidity", "aspect_ratio", "turning_angle_std_rad")

CONFOUNDER_FIELDS = (
    "hand",
    "leg",
    "hip",
    "shoulder",
    "mixed",
    "hardware",
    "multiscan",
    "frontal",
    "lateral",
    "oblique",
)

BONE_CANDIDATE = "bone_candidate"
SOFT_TISSUE_CANDIDATE = "soft_tissue_candidate"
VERIFIED_BONE = "verified_bone"
VERIFIED_SOFT_TISSUE = "verified_soft_tissue"
FRACTURE_REGION = "fracture_region"
EXTERNAL_RGB_LIMB = "external_rgb_limb_silhouette"

CANDIDATE_CONTOURS = {BONE_CANDIDATE, SOFT_TISSUE_CANDIDATE}
ANATOMICAL_CONTOURS = CANDIDATE_CONTOURS | {VERIFIED_BONE, VERIFIED_SOFT_TISSUE}
# Fracture polygons and external limb photos are different representations.
NOT_ANATOMICAL_CONTOURS = {FRACTURE_REGION, EXTERNAL_RGB_LIMB, "fracture_mask"}

EXCLUSION_ACTIONS = {
    "exclude_unusable",
    "exclude_pending_review",
    "exclude_duplicate",
}
