# Feature export schema 1.1

Schema 1.1 adds representation flags, alignment policy, contour quality, normalized centroids and `turning_angle_std_rad`. The classifier CSV still exports the original pixel geometry columns, plus `contour_type` and any manifest columns for hand, leg, hip, shoulder, mixed, hardware, multiscan, frontal, lateral or oblique. Ground-truth fracture overlap and boundary distance stay out of that CSV.

# Feature export schema 1.0

An export document has `schema_version`, `image_id`, `image_shape` (height, width), `records`, and `rejected`. No accepted contour yields an empty records list, not zero-valued measurements.

Each record includes:

| Field | Meaning |
| --- | --- |
| `contour_id` | Index in deterministic area-sorted extraction output |
| `contour_type` | `bone_candidate`, `soft_tissue_candidate`, or explicit verified anatomical type |
| `points` | Original image `[x,y]` pixel coordinates; x rightward, y downward |
| `features` | Geometry fields, with unit suffixes |
| `provenance` | SHA-256 of source image, method, parameters, software version, warnings, uncertainty |
| `fracture_relationship` | Ground-truth fracture centroid signed boundary distance and overlap fraction; kept outside classifier features |
| `relationship_source` | Explicit ground-truth source, unsuitable as a fracture-presence predictor |
| `modality` | `radiograph` |
| `clinically_validated` | `false` |
| `rgb_transfer.status` | `experimental_unvalidated` |

Distances/axes/perimeters are in pixels and areas in square pixels unless physical spacing was explicitly supplied to the Python geometry API. Positive signed boundary distance means inside the anatomical candidate. Orientation is the major axis of the minimum-area rectangle modulo 180 degrees, not a clinical anatomical axis. Square/symmetric shapes have ambiguous orientation. Uncertainty is qualitative and uncalibrated, never a fabricated confidence score. JSON rejects NaN/Infinity.

For video integration, retain image identity, coordinate frame, contour type and provenance. Do not reinterpret radiographic measurements as RGB/video measurements without a separately validated mapping. No such mapping is implemented or validated here.
