# Integration review

The executable path is a single Python package, `fracatlas`, with one CLI. Original notebooks and YOLOv8 checkpoints are not imported by that package.

## Contracts

| Stage | Producer | Consumer | Identity preserved |
| --- | --- | --- | --- |
| Inventory | `Dataset` | audit, prepare, extract, dashboard | `image_id` is the filename. It is not a patient id |
| Fracture boxes and polygons | COCO, converted by `annotations.py` | YOLO labels, fracture mask, dashboard overlay | Category 1 only. RLE and multipart instance lines are rejected. Masks may union several polygons |
| Annotation frame | `interpret_image` | prepare, extract, dashboard | Stored pixels, unless EXIF orientation is the unique match to the annotation size |
| YOLO prediction | `prediction_record` | dashboard overlay | `image_id`, width, height, task, checkpoint hash. Coordinates are documented as original-image pixels. That Ultralytics mapping was not executed |
| Contour candidate | `extract_document` | geometry JSON, dashboard, `export-table` | Contour type, quality, warnings, source hash, alignment |
| Descriptive fracture relationship | `fracture_relationship` | JSON only | Marked `usable_as_fracture_presence_feature: false` |
| Classifier table | `export_table` | `read_table` | Reviewed contour id, patient id, split, label, geometry fields, optional confounders |
| Neural masks | verified manifest | U-Net or nnU-Net export | `verified=true`, anatomical type, patient-disjoint splits |

`extraction.py` is the shared contour document. The CLI and the dashboard both call it, so the overlay and the downloaded JSON use the same alignment and the same contour ids.

## What does not connect

- Fracture polygons are not accepted as U-Net or nnU-Net labels.
- `bone_candidate` and `soft_tissue_candidate` cannot be selected under the other name. Verified bone and verified soft tissue cannot be mixed in one manifest.
- External RGB limb silhouettes are named in the schema and are not produced.
- Relationship training refuses feature names that contain fracture, target, label, overlap, or distance-to-boundary tokens. Fit receives training rows only. Test rows remain in the CSV so leakage across patients can be checked.
- YOLO evaluation refuses a missing or empty prepared split before importing Ultralytics. A missing checkpoint is rejected before that import as well.

## Backend status

| Backend | Code | This environment |
| --- | --- | --- |
| Ultralytics YOLO11 / YOLO26 | Config names, prepare, guarded train, predict, val | Not installed. Integration blocked |
| scikit-learn logistic regression | Guarded fit | Installed. Not fitted |
| XGBoost | Guarded fit | Not installed. Not fitted |
| PyTorch U-Net / Siamese | Definitions and guarded CPU trainer | Definitions import torch only when training or inference is called. Not run |
| nnU-Net v2 | PNG export and dataset JSON | Package not installed. No planning or training |
| Streamlit | Dashboard | Helpers import without starting the app |

Prepared YOLO folders record a filename map. An EXIF-aligned image is saved as a PNG whose stem matches the original id. The source JPEG is copied only when its stored size already matches the annotation. Provenance hashes cover the prepared files and are checked again before a future training call.

## Dependency direction

`cli` dispatches to `config`, `data`, `integrity`, `yolo`, `extraction`, `exports`, `analysis`, and the neural modules. `analysis` no longer imports `yolo`. `guards.py` is the shared training latch. `geometry` depends on contour objects. `contours` depends on preprocessing. Nothing in that chain imports the CLI. `neural.py` is imported only inside training and inference functions.

There is no import cycle among these modules. The original notebooks are a separate, untouched baseline. They are not on this path.
