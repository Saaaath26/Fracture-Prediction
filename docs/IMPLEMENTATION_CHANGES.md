# Implementation changes

Package version is 0.1.1. Schema version for new contour and prediction documents is 1.1. Original dataset files, notebooks, and YOLOv8 checkpoints were not modified. SHA-256 of the 15 baseline paths still matches `reports/baseline_checksums.json`.

## Behavior

- Unassigned JPEG decode failures no longer block `prepare`. In-split corruption still does.
- EXIF orientation is applied only when the oriented size is the unique match to the annotation size. The source file is not rewritten. The derived file is a PNG.
- `fracatlas integrity` writes exclusion, derived-split, duplicate, truncation, and orientation reports. The generated directory is `reports/integrity/`.
- Official YOLO configs do not reference that manifest, so they still stop on cross-split duplicates.
- `configs/derived/` references the manifest. A re-audit with it returns no blocking errors and the counts 559 / 68 / 63. Patient leakage stays unresolved.
- Contour records carry abstention, representation flags, and a projection caveat. Geometry adds normalized centroids and `turning_angle_std_rad`.
- Classifier export adds `contour_type` and optional anatomy/view columns. It still omits fracture-overlap features.
- Prediction records name the checkpoint and state that the Ultralytics coordinate mapping was not verified in this environment.
- The dashboard shows YOLO11 and YOLO26 as untrained when no run checkpoint contains those names. It does not invent predictions.
- Training functions still raise `PermissionError` without `--authorize-training`, and they do so before fitting.

## Files

New modules: `guards.py`, `devices.py`, `schemas.py`, `extraction.py`, `integrity.py`.

Updated modules: `data.py`, `preprocessing.py`, `annotations.py`, `config.py`, `contours.py`, `geometry.py`, `serialization.py`, `exports.py`, `analysis.py`, `yolo.py`, `cli.py`, `dashboard.py`, `visualization.py`, `neural_training.py`, `neural_inference.py`.

New tests: `tests/conftest.py`, `tests/test_contracts.py`, plus geometry cases in `tests/test_core.py`.

Generated data: `reports/integrity/`. Derived configs: `configs/derived/`.

## Verification

- Pre-change: `python3 -m pytest -q` — 28 passed.
- Post-change: `python3 -m pytest -q` — 44 passed. `pyflakes src tests` reported no issues. `black` was applied to the edited Python.
- Real integrity report regenerated from the dataset. Blocking finding on the official split: cross-split decoded-pixel duplicates.
- Second audit using `reports/integrity/exclusion_manifest.csv`: `errors == []`, splits 559 / 68 / 63, 59 truncation warnings, patient leakage unresolved, one EXIF correction.
- Full `prepare` of the derived cohort was not run. It would copy the radiographs into `artifacts/`.
- Ultralytics was not imported. No checkpoint was downloaded. No model was constructed for training and no `.fit` or `model.train` call was executed.

## Not claimed

Passing these tests does not mean YOLO11, YOLO26, U-Net, or a relationship model is operational. It means the guards, schemas, geometry, and dataset gates behave as specified on fixtures and on the audited cohort.
