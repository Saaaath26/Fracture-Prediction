# Future commands — NOT EXECUTED

**Every training command below is NOT EXECUTED. Run only after explicit authorization, dependency setup and resolving dataset blockers.** Commands assume an installed package and a shell in the repository root.

The original configs still fail preparation because exact decoded-pixel duplicates cross the official splits. A derived cohort is available after the integrity audit:

```bash
PYTHONPATH=src python3 -m fracatlas.cli integrity --config configs/yolo11_detect.yaml --output reports/integrity_rerun
fracatlas prepare --config configs/derived/yolo11_detect.yaml
fracatlas prepare --config configs/derived/yolo11_segment.yaml
```

`reports/integrity/` already contains one generated manifest. Do not point a second integrity run at that existing directory. The derived manifest removes 88 images: 59 truncated non-fracture JPEGs, 16 cross-split exact duplicates, and 13 within-split exact duplicates whose fracture boxes disagree. The resulting counts are 559 train, 68 validation and 63 test. Patient identifiers are still absent, so YOLO training on that cohort still requires `--acknowledge-unresolved-leakage`. Preparation copies images; it was not run during the audit, to avoid duplicating the cohort on disk. Set `device` to `cpu`, `mps`, `cuda`, `cuda:0`, or `auto` deliberately. `auto` only checks which devices exist. It does not benchmark them. Batch size 8 at 640 pixels is an unbenchmarked starting point for a 16 GB GPU.

## YOLO11 and YOLO26

First configure a reviewed, valid cohort or corrected dataset copy. Prepare each task once (safe data conversion; not training):

```bash
fracatlas prepare --config configs/yolo11_detect.yaml
fracatlas prepare --config configs/yolo11_segment.yaml
```

The YOLO26 defaults share those prepared task directories. If configurations use different cohorts, use separate preparation directories.

**Training — NOT EXECUTED:**

```bash
fracatlas train-yolo --config configs/yolo11_detect.yaml --authorize-training
fracatlas train-yolo --config configs/yolo11_segment.yaml --authorize-training
fracatlas train-yolo --config configs/yolo26_detect.yaml --authorize-training
fracatlas train-yolo --config configs/yolo26_segment.yaml --authorize-training
```

For an expressly authorized reproduction with unavailable patient IDs, append `--acknowledge-unresolved-leakage`. This flag does not bypass any image, annotation or exact-duplicate error and does not make an experiment leakage-free. Training defaults to CPU, 100 epochs and seed 42. Change configuration deliberately for the future host. Outputs must be new directories. Ultralytics may download the official checkpoint when a future authorized training run constructs the model.

Future inference and evaluation using local trained weights (also not executed):

```bash
fracatlas predict --config configs/yolo11_detect.yaml --checkpoint runs/yolo11_detect/train/weights/best.pt --source "../FractAtlas Dataset/FracAtlas/images/Fractured/IMG0000019.jpg" --output reports/prediction.json
fracatlas evaluate-yolo --config configs/yolo11_detect.yaml --checkpoint runs/yolo11_detect/train/weights/best.pt --split test --output reports/yolo11_test.json
```

The prediction/evaluation adapter also accepts the preserved author weights for baseline comparisons, with a matching detect/segment config. No existing weight was loaded during delivery.

## Geometry classifiers

Create real feature files and `reviewed/contours.csv` with reviewed patient IDs and contour selections:

```bash
fracatlas export-table --features reports/features/*.json --manifest reviewed/contours.csv --output reviewed/features.csv
```

**Fitting — NOT EXECUTED:**

```bash
fracatlas train-relationship --table reviewed/features.csv --features area_px2 perimeter_px circularity solidity aspect_ratio --model logistic --output runs/logistic --authorize-training
fracatlas train-relationship --table reviewed/features.csv --features area_px2 perimeter_px circularity solidity aspect_ratio --model xgboost --output runs/xgboost --authorize-training
```

Do not combine bone/soft-tissue candidates or anatomical sites without a prespecified design. Pixel-scale features are resolution-sensitive; consider normalized measures or verified spacing before interpreting them. Future held-out evaluation:

```bash
fracatlas evaluate-relationship --table reviewed/features.csv --model-dir runs/logistic --output reports/logistic_test.json
fracatlas evaluate-relationship --table reviewed/features.csv --model-dir runs/xgboost --output reports/xgboost_test.json
```

## U-Net and Siamese CNN

Provide real verified annotation manifests following the header-only templates. Paths in manifests resolve relative to their CSV. Anatomical masks must be binary, same size as source images, and marked `verified_bone` or `verified_soft_tissue`; `verified` must be `true`. Pair manifests require both `patient_id_a` and `patient_id_b`, independently verified binary labels and annotation-source provenance.

**Training — NOT EXECUTED:**

```bash
fracatlas train-neural --manifest reviewed/anatomy.csv --kind unet --output runs/unet --epochs 100 --batch 4 --size 256 --authorize-training
fracatlas train-neural --manifest reviewed/pairs.csv --kind siamese --output runs/siamese --epochs 100 --batch 4 --size 256 --authorize-training
```

Future evaluation:

```bash
fracatlas evaluate-neural --checkpoint runs/unet/best.pt --manifest reviewed/anatomy.csv --split test --output reports/unet_test.json
fracatlas evaluate-neural --checkpoint runs/siamese/best.pt --manifest reviewed/pairs.csv --split test --output reports/siamese_test.json
```

## nnU-Net

These preparation/planning/training commands are **all NOT EXECUTED**. Obtain verified anatomy masks first and install the separate nnU-Net dependency.

```bash
export nnUNet_raw="$PWD/artifacts/nnunet_raw"
export nnUNet_preprocessed="$PWD/artifacts/nnunet_preprocessed"
export nnUNet_results="$PWD/runs/nnunet"
fracatlas prepare-nnunet --manifest reviewed/anatomy.csv --output "$nnUNet_raw/Dataset501_FracAtlasBone"
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
cp "$nnUNet_raw/Dataset501_FracAtlasBone/splits_final.json" "$nnUNet_preprocessed/Dataset501_FracAtlasBone/splits_final.json"
nnUNetv2_train 501 2d 0
```

Copying `splits_final.json` into the preprocessed dataset preserves the reviewed split rather than allowing automatic folds. Use a distinct dataset ID and separate model for soft tissue. The final command invokes external training directly; it is not covered by the Python CLI acknowledgement guard and must only be run after explicit authorization.
