# FracAtlas architecture

This diagram matches the package after the 2026-10-28 audit. A box is included only if the repository contains the corresponding code or data. Untrained models and missing annotations are labeled as such.

- [FRACATLAS_ARCHITECTURE.mmd](FRACATLAS_ARCHITECTURE.mmd)
- [FRACATLAS_ARCHITECTURE.svg](FRACATLAS_ARCHITECTURE.svg)

## Branches

1. **Original data.** `FractAtlas Dataset/FracAtlas` is read only. The original repository contributes notebooks, split CSVs, and two YOLOv8 checkpoints. Those checkpoints are not loaded by the new package.
2. **Integrity.** `Dataset.audit` and `fracatlas integrity` record truncated JPEGs, the one EXIF correction, exact-pixel duplicates, and dHash candidates. The derived exclusion manifest is an explicit input to later preparation. It does not invent patient ids.
3. **YOLO11 and YOLO26.** Detection and instance segmentation are separate configs and separate label files. Both are implemented and untrained. Ultralytics is not installed here, so the backend is blocked.
4. **Contours.** Classical extraction produces bone candidates or soft-tissue candidates. Verified U-Net and nnU-Net paths exist and refuse fracture masks. No verified anatomical masks are in the dataset, so those trainers stay blocked.
5. **Geometry and relationships.** Pixel and scale-free measurements, plus a descriptive ground-truth fracture relationship that is not a classifier input. Logistic regression and XGBoost are guarded and were not fit.
6. **Dashboard and exports.** One contour document feeds the viewer and the CSV export. The viewer shows YOLO11 and YOLO26 as untrained unless a future run checkpoint appears.
7. **RGB boundary.** Schema fields reserve an external limb silhouette. No paired photographs and no transfer model exist.

## Dependency map

| Module | Calls or consumes |
| --- | --- |
| `cli` | `config`, `data`, `integrity`, `yolo`, `extraction`, `exports`, `analysis`, `neural_training`, `neural_inference` |
| `config` | `devices` |
| `data` | `annotations`, `preprocessing`, `schemas`, `serialization` |
| `yolo` | `devices`, `guards`, `evaluation`, `serialization` |
| `extraction` | `annotations`, `contours`, `geometry`, `preprocessing`, `serialization` |
| `contours` | `preprocessing`, `schemas` |
| `geometry` | contour objects |
| `analysis` | `guards`, `evaluation`, `serialization` |
| `exports` | `schemas` |
| `integrity` | `serialization`, dataset objects |
| `neural_training` | `guards`, `serialization`; imports `neural` only inside an authorized run |
| `neural_inference` | `evaluation`, `serialization`; imports `neural` only when a checkpoint is loaded |
| `dashboard` | `data`, `extraction`, `visualization` when the app is launched |
| `evaluation` | `serialization` |

`analysis` does not import `yolo`. `neural.py` is not imported at package import. The notebooks do not import `fracatlas`. No cycle was found.

## Status key

| Mark | Meaning in this repository |
| --- | --- |
| Implemented | Runs without training: audit, convert, contours, geometry, exports, dashboard |
| Configured, untrained | YOLO11/26, logistic regression, XGBoost, U-Net, Siamese |
| Blocked on data | Verified anatomy masks, patient ids, Siamese pair labels |
| Blocked on package | Ultralytics, XGBoost, nnU-Net |
| Not implemented | Paired RGB or video limb assessment |
