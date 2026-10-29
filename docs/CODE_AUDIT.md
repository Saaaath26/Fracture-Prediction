# FracAtlas code audit

Audit date 2026-10-28. The pre-change suite was 28 passing tests. No training or inference was run before or during this audit. All 15 original-repository checksums in `reports/baseline_checksums.json` still match.

## Method

Every module under `src/fracatlas/` was read against `RESEARCH.md`, `docs/IMPLEMENTATION_REPORT.md` and the saved dataset audit. Flagged JPEGs, `IMG0002628.jpg`, and the eight cross-split duplicate groups were opened individually. Ultralytics, XGBoost and nnU-Net are not installed, so those backends are blocked rather than marked passed.

## Issue inventory

| ID | Severity | Location | Impact | Remedy | Result |
| --- | --- | --- | --- | --- | --- |
| A1 | P0 | `data.py` audit gate | 59 truncated non-fracture JPEGs, none of them in a split, blocked preparation of the fractured cohort | Unassigned decode failures are warnings. Strict Pillow loading stays on. Partial recovery is recorded and rejected | Fixed and tested |
| A2 | P0 | `IMG0002628.jpg`, COCO and VOC | Stored size is 373×454 with EXIF orientation 8. Annotation size is 454×373. Training on stored pixels would use the wrong frame | Apply EXIF only when it is the unique size match. Other orientation tags are left alone | Fixed. One image. Original file untouched |
| A3 | P0 | Eight cross-split groups | Same decoded pixels in train and validation, different boxes, different file bytes | Quarantine every member. Do not choose a label | Fixed. Official config still fails until the manifest is used |
| A4 | P0 | Six within-split groups | Same pixels repeated inside train or validation with disagreeing boxes | Same quarantine rule | Fixed. 13 additional images removed from the derived split |
| B1 | P1 | Contour pipeline | Candidates could be mistaken for fracture masks, verified bone, or an external limb silhouette | Explicit type checks, abstention flags, and export fields | Fixed |
| B2 | P1 | `cli.py` extract and dashboard | Two copies of the contour document, and a missing image id raised `KeyError` | Shared `extraction.py`. Unknown ids exit with a clear error | Fixed |
| B3 | P1 | Prediction JSON | No statement of coordinate frame, checkpoint, or that the mapping was not checked against Ultralytics | Schema 1.1 prediction record | Fixed. Live Ultralytics mapping remains unverified |
| B4 | P1 | Dashboard | No untrained-model status. Importing the module started Streamlit | Status panel. Streamlit stays inside `main()` | Fixed |
| B5 | P1 | Relationship export | Anatomy and view columns in `dataset.csv` had no path into the analysis table | Optional confounder columns. Fracture-derived fields stay forbidden | Fixed |
| B6 | P1 | `analysis.py` | Relationship code imported the YOLO module only to reach the training guard | `guards.py` | Fixed |
| C1 | P2 | Audit I/O | Each JPEG was read twice | Hash and decode share one byte buffer | Fixed |
| C2 | P2 | Prepared images | A hard link would let a later write change the original dataset | Stored pixels are copied. EXIF-aligned frames are written as new PNGs | Fixed by refusing links |
| C3 | P2 | Dashboard | Re-decoded the selected radiograph on every rerun | Session cache for the dataset index and the selected image | Fixed |
| C4 | P2 | Tests | Dashboard status, EXIF alignment, exclusions, and schema contracts were untested | 44 tests, up from 28 | Fixed |
| D1 | P3 | Neural trainer | GPU execution cannot be verified without a training run | YOLO accepts `cpu`, `mps`, `cuda`, or `auto`. The reference U-Net/Siamese loop stays CPU until an authorized run | Left blocked |
| D2 | P3 | 117 dHash groups | Possible near duplicates, not proof of the same exposure | Reported, not removed | Open |
| D3 | P3 | Three exact-duplicate groups with only one member in a split | Not current leakage. A later split edit could make it leakage | Reported in the audit summary, not auto-removed | Open |
| D4 | P3 | Patient ids, verified anatomy masks, paired RGB | Required for the stated study and absent from FracAtlas | Interfaces reject unverified masks. No synthetic labels were created | Open |

## Dataset findings that were checked directly

Truncated files: 59, all under `images/Non_fractured/`, fractured flag 0, in no official split. Strict decode fails. Pillow's truncated-image loader can return a partial raster for all 59. None of those partial rasters were accepted or written back.

`IMG0002628.jpg`: stored 373×454, EXIF orientation 8, oriented size 454×373, matching COCO and VOC. The fracture box also fits the stored raster, so a transpose is not justified by the box alone. It is justified because the oriented size is the only size that matches the annotation record. Twenty-four other images carry a non-default EXIF tag whose stored size already matches the annotation, so those tags are not applied.

Cross-split groups: 8. Decoded pixels match. File bytes do not. Fracture boxes disagree in every group. Within-split conflicting groups: 6, covering 13 images. Exact pixel-duplicate groups overall: 78. Of those, 61 are entirely outside the official split and 3 have only one member in a split.

## Gate after the manifest

`reports/integrity/exclusion_manifest.csv` lists 88 images. Re-auditing with that manifest returns no blocking errors and counts 559 / 68 / 63. Patient leakage remains `unresolved: no patient identifiers`. The 59 truncated files remain warnings. `prepare` was not executed on this cohort.

## Modules that were not placeholders

Training entry points exist and raise `PermissionError` unless `--authorize-training` is passed. That was true before this audit and is still tested. U-Net, Siamese, logistic regression, XGBoost and nnU-Net preparation are real code paths, not empty functions. nnU-Net planning and training are external commands and were not run. YOLO11 and YOLO26 names are checked against the documented `yolo{11,26}{n,s,m,l,x}` and `-seg` patterns. The installed package was not available, so weight loading was not confirmed.
