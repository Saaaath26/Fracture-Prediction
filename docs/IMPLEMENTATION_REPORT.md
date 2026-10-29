# FracAtlas implementation report

Delivered 2026-10-28. **Implementation and safe verification only. No training, fitting, hyperparameter optimization, training smoke tests, GPU benchmarking or model inference were executed. No clinical validation or model performance is claimed.**

Audit revision 2026-10-28: see [code audit](CODE_AUDIT.md), [changes](IMPLEMENTATION_CHANGES.md), and [architecture](architecture/FRACATLAS_ARCHITECTURE.md). The original 15-file checksum baseline remains unchanged. The official split still fails preparation until `reports/integrity/exclusion_manifest.csv` is selected. That manifest does not establish patient-level independence.

## 1. Original repository and dataset audit

The actual repository is `FracAtlas/`; the dataset is `../FractAtlas Dataset/FracAtlas/`. The implementation uses configurable paths and supports spaces. No pre-existing architecture specification or AGENTS.md was found inside the workspace; the supplied research requirements guided the modules.

All 15 original repository files are byte-for-byte unchanged, verified with SHA-256 against `reports/baseline_checksums.json`. This includes the authors' two notebooks, two YOLOv8 checkpoints, README/license, data YAML, split CSVs and assets. The original dataset was read only. New files are uncommitted additions.

The original notebooks contain environment-specific paths and training commands, and the prediction notebook's localization example uses a segmentation task flag. These were inspected as text and preserved, not executed or rewritten. New adapters accept local author checkpoints with the matching task for future baseline evaluation.

Full image decoding, metadata, COCO geometry and duplicate/split audit results are stored in `reports/dataset_audit.json`:

| Finding | Result |
| --- | --- |
| Physical JPEG files | 4,085 |
| Unique image filenames / metadata rows | 4,083 |
| Fractured / non-fractured images | 719 / 3,364 |
| COCO images / fracture instances | 719 / 924 |
| Downloaded dataset train / validation / test | 574 / 82 / 63 |
| Original notebook train / validation / test | 574 / 82 / 61 |
| Unassigned normal images under default split | 3,364 |
| Truncated JPEGs rejected by strict Pillow decoding | 59 |
| COCO dimension mismatch | `IMG0002628.jpg`: annotation 454×373, actual 373×454 |
| Identical decoded-pixel duplicate groups | 78 |
| Exact duplicate groups crossing default splits | 8 |
| dHash collision groups requiring review | 117 |
| Metadata multiscan cases | 396 |
| Patient IDs | Absent; patient-level leakage unresolved |
| Verified anatomical masks / Siamese pair labels | Not provided |

`IMG0003375.jpg` and `IMG0003376.jpg` each occur in both class folders as byte-identical copies. The loader records both paths and selects the metadata-consistent folder. These are also the two images present in the downloaded test CSV but absent from the notebook test CSV. Train and validation IDs agree between the two sources; byte-level CSV formatting differs. The default loader explicitly follows the downloaded dataset split.

Two polygon coordinates exceeded the image height by approximately 5e-13 pixels. Conversion now tolerates and snaps only numerical boundary noise up to 1e-6 pixels; material violations still fail. This behavior is tested and does not change original annotations. The 61 final audit errors comprise 59 decode failures, one dimension mismatch, and one aggregate cross-split-duplicate error. All affected filenames and duplicate groups are listed in the JSON.

Original annotation inventory: 4,083 VOC files, 4,084 files in the YOLO directory, one VGG JSON and one COCO JSON. New training labels are derived explicitly from validated COCO records. No equivalence claim is made across every original annotation format.

## 2. Code added

| Area | Added files / behavior |
| --- | --- |
| Packaging | `pyproject.toml`, import-safe `src/fracatlas/`, optional dependency groups, `fracatlas` CLI |
| Configuration | `config.py`; four YOLO YAMLs, relative path resolution, task/checkpoint checks, source-output separation |
| Dataset | `data.py`; COCO index, metadata validation, duplicate audits, reviewed patient split selection, isolated YOLO preparation and integrity hashes |
| Annotation conversion | `annotations.py`; box/polygon validation, detection labels, instance segmentation labels, fracture masks; multipart/RLE rejected explicitly |
| Preprocessing | `preprocessing.py`; deterministic grayscale normalization and optional CLAHE |
| Contours / geometry | `contours.py`, `geometry.py`; typed candidates, validity filters, interpretable measurements and fracture spatial relationships |
| Export | `serialization.py`, `exports.py`; versioned JSON, source provenance, reviewed geometry CSV |
| YOLO workflows | `yolo.py`; guarded training, existing-local-weight prediction/evaluation |
| Relationship workflows | `analysis.py`; logistic regression/XGBoost pipelines, split/target-leakage checks, held-out evaluation |
| Neural workflows | `neural.py`, `neural_training.py`, `neural_inference.py`; compact U-Net/Siamese definitions, guarded training, local checkpoint prediction/evaluation, verified nnU-Net data export |
| Evaluation | `evaluation.py`; common reports, segmentation/classification metrics with explicit undefined/empty cases |
| UI | `visualization.py`, `dashboard.py`; original/annotation/prediction/contour overlays and feature inspection; no training UI |
| Tests | `tests/test_core.py`, `tests/test_safety_data.py`; 28 tests |
| Documentation | `RESEARCH.md`, this report, future commands, feature schema and checkpoint sources |
| Audit evidence | `reports/dataset_audit.json`, baseline checksums, environment versions, deterministic example features, verification summary |

Original files were not modified. No model artifacts were created. `runs/` and `artifacts/` contain only ignore files until future explicit workflows generate outputs.

## 3. YOLO11 / YOLO26 readiness

Official model/task/checkpoint names were verified against [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) and [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/). Detection and instance-segmentation names for n/s/m/l/x scales are validated. Supplied configurations use nano variants. Checkpoint existence was verified through official documentation; no checkpoint was downloaded or loaded.

Future training requires `--authorize-training`, a matching prepared dataset, valid provenance and unchanged prepared file hashes. Missing patient identity requires a separate explicit acknowledgement for original-split reproductions. That acknowledgement cannot bypass dataset audit failures. The new code uses COCO-to-YOLO conversion in separate output folders, never a blanket reuse of fracture masks as anatomy masks.

Ultralytics is not installed in this environment. Backend loading, optimization, inference and evaluation are therefore implemented but not integration-tested against the actual model package. This is explicitly different from successful static configuration validation.

## 4. Anatomical contour extraction

Classical processing creates bone or soft-tissue **candidates**, using separate thresholds and explicit semantic types. Border-truncated, degenerate and excessively large candidates cannot produce geometry. No method here establishes clinically correct anatomical identity. Expert verification remains necessary.

Geometry includes area, perimeter, centroid, major/minor minimum-area rectangle axes, aspect ratio, orientation, circularity and solidity. Optional verified pixel spacing produces physical area and perimeter; JPEG size alone never establishes millimeters. Qualitative uncertainty, source hash, method parameters and warnings accompany exported features.

Deterministic extraction on `IMG0000019.jpg` produced two usable geometric candidates and two rejected candidates, recorded in `reports/example_features.json`. These are image-processing outputs, not model performance measurements or expert anatomical annotations.

Optional U-Net and nnU-Net workflows require expert-verified masks with correct source geometry and patient grouping. They reject fracture contour types and mixed bone/soft-tissue targets. No verified anatomical masks are presently available.

## 5. Geometric relationship analysis

The code computes signed distance from the fracture-mask centroid to an anatomical candidate boundary and fracture-pixel overlap with that candidate. These descriptive relationships retain their ground-truth annotation provenance and are kept separate from predictor inputs.

Logistic regression and XGBoost support a future fracture-presence experiment using selected anatomical geometry. Imputation/scaling fit on training rows only. Validation and test are separate. Explicit patient IDs and split integrity are required; fracture labels/counts and target-derived relationship features are disallowed as presence-classifier inputs. Interpretation still requires controlling anatomy, resolution, view and acquisition confounders.

The optional Siamese CNN is an experimental pair classifier. Pair labels, annotation provenance, both patient identities and study semantics must be provided before training. No fracture prognosis, smartphone transfer model or clinical interpretation was invented.

## 6. Verification results

- `python3 -m pytest -q`: **28 passed**. Tests cover preprocessing, annotation conversion, polygon boundary tolerance, geometry, serialization, configuration, duplicate/split leakage, preparation, prepared-data integrity, training guards, reviewed table export and annotation requirements.
- `python3 -m compileall -q src tests`: passed.
- `python3 -m pyflakes src tests`: passed.
- `python3 -m black --check src tests`: passed for 20 Python files.
- All four experiment configurations loaded without model construction or downloads.
- Streamlit AppTest: default dashboard load passed; selecting a known truncated image displayed the expected error without an application exception.
- Full dataset audit completed with the real data findings above. Audit findings are not test failures; they intentionally prevent unsafe preparation.
- SHA-256 comparison: all 15 original repository files unchanged.
- Deterministic real-image feature extraction succeeded.

Synthetic shapes and tiny datasets exist only in temporary unit-test fixtures. They are not research annotations, training examples or claimed model results. No training body was exercised, including a single batch or epoch. Guard tests stop before imports, data loading or fitting.

## 7. Missing annotations and dependencies

Available package versions are in `reports/environment.json`. NumPy, OpenCV, Pillow, PyYAML, PyTorch, scikit-learn, Streamlit and pytest are present. **Ultralytics, XGBoost and nnunetv2 are absent** and declared as optional extras. No dependency installation was performed.

Missing research inputs include verified anatomical masks, reliable patient/study identifiers, an independently justified Siamese pair-label definition, reviewed contour selections, and physical pixel spacing where physical measurements are desired. Smartphone/video correspondence data and cross-modality validation are absent. RGB transfer remains explicitly experimental and unvalidated.

## 8. Exact future commands — NOT EXECUTED

[Future training instructions](FUTURE_TRAINING.md) contain exact preparation, YOLO11/YOLO26, logistic regression, XGBoost, U-Net, Siamese and nnU-Net commands, plus subsequent inference/evaluation. **Every training command is NOT EXECUTED.** Header-only manifest templates are in `configs/`; the referenced reviewed CSVs intentionally do not exist until real verified data is supplied.

## 9. Remaining blockers before training

1. Reacquire or explicitly exclude the 59 truncated images; resolve `IMG0002628.jpg` geometry with source/expert evidence. No automatic rotation or relabeling was assumed.
2. Resolve exact cross-split duplicates and manually review near-duplicate candidates. Obtain reliable patient grouping, keep repeated views/studies together, and freeze a reviewed partition.
3. Choose the precise baseline split version and record it. Include normal cases in new specificity/classification experiments instead of silently attaching them to the fracture-only split.
4. Install the required optional backends and record a reproducible environment; verify compatibility on the intended host before any authorized training.
5. Provide verified anatomical masks, pair labels and reviewed geometry manifests for the optional tasks. Establish outcome definitions and confounder controls.
6. Explicitly authorize training. The implementation session does not supply that authorization.

The research software and safety checks are implemented. Real-data corrections, backend integration validation, training, experimental results and clinical validation remain separate future work.
