# Optimization report

Target machines are an Apple Silicon MacBook and an NVIDIA workstation with 16 GB of GPU memory. No kernel was timed and no GPU benchmark was run.

## Changes that reduce work

The audit hashes a JPEG from the same buffer it decodes. Label validation for both YOLO tasks happens in that pass. `prepare` does not decode the cohort a second time except for the rare EXIF-aligned PNG.

The dashboard keeps the dataset index and the selected radiograph in session state. Changing a checkbox does not rebuild the index or re-decode the file. Only the selected image is decoded. The full JPEG set is not loaded as arrays.

CSV readers use context managers. Contour extraction for the CLI and the dashboard is one function, so the two interfaces do not each grow a private copy.

`device: auto` asks PyTorch which of CUDA, MPS, and CPU exist, in that order, and returns a name. It does not allocate a network or time a convolution. Explicit `cpu`, `mps`, `cuda`, and `cuda:0` are accepted and not overridden. Current YOLO configs stay on CPU until someone changes them. Batch size 8 and image size 640 are unchanged starting values for a 16 GB card. They are not a measured recommendation.

## Choices that keep the original data safe

Prepared images are copies, not hard links and not symlinks. A hard link shares permissions and bytes with the source file. A later training cache that rewrote an image would modify FracAtlas. The extra disk use is the cost of that separation. EXIF correction writes a new PNG for `IMG0002628.jpg` only.

Truncated-image mode is never left enabled. The integrity report turns it on only inside a `try/finally` while describing the 59 partial rasters, then restores the previous flag. Those rasters are not saved.

## What was not added

No process pool, no service queue, and no dataset-wide in-memory cache. The audit already streams one image at a time, and the dashboard only needs one image. A persistent contour cache would be useful after the first authorized experiment, keyed by source hash and method version. It is not required to audit the code.

The reference U-Net and Siamese trainer remains on CPU. Moving it to MPS or CUDA would be an untested training loop, which this audit is not allowed to execute. YOLO already forwards the resolved device string to Ultralytics for a future authorized run.

Ultralytics, XGBoost, and nnU-Net were not installed. Installing them would not have been exercised without weight downloads or fitting.
