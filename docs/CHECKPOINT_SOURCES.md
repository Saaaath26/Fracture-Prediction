# Checkpoint verification

Verified against official Ultralytics documentation on 2026-10-28:

- YOLO11: https://docs.ultralytics.com/models/yolo11/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- Official YOLO26 repository: https://github.com/ultralytics/yolo26

Documented detection names: `yolo11{n,s,m,l,x}.pt`, `yolo26{n,s,m,l,x}.pt`.
Documented instance-segmentation names: `yolo11{n,s,m,l,x}-seg.pt`, `yolo26{n,s,m,l,x}-seg.pt`.

Four supplied experiment configurations use `yolo11n.pt`, `yolo11n-seg.pt`, `yolo26n.pt`, and `yolo26n-seg.pt`. This verifies official existence, naming and task support; weights were not downloaded, checksummed or loaded. Ultralytics is absent in the delivery environment, so end-to-end backend compatibility has not been exercised. Install the optional YOLO extra and freeze the successful environment before future experiments.

nnU-Net dataset and split conventions follow official documentation:

- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md
- https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/manual_data_splits.md

All sources support implementation conventions, not fracture-specific performance claims.
