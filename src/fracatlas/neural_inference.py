"""Inference/evaluation on local research checkpoints; no optimization or fitting."""

from pathlib import Path
import numpy as np
from PIL import Image
from .serialization import write_json
from .evaluation import mask_metrics, classification_metrics, save_evaluation


def load_model(checkpoint):
    import torch
    from .neural import UNet, SiameseCNN

    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    kind = data["kind"]
    if kind not in {"unet", "siamese"}:
        raise ValueError("Unknown checkpoint kind")
    model = UNet() if kind == "unet" else SiameseCNN()
    model.load_state_dict(data["model"])
    model.eval()
    return model, kind, data["size"]


def _tensor(path, size):
    import torch

    with Image.open(path) as im:
        a = np.asarray(
            im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        ).copy()
    return torch.from_numpy(a.astype(np.float32) / 255)[None, None]


def predict_neural(checkpoint, image, output, image_b=None):
    import torch

    model, kind, size = load_model(checkpoint)
    with torch.inference_mode():
        if kind == "unet":
            if image_b:
                raise ValueError("U-Net takes one image")
            with Image.open(image) as im:
                original_size = im.size
            probabilities = torch.sigmoid(model(_tensor(image, size)))[0, 0].numpy()
            # Float probability interpolation back into original pixel coordinates.
            probability = np.asarray(
                Image.fromarray(probabilities).resize(
                    original_size, Image.Resampling.BILINEAR
                )
            )
            Image.fromarray((probability >= 0.5).astype(np.uint8) * 255).save(output)
            write_json(
                Path(str(output) + ".provenance.json"),
                dict(
                    schema_version="1.1",
                    record_type="neural_prediction",
                    kind="unet",
                    checkpoint=str(checkpoint),
                    source=str(image),
                    threshold=0.5,
                    anatomical_identity="not_established_without_verified_masks",
                    clinically_validated=False,
                    training_executed=False,
                ),
            )
        else:
            if not image_b:
                raise ValueError("Siamese prediction needs two images")
            probability = float(
                torch.sigmoid(
                    model(_tensor(image, size), _tensor(image_b, size))
                ).item()
            )
            write_json(
                output,
                dict(
                    schema_version="1.1",
                    record_type="neural_prediction",
                    kind="siamese",
                    pair_probability=probability,
                    experimental_unvalidated=True,
                    clinically_validated=False,
                    training_executed=False,
                ),
            )


def evaluate_neural(checkpoint, manifest, split, output):
    import torch
    from .neural_training import verified_manifest

    if split not in {"val", "test"}:
        raise ValueError("Use val or test")
    model, kind, size = load_model(checkpoint)
    rows = [r for r in verified_manifest(manifest, kind) if r["split"] == split]
    if not rows:
        raise ValueError("No evaluation samples")
    per_case = []
    truth = []
    probabilities = []
    with torch.inference_mode():
        for r in rows:
            if kind == "unet":
                with Image.open(r["mask"]) as im:
                    target = (np.asarray(im) > 0).astype(np.uint8)
                p = torch.sigmoid(model(_tensor(r["image"], size)))[0, 0].numpy()
                p = np.asarray(
                    Image.fromarray(p).resize(
                        (target.shape[1], target.shape[0]), Image.Resampling.BILINEAR
                    )
                )
                per_case.append(
                    dict(
                        image=r["image"],
                        patient_id=r["patient_id"],
                        **mask_metrics(target, (p >= 0.5).astype(np.uint8)),
                    )
                )
            else:
                probabilities.append(
                    float(
                        torch.sigmoid(
                            model(
                                _tensor(r["image_a"], size), _tensor(r["image_b"], size)
                            )
                        ).item()
                    )
                )
                truth.append(int(r["label"]))
    metrics = (
        dict(
            mean_dice=float(np.mean([r["dice"] for r in per_case])),
            mean_iou=float(np.mean([r["iou"] for r in per_case])),
            per_case=per_case,
        )
        if kind == "unet"
        else classification_metrics(truth, probabilities)
    )
    save_evaluation(
        output,
        kind,
        split,
        metrics,
        dict(checkpoint=str(checkpoint), manifest=str(manifest), threshold=0.5),
    )
