"""Deferred CPU training for verified anatomical masks or independently labeled pairs."""

import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .guards import require_training
from .serialization import write_json


def verified_manifest(path, kind):
    path = Path(path).resolve()
    rows = list(csv.DictReader(path.open()))
    if kind not in {"unet", "siamese"} or not rows:
        raise ValueError("Invalid manifest")
    groups = {s: set() for s in ("train", "val", "test")}
    images = {s: set() for s in groups}
    for r in rows:
        if r.get("verified") != "true" or not r.get("annotation_source"):
            raise ValueError("Expert verification and annotation_source required")
        if r.get("split") not in groups:
            raise ValueError("Invalid split")
        patient_fields = (
            ("patient_id",) if kind == "unet" else ("patient_id_a", "patient_id_b")
        )
        for key in patient_fields:
            if not r.get(key):
                raise ValueError("Patient identifiers required")
            groups[r["split"]].add(r[key])
        keys = ("image", "mask") if kind == "unet" else ("image_a", "image_b")
        for key in keys:
            r[key] = str((path.parent / r[key]).resolve())
            if not Path(r[key]).is_file():
                raise FileNotFoundError(r[key])
            if key != "mask":
                with Image.open(r[key]) as im:
                    pixels = np.asarray(im.convert("L"))
                digest = hashlib.sha256(
                    str(pixels.shape).encode() + pixels.tobytes()
                ).hexdigest()
                images[r["split"]].add(digest)
        if kind == "unet":
            if r.get("contour_type") not in {"verified_bone", "verified_soft_tissue"}:
                raise ValueError(
                    "Verified anatomical masks required, never fracture masks"
                )
            with Image.open(r["mask"]) as im:
                mask = np.asarray(im)
                shape = im.size
            with Image.open(r["image"]) as im:
                image_shape = im.size
            if (
                mask.ndim != 2
                or not np.isin(mask, [0, 1, 255]).all()
                or shape != image_shape
            ):
                raise ValueError("Binary aligned anatomical masks required")
        elif r.get("label") not in {"0", "1"}:
            raise ValueError("Binary verified pair labels required")
    for sets in (groups, images):
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            if sets[a] & sets[b]:
                raise ValueError("Patient/image leakage across neural splits")
    if kind == "unet" and len({r["contour_type"] for r in rows}) != 1:
        raise ValueError("Train separate bone and soft-tissue models")
    if not groups["train"] or not groups["val"]:
        raise ValueError("Train and validation sets required")
    return rows


def train_neural(
    manifest, kind, output, epochs=100, batch=4, size=256, authorized=False
):
    require_training(authorized)
    rows = verified_manifest(manifest, kind)
    if min(epochs, batch, size) <= 0 or size < 8:
        raise ValueError("Invalid training dimensions")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    import torch
    from torch.utils.data import Dataset, DataLoader
    from .neural import UNet, SiameseCNN

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)

    class Samples(Dataset):
        def __init__(self, split):
            self.rows = [r for r in rows if r["split"] == split]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]

            def tensor(key, mask=False):
                with Image.open(r[key]) as im:
                    im = im.convert("L").resize(
                        (size, size),
                        Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR,
                    )
                    a = np.asarray(im).copy()
                return torch.from_numpy(
                    (a > 0).astype(np.float32) if mask else a.astype(np.float32) / 255
                ).unsqueeze(0)

            return (
                (tensor("image"), tensor("mask", True))
                if kind == "unet"
                else (
                    tensor("image_a"),
                    tensor("image_b"),
                    torch.tensor(float(r["label"])),
                )
            )

    model = UNet() if kind == "unet" else SiameseCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loaders = {
        s: DataLoader(Samples(s), batch_size=batch, shuffle=s == "train", num_workers=0)
        for s in ("train", "val")
    }
    output.mkdir(parents=True)
    best = float("inf")
    history = []
    for epoch in range(epochs):
        losses = {}
        for split, loader in loaders.items():
            model.train(split == "train")
            total = 0.0
            count = 0
            with torch.set_grad_enabled(split == "train"):
                for sample in loader:
                    inputs, target = sample[:-1], sample[-1]
                    pred = model(*inputs)
                    loss = loss_fn(pred, target)
                    if split == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total += loss.item() * len(target)
                    count += len(target)
            losses[split] = total / count
        history.append(dict(epoch=epoch + 1, **losses))
        if losses["val"] < best:
            best = losses["val"]
            torch.save(
                dict(model=model.state_dict(), kind=kind, size=size), output / "best.pt"
            )
    write_json(output / "history.json", history)
    write_json(
        output / "experiment.json",
        dict(
            manifest=str(Path(manifest).resolve()),
            kind=kind,
            epochs=epochs,
            batch=batch,
            size=size,
            seed=42,
            clinically_validated=False,
        ),
    )


def prepare_nnunet(manifest, destination):
    """Prepare verified 2D PNG cases; never call nnU-Net preprocessing or training."""

    rows = verified_manifest(manifest, "unet")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    for folder in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (destination / folder).mkdir(parents=True)
    train = []
    val = []
    for i, r in enumerate(rows):
        case = f"case_{i:05d}"
        suffix = "Ts" if r["split"] == "test" else "Tr"
        with Image.open(r["image"]) as im:
            im.convert("L").save(destination / f"images{suffix}" / f"{case}_0000.png")
        with Image.open(r["mask"]) as im:
            Image.fromarray((np.asarray(im) > 0).astype(np.uint8)).save(
                destination / f"labels{suffix}" / f"{case}.png"
            )
        if r["split"] == "train":
            train.append(case)
        if r["split"] == "val":
            val.append(case)
    write_json(
        destination / "dataset.json",
        dict(
            channel_names={"0": "radiograph"},
            labels={"background": 0, "anatomy": 1},
            numTraining=len(train) + len(val),
            file_ending=".png",
            description=(
                "Verified anatomical masks only. FracAtlas fracture polygons are not valid labels."
            ),
        ),
    )
    write_json(destination / "splits_final.json", [dict(train=train, val=val)])
    write_json(destination / "source_manifest.json", rows)
