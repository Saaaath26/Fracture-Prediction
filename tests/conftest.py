import csv
import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny(tmp_path):
    root = tmp_path / "source with spaces"
    (root / "images/Fractured").mkdir(parents=True)
    (root / "Annotations/COCO JSON").mkdir(parents=True)
    (root / "Utilities/Fracture Split").mkdir(parents=True)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "fractured"}],
    }
    with (root / "dataset.csv").open("w") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_id", "fractured", "fracture_count", "multiscan"]
        )
        writer.writeheader()
        for index, split in enumerate(["train", "valid", "test"]):
            name = f"{index}.jpg"
            image = np.zeros((20, 20), np.uint8)
            image[2 + index : 10 + index, 3:15] = 100 + index * 50
            Image.fromarray(image).save(root / "images/Fractured" / name)
            writer.writerow(
                dict(image_id=name, fractured=1, fracture_count=1, multiscan=0)
            )
            (root / f"Utilities/Fracture Split/{split}.csv").write_text(
                "image_id\n" + name + "\n"
            )
            coco["images"].append(dict(id=index, file_name=name, width=20, height=20))
            coco["annotations"].append(
                dict(
                    id=index,
                    image_id=index,
                    category_id=1,
                    bbox=[2, 2, 8, 8],
                    segmentation=[[2, 2, 10, 2, 10, 10, 2, 10]],
                )
            )
    (root / "Annotations/COCO JSON/COCO_fracture_masks.json").write_text(json.dumps(coco))
    return root
