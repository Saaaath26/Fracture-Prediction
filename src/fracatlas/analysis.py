"""Future relationship classifiers. Fit preprocessing exclusively on training rows.
Features must be available at the intended deployment time. Ground-truth fracture
relationships must never be used to predict fracture presence.
"""

import csv
from pathlib import Path
import numpy as np
from .guards import require_training
from .serialization import write_json
from .evaluation import classification_metrics, save_evaluation

FORBIDDEN = {
    "label",
    "fractured",
    "fracture_count",
    "fracture_present",
    "patient_id",
    "split",
    "image_id",
}


def read_table(path, feature_names):
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Explicit unique features required")
    if any(
        n in FORBIDDEN
        or any(
            s in n.lower()
            for s in ("fracture", "target", "label", "overlap", "distance_to_boundary")
        )
        for n in feature_names
    ):
        raise ValueError(
            "Target-derived features forbidden in fracture-presence classification"
        )
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        raise ValueError("Empty feature table")
    if any(not r.get("patient_id") or not r.get("image_id") for r in rows):
        raise ValueError("Verified patient_id and image_id required")
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image IDs")
    for field in ("patient_id", "image_id"):
        sets = {
            s: {r[field] for r in rows if r["split"] == s}
            for s in ("train", "val", "test")
        }
        if (
            sets["train"] & sets["val"]
            or sets["train"] & sets["test"]
            or sets["val"] & sets["test"]
        ):
            raise ValueError(f"{field} leakage across splits")
    if any(r["split"] not in {"train", "val", "test"} for r in rows):
        raise ValueError("Invalid split")
    X = np.array(
        [[float(r[n]) if r[n] else np.nan for n in feature_names] for r in rows]
    )
    y = np.array([int(r["label"]) for r in rows])
    split = np.array([r["split"] for r in rows])
    if not np.isin(y, [0, 1]).all() or np.isinf(X).any():
        raise ValueError("Invalid labels/features")
    return X, y, split


def train_relationship(table, features, model_name, output, authorized=False):
    require_training(authorized)
    X, y, split = read_table(table, features)
    train = split == "train"
    val = split == "val"
    if len(set(y[train])) != 2 or not val.any():
        raise ValueError("Two training classes and validation rows required")
    if np.isnan(X[train]).all(axis=0).any():
        raise ValueError("Entirely missing training feature")
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
        )
    else:
        raise ValueError("Unknown relationship model")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )
    # Held-out test labels may be present for leakage checks. Fit sees training rows only.
    pipeline.fit(X[train], y[train])
    output.mkdir(parents=True)
    import joblib

    joblib.dump(pipeline, output / "model.joblib")
    write_json(
        output / "features.json",
        dict(
            features=features,
            model=model_name,
            seed=42,
            target="fracture_presence",
            deployment="radiograph; RGB transfer unvalidated",
        ),
    )
    save_evaluation(
        output / "validation.json",
        "classification",
        "val",
        classification_metrics(y[val], pipeline.predict_proba(X[val])[:, 1]),
        dict(table=str(table), features=features),
    )


def evaluate_relationship(table, model_dir, output):
    import json, joblib

    model_dir = Path(model_dir)
    features = json.loads((model_dir / "features.json").read_text())["features"]
    X, y, split = read_table(table, features)
    test = split == "test"
    if not test.any():
        raise ValueError("No held-out test rows")
    pipeline = joblib.load(model_dir / "model.joblib")
    save_evaluation(
        output,
        "classification",
        "test",
        classification_metrics(y[test], pipeline.predict_proba(X[test])[:, 1]),
        dict(model=str(model_dir), features=features),
    )
