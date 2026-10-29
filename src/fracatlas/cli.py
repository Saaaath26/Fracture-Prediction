"""Explicit command dispatch; validation and audit cannot invoke training."""

import argparse
import importlib.metadata
from .config import load_config
from .serialization import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fracatlas")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate",
        "audit",
        "prepare",
        "train-yolo",
        "predict",
        "evaluate-yolo",
        "extract",
        "integrity",
    ):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command in {"audit", "predict", "evaluate-yolo", "extract", "integrity"}:
            p.add_argument("--output", required=True)
        if command in {"predict", "evaluate-yolo"}:
            p.add_argument("--checkpoint", required=True)
        if command == "predict":
            p.add_argument("--source", required=True)
        if command == "evaluate-yolo":
            p.add_argument("--split", choices=["val", "test"], default="val")
        if command == "train-yolo":
            p.add_argument("--authorize-training", action="store_true")
            p.add_argument("--acknowledge-unresolved-leakage", action="store_true")
        if command == "extract":
            p.add_argument("--image-id", required=True)
            p.add_argument(
                "--contour-type",
                choices=["bone_candidate", "soft_tissue_candidate"],
                default="bone_candidate",
            )
    p = sub.add_parser("train-relationship")
    p.add_argument("--table", required=True)
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--model", choices=["logistic", "xgboost"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--authorize-training", action="store_true")
    p = sub.add_parser("evaluate-relationship")
    p.add_argument("--table", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("train-neural")
    p.add_argument("--manifest", required=True)
    p.add_argument("--kind", choices=["unet", "siamese"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--authorize-training", action="store_true")
    p = sub.add_parser("prepare-nnunet")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("predict-neural")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--source-b")
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate-neural")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", required=True)
    p = sub.add_parser("export-table")
    p.add_argument("--features", nargs="+", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "export-table":
        from .exports import export_table

        return export_table(args.features, args.manifest, args.output)
    if args.command == "predict-neural":
        from .neural_inference import predict_neural

        return predict_neural(args.checkpoint, args.source, args.output, args.source_b)
    if args.command == "evaluate-neural":
        from .neural_inference import evaluate_neural

        return evaluate_neural(args.checkpoint, args.manifest, args.split, args.output)
    if args.command == "train-relationship":
        from .analysis import train_relationship

        return train_relationship(
            args.table, args.features, args.model, args.output, args.authorize_training
        )
    if args.command == "evaluate-relationship":
        from .analysis import evaluate_relationship

        return evaluate_relationship(args.table, args.model_dir, args.output)
    if args.command == "train-neural":
        from .neural_training import train_neural

        return train_neural(
            args.manifest,
            args.kind,
            args.output,
            args.epochs,
            args.batch,
            args.size,
            args.authorize_training,
        )
    if args.command == "prepare-nnunet":
        from .neural_training import prepare_nnunet

        return prepare_nnunet(args.manifest, args.output)
    config = load_config(args.config)
    if args.command == "integrity":
        from .data import Dataset
        from .integrity import write_integrity_reports

        dataset = Dataset(
            config.dataset_root, config.split_manifest, config.exclusion_manifest
        )
        inventory = write_integrity_reports(dataset, args.output)
        print(
            dict(
                output=args.output,
                blocking_errors=len(inventory["blocking_errors"]),
                exclusions=inventory["exclusions"],
                training_executed=False,
            )
        )
        return
    if args.command == "validate":
        packages = {}
        for name in (
            "ultralytics",
            "torch",
            "opencv-python",
            "numpy",
            "scikit-learn",
            "xgboost",
        ):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not installed"
        print(
            dict(
                config=config.serializable(), packages=packages, training_executed=False
            )
        )
        return
    if args.command in {"train-yolo", "predict", "evaluate-yolo"}:
        from . import yolo

        if args.command == "train-yolo":
            return yolo.train(
                config, args.authorize_training, args.acknowledge_unresolved_leakage
            )
        if args.command == "predict":
            return yolo.predict(config, args.checkpoint, args.source, args.output)
        return yolo.evaluate(config, args.checkpoint, args.split, args.output)
    from .data import Dataset, prepare

    dataset = Dataset(
        config.dataset_root, config.split_manifest, config.exclusion_manifest
    )
    if args.command == "audit":
        return write_json(args.output, dataset.audit())
    if args.command == "prepare":
        print(prepare(dataset, config.prepared_root, config.task))
        return
    from .extraction import extract_document

    try:
        document, _, _ = extract_document(dataset, args.image_id, args.contour_type)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_json(args.output, document)


if __name__ == "__main__":
    main()
