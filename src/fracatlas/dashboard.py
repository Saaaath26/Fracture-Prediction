"""Launch with streamlit run src/fracatlas/dashboard.py. No training controls."""

import json
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


def default_dataset_root():
    anchor = repo_root()
    workspace = anchor.parent
    for base in (workspace, anchor):
        for name in ("FractAtlas Dataset", "FracAtlas Dataset"):
            candidate = base / name / "FracAtlas"
            if (candidate / "dataset.csv").is_file():
                return candidate
    return workspace / "FractAtlas Dataset" / "FracAtlas"


def model_status(root=None):
    root = repo_root() if root is None else Path(root)
    weights_dir = root / "weights"
    author = (
        sorted(path.name for path in weights_dir.glob("*.pt"))
        if weights_dir.is_dir()
        else []
    )
    runs = root / "runs"
    trained = (
        sorted(str(path.relative_to(root)) for path in runs.rglob("*.pt"))
        if runs.is_dir()
        else []
    )

    def state(token):
        return (
            "checkpoint_present_not_validated_here"
            if any(token in path for path in trained)
            else "untrained_unavailable"
        )

    return dict(
        yolo11=state("yolo11"),
        yolo26=state("yolo26"),
        author_yolov8_checkpoints=author,
        run_checkpoints=trained,
        inference_executed_by_dashboard=False,
        training_executed_by_dashboard=False,
        clinical_validation=False,
    )


def main():
    import streamlit as st
    from fracatlas.data import Dataset
    from fracatlas.extraction import extract_document
    from fracatlas.visualization import overlay

    st.set_page_config(page_title="FracAtlas research", layout="wide")
    st.title("FracAtlas research explorer")
    st.caption(
        "Research viewer. No training, no clinical validation, and no fabricated predictions. "
        "Contours are unverified candidates, not fracture masks or external limb silhouettes."
    )
    status = model_status()
    st.sidebar.markdown("**Model status**")
    st.sidebar.write(f"YOLO11: {status['yolo11'].replace('_', ' ')}")
    st.sidebar.write(f"YOLO26: {status['yolo26'].replace('_', ' ')}")
    if status["author_yolov8_checkpoints"]:
        st.sidebar.write(
            "Author YOLOv8 checkpoints are on disk and are not executed by this page."
        )
    else:
        st.sidebar.write("No author checkpoints found.")
    st.sidebar.caption(
        "Uploading a prediction JSON displays that file. It is not a result from this session."
    )
    inventory = repo_root() / "reports" / "integrity" / "dataset_inventory.json"
    if inventory.is_file():
        summary = json.loads(inventory.read_text())
        st.sidebar.caption(
            f"Integrity report: {summary.get('exclusions', 0)} derived exclusions, "
            f"{len(summary.get('blocking_errors', []))} blocking findings before exclusions."
        )

    root = st.sidebar.text_input("Dataset root", str(default_dataset_root()))
    if st.session_state.get("dataset_root") != root:
        try:
            st.session_state["dataset"] = Dataset(root)
            st.session_state["dataset_root"] = root
            st.session_state.pop("view_key", None)
        except (OSError, ValueError, KeyError) as exc:
            st.error(str(exc))
            st.stop()
    dataset = st.session_state["dataset"]
    name = st.sidebar.selectbox("Radiograph", sorted(dataset.images))
    kind = st.sidebar.selectbox(
        "Candidate contour type", ["bone_candidate", "soft_tissue_candidate"]
    )
    show_annotations = st.sidebar.checkbox("Show fracture annotations", value=True)
    view_key = (name, kind, root)
    if st.session_state.get("view_key") != view_key:
        try:
            document, image, contours = extract_document(dataset, name, kind)
        except OSError as exc:
            st.error(f"Image cannot be decoded: {exc}. Corrupt samples stay excluded.")
            st.stop()
        st.session_state["view_key"] = view_key
        st.session_state["document"] = document
        st.session_state["image"] = image
        st.session_state["contours"] = contours
    document = st.session_state["document"]
    image = st.session_state["image"]
    contours = st.session_state["contours"]
    predictions = None
    upload = st.sidebar.file_uploader(
        "Prediction JSON from fracatlas predict", type=["json"]
    )
    if upload:
        try:
            predictions = json.load(upload)
            if (
                predictions.get("image_id") != name
                or (
                    predictions.get("height"),
                    predictions.get("width"),
                )
                != image.shape
            ):
                raise ValueError("Prediction image/shape mismatch")
            if predictions.get("record_type") not in {None, "model_prediction"}:
                raise ValueError("File is not a prediction record")
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            st.error(str(exc))
            predictions = None
    if predictions:
        st.info(
            "Showing an uploaded prediction. Checkpoint "
            f"{predictions.get('checkpoint', 'unspecified')}. "
            "This is not evidence that YOLO11 or YOLO26 was trained or validated."
        )
    elif (
        status["yolo11"] == "untrained_unavailable"
        and status["yolo26"] == "untrained_unavailable"
    ):
        st.warning(
            "YOLO11 and YOLO26 checkpoints are untrained and unavailable. No prediction is shown."
        )
    left, right = st.columns(2)
    left.image(image, caption="Annotation-frame radiograph", use_container_width=True)
    right.image(
        overlay(
            image,
            dataset.annotations.get(name, []) if show_annotations else [],
            contours,
            predictions,
        ),
        caption=(
            "Orange: fracture annotation. Green: bone candidate. Blue: soft-tissue candidate. "
            "Gray: abstained. Purple: uploaded prediction."
        ),
        use_container_width=True,
    )
    st.caption(document["projection_caveat"])
    if document["alignment"]["exif_applied"]:
        st.info(
            "EXIF orientation was applied because it is the unique match to the annotation size."
        )
    elif document["alignment"]["exif_orientation"] not in (0, 1):
        st.caption(
            "An EXIF orientation tag is present and was not applied; stored pixels match the annotation frame."
        )
    st.download_button(
        "Download contour JSON",
        json.dumps(document, indent=2, default=_json_default),
        file_name=f"{Path(name).stem}_{kind}.json",
        mime="application/json",
    )
    st.json(dataset.metadata[name])
    for index, contour in enumerate(contours):
        state = "candidate geometry" if contour.valid else "abstained"
        with st.expander(f"Contour {index}: {state}"):
            st.write(contour.warnings)
            st.write(contour.uncertainty)
            st.json(contour.quality)
            if contour.valid:
                match = next(
                    row for row in document["records"] if row["contour_id"] == index
                )
                st.json(match["features"])
                st.caption(match["relationship_source"])


def _json_default(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


if __name__ == "__main__":
    main()
