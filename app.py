import os
from pathlib import Path
from typing import Any

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Sex-specific ICH Prognosis Prediction Model",
    page_icon="🧠",
    layout="wide",
)

APP_TITLE = "Sex-specific ICH Prognosis Prediction Model"
APP_SUBTITLE = "Switch between male and female models and run prognosis prediction."
CLASS_LABELS = {0: "Favorable outcome", 1: "Poor outcome"}

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "models"

# Recommended project structure:
# app.py
# models/
#   male/
#     metadata.json
#     tabpfn_male_model.pkl
#     tabpfn-v2-classifier-finetuned-vutqq28w-boexhu6h.ckpt
#   female/
#     metadata.json
#     tabpfn_female_model.pkl
#     tabpfn-v2-classifier-finetuned-llderlii-oyd7ul21.ckpt

MODEL_CONFIG = {
    "Male": {
        "model_dir": MODEL_ROOT / "male",
        "metadata_file": "metadata.json",
    },
    "Female": {
        "model_dir": MODEL_ROOT / "female",
        "metadata_file": "metadata.json",
    },
}

BINARY_FEATURES = {"IVH", "SAP"}

FEATURE_DESCRIPTIONS = {
    "Age": "Age, years",
    "NIHSS": "National Institutes of Health Stroke Scale score",
    "GCS": "Glasgow Coma Scale score",
    "SBP": "Systolic blood pressure, mmHg",
    "Hematoma Volume": "Hematoma volume, mL",
    "HB": "Hemoglobin, g/L",
    "LYM": "Lymphocyte count, 10⁹/L",
    "NLR": "Neutrophil-to-lymphocyte ratio",
    "sCr": "Serum creatinine, μmol/L",
    "IVH": "Intraventricular hemorrhage",
    "SAP": "Stroke-associated pneumonia",
    "Time to CT": "Time from symptom onset to CT scan, hours",
    "AST": "Aspartate aminotransferase, U/L",
}

FEATURE_STATS = {
    "Female": {
        "Age": {"min": 18.0, "max": 93.0, "median": 63.0, "step": 1.0},
        "NIHSS": {"min": 0.0, "max": 40.0, "median": 10.0, "step": 1.0},
        "GCS": {"min": 3.0, "max": 15.0, "median": 14.0, "step": 1.0},
        "Hematoma Volume": {"min": 0.087, "max": 133.48, "median": 10.617, "step": 0.1},
        "Time to CT": {"min": 0.283333332510665, "max": 71.835000000021, "median": 4.94027777784504, "step": 0.1},
        "NLR": {"min": 0.536379018612521, "max": 95.3, "median": 6.72727272727273, "step": 0.1},
        "AST": {"min": 10.0, "max": 288.0, "median": 26.0, "step": 1.0},
        "SAP": {"min": 0.0, "max": 1.0, "median": 0.0, "step": 1.0},
    },
    "Male": {
        "Age": {"min": 18.0, "max": 92.0, "median": 61.0, "step": 1.0},
        "NIHSS": {"min": 0.0, "max": 43.0, "median": 8.0, "step": 1.0},
        "GCS": {"min": 3.0, "max": 15.0, "median": 14.0, "step": 1.0},
        "SBP": {"min": 100.0, "max": 268.0, "median": 169.0, "step": 1.0},
        "Hematoma Volume": {"min": 0.06, "max": 181.98, "median": 11.433, "step": 0.1},
        "HB": {"min": 73.0, "max": 196.0, "median": 144.0, "step": 0.1},
        "LYM": {"min": 0.06941, "max": 8.0837, "median": 1.11222, "step": 0.1},
        "NLR": {"min": 0.747645951035782, "max": 85.7272727272727, "median": 6.63709677419355, "step": 0.1},
        "sCr": {"min": 6.5, "max": 1407.0, "median": 74.0, "step": 0.1},
        "IVH": {"min": 0.0, "max": 1.0, "median": 0.0, "step": 1.0},
        "SAP": {"min": 0.0, "max": 1.0, "median": 0.0, "step": 1.0},
        "Time to CT": {"min": 0.191388889797963, "max": 71.7697222211282, "median": 6.55583333439426, "step": 0.1},
    },
}


def format_class_label(value: Any) -> str:
    try:
        ivalue = int(value)
    except Exception:
        return str(value)
    return CLASS_LABELS.get(ivalue, f"Class {ivalue}")


def read_metadata(model_choice: str) -> dict[str, Any]:
    config = MODEL_CONFIG[model_choice]
    metadata_path = config["model_dir"] / config["metadata_file"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_model_step_with_model_path(model: Any) -> tuple[str | None, Any]:
    steps = getattr(model, "named_steps", {})
    for step_name, step in steps.items():
        if not hasattr(step, "get_params"):
            continue
        try:
            step_params = step.get_params(deep=False)
        except Exception:
            continue
        if "model_path" in step_params:
            return step_name, step
    return None, None


def _try_set_tabpfn_model_path(model: Any, ckpt_path: Path) -> str | None:
    """
    Reset TabPFN's model_path after loading a migrated pkl.
    This avoids using the original absolute Windows path saved inside the model.
    """
    ckpt_path = ckpt_path.resolve()
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(ckpt_path.parent)

    step_name, _ = _find_model_step_with_model_path(model)
    if step_name is not None:
        try:
            model.set_params(**{f"{step_name}__model_path": str(ckpt_path)})
            return str(ckpt_path)
        except Exception:
            pass

    if hasattr(model, "get_params"):
        try:
            top_params = model.get_params(deep=False)
            if "model_path" in top_params:
                model.set_params(model_path=str(ckpt_path))
                return str(ckpt_path)
        except Exception:
            pass

    return None


@st.cache_resource
def load_model_and_metadata(model_choice: str) -> tuple[Any, dict[str, Any], Path, Path]:
    metadata = read_metadata(model_choice)
    model_dir = MODEL_CONFIG[model_choice]["model_dir"]

    model_file = metadata["fitted_model_filename"]
    ckpt_file = metadata["ckpt_filename"]
    model_path = model_dir / model_file
    ckpt_path = model_dir / ckpt_file

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"CKPT file not found: {ckpt_path}")

    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(model_dir.resolve())
    model = joblib.load(model_path)
    _try_set_tabpfn_model_path(model, ckpt_path)
    return model, metadata, model_path, ckpt_path


def get_feature_stats(model_choice: str, feature: str) -> dict[str, float]:
    try:
        return FEATURE_STATS[model_choice][feature]
    except KeyError as exc:
        raise KeyError(f"Missing feature stats for {model_choice} / {feature}") from exc


def predict_with_model(model: Any, input_df: pd.DataFrame) -> dict[str, Any]:
    classes = None
    proba_vector = None

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(input_df))
        if proba.ndim == 2 and proba.shape[0] == 1:
            proba_vector = proba[0].astype(float)
            classes = getattr(model, "classes_", np.arange(proba_vector.shape[0]))

    pred = model.predict(input_df)[0]

    result = {
        "prediction": pred,
        "classes": classes,
        "proba_vector": proba_vector,
        "prob_class_1": None,
        "prob_class_0": None,
    }

    if proba_vector is not None and classes is not None:
        class_list = list(classes)
        if 1 in class_list:
            result["prob_class_1"] = float(proba_vector[class_list.index(1)])
        if 0 in class_list:
            result["prob_class_0"] = float(proba_vector[class_list.index(0)])
        if result["prob_class_1"] is None and result["prob_class_0"] is None and proba_vector.shape[0] == 2:
            result["prob_class_0"] = float(proba_vector[0])
            result["prob_class_1"] = float(proba_vector[1])

    return result


def render_sidebar(model_choice: str, metadata: dict[str, Any]) -> None:
    st.sidebar.title("Model Settings")
    st.sidebar.markdown(f"**System:** {APP_TITLE}")
    st.sidebar.markdown(f"**Selected model:** {model_choice}")
    st.sidebar.markdown(f"**Model file:** `{metadata['fitted_model_filename']}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Variable Definitions")
    for f in metadata["feature_names"]:
        st.sidebar.markdown(f"- **{f}**: {FEATURE_DESCRIPTIONS.get(f, f)}")


def render_input_widget(model_choice: str, feature: str) -> float:
    stat = get_feature_stats(model_choice, feature)
    label = f"{feature} ({FEATURE_DESCRIPTIONS.get(feature, feature)})"

    if feature in BINARY_FEATURES:
        default_value = int(round(float(stat["median"])))
        default_index = 1 if default_value == 1 else 0
        selected = st.selectbox(
            label=label,
            options=["No", "Yes"],
            index=default_index,
            key=f"{model_choice}_{feature}",
        )
        return float(1 if selected == "Yes" else 0)

    step = float(stat["step"])
    number_format = "%.0f" if step.is_integer() else "%.4f"
    return float(
        st.number_input(
            label=label,
            min_value=float(stat["min"]),
            max_value=float(stat["max"]),
            value=float(stat["median"]),
            step=step,
            format=number_format,
            key=f"{model_choice}_{feature}",
        )
    )


def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    model_choice = st.radio("Sex", options=["Male", "Female"], horizontal=True)

    try:
        model, metadata, model_path, ckpt_path = load_model_and_metadata(model_choice)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.info(
            "Please check that the model folder contains metadata.json, the fitted .pkl model, "
            "and the matching .ckpt file."
        )
        return

    features = list(metadata["feature_names"])
    render_sidebar(model_choice, metadata)

    st.subheader("Input Features")
    st.write("Enter patient features below. Binary variables use No/Yes selection.")

    user_inputs: dict[str, float] = {}
    with st.form("prediction_form"):
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                user_inputs[feature] = render_input_widget(model_choice, feature)

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if not submitted:
        return

    input_df = pd.DataFrame([user_inputs], columns=features)
    for c in input_df.columns:
        input_df[c] = pd.to_numeric(input_df[c], errors="coerce")

    if input_df.isnull().any().any():
        st.error("Invalid numeric input detected. Please check all fields.")
        return

    try:
        result = predict_with_model(model, input_df)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write("Input data sent to the model:")
        st.dataframe(input_df, use_container_width=True)
        return

    st.subheader("Prediction Result")
    st.metric("Predicted Outcome", format_class_label(result["prediction"]))

    if result["prob_class_0"] is not None and result["prob_class_1"] is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Probability of Favorable outcome**")
            st.progress(float(result["prob_class_0"]))
            st.write(f"{result['prob_class_0']:.2%}")
        with c2:
            st.markdown("**Probability of Poor outcome**")
            st.progress(float(result["prob_class_1"]))
            st.write(f"{result['prob_class_1']:.2%}")
    elif result["proba_vector"] is not None and result["classes"] is not None:
        proba_df = pd.DataFrame(
            {
                "Outcome": [format_class_label(c) for c in result["classes"]],
                "Probability": result["proba_vector"],
            }
        )
        st.markdown("**Outcome Probabilities**")
        st.dataframe(proba_df, use_container_width=True)
    else:
        st.warning("Model does not expose predict_proba; only class prediction is shown.")

    with st.expander("Input values used for prediction"):
        st.dataframe(input_df, use_container_width=True)


if __name__ == "__main__":
    main()
