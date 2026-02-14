import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import shap
except Exception:
    shap = None

st.set_page_config(
    page_title="Sex-specific ICH Prognosis Prediction Model",
    page_icon="ICH",
    layout="wide",
)

APP_TITLE = "Sex-specific ICH Prognosis Prediction Model"
APP_SUBTITLE = "Switch between male and female models and run prognosis prediction."
CLASS_LABELS = {0: "Favorable outcome", 1: "Poor outcome"}

MALE_FEATURES = ["Age", "NIHSS", "GCS", "Hematoma volume", "WBC", "SIRI"]
FEMALE_FEATURES = ["BUN", "Age", "NIHSS", "GCS", "Hematoma volume", "Time to CT"]

FEATURE_DESCRIPTIONS = {
    "Age": "Age",
    "NIHSS": "NIHSS score",
    "GCS": "GCS score",
    "Hematoma volume": "Hematoma volume",
    "WBC": "White blood cell count",
    "SIRI": "Systemic inflammatory response index",
    "BUN": "Blood urea nitrogen",
    "Time to CT": "Time from symptom onset to CT scan",
}

MODEL_CONFIG = {
    "Male": {
        "model_file": "tabpfn_model_male.pkl",
        "features": MALE_FEATURES,
    },
    "Female": {
        "model_file": "tabpfn_model_female.pkl",
        "features": FEMALE_FEATURES,
    },
}

BASE_DIR = Path(__file__).resolve().parent

HARD_CODED_FEATURE_STATS = {
    "Male": {
        "Age": {"min": 23.0, "max": 91.0, "median": 61.0, "step": 1.0},
        "NIHSS": {"min": 0.0, "max": 40.0, "median": 8.0, "step": 1.0},
        "GCS": {"min": 3.0, "max": 15.0, "median": 14.0, "step": 1.0},
        "Hematoma volume": {"min": 0.16, "max": 147.49, "median": 12.8125, "step": 0.1},
        "WBC": {"min": 3.56, "max": 25.9, "median": 9.0, "step": 0.1},
        "SIRI": {
            "min": 0.2580357180156658,
            "max": 46.52056799999999,
            "median": 2.545434663432016,
            "step": 0.01,
        },
    },
    "Female": {
        "BUN": {"min": 1.9, "max": 18.2, "median": 5.0, "step": 0.1},
        "Age": {"min": 21.0, "max": 91.0, "median": 64.0, "step": 1.0},
        "NIHSS": {"min": 0.0, "max": 40.0, "median": 11.0, "step": 1.0},
        "GCS": {"min": 3.0, "max": 15.0, "median": 13.0, "step": 1.0},
        "Hematoma volume": {"min": 0.164, "max": 111.33, "median": 12.2235, "step": 0.1},
        "Time to CT": {
            "min": 0.283333332510665,
            "max": 71.835000000021,
            "median": 3.313611110381315,
            "step": 0.1,
        },
    },
}


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


def _try_set_tabpfn_model_path(model: Any) -> str | None:
    # Use a local cache dir by default; can be overridden by TABPFN_MODEL_DIR.
    tabpfn_dir = os.getenv("TABPFN_MODEL_DIR", str(BASE_DIR / "tabpfn_models"))
    os.makedirs(tabpfn_dir, exist_ok=True)

    step_name, _ = _find_model_step_with_model_path(model)
    if step_name is not None:
        model.set_params(**{f"{step_name}__model_path": tabpfn_dir})
        return tabpfn_dir

    if hasattr(model, "get_params"):
        try:
            top_params = model.get_params(deep=False)
            if "model_path" in top_params:
                model.set_params(model_path=tabpfn_dir)
                return tabpfn_dir
        except Exception:
            pass
    return None


@st.cache_resource
def load_model(model_file: str) -> tuple[Any, str | None]:
    model_path = BASE_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    effective_model_dir = _try_set_tabpfn_model_path(model)
    return model, effective_model_dir


def get_feature_stats(model_choice: str, feature: str) -> dict[str, float]:
    return HARD_CODED_FEATURE_STATS[model_choice][feature]


@st.cache_data
def build_background_data(model_choice: str, features: list[str], n_samples: int = 80) -> pd.DataFrame:
    stats = HARD_CODED_FEATURE_STATS[model_choice]
    rng = np.random.default_rng(42 if model_choice == "Male" else 43)
    data: dict[str, np.ndarray] = {}
    for f in features:
        low = float(stats[f]["min"])
        high = float(stats[f]["max"])
        median = float(stats[f]["median"])
        if high <= low:
            vals = np.full(n_samples, median, dtype=float)
        else:
            vals = rng.uniform(low, high, size=n_samples).astype(float)
            vals[0] = median
            if n_samples > 1:
                vals[1] = low
            if n_samples > 2:
                vals[2] = high
        data[f] = vals
    return pd.DataFrame(data)


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
            idx_1 = class_list.index(1)
            result["prob_class_1"] = float(proba_vector[idx_1])
        if 0 in class_list:
            idx_0 = class_list.index(0)
            result["prob_class_0"] = float(proba_vector[idx_0])
        if (
            result["prob_class_1"] is None
            and result["prob_class_0"] is None
            and proba_vector.shape[0] == 2
        ):
            result["prob_class_0"] = float(proba_vector[0])
            result["prob_class_1"] = float(proba_vector[1])

    return result


def format_class_label(value: Any) -> str:
    try:
        ivalue = int(value)
    except Exception:
        return str(value)
    return CLASS_LABELS.get(ivalue, f"Class {ivalue}")


def _predict_positive_class_probability(model: Any, features: list[str], data: Any) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        data_df = data.copy()
    else:
        data_df = pd.DataFrame(data, columns=features)

    proba = np.asarray(model.predict_proba(data_df))
    if proba.ndim == 1:
        return proba.astype(float)

    classes = getattr(model, "classes_", None)
    if classes is not None:
        class_list = list(classes)
        if 1 in class_list:
            return proba[:, class_list.index(1)].astype(float)

    if proba.shape[1] >= 2:
        return proba[:, 1].astype(float)
    return proba[:, 0].astype(float)


def render_shap_force_plot(
    model: Any,
    background_df: pd.DataFrame,
    features: list[str],
    input_df: pd.DataFrame,
) -> None:
    st.subheader("SHAP Force Plot")
    if shap is None:
        st.warning("SHAP is not installed in the current environment.")
        return

    if not hasattr(model, "predict_proba"):
        st.warning("Current model does not support `predict_proba`, so SHAP force plot is unavailable.")
        return

    background_df = background_df[features].dropna().copy()
    if background_df.empty:
        background_df = input_df.copy()

    predict_fn = lambda x: _predict_positive_class_probability(model, features, x)
    try:
        with st.spinner("Computing SHAP force plot..."):
            explainer = shap.KernelExplainer(predict_fn, background_df)
            shap_values = explainer.shap_values(input_df, nsamples=120)

        if isinstance(shap_values, list):
            shap_row = np.asarray(shap_values[0])[0]
        else:
            shap_row = np.asarray(shap_values)[0]

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(np.asarray(expected_value).reshape(-1)[0])
        else:
            expected_value = float(expected_value)

        # Force JS-based rendering; do not use matplotlib backend on cloud.
        try:
            force_plot = shap.force_plot(
                expected_value,
                shap_row,
                input_df.iloc[0],
                feature_names=features,
                matplotlib=False,
                show=False,
            )
        except TypeError:
            # Compatibility with older/newer SHAP signatures.
            force_plot = shap.force_plot(
                expected_value,
                shap_row,
                input_df.iloc[0],
                feature_names=features,
                matplotlib=False,
            )

        if hasattr(force_plot, "html"):
            html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(html, height=420, scrolling=False)
        else:
            st.warning("SHAP force plot object has no HTML renderer in this environment.")
    except Exception as exc:
        st.warning(f"Failed to generate SHAP force plot: {exc}")


def render_sidebar(model_choice: str, config: dict[str, Any]) -> None:
    st.sidebar.title("Model Settings")
    st.sidebar.markdown(f"**System:** {APP_TITLE}")
    st.sidebar.markdown(f"**Selected model:** {model_choice}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Variable Definitions")
    for f in config["features"]:
        st.sidebar.markdown(f"- **{f}**: {FEATURE_DESCRIPTIONS.get(f, f)}")


def main() -> None:
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    model_choice = st.radio("Sex", options=["Male", "Female"], horizontal=True)
    config = MODEL_CONFIG[model_choice]
    features = config["features"]

    try:
        model, tabpfn_dir = load_model(config["model_file"])
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.info(
            "If you see `No module named tabpfn`, install the required runtime "
            "dependencies for this model first."
        )
        return

    render_sidebar(model_choice, config)
    st.subheader("Input Features")
    st.write("Enter patient features below. The hints use built-in min/max ranges.")

    stats = {f: get_feature_stats(model_choice, f) for f in features}
    user_inputs: dict[str, float] = {}

    with st.form("prediction_form"):
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            stat = stats[feature]
            with cols[idx % 2]:
                user_inputs[feature] = st.number_input(
                    label=f"{feature} ({FEATURE_DESCRIPTIONS.get(feature, feature)})",
                    value=float(stat["median"]),
                    step=float(stat["step"]),
                    format="%.4f",
                    key=f"{model_choice}_{feature}",
                )

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
        st.warning("Model does not expose `predict_proba`; only class prediction is shown.")

    background_df = build_background_data(model_choice, features)
    render_shap_force_plot(model, background_df, features, input_df)


if __name__ == "__main__":
    main()
