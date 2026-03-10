"""
SHAP Explainability Module
Generates SHAP values and visualizations for heart failure prediction model.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import os


def get_shap_explainer(model, X_train: np.ndarray):
    """
    Create the appropriate SHAP explainer based on model type.
    - TreeExplainer for tree-based models (RF, XGBoost, LightGBM)
    - LinearExplainer for Logistic Regression
    """
    model_name = type(model).__name__.lower()

    if any(name in model_name for name in ["forest", "xgb", "lgbm", "gradient", "tree"]):
        explainer = shap.TreeExplainer(model)
    else:
        # Logistic Regression or other linear models
        explainer = shap.LinearExplainer(model, X_train)

    return explainer


def compute_shap_values(explainer, X: np.ndarray):
    """
    Compute SHAP values for a given dataset.
    Returns raw shap_values array.
    """
    shap_values = explainer.shap_values(X)

    # For binary classification, tree models return a list [class0, class1]
    # We want class 1 (positive = heart failure death)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def plot_summary(shap_values, X, feature_names: list, save_path: str = None):
    """
    SHAP Summary Plot (Beeswarm) — shows feature importance and impact direction.
    """
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        plot_size=(10, 6)
    )
    plt.title("SHAP Summary Plot — Feature Impact on Heart Failure Prediction", pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Summary plot saved → {save_path}")

    plt.close()


def plot_bar_importance(shap_values, feature_names: list, save_path: str = None):
    """
    SHAP Bar Plot — global feature importance (mean |SHAP value|).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color="#c0392b"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP Value| (average impact on model output)")
    ax.set_title("Global Feature Importance — SHAP")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Bar importance plot saved → {save_path}")

    plt.close()


def plot_waterfall_single(explainer, X_single, feature_names: list, save_path: str = None):
    """
    SHAP Waterfall Plot for a single patient prediction.
    Shows which features push the prediction higher or lower.

    X_single : 1D array or DataFrame row (single patient)
    """
    # shap.Explanation object needed for waterfall
    shap_explanation = explainer(X_single)

    # Handle binary classification list output
    if shap_explanation.values.ndim == 3:
        # shape (1, n_features, n_classes) → take class 1
        vals = shap_explanation.values[0, :, 1]
        base = shap_explanation.base_values[0, 1]
    elif shap_explanation.values.ndim == 2 and shap_explanation.values.shape[-1] == 2:
        vals = shap_explanation.values[0, 1] if shap_explanation.values.ndim == 2 else shap_explanation.values[:, 1]
        base = shap_explanation.base_values[0] if np.ndim(shap_explanation.base_values) > 1 else shap_explanation.base_values
    else:
        vals = shap_explanation.values[0] if shap_explanation.values.ndim > 1 else shap_explanation.values
        base = shap_explanation.base_values[0] if np.ndim(shap_explanation.base_values) > 0 else shap_explanation.base_values

    explanation = shap.Explanation(
        values=vals,
        base_values=float(np.mean(base)) if np.ndim(base) > 0 else float(base),
        data=X_single.values.flatten() if hasattr(X_single, "values") else np.array(X_single).flatten(),
        feature_names=feature_names
    )

    plt.figure()
    shap.waterfall_plot(explanation, show=False, max_display=12)
    plt.title("SHAP Waterfall — Individual Patient Explanation", pad=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SHAP] Waterfall plot saved → {save_path}")

    plt.close()


def get_top_features(shap_values, feature_names: list, top_n: int = 5) -> list:
    """
    Returns the top N most important features based on mean |SHAP value|.
    Useful for displaying insights in the Streamlit interface.

    Returns: list of (feature_name, mean_abs_shap) tuples, sorted descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    return [(feature_names[i], round(float(mean_abs[i]), 4)) for i in sorted_idx[:top_n]]


def explain_patient(model, explainer, shap_values_train, X_patient,
                    feature_names: list, save_dir: str = "outputs/shap"):
    """
    Full explanation pipeline for a single patient:
    1. Compute prediction probability
    2. Generate waterfall plot
    3. Return top contributing features

    Returns dict with: probability, top_features, waterfall_path
    """
    prob = model.predict_proba(X_patient)[0][1]

    waterfall_path = os.path.join(save_dir, "patient_waterfall.png")
    plot_waterfall_single(explainer, X_patient, feature_names, save_path=waterfall_path)

    # Per-patient SHAP values
    patient_shap = compute_shap_values(explainer, X_patient)
    if patient_shap.ndim > 1:
        patient_shap = patient_shap[0]

    top = sorted(
        zip(feature_names, patient_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    return {
        "probability": round(float(prob), 4),
        "top_features": [(name, round(float(val), 4)) for name, val in top],
        "waterfall_path": waterfall_path
    }