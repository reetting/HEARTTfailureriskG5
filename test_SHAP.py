"""
tests/test_shap_explainability.py
Automated tests for the SHAP explainability module.
Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from shap_explainability import (
    get_shap_explainer,
    compute_shap_values,
    get_top_features,
    plot_summary,
    plot_bar_importance,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]

@pytest.fixture(scope="module")
def trained_model_and_data():
    X, y = make_classification(
        n_samples=100, n_features=12, n_informative=6,
        random_state=42
    )
    X = pd.DataFrame(X, columns=FEATURE_NAMES)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_explainer_creation(trained_model_and_data):
    """get_shap_explainer should return a TreeExplainer for RandomForest."""
    import shap
    model, X, _ = trained_model_and_data
    explainer = get_shap_explainer(model, X)
    assert isinstance(explainer, shap.TreeExplainer), \
        "Expected TreeExplainer for RandomForestClassifier"


def test_shap_values_shape(trained_model_and_data):
    """SHAP values array must have shape (n_samples, n_features)."""
    model, X, _ = trained_model_and_data
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)
    assert shap_values.shape == (len(X), X.shape[1]), \
        f"Expected shape {(len(X), X.shape[1])}, got {shap_values.shape}"


def test_shap_values_not_all_zero(trained_model_and_data):
    """SHAP values should not all be zero — model must differentiate features."""
    model, X, _ = trained_model_and_data
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)
    assert np.abs(shap_values).sum() > 0, "All SHAP values are zero — check model/explainer"


def test_get_top_features(trained_model_and_data):
    """get_top_features should return exactly top_n tuples, sorted by importance."""
    model, X, _ = trained_model_and_data
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)

    top5 = get_top_features(shap_values, FEATURE_NAMES, top_n=5)
    assert len(top5) == 5, "Should return exactly 5 features"

    # Verify descending order
    scores = [score for _, score in top5]
    assert scores == sorted(scores, reverse=True), "Features not sorted by importance"

    # Verify all names are valid
    for name, _ in top5:
        assert name in FEATURE_NAMES, f"Unknown feature name: {name}"


def test_plot_summary_saves_file(tmp_path, trained_model_and_data):
    """plot_summary should save a PNG file at the given path."""
    model, X, _ = trained_model_and_data
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)

    out = str(tmp_path / "summary.png")
    plot_summary(shap_values, X, FEATURE_NAMES, save_path=out)
    assert os.path.exists(out), "Summary plot file was not created"
    assert os.path.getsize(out) > 0, "Summary plot file is empty"


def test_plot_bar_importance_saves_file(tmp_path, trained_model_and_data):
    """plot_bar_importance should save a PNG file at the given path."""
    model, X, _ = trained_model_and_data
    explainer   = get_shap_explainer(model, X)
    shap_values = compute_shap_values(explainer, X)

    out = str(tmp_path / "bar.png")
    plot_bar_importance(shap_values, FEATURE_NAMES, save_path=out)
    assert os.path.exists(out), "Bar importance plot file was not created"
    assert os.path.getsize(out) > 0, "Bar importance plot file is empty"