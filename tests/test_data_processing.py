"""
tests/test_data_processing.py
Tests automatisés pour src/data_processing.py
Exécution : pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing import (
    load_data,
    optimize_memory,
    handle_outliers,
    preprocess,
)

# ── Fixture : DataFrame synthétique identique au dataset UCI ─────────────────
@pytest.fixture
def sample_df():
    """Crée un DataFrame synthétique qui imite heart_failure_clinical_records."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "age":                      np.random.uniform(40, 95, n).astype("float64"),
        "anaemia":                  np.random.randint(0, 2, n).astype("int64"),
        "creatinine_phosphokinase": np.random.uniform(20, 8000, n).astype("float64"),
        "diabetes":                 np.random.randint(0, 2, n).astype("int64"),
        "ejection_fraction":        np.random.uniform(14, 80, n).astype("float64"),
        "high_blood_pressure":      np.random.randint(0, 2, n).astype("int64"),
        "platelets":                np.random.uniform(25000, 850000, n).astype("float64"),
        "serum_creatinine":         np.random.uniform(0.5, 9.4, n).astype("float64"),
        "serum_sodium":             np.random.uniform(113, 148, n).astype("float64"),
        "sex":                      np.random.randint(0, 2, n).astype("int64"),
        "smoking":                  np.random.randint(0, 2, n).astype("int64"),
        "time":                     np.random.randint(4, 285, n).astype("int64"),
        "DEATH_EVENT":              np.random.randint(0, 2, n).astype("int64"),
    })


# ── Tests : load_data ─────────────────────────────────────────────────────────

def test_load_data_returns_dataframe(tmp_path, sample_df):
    """load_data doit retourner un DataFrame non vide."""
    path = str(tmp_path / "test.csv")
    sample_df.to_csv(path, index=False)
    df = load_data(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_load_data_has_correct_columns(tmp_path, sample_df):
    """load_data doit contenir toutes les colonnes attendues."""
    path = str(tmp_path / "test.csv")
    sample_df.to_csv(path, index=False)
    df = load_data(path)
    expected_cols = [
        "age", "anaemia", "creatinine_phosphokinase", "diabetes",
        "ejection_fraction", "high_blood_pressure", "platelets",
        "serum_creatinine", "serum_sodium", "sex", "smoking",
        "time", "DEATH_EVENT"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Colonne manquante : {col}"


# ── Tests : optimize_memory ───────────────────────────────────────────────────

def test_optimize_memory_reduces_memory(sample_df):
    """optimize_memory doit réduire l'usage mémoire."""
    before = sample_df.memory_usage(deep=True).sum()
    df_opt = optimize_memory(sample_df)
    after  = df_opt.memory_usage(deep=True).sum()
    assert after < before, "La mémoire n'a pas été réduite"

def test_optimize_memory_converts_float64_to_float32(sample_df):
    """Les colonnes float64 doivent devenir float32."""
    df_opt = optimize_memory(sample_df)
    float_cols = [c for c in sample_df.columns if sample_df[c].dtype == "float64"]
    for col in float_cols:
        assert df_opt[col].dtype == np.float32, f"{col} n'est pas float32"

def test_optimize_memory_converts_int64_to_int32(sample_df):
    """Les colonnes int64 doivent devenir int32."""
    df_opt = optimize_memory(sample_df)
    int_cols = [c for c in sample_df.columns if sample_df[c].dtype == "int64"]
    for col in int_cols:
        assert df_opt[col].dtype == np.int32, f"{col} n'est pas int32"

def test_optimize_memory_preserves_values(sample_df):
    """optimize_memory ne doit pas modifier les valeurs."""
    df_opt = optimize_memory(sample_df)
    assert df_opt.shape == sample_df.shape
    for col in sample_df.columns:
        assert np.allclose(sample_df[col].values, df_opt[col].values, rtol=1e-3), \
            f"Valeurs modifiées dans la colonne {col}"

def test_optimize_memory_does_not_modify_original(sample_df):
    """optimize_memory ne doit pas modifier le DataFrame original."""
    original_dtypes = sample_df.dtypes.copy()
    optimize_memory(sample_df)
    for col in sample_df.columns:
        assert sample_df[col].dtype == original_dtypes[col], \
            f"Le DataFrame original a été modifié pour {col}"


# ── Tests : valeurs manquantes ────────────────────────────────────────────────

def test_no_missing_values_in_dataset(sample_df):
    """Le dataset ne doit pas contenir de valeurs manquantes."""
    missing = sample_df.isnull().sum().sum()
    assert missing == 0, f"{missing} valeurs manquantes détectées"

def test_optimize_memory_handles_missing_values(sample_df):
    """optimize_memory doit fonctionner même si des NaN sont présents."""
    df_with_nan = sample_df.copy()
    df_with_nan.loc[0, "age"] = np.nan
    try:
        df_opt = optimize_memory(df_with_nan)
        assert df_opt is not None
    except Exception as e:
        pytest.fail(f"optimize_memory a planté avec des NaN : {e}")


# ── Tests : handle_outliers ───────────────────────────────────────────────────

def test_handle_outliers_preserves_shape(sample_df):
    """handle_outliers ne doit pas changer le nombre de lignes/colonnes."""
    df_out = handle_outliers(sample_df.copy())
    assert df_out.shape == sample_df.shape

def test_handle_outliers_does_not_touch_death_event(sample_df):
    """handle_outliers ne doit pas modifier la colonne DEATH_EVENT."""
    df_out = handle_outliers(sample_df.copy())
    pd.testing.assert_series_equal(
        sample_df["DEATH_EVENT"].reset_index(drop=True),
        df_out["DEATH_EVENT"].reset_index(drop=True),
    )

def test_handle_outliers_clips_values(sample_df):
    """Après handle_outliers, aucune valeur ne doit dépasser les bornes IQR."""
    cols  = [c for c in sample_df.columns if c != "DEATH_EVENT"]
    Q1    = sample_df[cols].quantile(0.25)
    Q3    = sample_df[cols].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df_out = handle_outliers(sample_df.copy())
    for col in cols:
        assert df_out[col].min() >= lower[col] - 1e-6, f"Valeur sous borne basse : {col}"
        assert df_out[col].max() <= upper[col] + 1e-6, f"Valeur au-dessus borne haute : {col}"


# ── Tests : preprocess (pipeline complet) ────────────────────────────────────

def test_preprocess_returns_four_splits(tmp_path, sample_df):
    """preprocess doit retourner X_train, X_test, y_train, y_test."""
    path = str(tmp_path / "test.csv")
    sample_df.to_csv(path, index=False)
    result = preprocess(path)
    assert len(result) == 4, "preprocess doit retourner exactement 4 éléments"

def test_preprocess_no_death_event_in_X(tmp_path, sample_df):
    """DEATH_EVENT ne doit pas apparaître dans les features X."""
    path = str(tmp_path / "test.csv")
    sample_df.to_csv(path, index=False)
    X_train, X_test, _, _ = preprocess(path)
    assert "DEATH_EVENT" not in X_train.columns
    assert "DEATH_EVENT" not in X_test.columns

def test_preprocess_balanced_after_smote(tmp_path, sample_df):
    """Après SMOTE, les classes doivent être équilibrées dans y_train."""
    path = str(tmp_path / "test.csv")
    sample_df.to_csv(path, index=False)
    _, _, y_train, _ = preprocess(path)
    counts = pd.Series(y_train).value_counts()
    assert counts[0] == counts[1], "Classes déséquilibrées après SMOTE"