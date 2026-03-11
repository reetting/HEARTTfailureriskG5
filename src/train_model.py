
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score
from src.data_processing import load_data, handle_outliers, optimize_memory
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score

MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, eval_metric="logloss", random_state=42
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, verbose=-1
    ),
}


def train_all_models(X_train, y_train):
    trained = {}
    for name, model in MODELS.items():
        print(f"Entraînement : {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def evaluate_all_models(trained_models: dict, X_test, y_test) -> None:
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(
            model, X_test, y_test, cv=kf, scoring='roc_auc'
        )

        print(f"\n{'='*35}")
        print(f"  Modèle : {name}")
        print(f"{'='*35}")
        print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob):.3f}")
        print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
        print(f"  Precision : {precision_score(y_test, y_pred):.3f}")
        print(f"  Recall    : {recall_score(y_test, y_pred):.3f}")
        print(f"  F1-Score  : {f1_score(y_test, y_pred):.3f}")
        print(f"  CV-Score  : {cv_scores.mean():.3f} ± {cv_scores.std():.3f} ➕")
