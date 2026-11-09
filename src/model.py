"""
Model Training and Evaluation Module.
Supports LightGBM, Random Forest, and standard evaluation metrics.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)


def build_classifier(
    model_type: str = "lightgbm",
    random_state: int = 42,
    **kwargs,
):
    """
    Constructs a classifier instance.
    """
    if model_type.lower() == "lightgbm":
        params = {
            "n_estimators": 120,
            "learning_rate": 0.08,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": random_state,
            "verbose": -1,
        }
        params.update(kwargs)
        return LGBMClassifier(**params)
    elif model_type.lower() == "random_forest":
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": random_state,
            "n_jobs": -1,
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_type: str = "lightgbm",
    sample_weight: Optional[np.ndarray] = None,
    random_state: int = 42,
    **kwargs,
):
    """
    Fits a classifier on training data, optionally with sample weights.
    """
    clf = build_classifier(model_type=model_type, random_state=random_state, **kwargs)
    if sample_weight is not None:
        clf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)
    return clf


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes a comprehensive suite of binary classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    return metrics


def evaluate_model(
    clf: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on test data, returning predictions, probabilities, and metrics.
    """
    y_pred = clf.predict(X_test)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    return {
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
    }


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_adult_dataset
    data = load_adult_dataset()
    model = train_model(data["X_train"], data["y_train"])
    results = evaluate_model(model, data["X_test"], data["y_test"])
    print("Baseline Model Evaluation:")
    for k, v in results["metrics"].items():
        print(f"  {k}: {v:.4f}")
