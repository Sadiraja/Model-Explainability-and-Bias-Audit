"""
Unit tests for model training and evaluation module.
"""

import pytest
import numpy as np
import pandas as pd
from src.model import build_classifier, train_model, evaluate_predictions, evaluate_model


def test_build_classifier():
    clf = build_classifier("lightgbm", n_estimators=10)
    assert clf is not None
    assert clf.n_estimators == 10

    rf = build_classifier("random_forest", n_estimators=10)
    assert rf is not None
    assert rf.n_estimators == 10

    with pytest.raises(ValueError):
        build_classifier("unsupported_model")


def test_evaluate_predictions():
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
