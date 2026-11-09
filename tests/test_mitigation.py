"""
Unit tests for bias mitigation techniques.
"""

import pytest
import numpy as np
import pandas as pd
from src.mitigation import compute_sample_weights, train_reweighted_model


def test_compute_sample_weights():
    # 4 balanced combinations
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0])

    weights = compute_sample_weights(y, a)
    assert len(weights) == len(y)
    assert np.all(weights > 0)
    # Sum of weights should equal sample size
    assert pytest.approx(np.sum(weights), 0.01) == len(y)


def test_train_reweighted_model():
    # Small synthetic dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = np.random.randint(0, 2, size=100)
    a = np.random.randint(0, 2, size=100)

    clf, weights = train_reweighted_model(X, y, a, model_params={"n_estimators": 10})
    assert clf is not None
    assert len(weights) == 100
    preds = clf.predict(X)
    assert len(preds) == 100
