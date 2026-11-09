"""
Unit tests for data loader module.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_adult_dataset, clean_feature_names


def test_clean_feature_names():
    df = pd.DataFrame(columns=["Age", "Education-Num", "Marital Status", "Hours per week"])
    cleaned = clean_feature_names(df)
    assert list(cleaned.columns) == ["Age", "Education_Num", "Marital_Status", "Hours_per_week"]


def test_load_adult_dataset():
    data = load_adult_dataset()

    assert "X_train" in data
    assert "X_test" in data
    assert "y_train" in data
    assert "y_test" in data
    assert "A_train" in data
    assert "A_test" in data

    # Check shapes and types
    assert len(data["X_train"]) > 20000
    assert len(data["X_test"]) > 5000
    assert len(data["y_train"]) == len(data["X_train"])
    assert len(data["y_test"]) == len(data["X_test"])

    # Check target is binary 0 or 1
    assert set(np.unique(data["y_train"])).issubset({0, 1})

    # Check sensitive attributes presence
    assert "sex_binary" in data["A_train"].columns
    assert "sex_name" in data["A_train"].columns
    assert "race_name" in data["A_train"].columns
    assert "intersection" in data["A_train"].columns
