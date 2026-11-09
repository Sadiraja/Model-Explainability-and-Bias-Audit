"""
Unit tests for fairness audit metrics and calculations.
"""

import pytest
import numpy as np
import pandas as pd
from src.fairness_audit import calculate_group_confusion_rates, audit_sensitive_attribute


def test_calculate_group_confusion_rates():
    # 2 TP, 1 FP, 3 TN, 1 FN
    # y_true: [1, 1, 1, 0, 0, 0, 0]
    # y_pred: [1, 1, 0, 1, 0, 0, 0]
    y_true = np.array([1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0])

    rates = calculate_group_confusion_rates(y_true, y_pred)
    assert rates["TP"] == 2
    assert rates["FN"] == 1
    assert rates["FP"] == 1
    assert rates["TN"] == 3
    assert pytest.approx(rates["TPR"], 0.01) == 2 / 3
    assert pytest.approx(rates["FPR"], 0.01) == 1 / 4


def test_audit_sensitive_attribute():
    y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    sensitive = pd.Series(["Male", "Male", "Male", "Male", "Female", "Female", "Female", "Female"])

    audit = audit_sensitive_attribute(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_group=sensitive,
        attribute_name="Sex",
        privileged_group="Male",
        unprivileged_group="Female",
    )

    assert "disaggregated_df" in audit
    assert "disparity_summary" in audit
    assert len(audit["disaggregated_df"]) == 2
    assert "demographic_parity_difference" in audit["disparity_summary"]
    assert "disparate_impact_four_fifths_violation" in audit["disparity_summary"]
