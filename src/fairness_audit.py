"""
Fairness Audit Module.
Calculates disaggregated group metrics, demographic parity, equalized odds,
disparate impact (4/5ths rule), predictive parity, and intersectional audits.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    true_positive_rate,
    false_positive_rate,
)

# Silence loky warning on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def calculate_group_confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes TPR, FPR, TNR, FNR, PPV, NPV from confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "TPR": float(tpr),
        "FPR": float(fpr),
        "TNR": float(tnr),
        "FNR": float(fnr),
        "PPV": float(ppv),
        "NPV": float(npv),
    }


def audit_sensitive_attribute(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_group: Union[pd.Series, np.ndarray],
    y_prob: Optional[np.ndarray] = None,
    attribute_name: str = "Sensitive Attribute",
    privileged_group: Optional[Any] = None,
    unprivileged_group: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Performs a full fairness audit for a given sensitive attribute.

    Returns:
        Dict containing:
            - disaggregated_df: Table of performance & selection metrics per group
            - disparity_metrics: Demographic parity, Equalized odds, Disparate impact, etc.
            - fairlearn_frame: Fairlearn MetricFrame object
    """
    sensitive_group = pd.Series(sensitive_group).reset_index(drop=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_groups = sensitive_group.unique()
    group_rows = []

    for grp in unique_groups:
        mask = (sensitive_group == grp).values
        n_group = int(np.sum(mask))
        if n_group == 0:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]
        conf = calculate_group_confusion_rates(y_t, y_p)

        base_rate = float(np.mean(y_t))
        sel_rate = float(np.mean(y_p))
        acc = float(accuracy_score(y_t, y_p))
        bal_acc = float(balanced_accuracy_score(y_t, y_p))
        prec = float(precision_score(y_t, y_p, zero_division=0))
        rec = float(recall_score(y_t, y_p, zero_division=0))
        f1 = float(f1_score(y_t, y_p, zero_division=0))

        row = {
            attribute_name: grp,
            "Count": n_group,
            "Group_Share": n_group / len(y_true),
            "Base_Rate": base_rate,
            "Selection_Rate": sel_rate,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Precision": prec,
            "Recall_TPR": rec,
            "FPR": conf["FPR"],
            "FNR": conf["FNR"],
            "F1_Score": f1,
        }

        if y_prob is not None:
            y_pr = y_prob[mask]
            # ROC AUC only valid if both classes are present in group
            if len(np.unique(y_t)) > 1:
                row["ROC_AUC"] = float(roc_auc_score(y_t, y_pr))
            else:
                row["ROC_AUC"] = np.nan

        group_rows.append(row)

    disaggregated_df = pd.DataFrame(group_rows)

    # Use Fairlearn MetricFrame for standard calculations
    metrics_dict = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
    }
    metric_frame = MetricFrame(
        metrics=metrics_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_group,
    )

    dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_group))
    dp_ratio = float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive_group))
    eo_diff = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_group))
    eo_ratio = float(equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_group))

    # Binary pairwise disparity calculations if privileged / unprivileged specified
    pairwise_disparities = {}
    if privileged_group is not None and unprivileged_group is not None:
        p_mask = (sensitive_group == privileged_group).values
        u_mask = (sensitive_group == unprivileged_group).values

        if np.sum(p_mask) > 0 and np.sum(u_mask) > 0:
            p_sel = float(np.mean(y_pred[p_mask]))
            u_sel = float(np.mean(y_pred[u_mask]))
            p_conf = calculate_group_confusion_rates(y_true[p_mask], y_pred[p_mask])
            u_conf = calculate_group_confusion_rates(y_true[u_mask], y_pred[u_mask])

            pairwise_disparities = {
                "privileged_group": privileged_group,
                "unprivileged_group": unprivileged_group,
                "selection_rate_priv": p_sel,
                "selection_rate_unpriv": u_sel,
                "disparate_impact_ratio": u_sel / p_sel if p_sel > 0 else np.nan,
                "demographic_parity_diff": p_sel - u_sel,
                "tpr_priv": p_conf["TPR"],
                "tpr_unpriv": u_conf["TPR"],
                "equal_opportunity_diff": p_conf["TPR"] - u_conf["TPR"],
                "fpr_priv": p_conf["FPR"],
                "fpr_unpriv": u_conf["FPR"],
                "predictive_equality_diff": p_conf["FPR"] - u_conf["FPR"],
                "equalized_odds_diff": max(
                    abs(p_conf["TPR"] - u_conf["TPR"]),
                    abs(p_conf["FPR"] - u_conf["FPR"]),
                ),
                "accuracy_diff": float(accuracy_score(y_true[p_mask], y_pred[p_mask]))
                - float(accuracy_score(y_true[u_mask], y_pred[u_mask])),
            }

    disparity_summary = {
        "demographic_parity_difference": dp_diff,
        "demographic_parity_ratio": dp_ratio,
        "disparate_impact_four_fifths_violation": bool(dp_ratio < 0.80),
        "equalized_odds_difference": eo_diff,
        "equalized_odds_ratio": eo_ratio,
        "pairwise": pairwise_disparities,
    }

    return {
        "attribute_name": attribute_name,
        "disaggregated_df": disaggregated_df,
        "disparity_summary": disparity_summary,
        "metric_frame": metric_frame,
    }


def audit_model_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    A_df: pd.DataFrame,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Performs full multi-attribute fairness audit: Sex, Race, and Intersectional (Sex x Race).
    """
    audit_sex = audit_sensitive_attribute(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_group=A_df["sex_name"],
        y_prob=y_prob,
        attribute_name="Sex",
        privileged_group="Male",
        unprivileged_group="Female",
    )

    audit_race = audit_sensitive_attribute(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_group=A_df["race_name"],
        y_prob=y_prob,
        attribute_name="Race",
        privileged_group="White",
        unprivileged_group="Black",
    )

    audit_intersection = audit_sensitive_attribute(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_group=A_df["intersection"],
        y_prob=y_prob,
        attribute_name="Sex_x_Race",
    )

    return {
        "sex_audit": audit_sex,
        "race_audit": audit_race,
        "intersection_audit": audit_intersection,
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_adult_dataset
    from src.model import train_model, evaluate_model

    data = load_adult_dataset()
    model = train_model(data["X_train"], data["y_train"])
    eval_res = evaluate_model(model, data["X_test"], data["y_test"])

    audit_res = audit_model_fairness(
        y_true=data["y_test"],
        y_pred=eval_res["y_pred"],
        A_df=data["A_test"],
        y_prob=eval_res["y_prob"],
    )

    print("\n--- SEX DISAGGREGATED AUDIT ---")
    print(audit_res["sex_audit"]["disaggregated_df"].to_string(index=False))
    print("\n--- SEX FAIRNESS DISPARITY SUMMARY ---")
    for k, v in audit_res["sex_audit"]["disparity_summary"].items():
        if k != "pairwise":
            print(f"  {k}: {v}")
    print("  Pairwise details:", audit_res["sex_audit"]["disparity_summary"]["pairwise"])
