"""
Fairness Mitigation Module.
Implements Pre-processing (Sample Reweighing), In-processing (Exponentiated Gradient),
and Post-processing (Threshold Optimizer) to mitigate demographic and equalized odds disparities.
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer

# Silence loky warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def compute_sample_weights(
    y: np.ndarray,
    sensitive_features: np.ndarray,
) -> np.ndarray:
    """
    Computes Kamiran & Calders reweighing weights:
    w(a, y) = [P(A=a) * P(Y=y)] / P(A=a, Y=y)
    """
    y = np.asarray(y)
    a_arr = np.asarray(sensitive_features)
    n = len(y)
    weights = np.ones(n, dtype=float)

    unique_a = np.unique(a_arr)
    unique_y = np.unique(y)

    for a_val in unique_a:
        p_a = np.mean(a_arr == a_val)
        for y_val in unique_y:
            p_y = np.mean(y == y_val)
            p_ay = np.mean((a_arr == a_val) & (y == y_val))
            if p_ay > 0:
                expected_joint = p_a * p_y
                weight = expected_joint / p_ay
            else:
                weight = 1.0

            mask = (a_arr == a_val) & (y == y_val)
            weights[mask] = weight

    # Normalize weights so sum equals n
    weights = weights * (n / np.sum(weights))
    return weights


def train_reweighted_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    model_params: Optional[Dict[str, Any]] = None,
) -> Tuple[LGBMClassifier, np.ndarray]:
    """
    Pre-processing Mitigation: Trains LightGBM with sample reweighing.
    """
    weights = compute_sample_weights(y_train, sensitive_train)
    params = {
        "n_estimators": 120,
        "learning_rate": 0.08,
        "max_depth": 6,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1,
    }
    if model_params:
        params.update(model_params)

    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train, sample_weight=weights)
    return clf, weights


def train_inprocessing_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    constraint_type: str = "equalized_odds",
    eps: float = 0.015,
    max_iter: int = 15,
) -> ExponentiatedGradient:
    """
    In-processing Mitigation: Trains Exponentiated Gradient reduction on LightGBM.
    """
    if constraint_type.lower() == "equalized_odds":
        constraint = EqualizedOdds()
    elif constraint_type.lower() == "demographic_parity":
        constraint = DemographicParity()
    else:
        raise ValueError(f"Unknown constraint: {constraint_type}")

    base_estimator = LGBMClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        verbose=-1,
    )

    mitigator = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=eps,
        max_iter=max_iter,
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator


def train_postprocessing_model(
    base_model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    constraint: str = "equalized_odds",
) -> ThresholdOptimizer:
    """
    Post-processing Mitigation: Fits group-specific decision thresholds using ThresholdOptimizer.
    """
    postprocessor = ThresholdOptimizer(
        estimator=base_model,
        constraints=constraint,
        prefit=True,
        predict_method="predict_proba",
    )
    postprocessor.fit(X_train, y_train, sensitive_features=sensitive_train)
    return postprocessor


def run_comprehensive_mitigation_benchmark(
    data: Dict[str, Any],
    base_model: Any,
    base_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Trains and benchmarks Baseline vs Pre-processing vs In-processing vs Post-processing models.
    Returns comparison tables and delta metrics.
    """
    from src.fairness_audit import audit_sensitive_attribute

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    A_tr_sex = data["A_train"]["sex_name"]
    A_te_sex = data["A_test"]["sex_name"]

    models_dict = {}
    preds_dict = {}
    probs_dict = {}

    # 1. Baseline
    models_dict["Baseline (Unmitigated)"] = base_model
    preds_dict["Baseline (Unmitigated)"] = base_eval["y_pred"]
    probs_dict["Baseline (Unmitigated)"] = base_eval["y_prob"]

    # 2. Pre-processing: Reweighing
    print("Training Pre-processing (Reweighing) model...")
    reweighted_model, sample_weights = train_reweighted_model(X_train, y_train, A_tr_sex.values)
    models_dict["Pre-processing (Reweighing)"] = reweighted_model
    preds_dict["Pre-processing (Reweighing)"] = reweighted_model.predict(X_test)
    probs_dict["Pre-processing (Reweighing)"] = reweighted_model.predict_proba(X_test)[:, 1]

    # 3. In-processing: Exponentiated Gradient
    print("Training In-processing (Exponentiated Gradient) model...")
    inproc_model = train_inprocessing_model(X_train, y_train, A_tr_sex.values, constraint_type="equalized_odds")
    models_dict["In-processing (ExpGrad)"] = inproc_model
    preds_dict["In-processing (ExpGrad)"] = inproc_model.predict(X_test)
    probs_dict["In-processing (ExpGrad)"] = None  # Random ensemble produces hard predictions

    # 4. Post-processing: Threshold Optimizer
    print("Fitting Post-processing (Threshold Optimizer) model...")
    postproc_model = train_postprocessing_model(base_model, X_train, y_train, A_tr_sex.values, constraint="equalized_odds")
    models_dict["Post-processing (ThresholdOpt)"] = postproc_model
    preds_dict["Post-processing (ThresholdOpt)"] = postproc_model.predict(X_test, sensitive_features=A_te_sex.values)
    probs_dict["Post-processing (ThresholdOpt)"] = None

    # Benchmark metrics compilation
    records = []
    audits = {}

    for name, y_pred in preds_dict.items():
        y_prob = probs_dict[name]
        audit = audit_sensitive_attribute(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_group=A_te_sex,
            y_prob=y_prob,
            attribute_name="Sex",
            privileged_group="Male",
            unprivileged_group="Female",
        )
        audits[name] = audit

        disp = audit["disparity_summary"]
        pw = disp["pairwise"]

        acc = float(accuracy_score(y_test, y_pred))
        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        records.append({
            "Model": name,
            "Accuracy": acc,
            "Balanced_Accuracy": bal_acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Selection_Rate_Male": pw.get("selection_rate_priv", np.nan),
            "Selection_Rate_Female": pw.get("selection_rate_unpriv", np.nan),
            "Demographic_Parity_Diff": disp["demographic_parity_difference"],
            "Disparate_Impact_Ratio": disp["demographic_parity_ratio"],
            "Equal_Opportunity_Diff": pw.get("equal_opportunity_diff", np.nan),
            "Equalized_Odds_Diff": disp["equalized_odds_difference"],
            "Four_Fifths_Violated": disp["disparate_impact_four_fifths_violation"],
        })

    summary_df = pd.DataFrame(records)

    # Compute deltas relative to baseline
    base_row = summary_df.loc[summary_df["Model"] == "Baseline (Unmitigated)"].iloc[0]
    deltas = []
    for _, row in summary_df.iterrows():
        deltas.append({
            "Model": row["Model"],
            "Delta_Accuracy": row["Accuracy"] - base_row["Accuracy"],
            "Delta_F1_Score": row["F1_Score"] - base_row["F1_Score"],
            "Delta_Demographic_Parity_Diff": row["Demographic_Parity_Diff"] - base_row["Demographic_Parity_Diff"],
            "Delta_Disparate_Impact_Ratio": row["Disparate_Impact_Ratio"] - base_row["Disparate_Impact_Ratio"],
            "Delta_Equalized_Odds_Diff": row["Equalized_Odds_Diff"] - base_row["Equalized_Odds_Diff"],
        })
    delta_df = pd.DataFrame(deltas)

    return {
        "models": models_dict,
        "predictions": preds_dict,
        "probabilities": probs_dict,
        "summary_df": summary_df,
        "delta_df": delta_df,
        "audits": audits,
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_adult_dataset
    from src.model import train_model, evaluate_model

    data = load_adult_dataset()
    base_model = train_model(data["X_train"], data["y_train"])
    base_eval = evaluate_model(base_model, data["X_test"], data["y_test"])

    res = run_comprehensive_mitigation_benchmark(data, base_model, base_eval)
    print("\n=== MITIGATION BENCHMARK SUMMARY ===")
    print(res["summary_df"].to_string(index=False))
    print("\n=== MITIGATION DELTAS VS BASELINE ===")
    print(res["delta_df"].to_string(index=False))
