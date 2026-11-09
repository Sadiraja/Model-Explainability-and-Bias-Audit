"""
Main Execution Script for Model Explainability and Bias Audit Pipeline.
Runs end-to-end:
1. Data loading & sensitive group profiling
2. Baseline model training & evaluation
3. Global and local explainability (SHAP & LIME)
4. Fairness audit (Demographic Parity, Equalized Odds, Disparate Impact)
5. Multi-technique bias mitigation benchmark (Pre-, In-, Post-processing)
6. Visualization generation & report compilation
"""

import os
import sys
import time
from typing import Dict, Any
import numpy as np
import pandas as pd
from tabulate import tabulate

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_adult_dataset
from src.model import train_model, evaluate_model
from src.explainability import ModelExplainer
from src.fairness_audit import audit_model_fairness
from src.mitigation import run_comprehensive_mitigation_benchmark
from src.visualizations import generate_all_visualizations


def run_pipeline() -> Dict[str, Any]:
    print("=" * 80)
    print("   MODEL EXPLAINABILITY AND BIAS AUDIT PIPELINE")
    print("=" * 80)
    start_time = time.time()

    # 1. Ingest Data
    print("\n[Step 1/6] Loading and preprocessing Adult Income dataset...")
    data = load_adult_dataset()
    print(f"  Training samples: {len(data['X_train']):,}, Test samples: {len(data['X_test']):,}")
    print(f"  Overall Positive Rate (>50K): {np.mean(data['y_train']) * 100:.2f}%")
    print(f"  Sensitive Group (Sex): Male = {np.mean(data['A_train']['sex_binary']) * 100:.1f}%, Female = {(1 - np.mean(data['A_train']['sex_binary'])) * 100:.1f}%")

    # 2. Train Baseline Classifier
    print("\n[Step 2/6] Training baseline LightGBM classifier...")
    base_model = train_model(data["X_train"], data["y_train"])
    base_eval = evaluate_model(base_model, data["X_test"], data["y_test"])
    print(f"  Baseline Accuracy: {base_eval['metrics']['accuracy'] * 100:.2f}%")
    print(f"  Baseline F1 Score: {base_eval['metrics']['f1']:.4f}")
    print(f"  Baseline ROC AUC:  {base_eval['metrics']['roc_auc']:.4f}")

    # 3. Model Explainability Engine
    print("\n[Step 3/6] Computing SHAP TreeExplainer values and LIME explanations...")
    explainer = ModelExplainer(
        model=base_model,
        X_train=data["X_train"],
        feature_names=data["feature_names"],
        categorical_features=data["categorical_indices"],
    )
    # Compute SHAP on representative test sample (1,000 instances for fast, high-fidelity curves)
    sample_size = min(1000, len(data["X_test"]))
    X_test_sample = data["X_test"].iloc[:sample_size]
    shap_explanation = explainer.compute_shap_values(X_test_sample)
    print(f"  Computed SHAP explanation tensor for {sample_size:,} evaluation instances.")

    # 4. Multi-Attribute Fairness Audit
    print("\n[Step 4/6] Executing algorithmic fairness audit...")
    audit_results = audit_model_fairness(
        y_true=data["y_test"],
        y_pred=base_eval["y_pred"],
        A_df=data["A_test"],
        y_prob=base_eval["y_prob"],
    )
    sex_audit_df = audit_results["sex_audit"]["disaggregated_df"]
    sex_disp = audit_results["sex_audit"]["disparity_summary"]

    print("\n--- Baseline Disaggregated Performance by Sex ---")
    print(tabulate(sex_audit_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    print("\n--- Baseline Disparity Summary (Sex) ---")
    print(f"  Demographic Parity Difference: {sex_disp['demographic_parity_difference']:.4f} ({sex_disp['demographic_parity_difference']*100:.2f}%)")
    print(f"  Disparate Impact Ratio:        {sex_disp['demographic_parity_ratio']:.4f} (4/5ths Rule Violated: {sex_disp['disparate_impact_four_fifths_violation']})")
    print(f"  Equalized Odds Difference:     {sex_disp['equalized_odds_difference']:.4f} ({sex_disp['equalized_odds_difference']*100:.2f}%)")

    # 5. Mitigation Benchmark
    print("\n[Step 5/6] Benchmarking Pre-, In-, and Post-processing Fairness Mitigations...")
    benchmark_res = run_comprehensive_mitigation_benchmark(data, base_model, base_eval)
    summary_df = benchmark_res["summary_df"]
    delta_df = benchmark_res["delta_df"]

    print("\n--- Mitigation Benchmark Summary ---")
    print(tabulate(summary_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    print("\n--- Mitigation Deltas vs. Baseline ---")
    print(tabulate(delta_df, headers="keys", tablefmt="github", floatfmt=".4f"))

    # Save metrics to disk
    os.makedirs("reports", exist_ok=True)
    summary_csv_path = os.path.join("reports", "audit_summary_metrics.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    delta_csv_path = os.path.join("reports", "mitigation_deltas.csv")
    delta_df.to_csv(delta_csv_path, index=False)

    sex_audit_csv_path = os.path.join("reports", "disaggregated_audit_sex.csv")
    sex_audit_df.to_csv(sex_audit_csv_path, index=False)
    print(f"\n  Saved metric tables to {summary_csv_path}, {delta_csv_path}, and {sex_audit_csv_path}")

    # 6. Generate Figures
    print("\n[Step 6/6] Rendering and saving publication-grade figures...")
    plots = generate_all_visualizations(
        data=data,
        base_eval=base_eval,
        benchmark_res=benchmark_res,
        explainer=explainer,
        shap_explanation=shap_explanation,
        output_dir="reports/figures",
    )
    print(f"  Successfully generated and saved {len(plots)} figures in reports/figures/:")
    for p in plots:
        print(f"   * {p}")

    elapsed = time.time() - start_time
    print(f"\nAudit pipeline finished successfully in {elapsed:.1f} seconds.")

    return {
        "data": data,
        "base_model": base_model,
        "base_eval": base_eval,
        "explainer": explainer,
        "shap_explanation": shap_explanation,
        "audit_results": audit_results,
        "benchmark_res": benchmark_res,
        "plots": plots,
    }


if __name__ == "__main__":
    run_pipeline()
