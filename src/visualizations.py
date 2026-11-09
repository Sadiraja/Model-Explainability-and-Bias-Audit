"""
Visualizations Module.
Generates publication-quality charts for SHAP/LIME explainability,
demographic disparities, confusion matrices, and mitigation trade-off Pareto frontiers.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Set global aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": False,
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def plot_selection_rate_disparity(
    sex_audit_df: pd.DataFrame,
    output_path: str = "reports/figures/fairness_selection_rate_by_group.png",
) -> None:
    """Plots base rate vs predicted selection rate across demographic groups."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sex_audit_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, sex_audit_df["Base_Rate"] * 100, width, label="True Base Rate (>50K)", color="#4575b4")
    rects2 = ax.bar(x + width/2, sex_audit_df["Selection_Rate"] * 100, width, label="Model Selection Rate (>50K)", color="#d73027")

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Base Rate vs. Model Selection Rate by Sex (Baseline Model)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sex_audit_df["Sex"], fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    ax.set_ylim(0, max(sex_audit_df["Selection_Rate"].max(), sex_audit_df["Base_Rate"].max()) * 125)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tpr_fpr_disparity(
    sex_audit_df: pd.DataFrame,
    output_path: str = "reports/figures/fairness_tpr_fpr_disparity.png",
) -> None:
    """Plots True Positive Rate (Equal Opportunity) and False Positive Rate across groups."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(sex_audit_df))
    width = 0.35

    rects1 = ax.bar(x - width/2, sex_audit_df["Recall_TPR"] * 100, width, label="True Positive Rate (TPR / Recall)", color="#31a354")
    rects2 = ax.bar(x + width/2, sex_audit_df["FPR"] * 100, width, label="False Positive Rate (FPR / Fall-out)", color="#e6550d")

    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Equal Opportunity & Predictive Equality Disparities by Sex", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(sex_audit_df["Sex"], fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    ax.set_ylim(0, 100)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_group_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_group: pd.Series,
    output_path: str = "reports/figures/confusion_matrices_by_group.png",
) -> None:
    """Plots side-by-side normalized confusion matrices for demographic groups."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    groups = ["Female", "Male"]
    for i, grp in enumerate(groups):
        mask = (sensitive_group == grp).values
        cm = confusion_matrix(y_true[mask], y_pred[mask], normalize="true")

        sns.heatmap(
            cm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            ax=axes[i],
            xticklabels=["<=50K (Pred)", ">50K (Pred)"],
            yticklabels=["<=50K (True)", ">50K (True)"],
            annot_kws={"size": 12, "weight": "bold"},
        )
        axes[i].set_title(f"Normalized Confusion Matrix: {grp} (N={np.sum(mask):,})", fontweight="bold")
        axes[i].set_ylabel("Actual Label", fontweight="bold")
        axes[i].set_xlabel("Predicted Label", fontweight="bold")

    plt.suptitle("Baseline Model Error Distribution by Sensitive Attribute", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_mitigation_benchmark_bar(
    summary_df: pd.DataFrame,
    output_path: str = "reports/figures/fairness_metric_comparison_bar.png",
) -> None:
    """Compares Accuracy, F1, Demographic Parity Diff, and Equalized Odds Diff across models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    models = summary_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.20

    m1 = ax.bar(x - 1.5*width, summary_df["Accuracy"] * 100, width, label="Accuracy (%)", color="#1f77b4")
    m2 = ax.bar(x - 0.5*width, summary_df["F1_Score"] * 100, width, label="F1 Score (%)", color="#2ca02c")
    m3 = ax.bar(x + 0.5*width, summary_df["Demographic_Parity_Diff"] * 100, width, label="Demog. Parity Diff (%)", color="#d62728")
    m4 = ax.bar(x + 1.5*width, summary_df["Equalized_Odds_Diff"] * 100, width, label="Equalized Odds Diff (%)", color="#9467bd")

    ax.set_ylabel("Metric Value (%)", fontweight="bold")
    ax.set_title("Model Performance vs. Fairness Disparities Across Mitigation Strategies", fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold", rotation=10, ha="right")
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    ax.set_ylim(0, 100)

    # Highlight the four-fifths rule threshold on parity
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_tradeoff_frontier(
    summary_df: pd.DataFrame,
    output_path: str = "reports/figures/mitigation_tradeoff_pareto.png",
) -> None:
    """Plots Pareto Trade-Off between Model Accuracy and Fairness Disparity (Equalized Odds)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#d73027", "#fc8d59", "#4575b4", "#91bfdb"]
    markers = ["o", "s", "^", "D"]

    for i, (_, row) in enumerate(summary_df.iterrows()):
        ax.scatter(
            row["Equalized_Odds_Diff"] * 100,
            row["Accuracy"] * 100,
            s=180,
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            edgecolor="black",
            linewidth=1.5,
            label=row["Model"],
            zorder=4,
        )
        # Annotation callout
        ax.annotate(
            row["Model"],
            (row["Equalized_Odds_Diff"] * 100, row["Accuracy"] * 100),
            textcoords="offset points",
            xytext=(8, 6),
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Equalized Odds Difference (% - Lower is Fairer)", fontweight="bold")
    ax.set_ylabel("Overall Test Accuracy (% - Higher is Better)", fontweight="bold")
    ax.set_title("Pareto Trade-Off Frontier: Predictive Performance vs. Algorithmic Fairness", fontweight="bold", pad=12)

    # Add visual quadrants / arrows
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_all_visualizations(
    data: Dict[str, Any],
    base_eval: Dict[str, Any],
    benchmark_res: Dict[str, Any],
    explainer: Any,
    shap_explanation: Any,
    output_dir: str = "reports/figures",
) -> List[str]:
    """Generates and saves the complete set of visual artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    saved_plots = []

    # 1. SHAP Beeswarm
    beeswarm_path = os.path.join(output_dir, "shap_summary_beeswarm.png")
    explainer.plot_shap_summary_beeswarm(shap_explanation, beeswarm_path)
    saved_plots.append(beeswarm_path)

    # 2. SHAP Bar
    bar_path = os.path.join(output_dir, "shap_feature_importance_bar.png")
    explainer.plot_shap_summary_bar(shap_explanation, bar_path)
    saved_plots.append(bar_path)

    # 3. SHAP Dependence: Capital Gain
    dep_cap_path = os.path.join(output_dir, "shap_dependence_capital_gain.png")
    explainer.plot_shap_dependence(
        shap_explanation,
        feature="Capital_Gain",
        interaction_feature="Sex",
        output_path=dep_cap_path,
        title="SHAP Dependence: Capital Gain colored by Sex",
    )
    saved_plots.append(dep_cap_path)

    # 4. SHAP Dependence: Age x Sex
    dep_age_path = os.path.join(output_dir, "shap_dependence_age_sex.png")
    explainer.plot_shap_dependence(
        shap_explanation,
        feature="Age",
        interaction_feature="Sex",
        output_path=dep_age_path,
        title="SHAP Dependence: Age colored by Sex",
    )
    saved_plots.append(dep_age_path)

    # 5. Local SHAP Waterfall (High income male profile)
    # Find male index with true positive prediction
    y_test = data["y_test"]
    A_sex = data["A_test"]["sex_name"].values
    y_pred = base_eval["y_pred"]

    male_pos_idx = np.where((A_sex == "Male") & (y_test == 1) & (y_pred == 1))[0][0]
    wf_male_path = os.path.join(output_dir, "shap_waterfall_male_pos.png")
    explainer.plot_shap_waterfall(
        shap_explanation,
        index=int(male_pos_idx),
        output_path=wf_male_path,
        title=f"SHAP Local Waterfall: Male High-Income Applicant (Index {male_pos_idx})",
    )
    saved_plots.append(wf_male_path)

    # 6. Local SHAP Waterfall (Female applicant profile)
    female_idx = np.where((A_sex == "Female") & (y_test == 0) & (y_pred == 0))[0][0]
    wf_fem_path = os.path.join(output_dir, "shap_waterfall_female_neg.png")
    explainer.plot_shap_waterfall(
        shap_explanation,
        index=int(female_idx),
        output_path=wf_fem_path,
        title=f"SHAP Local Waterfall: Female Applicant <=50K (Index {female_idx})",
    )
    saved_plots.append(wf_fem_path)

    # 7 & 8. LIME Local Explanations
    lime_male_path = os.path.join(output_dir, "lime_explanation_male.png")
    explainer.plot_lime_explanation(
        data["X_test"].iloc[male_pos_idx],
        output_path=lime_male_path,
        title=f"LIME Local Explanation: Male Applicant (Idx {male_pos_idx})",
    )
    saved_plots.append(lime_male_path)

    lime_fem_path = os.path.join(output_dir, "lime_explanation_female.png")
    explainer.plot_lime_explanation(
        data["X_test"].iloc[female_idx],
        output_path=lime_fem_path,
        title=f"LIME Local Explanation: Female Applicant (Idx {female_idx})",
    )
    saved_plots.append(lime_fem_path)

    # 9. Selection Rate Disparity
    base_sex_audit_df = benchmark_res["audits"]["Baseline (Unmitigated)"]["disaggregated_df"]
    sel_path = os.path.join(output_dir, "fairness_selection_rate_by_group.png")
    plot_selection_rate_disparity(base_sex_audit_df, sel_path)
    saved_plots.append(sel_path)

    # 10. TPR & FPR Disparity
    tpr_path = os.path.join(output_dir, "fairness_tpr_fpr_disparity.png")
    plot_tpr_fpr_disparity(base_sex_audit_df, tpr_path)
    saved_plots.append(tpr_path)

    # 11. Group Confusion Matrices
    cm_path = os.path.join(output_dir, "confusion_matrices_by_group.png")
    plot_group_confusion_matrices(data["y_test"], base_eval["y_pred"], data["A_test"]["sex_name"], cm_path)
    saved_plots.append(cm_path)

    # 12. Mitigation Comparison Bar
    bar_comp_path = os.path.join(output_dir, "fairness_metric_comparison_bar.png")
    plot_mitigation_benchmark_bar(benchmark_res["summary_df"], bar_comp_path)
    saved_plots.append(bar_comp_path)

    # 13. Pareto Frontier
    pareto_path = os.path.join(output_dir, "mitigation_tradeoff_pareto.png")
    plot_pareto_tradeoff_frontier(benchmark_res["summary_df"], pareto_path)
    saved_plots.append(pareto_path)

    return saved_plots


if __name__ == "__main__":
    print("Visualizations module compiled successfully.")
