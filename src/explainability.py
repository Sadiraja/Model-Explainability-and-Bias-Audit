"""
Explainability Module using SHAP and LIME.
Provides global feature attribution, beeswarm/bar plots, dependence plots,
and local instance explanations via SHAP waterfall and LIME tabular explainer.
"""

import os
import sys
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for clean plot saving
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer


class ModelExplainer:
    """
    Unified Explainability Engine wrapping SHAP TreeExplainer and LIME Tabular Explainer.
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names if feature_names is not None else list(X_train.columns)
        self.categorical_features = categorical_features if categorical_features is not None else []
        self.class_names = class_names if class_names is not None else ["<=50K", ">50K"]

        # Initialize SHAP TreeExplainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        # Initialize LIME Tabular Explainer
        self.lime_explainer = LimeTabularExplainer(
            training_data=np.asarray(self.X_train),
            feature_names=self.feature_names,
            categorical_features=self.categorical_features,
            class_names=self.class_names,
            mode="classification",
            random_state=42,
        )

    def compute_shap_values(self, X_eval: pd.DataFrame) -> shap.Explanation:
        """Computes SHAP Explanation object on given evaluation dataset."""
        return self.shap_explainer(X_eval)

    def plot_shap_summary_beeswarm(
        self,
        shap_explanation: shap.Explanation,
        output_path: str,
        max_display: int = 12,
        title: str = "SHAP Global Feature Importance (Beeswarm)",
    ) -> None:
        """Generates and saves SHAP beeswarm summary plot."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        shap.plots.beeswarm(shap_explanation, max_display=max_display, show=False)
        plt.title(title, fontsize=14, pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_shap_summary_bar(
        self,
        shap_explanation: shap.Explanation,
        output_path: str,
        max_display: int = 12,
        title: str = "SHAP Global Feature Importance (|mean SHAP value|)",
    ) -> None:
        """Generates and saves SHAP bar plot of mean absolute SHAP values."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_explanation, max_display=max_display, show=False)
        plt.title(title, fontsize=14, pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_shap_dependence(
        self,
        shap_explanation: shap.Explanation,
        feature: str,
        interaction_feature: Optional[str],
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Generates and saves a SHAP dependence plot."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(8, 5))
        if interaction_feature is not None:
            shap.plots.scatter(
                shap_explanation[:, feature],
                color=shap_explanation[:, interaction_feature],
                show=False,
            )
        else:
            shap.plots.scatter(shap_explanation[:, feature], show=False)

        if title:
            plt.title(title, fontsize=12, pad=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_shap_waterfall(
        self,
        shap_explanation: shap.Explanation,
        index: int,
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        """Generates and saves a SHAP waterfall plot for an individual prediction."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_explanation[index], max_display=10, show=False)
        if title:
            plt.title(title, fontsize=13, pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def explain_instance_lime(
        self,
        instance: pd.Series,
        num_features: int = 8,
        label: int = 1,
    ):
        """Generates a LIME local explanation for an individual instance."""
        def predict_fn(x):
            df_x = pd.DataFrame(x, columns=self.feature_names)
            return self.model.predict_proba(df_x)

        exp = self.lime_explainer.explain_instance(
            data_row=np.asarray(instance),
            predict_fn=predict_fn,
            num_features=num_features,
            labels=(label,),
        )
        return exp

    def plot_lime_explanation(
        self,
        instance: pd.Series,
        output_path: str,
        num_features: int = 8,
        label: int = 1,
        title: str = "LIME Local Explanation",
    ) -> None:
        """Generates and saves a LIME explanation plot."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        exp = self.explain_instance_lime(instance, num_features=num_features, label=label)
        fig = exp.as_pyplot_figure(label=label)
        plt.title(title, fontsize=12, pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.data_loader import load_adult_dataset
    from src.model import train_model

    data = load_adult_dataset()
    model = train_model(data["X_train"], data["y_train"])

    explainer = ModelExplainer(
        model=model,
        X_train=data["X_train"],
        feature_names=data["feature_names"],
        categorical_features=data["categorical_indices"],
    )

    # Compute on sample of test set for speed in test
    sample_X_test = data["X_test"].iloc[:200]
    shap_vals = explainer.compute_shap_values(sample_X_test)
    print("SHAP values computed successfully, shape:", shap_vals.shape)

    test_plot = os.path.join("reports", "figures", "test_beeswarm.png")
    explainer.plot_shap_summary_beeswarm(shap_vals, test_plot)
    print(f"Test beeswarm plot saved to {test_plot}")
