"""
Data Loader Module for Adult Census Income Dataset.
Handles loading, pre-processing, sensitive attribute extraction,
and stratified train-test splitting.
"""

import os
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shap


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by replacing spaces and hyphens with underscores."""
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def load_adult_dataset(
    cache_dir: str = "data",
    random_state: int = 42,
    test_size: float = 0.20,
) -> Dict[str, Any]:
    """
    Loads the UCI Adult Census Income dataset via SHAP, formats features and sensitive attributes,
    and returns stratified train-test splits.

    Returns:
        Dict containing:
            - X_train, X_test: Numeric/encoded feature DataFrames for model training
            - X_display_train, X_display_test: Human-readable DataFrames for LIME & reporting
            - y_train, y_test: Binary labels (1: >50K, 0: <=50K)
            - A_train, A_test: Sensitive attribute DataFrames (Sex, Race, Intersectional)
            - feature_names: List of feature names
            - categorical_features: List of categorical feature names
            - categorical_indices: Indices of categorical columns in X
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file_x = os.path.join(cache_dir, "adult_X.csv")
    cache_file_y = os.path.join(cache_dir, "adult_y.npy")
    cache_file_disp = os.path.join(cache_dir, "adult_X_display.csv")

    if os.path.exists(cache_file_x) and os.path.exists(cache_file_y) and os.path.exists(cache_file_disp):
        X = pd.read_csv(cache_file_x)
        y = np.load(cache_file_y)
        X_display = pd.read_csv(cache_file_disp)
    else:
        # Load raw dataset and display version from SHAP
        X, y = shap.datasets.adult()
        X_display, _ = shap.datasets.adult(display=True)

        # Standardize strings in X_display
        for col in X_display.select_dtypes(include=["object"]).columns:
            X_display[col] = X_display[col].astype(str).str.strip()

        # Save to local cache
        X.to_csv(cache_file_x, index=False)
        np.save(cache_file_y, y)
        X_display.to_csv(cache_file_disp, index=False)

    # Standardize column names
    X = clean_feature_names(X)
    X_display = clean_feature_names(X_display)

    # Ensure y is binary integer (1 for >50K, 0 for <=50K)
    y_binary = y.astype(int)

    # Sensitive attributes extraction
    # Sex: In Adult dataset, Sex=1 is Male, Sex=0 is Female
    # Race: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other
    A = pd.DataFrame(index=X.index)
    A["sex_binary"] = X["Sex"].astype(int)
    A["sex_name"] = X_display["Sex"].astype(str).str.strip()
    A["race_name"] = X_display["Race"].astype(str).str.strip()
    A["race_binary"] = (A["race_name"] == "White").astype(int)
    A["intersection"] = A["sex_name"] + " - " + A["race_name"]

    # Identify categorical columns
    categorical_cols = [
        "Workclass",
        "Marital_Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Country",
    ]
    categorical_indices = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]

    # Combined stratification vector (Target + Sensitive Sex)
    strat_key = y_binary.astype(str) + "_" + A["sex_binary"].astype(str)

    # Perform stratified split
    indices = np.arange(len(X))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )

    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_test = X.iloc[idx_test].reset_index(drop=True)

    X_disp_train = X_display.iloc[idx_train].reset_index(drop=True)
    X_disp_test = X_display.iloc[idx_test].reset_index(drop=True)

    y_train = y_binary[idx_train]
    y_test = y_binary[idx_test]

    A_train = A.iloc[idx_train].reset_index(drop=True)
    A_test = A.iloc[idx_test].reset_index(drop=True)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_display_train": X_disp_train,
        "X_display_test": X_disp_test,
        "y_train": y_train,
        "y_test": y_test,
        "A_train": A_train,
        "A_test": A_test,
        "feature_names": list(X.columns),
        "categorical_features": categorical_cols,
        "categorical_indices": categorical_indices,
    }


if __name__ == "__main__":
    data = load_adult_dataset()
    print("Dataset loaded successfully:")
    print(f"X_train shape: {data['X_train'].shape}, X_test shape: {data['X_test'].shape}")
    print(f"Class balance: >50K: {np.mean(data['y_train']):.3f}")
    print(f"Sex distribution: Male: {np.mean(data['A_train']['sex_binary']):.3f}")
    print(f"Features: {data['feature_names']}")
