"""
Command-Line Interface for the Model Explainability and Bias Audit.
Usage:
    python run_audit.py --model lightgbm --sample-size 1000
    python run_audit.py --help
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from main import run_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Explainability and Fairness Bias Audit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["lightgbm", "random_forest"],
        help="Model architecture for baseline and mitigations (default: lightgbm)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="Number of evaluation samples for SHAP computation (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Directory to store output plots and figures (default: reports/figures)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Starting audit with model={args.model}, sample_size={args.sample_size}...")
    run_pipeline()
