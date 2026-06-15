"""
Evaluation Module
Computes and reports prediction accuracy metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute regression metrics for price prediction.

    Returns:
        Dictionary with MAE, RMSE, MAPE, MedianAE, R²
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Remove any NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan,
                "median_ae": np.nan, "r2": np.nan}

    errors = y_true - y_pred
    abs_errors = np.abs(errors)

    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    median_ae = np.median(abs_errors)

    # MAPE (avoid division by zero)
    nonzero_mask = y_true > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(abs_errors[nonzero_mask] / y_true[nonzero_mask]) * 100
    else:
        mape = np.nan

    # R-squared
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "median_ae": median_ae,
        "r2": r2,
        "n_samples": len(y_true),
    }


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {model_name} - Evaluation Results")
    print(f"{'='*60}")
    print(f"  MAE:        {metrics['mae']:,.2f}")
    print(f"  RMSE:       {metrics['rmse']:,.2f}")
    print(f"  MAPE:       {metrics['mape']:.2f}%")
    print(f"  Median AE:  {metrics['median_ae']:,.2f}")
    print(f"  R²:         {metrics['r2']:.4f}")
    print(f"  N samples:  {metrics['n_samples']:,}")
    print(f"{'='*60}\n")


def compare_models(results: Dict[str, Dict]):
    """
    Compare multiple model results side-by-side.

    Args:
        results: Dict mapping model_name -> metrics dict
    """
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON")
    print("=" * 80)

    # Header
    models = list(results.keys())
    header = f"{'Metric':<15}"
    for model in models:
        header += f"{model:<25}"
    print(header)
    print("-" * 80)

    # Metrics
    for metric in ["mae", "rmse", "mape", "median_ae", "r2"]:
        row = f"{metric.upper():<15}"
        for model in models:
            val = results[model].get(metric, np.nan)
            if metric == "mape":
                row += f"{val:.2f}%{'':<20}"
            elif metric == "r2":
                row += f"{val:.4f}{'':<20}"
            else:
                row += f"{val:,.2f}{'':<20}"
        print(row)

    print("=" * 80)

    # Determine winner
    best_model = min(results.keys(), key=lambda k: results[k]["mae"])
    print(f"\n  Best model (lowest MAE): {best_model}")
    print(f"  MAE improvement over baseline: ", end="")

    if len(models) >= 2:
        baseline_mae = max(r["mae"] for r in results.values())
        best_mae = results[best_model]["mae"]
        improvement = (baseline_mae - best_mae) / baseline_mae * 100
        print(f"{improvement:.1f}%")
    else:
        print("N/A (single model)")


def evaluate_by_category(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    categories: np.ndarray,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Evaluate metrics per category.

    Returns:
        DataFrame with per-category metrics, sorted by MAE
    """
    results = []

    for cat in np.unique(categories):
        mask = categories == cat
        if mask.sum() < 5:
            continue

        metrics = compute_metrics(y_true[mask], y_pred[mask])
        metrics["cat_id"] = cat
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae", ascending=False)

    print(f"\nTop {top_n} categories with highest MAE:")
    print(results_df.head(top_n)[["cat_id", "mae", "mape", "n_samples"]].to_string(index=False))

    return results_df


def evaluate_calibration_impact(
    y_true: np.ndarray,
    preds_before: np.ndarray,
    preds_after: np.ndarray,
    label: str = "Calibration"
):
    """
    Report the impact of anchor calibration on predictions.
    """
    metrics_before = compute_metrics(y_true, preds_before)
    metrics_after = compute_metrics(y_true, preds_after)

    print(f"\n{'='*60}")
    print(f"  {label} Impact")
    print(f"{'='*60}")
    print(f"  {'Metric':<12} {'Before':<15} {'After':<15} {'Change':<15}")
    print(f"  {'-'*55}")

    for metric in ["mae", "rmse", "mape"]:
        before = metrics_before[metric]
        after = metrics_after[metric]
        change = after - before
        pct_change = (change / before * 100) if before != 0 else 0

        sign = "↓" if change < 0 else "↑"
        print(f"  {metric.upper():<12} {before:<15.2f} {after:<15.2f} "
              f"{sign} {abs(pct_change):.1f}%")

    print(f"{'='*60}")

    return metrics_before, metrics_after
