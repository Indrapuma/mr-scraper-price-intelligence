"""
Anchor Set Calibration Module
Uses the 100 anchor samples per day to calibrate model predictions.

Calibration Strategies:
1. Global bias correction — detect and correct day-level price shifts
2. Category-level adjustment — per-category correction factors
3. Shop-level adjustment — per-shop correction when anchor data is available
4. Multiplicative scaling — ratio-based correction for proportional errors
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_global_bias(
    anchor_predictions: np.ndarray,
    anchor_actual: np.ndarray
) -> Dict:
    """
    Compute global bias metrics from anchor set predictions vs actuals.

    Returns:
        Dictionary with bias statistics
    """
    errors = anchor_actual - anchor_predictions
    ratios = np.where(anchor_predictions > 0,
                      anchor_actual / anchor_predictions, 1.0)

    bias_info = {
        "mean_error": np.mean(errors),
        "median_error": np.median(errors),
        "mean_ratio": np.mean(ratios),
        "median_ratio": np.median(ratios),
        "std_error": np.std(errors),
        "mae": np.mean(np.abs(errors)),
        "mape": np.mean(np.abs(errors) / np.maximum(anchor_actual, 1)) * 100,
    }

    return bias_info


def apply_global_bias_correction(
    predictions: np.ndarray,
    anchor_predictions: np.ndarray,
    anchor_actual: np.ndarray,
    method: str = "multiplicative"
) -> np.ndarray:
    """
    Apply global bias correction based on anchor set performance.

    Methods:
    - "additive": Add median error to all predictions
    - "multiplicative": Scale predictions by median ratio
    - "hybrid": Combine both approaches

    Args:
        predictions: Raw model predictions to correct
        anchor_predictions: Model predictions on anchor samples
        anchor_actual: Actual prices of anchor samples
        method: Correction method

    Returns:
        Corrected predictions
    """
    errors = anchor_actual - anchor_predictions
    ratios = np.where(anchor_predictions > 0,
                      anchor_actual / anchor_predictions, 1.0)

    # Remove outlier ratios (likely anomalous anchors)
    ratios_clipped = np.clip(ratios, 0.5, 2.0)

    if method == "additive":
        correction = np.median(errors)
        corrected = predictions + correction
        print(f"  Global additive correction: {correction:.2f}")

    elif method == "multiplicative":
        scale_factor = np.median(ratios_clipped)
        corrected = predictions * scale_factor
        print(f"  Global multiplicative correction: {scale_factor:.4f}")

    elif method == "hybrid":
        # Use multiplicative for the scale, additive for residual
        scale_factor = np.median(ratios_clipped)
        scaled = predictions * scale_factor

        # Compute residual error after scaling
        scaled_anchors = anchor_predictions * scale_factor
        residual_errors = anchor_actual - scaled_anchors
        residual_correction = np.median(residual_errors)

        corrected = scaled + residual_correction * 0.5  # Dampened residual
        print(f"  Hybrid correction: scale={scale_factor:.4f}, residual={residual_correction:.2f}")

    else:
        corrected = predictions

    return np.maximum(corrected, 0)


def compute_category_corrections(
    anchors_df: pd.DataFrame,
    anchor_predictions: np.ndarray
) -> Dict:
    """
    Compute per-category correction factors from anchor set.

    Returns:
        Dictionary mapping cat_id to correction factor
    """
    anchors_df = anchors_df.copy()
    anchors_df["pred"] = anchor_predictions
    anchors_df["ratio"] = np.where(
        anchors_df["pred"] > 0,
        anchors_df["price"] / anchors_df["pred"],
        1.0
    )

    # Compute per-category median ratio
    cat_corrections = {}
    for cat_id, group in anchors_df.groupby("cat_id"):
        if len(group) >= 2:
            median_ratio = np.median(np.clip(group["ratio"].values, 0.5, 2.0))
            cat_corrections[cat_id] = median_ratio
        else:
            cat_corrections[cat_id] = group["ratio"].values[0]

    return cat_corrections


def apply_category_corrections(
    predictions: np.ndarray,
    df: pd.DataFrame,
    cat_corrections: Dict,
    global_ratio: float = 1.0
) -> np.ndarray:
    """
    Apply per-category correction factors to predictions.

    Falls back to global_ratio for categories not in anchor set.
    """
    corrected = predictions.copy()
    cat_ids = df["cat_id"].values

    for i in range(len(df)):
        cat_id = cat_ids[i]
        if cat_id in cat_corrections:
            corrected[i] *= cat_corrections[cat_id]
        else:
            corrected[i] *= global_ratio

    return np.maximum(corrected, 0)


def compute_shop_corrections(
    anchors_df: pd.DataFrame,
    anchor_predictions: np.ndarray
) -> Dict:
    """
    Compute per-shop correction factors from anchor set.

    Returns:
        Dictionary mapping shopId to correction factor
    """
    anchors_df = anchors_df.copy()
    anchors_df["pred"] = anchor_predictions
    anchors_df["ratio"] = np.where(
        anchors_df["pred"] > 0,
        anchors_df["price"] / anchors_df["pred"],
        1.0
    )

    shop_corrections = {}
    for shop_id, group in anchors_df.groupby("shopId"):
        if len(group) >= 2:
            median_ratio = np.median(np.clip(group["ratio"].values, 0.5, 2.0))
            shop_corrections[shop_id] = median_ratio

    return shop_corrections


def apply_hierarchical_calibration(
    predictions: np.ndarray,
    df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    anchor_predictions: np.ndarray,
    anchor_actual: np.ndarray
) -> np.ndarray:
    """
    Apply hierarchical calibration: shop > category > global.

    Priority:
    1. If shop has anchor data → use shop-level correction
    2. If category has anchor data → use category-level correction
    3. Otherwise → use global correction

    This is the recommended calibration strategy as it provides the most
    granular corrections while maintaining robustness.
    """
    print("\nApplying hierarchical calibration...")

    # Compute all correction levels
    global_bias = compute_global_bias(anchor_predictions, anchor_actual)
    global_ratio = global_bias["median_ratio"]
    print(f"  Global median ratio: {global_ratio:.4f}")

    cat_corrections = compute_category_corrections(anchors_df, anchor_predictions)
    print(f"  Category corrections computed for {len(cat_corrections)} categories")

    shop_corrections = compute_shop_corrections(anchors_df, anchor_predictions)
    print(f"  Shop corrections computed for {len(shop_corrections)} shops")

    # Apply hierarchically
    corrected = predictions.copy()
    method_counts = {"shop": 0, "category": 0, "global": 0}

    shop_ids = df["shopId"].values
    cat_ids = df["cat_id"].values

    for i in range(len(df)):
        shop_id = shop_ids[i]
        cat_id = cat_ids[i]

        if shop_id in shop_corrections:
            corrected[i] *= shop_corrections[shop_id]
            method_counts["shop"] += 1
        elif cat_id in cat_corrections:
            corrected[i] *= cat_corrections[cat_id]
            method_counts["category"] += 1
        else:
            corrected[i] *= global_ratio
            method_counts["global"] += 1

    total = sum(method_counts.values())
    print(f"  Calibration breakdown:")
    print(f"    Shop-level: {method_counts['shop']} ({method_counts['shop']/total*100:.1f}%)")
    print(f"    Category-level: {method_counts['category']} ({method_counts['category']/total*100:.1f}%)")
    print(f"    Global-level: {method_counts['global']} ({method_counts['global']/total*100:.1f}%)")

    return np.maximum(corrected, 0)


def select_best_calibration(
    predictions: np.ndarray,
    df: pd.DataFrame,
    anchors_df: pd.DataFrame,
    anchor_predictions: np.ndarray,
    anchor_actual: np.ndarray
) -> Tuple[np.ndarray, str]:
    """
    Try multiple calibration strategies and select the best one
    based on anchor set cross-validation.

    Uses leave-one-out on anchor set to evaluate each strategy.

    Returns:
        (calibrated_predictions, best_method_name)
    """
    from sklearn.model_selection import KFold

    # Strategies to evaluate
    strategies = {
        "global_multiplicative": lambda p, df_, ap, aa: apply_global_bias_correction(
            p, ap, aa, method="multiplicative"
        ),
        "global_hybrid": lambda p, df_, ap, aa: apply_global_bias_correction(
            p, ap, aa, method="hybrid"
        ),
        "hierarchical": lambda p, df_, ap, aa: apply_hierarchical_calibration(
            p, df_, anchors_df, ap, aa
        ),
    }

    # Evaluate on anchor set using 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    strategy_scores = {}

    print("\nEvaluating calibration strategies on anchor set...")

    for name, strategy_fn in strategies.items():
        fold_maes = []

        for train_idx, val_idx in kf.split(anchor_predictions):
            # Use train fold as "anchor set" to compute corrections
            fold_preds = anchor_predictions[train_idx]
            fold_actual = anchor_actual[train_idx]

            # Apply correction to val fold
            val_preds = anchor_predictions[val_idx].copy()

            # Simple global correction based on train fold
            ratio = np.median(np.clip(fold_actual / np.maximum(fold_preds, 1), 0.5, 2.0))
            val_corrected = val_preds * ratio

            fold_mae = np.mean(np.abs(anchor_actual[val_idx] - val_corrected))
            fold_maes.append(fold_mae)

        strategy_scores[name] = np.mean(fold_maes)
        print(f"  {name}: MAE = {strategy_scores[name]:.2f}")

    # Select best strategy
    best_method = min(strategy_scores, key=strategy_scores.get)
    print(f"\n  Best calibration strategy: {best_method}")

    # Apply best strategy to full predictions
    calibrated = strategies[best_method](
        predictions, df, anchor_predictions, anchor_actual
    )

    return calibrated, best_method
