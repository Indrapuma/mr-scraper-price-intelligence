"""
Approach 2: Shop/Product Level Model
Trains models conditioned on shopId + itemId + modelId groupings.
Uses a hierarchical fallback strategy:
  1. Product-level model (shopId + itemId + modelId)
  2. Item-level model (shopId + itemId)
  3. Shop-level model (shopId)
  4. Global model fallback
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict
import joblib
import os
from typing import Dict, Tuple, Optional

from .feature_engineering import get_feature_columns


SEED = 42
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MIN_SAMPLES_FOR_MODEL = 10  # Minimum historical records to train a per-entity model


def get_product_key(row):
    """Generate composite product key."""
    return (row["shopId"], row["itemId"], row["modelId"])


def train_product_models(
    train_df: pd.DataFrame,
    min_samples: int = MIN_SAMPLES_FOR_MODEL
) -> Dict:
    """
    Train per-product price prediction using historical aggregates.
    Instead of training separate ML models per product (which would be too many),
    we use a sophisticated lookup + trend model.

    Strategy:
    - For each (shopId, itemId, modelId), compute price statistics
    - Capture recent trend (last N observations)
    - Store pricing patterns for calibration

    Returns:
        Dictionary of product-level price models/stats
    """
    print("Building product-level price models...")

    # Sort by time for trend calculation
    train_df = train_df.sort_values("capturedAt")

    product_models = {}

    # Group by product key
    grouped = train_df.groupby(["shopId", "itemId", "modelId"])

    for key, group in grouped:
        if len(group) < 2:
            continue

        prices = group["price"].values
        timestamps = group["capturedAt"].values

        # Recent price (most recent observation)
        recent_price = prices[-1]

        # Price statistics
        model_info = {
            "mean_price": np.mean(prices),
            "median_price": np.median(prices),
            "std_price": np.std(prices),
            "min_price": np.min(prices),
            "max_price": np.max(prices),
            "recent_price": recent_price,
            "n_observations": len(prices),
            "price_trend": _calculate_price_trend(prices),
            "last_5_prices": prices[-5:].tolist() if len(prices) >= 5 else prices.tolist(),
            "hist_price_cv": float(np.std(prices) / np.mean(prices)) if np.mean(prices) > 0 else 0.0,
            "has_discount_history": bool((group["has_discount"] if "has_discount" in group.columns
                                         else (group["priceBeforeDiscount"] > 0)).any()),
        }

        product_models[key] = model_info

    print(f"  Built models for {len(product_models)} products")
    return product_models


def _calculate_price_trend(prices: np.ndarray) -> float:
    """
    Calculate price trend as slope of linear regression on recent prices.
    Positive = prices increasing, Negative = prices decreasing.
    """
    if len(prices) < 3:
        return 0.0

    # Use last 10 observations max
    recent = prices[-10:]
    x = np.arange(len(recent))

    # Simple linear regression slope
    if np.std(recent) == 0:
        return 0.0

    slope = np.polyfit(x, recent, 1)[0]
    return float(slope)


def train_shop_level_model(
    train_df: pd.DataFrame,
    feature_cols: list = None
) -> Tuple[lgb.LGBMRegressor, list]:
    """
    Train a shop-conditioned model.
    This is a single LightGBM model but with shopId as a prominent feature,
    allowing it to learn shop-specific pricing patterns.

    The model focuses on per-shop pricing dynamics with a reduced feature set
    optimized for shop-level patterns.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    # Key features for shop-level model
    shop_model_features = [
        "shopId", "itemId", "modelId", "cat_id",
        "priceBeforeDiscount", "item_price_min", "item_price_max",
        "price_range", "price_midrange",
        "has_discount", "show_discount", "raw_discount", "has_promotion",
        "stock", "stock_ratio",
        "shop_rating", "shop_follower_count", "shop_quality_score",
        "review_rating", "total_rating_count",
        "hist_price_mean", "hist_price_median", "hist_price_last",
        "hist_price_std", "hist_count", "hist_price_cv",
        "shop_avg_price", "shop_price_std",
        "cat_avg_price",
        "day_of_week", "is_weekend", "date_ordinal",
    ]

    available_features = [c for c in shop_model_features if c in train_df.columns]
    print(f"Training Shop-Level Model with {len(available_features)} features...")

    X_train = train_df[available_features].fillna(-1)
    y_train = train_df["price"]

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 255,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "max_depth": -1,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "n_estimators": 1500,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    return model, available_features


def predict_product_level(
    product_models: Dict,
    df: pd.DataFrame,
    use_trend: bool = True
) -> pd.Series:
    """
    Predict prices using product-level models (vectorized).

    Strategy:
    - Use recent_price as base prediction
    - Adjust with trend if available
    - Fall back to median_price if recent is unavailable

    Returns:
        Series with predictions (NaN for products without models)
    """
    predictions = pd.Series(index=df.index, dtype=float, data=np.nan)

    # Create product keys for vectorized lookup
    keys = list(zip(df["shopId"].values, df["itemId"].values, df["modelId"].values))

    for i, key in enumerate(keys):
        if key in product_models:
            info = product_models[key]

            # Base prediction: weighted combination of recent and median
            base_price = info["recent_price"] * 0.7 + info["median_price"] * 0.3

            # Apply trend adjustment
            if use_trend and abs(info["price_trend"]) > 0:
                trend_adj = info["price_trend"] * 0.5
                base_price += trend_adj

            predictions.iloc[i] = max(base_price, 0)

    coverage = predictions.notna().sum() / len(predictions) * 100
    print(f"  Product-level coverage: {coverage:.1f}%")

    return predictions


def predict_shop_level(
    model: lgb.LGBMRegressor,
    df: pd.DataFrame,
    feature_cols: list
) -> np.ndarray:
    """Predict using the shop-level LightGBM model."""
    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].fillna(-1)
    predictions = model.predict(X)
    return np.maximum(predictions, 0)


def ensemble_predictions(
    global_preds: np.ndarray,
    shop_preds: np.ndarray,
    product_preds: pd.Series,
    df: pd.DataFrame,
    product_models: Dict
) -> np.ndarray:
    """
    Ensemble predictions from all approaches with adaptive weighting.

    Weighting strategy:
    - Products with rich history: weight product-level more
    - Products with sparse history: weight global/shop models more
    - Always include global as stabilizer

    Returns:
        Final ensembled predictions
    """
    n = len(df)
    final_preds = np.zeros(n)

    # Vectorized key creation
    keys = list(zip(df["shopId"].values, df["itemId"].values, df["modelId"].values))
    product_preds_values = product_preds.values

    for i, key in enumerate(keys):
        has_product = not np.isnan(product_preds_values[i])

        if has_product and key in product_models:
            n_obs = product_models[key]["n_observations"]
            cv = product_models[key]["hist_price_cv"]

            if n_obs >= 20 and cv < 0.1:
                w_product, w_shop, w_global = 0.6, 0.25, 0.15
            elif n_obs >= 10:
                w_product, w_shop, w_global = 0.4, 0.35, 0.25
            else:
                w_product, w_shop, w_global = 0.25, 0.40, 0.35

            final_preds[i] = (
                w_product * product_preds_values[i] +
                w_shop * shop_preds[i] +
                w_global * global_preds[i]
            )
        else:
            final_preds[i] = 0.55 * shop_preds[i] + 0.45 * global_preds[i]

    return final_preds
