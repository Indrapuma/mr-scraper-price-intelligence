"""
Prediction Script
Run inference on new test data using trained models.

Usage:
    python predict.py --test_file data/test.csv --output predictions.csv
    python predict.py --test_file data/test_full.csv --output predictions_full.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, split_test_anchors
from src.feature_engineering import build_features, get_feature_columns
from src.model_global import predict_global
from src.model_shop import (
    predict_product_level, predict_shop_level, ensemble_predictions
)
from src.anchor_calibration import apply_hierarchical_calibration

SEED = 42
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_models():
    """Load pre-trained models from disk."""
    global_model_path = os.path.join(MODEL_DIR, "global_model.joblib")
    shop_model_path = os.path.join(MODEL_DIR, "shop_model.joblib")
    product_models_path = os.path.join(MODEL_DIR, "product_models.joblib")
    shop_features_path = os.path.join(MODEL_DIR, "shop_features.joblib")

    global_model = joblib.load(global_model_path)
    print(f"  Loaded global model from {global_model_path}")

    shop_model = joblib.load(shop_model_path)
    shop_features = joblib.load(shop_features_path)
    print(f"  Loaded shop model from {shop_model_path}")

    product_models = joblib.load(product_models_path)
    print(f"  Loaded {len(product_models)} product models")

    return global_model, shop_model, shop_features, product_models


def predict(test_file: str, output_file: str, train_file: str = None):
    """
    Run prediction pipeline on test data.

    Args:
        test_file: Path to test CSV file
        output_file: Path to save predictions
        train_file: Path to training data (for feature engineering reference)
    """
    print("=" * 70)
    print("  PRICE PREDICTION - INFERENCE PIPELINE")
    print("=" * 70)

    # Load training data for feature reference
    if train_file is None:
        train_file = os.path.join(DATA_DIR, "train.csv")

    print("\nLoading data...")
    dtype_map = {
        "shopId": "float64", "itemId": "float64", "modelId": "float64",
        "price": "float64", "priceBeforeDiscount": "float64",
        "promotionId": "float64", "cat_id": "float64",
        "stock": "float64", "normal_stock": "float64",
        "raw_discount": "float64", "show_discount": "float64",
        "is_free_shipping": "object", "is_pre_order": "object",
        "item_price_min": "float64", "item_price_max": "float64",
        "review_rating": "float64", "total_rating_count": "float64",
        "cmt_count": "float64", "shop_rating": "float64",
        "shop_response_rate": "float64", "shop_follower_count": "float64",
        "is_official_shop": "object", "is_verified": "object",
        "is_preferred_plus_seller": "object", "brand": "object",
    }

    train_df = pd.read_csv(train_file, dtype=dtype_map, parse_dates=["capturedAt"])
    test_df = pd.read_csv(test_file, dtype=dtype_map, parse_dates=["capturedAt"])

    # Convert booleans
    bool_cols = ["is_free_shipping", "is_pre_order", "is_official_shop",
                 "is_verified", "is_preferred_plus_seller"]
    for col in bool_cols:
        for df in [train_df, test_df]:
            if col in df.columns:
                df[col] = df[col].map({"t": True, "f": False}).astype(bool)

    # Split test into anchors and targets
    anchors_df = test_df[test_df["price"].notna()].copy()
    targets_df = test_df[test_df["price"].isna()].copy()

    print(f"  Test data: {test_df.shape[0]} rows")
    print(f"  Anchors: {anchors_df.shape[0]} rows")
    print(f"  Targets to predict: {targets_df.shape[0]} rows")

    # Load models
    print("\nLoading models...")
    global_model, shop_model, shop_features, product_models = load_models()

    # Feature engineering
    print("\nBuilding features...")
    anchors_feat = build_features(anchors_df, train_df, is_training=False)
    targets_feat = build_features(targets_df, train_df, is_training=False)

    # Generate predictions per day
    print("\nGenerating predictions per day...")

    targets_feat["date"] = targets_feat["capturedAt"].dt.date
    anchors_feat["date"] = anchors_feat["capturedAt"].dt.date

    all_predictions = np.zeros(len(targets_df))
    dates = sorted(targets_feat["date"].unique())

    for date in dates:
        print(f"\n  Processing {date}...")

        day_target_mask = targets_feat["date"] == date
        day_anchor_mask = anchors_feat["date"] == date

        day_targets = targets_feat[day_target_mask]
        day_anchors = anchors_feat[day_anchor_mask]
        day_anchor_actual = anchors_df.loc[day_anchors.index, "price"].values

        # Generate raw predictions
        global_preds = predict_global(global_model, day_targets)
        product_preds = predict_product_level(product_models, day_targets)
        shop_preds = predict_shop_level(shop_model, day_targets, shop_features)

        # Ensemble
        raw_preds = ensemble_predictions(
            global_preds, shop_preds, product_preds,
            day_targets, product_models
        )

        # Calibrate with day's anchors
        anchor_global = predict_global(global_model, day_anchors)
        anchor_product = predict_product_level(product_models, day_anchors)
        anchor_shop = predict_shop_level(shop_model, day_anchors, shop_features)
        anchor_ensemble = ensemble_predictions(
            anchor_global, anchor_shop, anchor_product,
            day_anchors, product_models
        )

        calibrated_preds = apply_hierarchical_calibration(
            raw_preds, day_targets,
            day_anchors, anchor_ensemble,
            day_anchor_actual
        )

        # Store predictions
        day_indices = np.where(day_target_mask.values)[0]
        all_predictions[day_indices] = calibrated_preds

        print(f"    Predicted {len(calibrated_preds)} prices "
              f"(mean: {calibrated_preds.mean():,.0f})")

    # Create output
    output_df = test_df.copy()
    output_df.loc[targets_df.index, "price"] = all_predictions
    output_df.to_csv(output_file, index=False)

    print(f"\n{'='*70}")
    print(f"  Predictions saved to: {output_file}")
    print(f"  Total rows: {len(output_df)}")
    print(f"  Predicted rows: {len(targets_df)}")
    print(f"  Mean predicted price: {all_predictions.mean():,.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price Prediction Inference")
    parser.add_argument("--test_file", type=str, default="data/test.csv",
                        help="Path to test CSV file")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Output file path for predictions")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Path to training data (optional)")

    args = parser.parse_args()
    predict(args.test_file, args.output, args.train_file)
