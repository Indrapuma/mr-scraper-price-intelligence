"""
Main Pipeline: Price Intelligence & Anomaly Detection
MrScraper AI Engineer Take Home Test

This script runs the full pipeline:
1. Load data
2. Feature engineering
3. Train & validate Global Model (Approach 1)
4. Train & validate Shop/Product Model (Approach 2)
5. Anchor calibration
6. Compare approaches
7. Generate predictions for test set
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, split_test_anchors, download_data, filter_invalid_rows
from src.feature_engineering import build_features, get_feature_columns
from src.model_global import (
    train_global_model, predict_global,
    create_time_based_split, simulate_anchor_validation
)
from src.model_shop import (
    train_product_models, train_shop_level_model,
    predict_product_level, predict_shop_level,
    ensemble_predictions
)
from src.anchor_calibration import (
    compute_global_bias, apply_global_bias_correction,
    apply_hierarchical_calibration, select_best_calibration
)
from src.evaluate import (
    compute_metrics, print_metrics, compare_models,
    evaluate_calibration_impact
)
import joblib

SEED = 42
np.random.seed(SEED)


def main():
    print("=" * 70)
    print("  PRICE INTELLIGENCE & ANOMALY DETECTION PIPELINE")
    print("  MrScraper AI Engineer Take Home Test")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 1: Loading Data")
    print("=" * 70)

    # Download if not present
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data files not found. Attempting download...")
        download_data()

    train_df, test_df = load_data(train_path, test_path)

    # Data quality filter — remove logically impossible rows
    print("\nApplying data quality filters...")
    train_df = filter_invalid_rows(train_df, is_train=True)
    test_df = filter_invalid_rows(test_df, is_train=False)

    # Split test into anchors and targets
    anchors_df, targets_df = split_test_anchors(test_df)

    # =========================================================================
    # STEP 2: Validation Split (simulate outage scenario)
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 2: Creating Validation Split")
    print("=" * 70)

    train_split, val_split = create_time_based_split(train_df, n_val_days=3)

    # Simulate anchor scenario on validation
    val_anchors, val_targets = simulate_anchor_validation(val_split, n_anchors=100)

    # =========================================================================
    # STEP 3: Feature Engineering
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 3: Feature Engineering")
    print("=" * 70)

    # Build features for training split
    print("\n--- Training split features ---")
    train_split_feat = build_features(train_split, train_split, is_training=True)

    # Build features for validation
    print("\n--- Validation target features ---")
    val_targets_feat = build_features(val_targets, train_split, is_training=False)

    print("\n--- Validation anchor features ---")
    val_anchors_feat = build_features(val_anchors, train_split, is_training=False)

    # =========================================================================
    # STEP 4: Train Global Model (Approach 1)
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 4: Training Global Marketplace Model (Approach 1)")
    print("=" * 70)

    # Train with validation for early stopping
    global_model, global_importance = train_global_model(
        train_split_feat, val_targets_feat, save_model=True
    )

    # Predict on validation targets
    global_val_preds = predict_global(global_model, val_targets_feat)
    global_val_metrics = compute_metrics(val_targets["price"].values, global_val_preds)
    print_metrics(global_val_metrics, "Global Model (Before Calibration)")

    # Predict on anchors for calibration
    global_anchor_preds = predict_global(global_model, val_anchors_feat)

    # Apply anchor calibration
    global_val_calibrated = apply_hierarchical_calibration(
        global_val_preds, val_targets_feat,
        val_anchors_feat, global_anchor_preds,
        val_anchors["price"].values
    )

    global_calibrated_metrics = compute_metrics(
        val_targets["price"].values, global_val_calibrated
    )
    print_metrics(global_calibrated_metrics, "Global Model (After Calibration)")

    evaluate_calibration_impact(
        val_targets["price"].values,
        global_val_preds,
        global_val_calibrated,
        "Global Model Anchor Calibration"
    )

    # =========================================================================
    # STEP 5: Train Shop/Product Model (Approach 2)
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 5: Training Shop/Product Level Model (Approach 2)")
    print("=" * 70)

    # Train product-level models
    product_models = train_product_models(train_split_feat)

    # Train shop-level LightGBM
    shop_model, shop_features = train_shop_level_model(train_split_feat)

    # Predict on validation
    product_val_preds = predict_product_level(product_models, val_targets_feat)
    shop_val_preds = predict_shop_level(shop_model, val_targets_feat, shop_features)

    # Ensemble
    ensemble_val_preds = ensemble_predictions(
        global_val_preds, shop_val_preds, product_val_preds,
        val_targets_feat, product_models
    )

    ensemble_metrics = compute_metrics(val_targets["price"].values, ensemble_val_preds)
    print_metrics(ensemble_metrics, "Ensemble (Before Calibration)")

    # Apply calibration to ensemble
    # Get anchor predictions for ensemble
    product_anchor_preds = predict_product_level(product_models, val_anchors_feat)
    shop_anchor_preds = predict_shop_level(shop_model, val_anchors_feat, shop_features)
    ensemble_anchor_preds = ensemble_predictions(
        global_anchor_preds, shop_anchor_preds, product_anchor_preds,
        val_anchors_feat, product_models
    )

    ensemble_val_calibrated = apply_hierarchical_calibration(
        ensemble_val_preds, val_targets_feat,
        val_anchors_feat, ensemble_anchor_preds,
        val_anchors["price"].values
    )

    ensemble_calibrated_metrics = compute_metrics(
        val_targets["price"].values, ensemble_val_calibrated
    )
    print_metrics(ensemble_calibrated_metrics, "Ensemble (After Calibration)")

    # =========================================================================
    # STEP 6: Model Comparison
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 6: Model Comparison")
    print("=" * 70)

    compare_models({
        "Global (raw)": global_val_metrics,
        "Global (calibrated)": global_calibrated_metrics,
        "Ensemble (raw)": ensemble_metrics,
        "Ensemble (calibrated)": ensemble_calibrated_metrics,
    })

    # =========================================================================
    # STEP 7: Final Model Training & Test Predictions
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 7: Final Training & Test Set Predictions")
    print("=" * 70)

    # Retrain on FULL training data
    print("\nRetraining on full training data...")
    full_train_feat = build_features(train_df, train_df, is_training=True)

    # Global model on full data
    final_global_model, _ = train_global_model(full_train_feat, save_model=True)

    # Product models on full data
    final_product_models = train_product_models(full_train_feat)

    # Shop model on full data
    final_shop_model, final_shop_features = train_shop_level_model(full_train_feat)

    # Save all models for inference
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(final_shop_model, os.path.join(model_dir, "shop_model.joblib"))
    joblib.dump(final_shop_features, os.path.join(model_dir, "shop_features.joblib"))
    joblib.dump(final_product_models, os.path.join(model_dir, "product_models.joblib"))
    print("All models saved to models/ directory")

    # Build features for test set
    print("\n--- Test set features ---")
    anchors_feat = build_features(anchors_df, train_df, is_training=False)
    targets_feat = build_features(targets_df, train_df, is_training=False)

    # Generate predictions for test targets
    print("\nGenerating predictions for test targets...")

    # Global predictions
    test_global_preds = predict_global(final_global_model, targets_feat)

    # Product-level predictions
    test_product_preds = predict_product_level(final_product_models, targets_feat)

    # Shop-level predictions
    test_shop_preds = predict_shop_level(final_shop_model, targets_feat, final_shop_features)

    # Ensemble
    test_ensemble_preds = ensemble_predictions(
        test_global_preds, test_shop_preds, test_product_preds,
        targets_feat, final_product_models
    )

    # Calibrate using actual anchor set
    anchor_global_preds = predict_global(final_global_model, anchors_feat)
    anchor_product_preds = predict_product_level(final_product_models, anchors_feat)
    anchor_shop_preds = predict_shop_level(final_shop_model, anchors_feat, final_shop_features)
    anchor_ensemble_preds = ensemble_predictions(
        anchor_global_preds, anchor_shop_preds, anchor_product_preds,
        anchors_feat, final_product_models
    )

    # Apply hierarchical calibration
    test_final_preds = apply_hierarchical_calibration(
        test_ensemble_preds, targets_feat,
        anchors_feat, anchor_ensemble_preds,
        anchors_df["price"].values
    )

    # =========================================================================
    # STEP 8: Save Predictions
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  STEP 8: Saving Predictions")
    print("=" * 70)

    # Create output: fill in predicted prices in the test dataframe
    output_df = test_df.copy()

    # Fill in predictions for target rows
    target_indices = targets_df.index
    output_df.loc[target_indices, "price"] = test_final_preds

    # Save full predictions
    output_path = os.path.join(os.path.dirname(__file__), "predictions.csv")
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    print(f"Total rows: {len(output_df)}")
    print(f"Anchor rows (unchanged): {len(anchors_df)}")
    print(f"Predicted rows: {len(targets_df)}")

    # Summary statistics of predictions
    print(f"\nPrediction Statistics:")
    print(f"  Mean predicted price: {test_final_preds.mean():,.2f}")
    print(f"  Median predicted price: {np.median(test_final_preds):,.2f}")
    print(f"  Min predicted price: {test_final_preds.min():,.2f}")
    print(f"  Max predicted price: {test_final_preds.max():,.2f}")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
