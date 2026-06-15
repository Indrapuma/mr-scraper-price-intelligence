"""
Approach 1: Global Marketplace Model
Trains a single LightGBM model on the entire historical dataset.
Uses anchor samples for global bias correction.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import joblib
import os

from .feature_engineering import get_feature_columns, build_features


SEED = 42
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def get_lgb_params():
    """LightGBM hyperparameters for the global model."""
    return {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 255,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 50,
        "max_depth": -1,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 2000,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }


def train_global_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame = None,
    save_model: bool = True
) -> tuple:
    """
    Train the global marketplace model.

    Args:
        train_df: Training dataframe with features already built
        val_df: Optional validation dataframe
        save_model: Whether to save the model to disk

    Returns:
        (model, feature_importance_df)
    """
    feature_cols = get_feature_columns()

    # Filter to available features
    available_features = [c for c in feature_cols if c in train_df.columns]
    print(f"Training Global Model with {len(available_features)} features...")

    X_train = train_df[available_features]
    y_train = train_df["price"]

    # Handle any remaining NaN in features
    X_train = X_train.fillna(-1)

    params = get_lgb_params()

    if val_df is not None:
        X_val = val_df[available_features].fillna(-1)
        y_val = val_df["price"]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100)
            ]
        )
        print(f"Best iteration: {model.best_iteration_}")
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

    # Feature importance
    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features:")
    print(importance_df.head(15).to_string(index=False))

    # Save model
    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "global_model.joblib")
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")

    return model, importance_df


def predict_global(
    model,
    df: pd.DataFrame,
    feature_cols: list = None
) -> np.ndarray:
    """
    Generate predictions using the global model.

    Args:
        model: Trained LightGBM model
        df: DataFrame with features
        feature_cols: Feature columns to use (defaults to get_feature_columns())

    Returns:
        Numpy array of predictions
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].fillna(-1)

    predictions = model.predict(X)

    # Ensure non-negative prices
    predictions = np.maximum(predictions, 0)

    return predictions


def create_time_based_split(
    train_df: pd.DataFrame,
    n_val_days: int = 3
) -> tuple:
    """
    Create a time-based validation split that simulates the test scenario.
    Uses the last N days of training data as validation (simulating an outage).

    Args:
        train_df: Full training dataframe
        n_val_days: Number of days to hold out for validation

    Returns:
        (train_split, val_split)
    """
    train_df = train_df.copy()
    train_df["date"] = train_df["capturedAt"].dt.date

    # Get unique dates sorted
    unique_dates = sorted(train_df["date"].unique())

    # Last n_val_days as validation
    val_dates = unique_dates[-n_val_days:]
    train_dates = unique_dates[:-n_val_days]

    print(f"Training dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)")
    print(f"Validation dates: {val_dates[0]} to {val_dates[-1]} ({len(val_dates)} days)")

    train_split = train_df[train_df["date"].isin(train_dates)].drop(columns=["date"])
    val_split = train_df[train_df["date"].isin(val_dates)].drop(columns=["date"])

    print(f"Train split: {train_split.shape[0]} rows")
    print(f"Val split: {val_split.shape[0]} rows")

    return train_split, val_split


def simulate_anchor_validation(
    val_df: pd.DataFrame,
    n_anchors: int = 100
) -> tuple:
    """
    Simulate the anchor scenario on validation data.
    Randomly select n_anchors per day as known prices, predict the rest.

    Returns:
        (anchors_df, targets_df)
    """
    val_df = val_df.copy()
    val_df["date"] = val_df["capturedAt"].dt.date

    anchors_list = []
    targets_list = []

    for date, group in val_df.groupby("date"):
        if len(group) <= n_anchors:
            anchors_list.append(group)
            continue

        anchor_idx = group.sample(n=n_anchors, random_state=SEED).index
        anchors_list.append(group.loc[anchor_idx])
        targets_list.append(group.loc[~group.index.isin(anchor_idx)])

    anchors_df = pd.concat(anchors_list, ignore_index=True)
    targets_df = pd.concat(targets_list, ignore_index=True)

    anchors_df = anchors_df.drop(columns=["date"])
    targets_df = targets_df.drop(columns=["date"])

    print(f"Simulated anchors: {anchors_df.shape[0]} rows")
    print(f"Simulated targets: {targets_df.shape[0]} rows")

    return anchors_df, targets_df
