"""
Feature Engineering Module
Extracts and transforms features for price prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from capturedAt timestamp."""
    df = df.copy()

    df["day_of_week"] = df["capturedAt"].dt.dayofweek
    df["day_of_month"] = df["capturedAt"].dt.day
    df["month"] = df["capturedAt"].dt.month
    df["hour"] = df["capturedAt"].dt.hour
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["capturedAt"].dt.isocalendar().week.astype(int)

    # Date as ordinal for trend capture
    df["date_ordinal"] = df["capturedAt"].dt.date.apply(lambda x: x.toordinal())

    return df


def create_discount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create discount-related derived features."""
    df = df.copy()

    # Discount depth: ratio of discount to original price
    df["has_discount"] = (df["priceBeforeDiscount"] > 0).astype(int)

    # Effective discount percentage (calculated, not just shown)
    df["effective_discount_pct"] = np.where(
        df["priceBeforeDiscount"] > 0,
        (df["priceBeforeDiscount"] - df["price"]) / df["priceBeforeDiscount"] * 100,
        0
    )
    # For prediction targets where price is NaN, use show_discount as proxy
    df["effective_discount_pct"] = df["effective_discount_pct"].fillna(df["show_discount"])

    # Has promotion
    df["has_promotion"] = (df["promotionId"] != 0).astype(int)

    return df


def create_price_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from item price range."""
    df = df.copy()

    # Price range width
    df["price_range"] = df["item_price_max"] - df["item_price_min"]

    # Price range ratio
    df["price_range_ratio"] = np.where(
        df["item_price_min"] > 0,
        df["item_price_max"] / df["item_price_min"],
        1
    )

    # Mid-range price as reference
    df["price_midrange"] = (df["item_price_max"] + df["item_price_min"]) / 2

    return df


def create_shop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create shop engagement and quality features."""
    df = df.copy()

    # Shop quality composite score
    df["shop_quality_score"] = (
        df["shop_rating"] * 0.4 +
        (df["shop_response_rate"] / 100) * 0.3 +
        df["is_official_shop"].astype(int) * 0.15 +
        df["is_verified"].astype(int) * 0.1 +
        df["is_preferred_plus_seller"].astype(int) * 0.05
    )

    # Log transform follower count
    df["log_shop_followers"] = np.log1p(df["shop_follower_count"])

    return df


def create_item_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create item engagement and popularity features."""
    df = df.copy()

    # Log transforms for skewed features
    df["log_total_ratings"] = np.log1p(df["total_rating_count"])
    df["log_cmt_count"] = np.log1p(df["cmt_count"])
    df["log_stock"] = np.log1p(df["stock"])

    # Rating to comment ratio (engagement depth)
    df["rating_comment_ratio"] = np.where(
        df["total_rating_count"] > 0,
        df["cmt_count"] / df["total_rating_count"],
        0
    )

    # Stock utilization ratio
    df["stock_ratio"] = np.where(
        df["normal_stock"] > 0,
        df["stock"] / df["normal_stock"],
        1
    )

    return df


def create_historical_price_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create historical price aggregation features per product.
    These capture the price history of each (shopId, itemId, modelId) combination.
    """
    df = df.copy()

    # Compute historical price stats per product
    product_stats = train_df.groupby(["shopId", "itemId", "modelId"]).agg(
        hist_price_mean=("price", "mean"),
        hist_price_std=("price", "std"),
        hist_price_min=("price", "min"),
        hist_price_max=("price", "max"),
        hist_price_median=("price", "median"),
        hist_price_last=("price", "last"),  # most recent price
        hist_count=("price", "count"),
    ).reset_index()

    # Fill NaN std with 0 (products with single observation)
    product_stats["hist_price_std"] = product_stats["hist_price_std"].fillna(0)

    # Price volatility
    product_stats["hist_price_cv"] = np.where(
        product_stats["hist_price_mean"] > 0,
        product_stats["hist_price_std"] / product_stats["hist_price_mean"],
        0
    )

    # Merge with target dataframe
    df = df.merge(product_stats, on=["shopId", "itemId", "modelId"], how="left")

    return df


def create_historical_shop_price_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """Create shop-level historical price statistics."""
    df = df.copy()

    shop_stats = train_df.groupby("shopId").agg(
        shop_avg_price=("price", "mean"),
        shop_price_std=("price", "std"),
        shop_product_count=("modelId", "nunique"),
    ).reset_index()

    shop_stats["shop_price_std"] = shop_stats["shop_price_std"].fillna(0)

    df = df.merge(shop_stats, on="shopId", how="left")

    return df


def create_category_price_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame
) -> pd.DataFrame:
    """Create category-level historical price statistics."""
    df = df.copy()

    cat_stats = train_df.groupby("cat_id").agg(
        cat_avg_price=("price", "mean"),
        cat_price_std=("price", "std"),
        cat_median_discount=("show_discount", "median"),
    ).reset_index()

    cat_stats["cat_price_std"] = cat_stats["cat_price_std"].fillna(0)

    df = df.merge(cat_stats, on="cat_id", how="left")

    return df


def get_feature_columns() -> list:
    """Return the list of feature columns used for modeling."""
    return [
        # Temporal
        "day_of_week", "day_of_month", "month", "hour", "is_weekend",
        "week_of_year", "date_ordinal",
        # Identifiers (encoded as numeric)
        "shopId", "itemId", "modelId", "cat_id",
        # Price-related
        "priceBeforeDiscount", "item_price_min", "item_price_max",
        "price_range", "price_range_ratio", "price_midrange",
        # Discount
        "has_discount", "show_discount", "raw_discount",
        "has_promotion", "effective_discount_pct",
        # Stock
        "stock", "normal_stock", "log_stock", "stock_ratio",
        # Shop features
        "shop_rating", "shop_response_rate", "shop_follower_count",
        "is_official_shop", "is_verified", "is_preferred_plus_seller",
        "shop_quality_score", "log_shop_followers",
        # Item engagement
        "review_rating", "total_rating_count", "cmt_count",
        "log_total_ratings", "log_cmt_count", "rating_comment_ratio",
        # Shipping/order
        "is_free_shipping", "is_pre_order",
        # Historical price features
        "hist_price_mean", "hist_price_std", "hist_price_min",
        "hist_price_max", "hist_price_median", "hist_price_last",
        "hist_count", "hist_price_cv",
        # Shop-level historical
        "shop_avg_price", "shop_price_std", "shop_product_count",
        # Category-level historical
        "cat_avg_price", "cat_price_std", "cat_median_discount",
    ]


def build_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    is_training: bool = False
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: DataFrame to transform
        train_df: Training data for computing historical features
        is_training: Whether this is the training data itself

    Returns:
        DataFrame with all engineered features
    """
    print("Building features...")

    # Temporal features
    df = create_temporal_features(df)
    print("  ✓ Temporal features")

    # Discount features
    df = create_discount_features(df)
    print("  ✓ Discount features")

    # Price range features
    df = create_price_range_features(df)
    print("  ✓ Price range features")

    # Shop features
    df = create_shop_features(df)
    print("  ✓ Shop features")

    # Item engagement features
    df = create_item_engagement_features(df)
    print("  ✓ Item engagement features")

    # Historical price features (use train_df as reference)
    reference_df = train_df if not is_training else df
    df = create_historical_price_features(df, reference_df)
    print("  ✓ Historical price features")

    # Shop-level historical
    df = create_historical_shop_price_features(df, reference_df)
    print("  ✓ Shop-level historical features")

    # Category-level historical
    df = create_category_price_features(df, reference_df)
    print("  ✓ Category-level historical features")

    # Convert boolean columns to int
    bool_cols = ["is_free_shipping", "is_pre_order", "is_official_shop",
                 "is_verified", "is_preferred_plus_seller"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    print(f"  Total features: {len(get_feature_columns())}")

    return df
