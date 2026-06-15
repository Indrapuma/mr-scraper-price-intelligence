"""
Exploratory Data Analysis (EDA)
Price Intelligence & Anomaly Detection

Run: python notebooks/eda.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import load_data, split_test_anchors

# Settings
plt.style.use("seaborn-v0_8-whitegrid")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_eda():
    print("=" * 70)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Load data
    train_df, test_df = load_data()
    anchors_df, targets_df = split_test_anchors(test_df)

    # =========================================================================
    # 1. Basic Statistics
    # =========================================================================
    print("\n\n--- 1. BASIC STATISTICS ---")
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Anchor samples: {anchors_df.shape[0]}")
    print(f"Prediction targets: {targets_df.shape[0]}")

    print(f"\nDate range (train): {train_df['capturedAt'].min()} to {train_df['capturedAt'].max()}")
    print(f"Date range (test): {test_df['capturedAt'].min()} to {test_df['capturedAt'].max()}")

    print(f"\nUnique shops: {train_df['shopId'].nunique()}")
    print(f"Unique items: {train_df['itemId'].nunique()}")
    print(f"Unique models: {train_df['modelId'].nunique()}")
    print(f"Unique categories: {train_df['cat_id'].nunique()}")
    print(f"Unique brands: {train_df['brand'].nunique()}")

    # =========================================================================
    # 2. Price Distribution
    # =========================================================================
    print("\n\n--- 2. PRICE DISTRIBUTION ---")
    print(f"\nPrice statistics:")
    print(train_df["price"].describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Price distribution (log scale)
    axes[0].hist(np.log1p(train_df["price"]), bins=100, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Log(Price + 1)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Price Distribution (Log Scale)")

    # Price by category (top 10)
    top_cats = train_df["cat_id"].value_counts().head(10).index
    cat_prices = train_df[train_df["cat_id"].isin(top_cats)]
    cat_prices.boxplot(column="price", by="cat_id", ax=axes[1])
    axes[1].set_title("Price by Top 10 Categories")
    axes[1].set_xlabel("Category ID")
    axes[1].set_ylabel("Price")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), dpi=150)
    plt.close()

    # =========================================================================
    # 3. Temporal Patterns
    # =========================================================================
    print("\n\n--- 3. TEMPORAL PATTERNS ---")

    train_df["date"] = train_df["capturedAt"].dt.date
    daily_prices = train_df.groupby("date").agg(
        mean_price=("price", "mean"),
        median_price=("price", "median"),
        count=("price", "count")
    ).reset_index()

    print(f"\nDaily scrape count:")
    print(daily_prices["count"].describe())

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(daily_prices["date"], daily_prices["mean_price"], label="Mean", alpha=0.8)
    axes[0].plot(daily_prices["date"], daily_prices["median_price"], label="Median", alpha=0.8)
    axes[0].set_title("Daily Price Trends")
    axes[0].set_ylabel("Price")
    axes[0].legend()

    axes[1].bar(daily_prices["date"], daily_prices["count"], alpha=0.7, color="steelblue")
    axes[1].set_title("Daily Scrape Volume")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "temporal_patterns.png"), dpi=150)
    plt.close()

    # =========================================================================
    # 4. Discount Analysis
    # =========================================================================
    print("\n\n--- 4. DISCOUNT ANALYSIS ---")

    has_discount = train_df["priceBeforeDiscount"] > 0
    print(f"\nItems with discount: {has_discount.sum()} ({has_discount.mean()*100:.1f}%)")

    has_promotion = train_df["promotionId"] != 0
    print(f"Items with promotion: {has_promotion.sum()} ({has_promotion.mean()*100:.1f}%)")

    print(f"\nDiscount percentage distribution:")
    print(train_df[train_df["show_discount"] > 0]["show_discount"].describe())

    # =========================================================================
    # 5. Price Stability per Product
    # =========================================================================
    print("\n\n--- 5. PRICE STABILITY ---")

    product_stats = train_df.groupby(["shopId", "itemId", "modelId"]).agg(
        price_mean=("price", "mean"),
        price_std=("price", "std"),
        price_count=("price", "count"),
    ).reset_index()

    product_stats["price_cv"] = product_stats["price_std"] / product_stats["price_mean"]
    product_stats["price_cv"] = product_stats["price_cv"].fillna(0)

    print(f"\nCoefficient of Variation (price stability):")
    print(product_stats["price_cv"].describe())
    print(f"\nProducts with CV < 0.01 (very stable): "
          f"{(product_stats['price_cv'] < 0.01).sum()} "
          f"({(product_stats['price_cv'] < 0.01).mean()*100:.1f}%)")
    print(f"Products with CV < 0.05 (stable): "
          f"{(product_stats['price_cv'] < 0.05).sum()} "
          f"({(product_stats['price_cv'] < 0.05).mean()*100:.1f}%)")
    print(f"Products with CV > 0.2 (volatile): "
          f"{(product_stats['price_cv'] > 0.2).sum()} "
          f"({(product_stats['price_cv'] > 0.2).mean()*100:.1f}%)")

    # =========================================================================
    # 6. Anchor Set Analysis
    # =========================================================================
    print("\n\n--- 6. ANCHOR SET ANALYSIS ---")
    print(f"\nAnchor samples per day:")
    anchors_df["date"] = anchors_df["capturedAt"].dt.date
    print(anchors_df.groupby("date").size())

    print(f"\nAnchor category coverage:")
    anchor_cats = anchors_df["cat_id"].nunique()
    total_cats = test_df["cat_id"].nunique()
    print(f"  Categories in anchors: {anchor_cats}/{total_cats} "
          f"({anchor_cats/total_cats*100:.1f}%)")

    anchor_shops = anchors_df["shopId"].nunique()
    total_shops = test_df["shopId"].nunique()
    print(f"  Shops in anchors: {anchor_shops}/{total_shops} "
          f"({anchor_shops/total_shops*100:.1f}%)")

    # =========================================================================
    # 7. Feature Correlations
    # =========================================================================
    print("\n\n--- 7. FEATURE CORRELATIONS WITH PRICE ---")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    correlations = train_df[numeric_cols].corrwith(train_df["price"]).abs().sort_values(ascending=False)
    print("\nTop correlations with price:")
    print(correlations.head(15))

    # =========================================================================
    # 8. Missing Values
    # =========================================================================
    print("\n\n--- 8. MISSING VALUES ---")
    missing = train_df.isnull().sum()
    missing_pct = missing / len(train_df) * 100
    missing_report = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_report = missing_report[missing_report["count"] > 0]
    if len(missing_report) > 0:
        print(missing_report)
    else:
        print("No missing values in training data!")

    # =========================================================================
    # Summary Insights
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHTS")
    print("=" * 70)
    print("""
    1. Most products have relatively stable prices (low CV), making historical
       price features highly predictive.

    2. Discounts and promotions affect a significant portion of items - these
       create temporary price drops that must be accounted for.

    3. The anchor set provides category and shop coverage that can be used
       for hierarchical calibration.

    4. Price distributions are highly skewed - log transform may be beneficial.

    5. Item_price_min and item_price_max are strongly correlated with price,
       providing excellent reference bounds.

    6. Temporal patterns show day-level price shifts that the anchor set
       can help detect and correct.
    """)

    print("\nEDA complete! Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
