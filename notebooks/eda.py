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
from src.data_loader import load_data, split_test_anchors, filter_invalid_rows

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

    # Filter before splitting so anchors_df/targets_df reflect clean data
    train_df = filter_invalid_rows(train_df, is_train=True)
    test_df  = filter_invalid_rows(test_df,  is_train=False)

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
    # 9. Data Quality — Invalid Rows
    # =========================================================================
    print("\n\n--- 9. DATA QUALITY — INVALID ROWS ---")

    checks = {
        "price <= 0":            train_df["price"].notna() & (train_df["price"] <= 0),
        "show_discount < 0":     train_df["show_discount"].notna() & (train_df["show_discount"] < 0),
        "show_discount > 100":   train_df["show_discount"].notna() & (train_df["show_discount"] > 100),
        "stock < 0":             train_df["stock"].notna() & (train_df["stock"] < 0),
        "normal_stock < 0":      train_df["normal_stock"].notna() & (train_df["normal_stock"] < 0),
        "review_rating < 0":     train_df["review_rating"].notna() & (train_df["review_rating"] < 0),
        "review_rating > 5":     train_df["review_rating"].notna() & (train_df["review_rating"] > 5),
        "shop_rating < 0":       train_df["shop_rating"].notna() & (train_df["shop_rating"] < 0),
        "shop_rating > 5":       train_df["shop_rating"].notna() & (train_df["shop_rating"] > 5),
        "shop_response_rate < 0":   train_df["shop_response_rate"].notna() & (train_df["shop_response_rate"] < 0),
        "shop_response_rate > 100": train_df["shop_response_rate"].notna() & (train_df["shop_response_rate"] > 100),
    }

    print(f"\n{'Violation':<30} {'Count':>8}  {'% of rows':>10}")
    print("-" * 52)
    total_invalid = pd.Series(False, index=train_df.index)
    for label, mask in checks.items():
        n = int(mask.sum())
        pct = n / len(train_df) * 100
        print(f"  {label:<28} {n:>8,}  {pct:>9.3f}%")
        total_invalid |= mask

    total_n = int(total_invalid.sum())
    print(f"\n  Total invalid rows: {total_n:,} ({total_n / len(train_df) * 100:.3f}% of training data)")
    print(f"  (These rows are removed by filter_invalid_rows() before training)")

    # =========================================================================
    # 10. Observations per Product
    # =========================================================================
    print("\n\n--- 10. OBSERVATIONS PER PRODUCT ---")

    obs_per_product = train_df.groupby(
        ["shopId", "itemId", "modelId"]
    )["price"].count().reset_index(name="n_obs")

    print(f"\nTotal unique products: {len(obs_per_product):,}")
    print(f"\nObservations per product:")
    print(obs_per_product["n_obs"].describe())

    thresholds = [1, 2, 5, 10, 20, 50]
    print(f"\n{'Threshold':<20} {'Count':>10} {'% products':>12}")
    print("-" * 44)
    for t in thresholds:
        n = int((obs_per_product["n_obs"] < t).sum())
        pct = n / len(obs_per_product) * 100
        print(f"  < {t} obs{'':<14} {n:>10,}  {pct:>10.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(obs_per_product["n_obs"].clip(upper=100), bins=50,
            color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Number of observations (clipped at 100)")
    ax.set_ylabel("Number of products")
    ax.set_title("Distribution of Historical Observations per Product\n"
                 "(determines cold-start risk and Tier 2 model reliability)")
    ax.axvline(5, color="orange", linestyle="--", label="min for Tier 2 (5 obs)")
    ax.axvline(20, color="red", linestyle="--", label="high-confidence Tier 2 (20 obs)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "obs_per_product.png"), dpi=150)
    plt.close()

    # =========================================================================
    # 11. Train-Test Product Overlap
    # =========================================================================
    print("\n\n--- 11. TRAIN-TEST PRODUCT OVERLAP ---")

    # Build obs lookup: (shopId, itemId, modelId) -> n_obs
    obs_map = {
        (row.shopId, row.itemId, row.modelId): row.n_obs
        for row in obs_per_product.itertuples(index=False)
    }

    # Count TARGET ROWS (not unique products) by history bucket
    obs_counts = np.array([
        obs_map.get((s, i, m), 0)
        for s, i, m in zip(
            targets_df["shopId"].values,
            targets_df["itemId"].values,
            targets_df["modelId"].values,
        )
    ])

    total_targets = len(targets_df)
    no_hist  = int((obs_counts == 0).sum())
    has_hist = total_targets - no_hist

    print(f"\nTarget rows with history in train : {has_hist:,} ({has_hist/total_targets*100:.1f}%)")
    print(f"Target rows with no history        : {no_hist:,} ({no_hist/total_targets*100:.1f}%)")

    buckets = [
        ("no history", obs_counts == 0),
        ("1 obs",      obs_counts == 1),
        ("2-4 obs",    (obs_counts >= 2)  & (obs_counts < 5)),
        ("5-9 obs",    (obs_counts >= 5)  & (obs_counts < 10)),
        ("10-19 obs",  (obs_counts >= 10) & (obs_counts < 20)),
        (">=20 obs",   obs_counts >= 20),
    ]

    print(f"\n{'Bucket':<14} {'Rows':>10} {'% of targets':>14}")
    print("-" * 42)
    for label, mask in buckets:
        n = int(mask.sum())
        print(f"  {label:<12} {n:>10,}  {n/total_targets*100:>12.1f}%")

    # =========================================================================
    # 12. Price: With Discount vs Without
    # =========================================================================
    print("\n\n--- 12. PRICE: WITH DISCOUNT VS WITHOUT ---")

    discounted = train_df[train_df["priceBeforeDiscount"] > 0]["price"]
    non_discounted = train_df[train_df["priceBeforeDiscount"] == 0]["price"]

    print(f"\nWith discount ({len(discounted):,} rows):")
    print(f"  Median price: {discounted.median():,.0f}")
    print(f"  Mean price:   {discounted.mean():,.0f}")
    print(f"\nWithout discount ({len(non_discounted):,} rows):")
    print(f"  Median price: {non_discounted.median():,.0f}")
    print(f"  Mean price:   {non_discounted.mean():,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overlapping log-price distributions
    axes[0].hist(np.log1p(non_discounted), bins=80, alpha=0.6,
                 color="steelblue", label=f"No discount (n={len(non_discounted):,})")
    axes[0].hist(np.log1p(discounted), bins=80, alpha=0.6,
                 color="salmon", label=f"Discounted (n={len(discounted):,})")
    axes[0].set_xlabel("Log(Price + 1)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Price Distribution: Discounted vs Non-Discounted")
    axes[0].legend()

    # Discount depth distribution
    discount_pcts = train_df[train_df["show_discount"] > 0]["show_discount"]
    axes[1].hist(discount_pcts, bins=50, color="salmon", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Discount %")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Discount Depth Distribution\n"
                      f"(median: {discount_pcts.median():.0f}%, "
                      f"mean: {discount_pcts.mean():.0f}%)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "discount_vs_price.png"), dpi=150)
    plt.close()

    # =========================================================================
    # Summary Insights
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("  KEY INSIGHTS")
    print("=" * 70)
    print("""
    1. Most products have relatively stable prices (low CV), making historical
       price features highly predictive.

    2. Discounts and promotions affect a significant portion of items — these
       create temporary price drops that must be accounted for.

    3. The anchor set provides category and shop coverage that can be used
       for hierarchical calibration.

    4. Price distributions are highly skewed — log transform may be beneficial,
       though LightGBM handles it without transformation.

    5. item_price_min and item_price_max are strongly correlated with price,
       providing excellent reference bounds.

    6. Temporal patterns show day-level price shifts that the anchor set
       can help detect and correct.

    7. Data quality violations (invalid rows) are rare but present — filtered
       before training to prevent hist_price_mean contamination.

    8. Observation count per product is skewed: a subset of products has very
       few historical observations, confirming the cold-start concern for Tier 2.

    9. Nearly all test target products (~98.9%) exist in training data, making
       this effectively a "price recall with context" problem.

    10. Discounted items have a distinct price distribution from non-discounted
        ones — bimodal behavior that the model captures via has_discount and
        show_discount features.
    """)

    print("\nEDA complete! Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
