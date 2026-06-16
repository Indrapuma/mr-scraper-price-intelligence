"""
Data Loader Module
Downloads and loads the training and test datasets.
"""

import os
import pandas as pd
import gdown


# Google Drive file IDs
TRAIN_FILE_ID = "1ZOYuyrcBJF7fvnG6kBva1m8og5urXHjU"
TEST_FILE_ID = "1Ni2aBrOaV1YEWspZVmBw__HV1a7M37cd"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download_data():
    """Download train and test datasets from Google Drive."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(train_path):
        print("Downloading training data...")
        gdown.download(
            f"https://drive.google.com/uc?id={TRAIN_FILE_ID}",
            train_path,
            quiet=False
        )
    else:
        print("Training data already exists.")

    if not os.path.exists(test_path):
        print("Downloading test data...")
        gdown.download(
            f"https://drive.google.com/uc?id={TEST_FILE_ID}",
            test_path,
            quiet=False
        )
    else:
        print("Test data already exists.")

    return train_path, test_path


def load_data(train_path=None, test_path=None):
    """Load and parse the datasets with proper dtypes."""
    if train_path is None:
        train_path = os.path.join(DATA_DIR, "train.csv")
    if test_path is None:
        test_path = os.path.join(DATA_DIR, "test.csv")

    # Define dtypes for efficient loading
    # Use float64 for all numeric columns to handle potential NaN values
    dtype_map = {
        "shopId": "float64",
        "itemId": "float64",
        "modelId": "float64",
        "price": "float64",
        "priceBeforeDiscount": "float64",
        "promotionId": "float64",
        "cat_id": "float64",
        "stock": "float64",
        "normal_stock": "float64",
        "raw_discount": "float64",
        "show_discount": "float64",
        "is_free_shipping": "object",
        "is_pre_order": "object",
        "item_price_min": "float64",
        "item_price_max": "float64",
        "review_rating": "float64",
        "total_rating_count": "float64",
        "cmt_count": "float64",
        "shop_rating": "float64",
        "shop_response_rate": "float64",
        "shop_follower_count": "float64",
        "is_official_shop": "object",
        "is_verified": "object",
        "is_preferred_plus_seller": "object",
        "brand": "object",
    }

    print("Loading training data...")
    train_df = pd.read_csv(train_path, dtype=dtype_map, parse_dates=["capturedAt"])

    print("Loading test data...")
    test_df = pd.read_csv(test_path, dtype=dtype_map, parse_dates=["capturedAt"])

    # Convert boolean columns
    bool_cols = ["is_free_shipping", "is_pre_order", "is_official_shop",
                 "is_verified", "is_preferred_plus_seller"]
    for col in bool_cols:
        for df in [train_df, test_df]:
            if col in df.columns:
                df[col] = df[col].map({"t": True, "f": False}).astype(bool)

    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")

    return train_df, test_df


def filter_invalid_rows(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Remove rows that are logically impossible regardless of business context.

    These are not outlier judgement calls — they are data quality violations
    that cannot represent real e-commerce observations:
      - price <= 0          (a sold item must have a positive price)
      - show_discount > 100 (percentage cannot exceed 100)
      - show_discount < 0   (percentage cannot be negative)
      - stock < 0           (inventory cannot be negative)
      - normal_stock < 0    (inventory cannot be negative)
      - review_rating > 5   (platform uses 0–5 scale)
      - review_rating < 0
      - shop_rating > 5
      - shop_rating < 0
      - shop_response_rate > 100
      - shop_response_rate < 0

    For test data, price filtering is skipped because target rows have NaN price
    by design — only anchor rows (price filled) are checked.

    Args:
        df: DataFrame to filter
        is_train: If True, price <= 0 rows are dropped.
                  If False, only rows with a known price that is <= 0 are dropped.

    Returns:
        Filtered DataFrame with a reset index.
    """
    original_len = len(df)
    mask_valid = pd.Series(True, index=df.index)

    # Price must be positive where known
    if "price" in df.columns:
        has_price = df["price"].notna()
        mask_valid &= ~(has_price & (df["price"] <= 0))

    # Discount percentage: 0–100
    if "show_discount" in df.columns:
        mask_valid &= ~(df["show_discount"].notna() & (df["show_discount"] < 0))
        mask_valid &= ~(df["show_discount"].notna() & (df["show_discount"] > 100))

    # Stock cannot be negative
    for col in ["stock", "normal_stock"]:
        if col in df.columns:
            mask_valid &= ~(df[col].notna() & (df[col] < 0))

    # Ratings are on a 0–5 scale
    for col in ["review_rating", "shop_rating"]:
        if col in df.columns:
            mask_valid &= ~(df[col].notna() & (df[col] < 0))
            mask_valid &= ~(df[col].notna() & (df[col] > 5))

    # Response rate is a percentage: 0–100
    if "shop_response_rate" in df.columns:
        mask_valid &= ~(df["shop_response_rate"].notna() & (df["shop_response_rate"] < 0))
        mask_valid &= ~(df["shop_response_rate"].notna() & (df["shop_response_rate"] > 100))

    filtered_df = df[mask_valid].reset_index(drop=True)
    n_removed = original_len - len(filtered_df)

    if n_removed > 0:
        print(f"  Data quality filter: removed {n_removed} invalid rows "
              f"({n_removed / original_len * 100:.2f}% of {original_len:,})")
    else:
        print(f"  Data quality filter: no invalid rows found ({original_len:,} rows clean)")

    return filtered_df


def split_test_anchors(test_df):
    """
    Split test data into anchor samples (price filled) and prediction targets (price blank).

    Returns:
        anchors_df: Rows with known prices (100 per day)
        targets_df: Rows that need price prediction
    """
    anchors_df = test_df[test_df["price"].notna()].copy()
    targets_df = test_df[test_df["price"].isna()].copy()

    print(f"Anchor samples: {anchors_df.shape[0]} rows")
    print(f"Prediction targets: {targets_df.shape[0]} rows")

    return anchors_df, targets_df


if __name__ == "__main__":
    download_data()
    train_df, test_df = load_data()
    anchors, targets = split_test_anchors(test_df)
    print("\nData loaded successfully!")
    print(f"Train columns: {list(train_df.columns)}")
    print(f"Train date range: {train_df['capturedAt'].min()} to {train_df['capturedAt'].max()}")
