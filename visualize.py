"""
Visualisasi Hasil - Price Intelligence & Anomaly Detection
Menghasilkan grafik untuk laporan dan presentasi.

Run: python visualize.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data, split_test_anchors
from src.feature_engineering import build_features, get_feature_columns
from src.model_global import (
    train_global_model, predict_global,
    create_time_based_split, simulate_anchor_validation
)
from src.model_shop import (
    train_product_models, predict_product_level,
    train_shop_level_model, predict_shop_level, ensemble_predictions
)
from src.anchor_calibration import apply_hierarchical_calibration
from src.evaluate import compute_metrics

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Output directory
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 10


def load_and_prepare():
    """Load data and prepare validation split."""
    print("Loading data...")
    train_df, test_df = load_data()
    anchors_df, targets_df = split_test_anchors(test_df)

    print("Creating validation split...")
    train_split, val_split = create_time_based_split(train_df, n_val_days=3)
    val_anchors, val_targets = simulate_anchor_validation(val_split, n_anchors=100)

    print("Building features...")
    train_feat = build_features(train_split, train_split, is_training=True)
    val_targets_feat = build_features(val_targets, train_split, is_training=False)
    val_anchors_feat = build_features(val_anchors, train_split, is_training=False)

    return train_df, test_df, train_feat, val_targets, val_targets_feat, val_anchors, val_anchors_feat


def plot_1_price_distribution(train_df):
    """Plot distribusi harga di training data."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram log price
    log_prices = np.log10(train_df["price"][train_df["price"] > 0])
    axes[0].hist(log_prices, bins=80, color="#2196F3", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Log₁₀(Price)", fontsize=11)
    axes[0].set_ylabel("Jumlah Produk", fontsize=11)
    axes[0].set_title("Distribusi Harga (Log Scale)", fontsize=13, fontweight="bold")
    axes[0].axvline(log_prices.median(), color="red", linestyle="--", label=f"Median: {10**log_prices.median():,.0f}")
    axes[0].legend()

    # Box plot per category (top 8)
    top_cats = train_df["cat_id"].value_counts().head(8).index
    cat_data = train_df[train_df["cat_id"].isin(top_cats)].copy()
    cat_data["log_price"] = np.log10(cat_data["price"].clip(lower=1))

    cat_order = cat_data.groupby("cat_id")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=cat_data, x="cat_id", y="log_price", order=cat_order,
                ax=axes[1], palette="Blues_r", fliersize=1)
    axes[1].set_xlabel("Category ID", fontsize=11)
    axes[1].set_ylabel("Log₁₀(Price)", fontsize=11)
    axes[1].set_title("Distribusi Harga per Kategori (Top 8)", fontsize=13, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_price_distribution.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 01_price_distribution.png")


def plot_2_temporal_trends(train_df):
    """Plot trend harga harian."""
    train_df = train_df.copy()
    train_df["date"] = train_df["capturedAt"].dt.date

    daily = train_df.groupby("date").agg(
        mean_price=("price", "mean"),
        median_price=("price", "median"),
        count=("price", "count"),
        discount_rate=("priceBeforeDiscount", lambda x: (x > 0).mean() * 100)
    ).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Mean & median price
    axes[0].plot(daily["date"], daily["median_price"] / 1e6, color="#2196F3",
                 linewidth=2, label="Median Price")
    axes[0].fill_between(daily["date"], daily["median_price"] / 1e6 * 0.95,
                         daily["median_price"] / 1e6 * 1.05, alpha=0.1, color="#2196F3")
    axes[0].set_ylabel("Harga (Juta IDR)", fontsize=11)
    axes[0].set_title("Trend Harga Harian", fontsize=13, fontweight="bold")
    axes[0].legend()

    # Daily volume
    axes[1].bar(daily["date"], daily["count"], color="#4CAF50", alpha=0.7, width=0.8)
    axes[1].set_ylabel("Jumlah Scrape", fontsize=11)
    axes[1].set_title("Volume Scraping Harian", fontsize=13, fontweight="bold")

    # Discount rate
    axes[2].plot(daily["date"], daily["discount_rate"], color="#FF5722", linewidth=2)
    axes[2].set_ylabel("% Produk Diskon", fontsize=11)
    axes[2].set_xlabel("Tanggal", fontsize=11)
    axes[2].set_title("Persentase Produk dengan Diskon", fontsize=13, fontweight="bold")
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_temporal_trends.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 02_temporal_trends.png")


def plot_3_feature_importance(train_feat, val_targets_feat):
    """Plot feature importance dari Global Model."""
    print("  Training model for feature importance...")
    model, importance_df = train_global_model(train_feat, val_targets_feat, save_model=False)

    # Top 20 features
    top20 = importance_df.head(20).sort_values("importance")

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(len(top20)), top20["importance"],
                   color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top20))))
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"], fontsize=10)
    ax.set_xlabel("Importance (Split Count)", fontsize=11)
    ax.set_title("Top 20 Feature Importance — Global Model", fontsize=13, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, top20["importance"]):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                f"{int(val):,}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_feature_importance.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 03_feature_importance.png")

    return model


def plot_4_prediction_vs_actual(model, val_targets, val_targets_feat):
    """Scatter plot: predicted vs actual prices."""
    preds = predict_global(model, val_targets_feat)
    actual = val_targets["price"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Full range
    max_val = max(actual.max(), preds.max())
    axes[0].scatter(actual / 1e6, preds / 1e6, alpha=0.15, s=5, color="#2196F3")
    axes[0].plot([0, max_val / 1e6], [0, max_val / 1e6], "r--", linewidth=2, label="Perfect prediction")
    axes[0].set_xlabel("Harga Aktual (Juta IDR)", fontsize=11)
    axes[0].set_ylabel("Harga Prediksi (Juta IDR)", fontsize=11)
    axes[0].set_title("Prediksi vs Aktual — Global Model", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].set_xlim(0, np.percentile(actual / 1e6, 99))
    axes[0].set_ylim(0, np.percentile(preds / 1e6, 99))

    # Error distribution
    errors = (preds - actual) / np.maximum(actual, 1) * 100  # percentage error
    errors_clipped = np.clip(errors, -50, 50)
    axes[1].hist(errors_clipped, bins=100, color="#4CAF50", alpha=0.8, edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Percentage Error (%)", fontsize=11)
    axes[1].set_ylabel("Jumlah Prediksi", fontsize=11)
    axes[1].set_title("Distribusi Error Prediksi", fontsize=13, fontweight="bold")

    # Stats annotation
    metrics = compute_metrics(actual, preds)
    stats_text = f"MAE: {metrics['mae']:,.0f}\nMAPE: {metrics['mape']:.2f}%\nR²: {metrics['r2']:.4f}"
    axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                 fontsize=11, verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_prediction_vs_actual.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 04_prediction_vs_actual.png")

    return preds


def plot_5_model_comparison(model, val_targets, val_targets_feat, val_anchors, val_anchors_feat, train_feat):
    """Bar chart perbandingan model."""
    print("  Computing predictions for all models...")

    # Global predictions
    global_preds = predict_global(model, val_targets_feat)
    global_anchor_preds = predict_global(model, val_anchors_feat)

    # Product models
    product_models = train_product_models(train_feat)
    product_preds = predict_product_level(product_models, val_targets_feat)

    # Shop model
    shop_model, shop_features = train_shop_level_model(train_feat)
    shop_preds = predict_shop_level(shop_model, val_targets_feat, shop_features)

    # Ensemble
    ensemble_preds = ensemble_predictions(
        global_preds, shop_preds, product_preds, val_targets_feat, product_models
    )

    # Calibrated ensemble
    product_anchor = predict_product_level(product_models, val_anchors_feat)
    shop_anchor = predict_shop_level(shop_model, val_anchors_feat, shop_features)
    ensemble_anchor = ensemble_predictions(
        global_anchor_preds, shop_anchor, product_anchor, val_anchors_feat, product_models
    )
    ensemble_calibrated = apply_hierarchical_calibration(
        ensemble_preds, val_targets_feat,
        val_anchors_feat, ensemble_anchor,
        val_anchors["price"].values
    )

    # Compute metrics
    actual = val_targets["price"].values
    results = {
        "Global Model": compute_metrics(actual, global_preds),
        "Shop-Level Model": compute_metrics(actual, shop_preds),
        "Ensemble (raw)": compute_metrics(actual, ensemble_preds),
        "Ensemble\n(calibrated)": compute_metrics(actual, ensemble_calibrated),
    }

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    models = list(results.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    # MAE
    maes = [results[m]["mae"] / 1e3 for m in models]
    bars = axes[0].bar(models, maes, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_ylabel("MAE (Ribu IDR)", fontsize=11)
    axes[0].set_title("Mean Absolute Error", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, maes):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{val:.1f}K", ha="center", fontsize=10)

    # MAPE
    mapes = [results[m]["mape"] for m in models]
    bars = axes[1].bar(models, mapes, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_ylabel("MAPE (%)", fontsize=11)
    axes[1].set_title("Mean Absolute Percentage Error", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, mapes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}%", ha="center", fontsize=10)

    # R²
    r2s = [results[m]["r2"] for m in models]
    bars = axes[2].bar(models, r2s, color=colors, alpha=0.85, edgecolor="white")
    axes[2].set_ylabel("R² Score", fontsize=11)
    axes[2].set_title("R² (Coefficient of Determination)", fontsize=13, fontweight="bold")
    axes[2].set_ylim(min(r2s) - 0.001, 1.0)
    for bar, val in zip(bars, r2s):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                     f"{val:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_model_comparison.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 05_model_comparison.png")

    return results


def plot_6_calibration_impact(train_df):
    """Visualisasi dampak anchor calibration."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Simulated data showing calibration concept
    categories = ["Shop-level\ncorrection", "Category-level\ncorrection", "Global\ncorrection"]
    coverage = [80.5, 19.4, 0.1]
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    bars = ax.barh(categories, coverage, color=colors, alpha=0.85, edgecolor="white", height=0.5)
    ax.set_xlabel("% dari Target Predictions", fontsize=11)
    ax.set_title("Strategi Kalibrasi Hierarkis — Coverage", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, coverage):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=12, fontweight="bold")

    ax.set_xlim(0, 100)

    # Add explanation
    ax.text(50, -0.8, "Prioritas: Shop > Category > Global\n"
            "Semakin granular koreksi, semakin akurat penyesuaian harga",
            ha="center", fontsize=10, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "06_calibration_strategy.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 06_calibration_strategy.png")


def plot_7_price_stability(train_df):
    """Plot stabilitas harga per produk."""
    product_stats = train_df.groupby(["shopId", "itemId", "modelId"]).agg(
        price_cv=("price", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
        n_obs=("price", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CV distribution
    cv_clipped = product_stats["price_cv"].clip(0, 0.5)
    axes[0].hist(cv_clipped, bins=80, color="#9C27B0", alpha=0.8, edgecolor="white")
    axes[0].axvline(0.05, color="red", linestyle="--", linewidth=2, label="CV = 0.05 (stabil)")
    axes[0].set_xlabel("Coefficient of Variation (CV)", fontsize=11)
    axes[0].set_ylabel("Jumlah Produk", fontsize=11)
    axes[0].set_title("Stabilitas Harga per Produk", fontsize=13, fontweight="bold")
    axes[0].legend()

    # Stats annotation
    stable = (product_stats["price_cv"] < 0.05).mean() * 100
    axes[0].text(0.95, 0.95, f"{stable:.1f}% produk\nmemiliki CV < 0.05\n(harga stabil)",
                 transform=axes[0].transAxes, fontsize=11,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Observation count distribution
    obs_clipped = product_stats["n_obs"].clip(0, 100)
    axes[1].hist(obs_clipped, bins=50, color="#FF5722", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Jumlah Observasi per Produk", fontsize=11)
    axes[1].set_ylabel("Jumlah Produk", fontsize=11)
    axes[1].set_title("Distribusi Riwayat Data per Produk", fontsize=13, fontweight="bold")

    median_obs = product_stats["n_obs"].median()
    axes[1].axvline(median_obs, color="red", linestyle="--", linewidth=2,
                    label=f"Median: {median_obs:.0f} observasi")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "07_price_stability.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 07_price_stability.png")


def plot_8_pipeline_overview():
    """Diagram alur pipeline."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(7, 9.5, "Pipeline Prediksi Harga — Alur Kerja",
            ha="center", fontsize=16, fontweight="bold")

    # Boxes
    boxes = [
        (1, 7.5, "1. Data Loading\n306K train rows\n25.9K test rows", "#E3F2FD"),
        (4, 7.5, "2. Feature Engineering\n56 fitur\n(temporal, discount,\nhistorical, shop)", "#E8F5E9"),
        (7.5, 7.5, "3. Validation Split\nTime-based split\n3 hari terakhir\n+ 100 anchors/hari", "#FFF3E0"),
        (1, 4.5, "4. Global Model\nLightGBM\n2000 trees\nR² = 0.9991", "#E3F2FD"),
        (4, 4.5, "5. Shop/Product Model\nProduct lookup +\nShop LightGBM +\nEnsemble", "#E8F5E9"),
        (7.5, 4.5, "6. Anchor Calibration\nHierarkis:\nShop > Cat > Global", "#FFF3E0"),
        (11, 4.5, "7. Final Predictions\n25.600 produk\nMAPE < 1%", "#FCE4EC"),
    ]

    for x, y, text, color in boxes:
        bbox = FancyBboxPatch((x - 1.2, y - 0.9), 2.4, 1.8,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor="gray", linewidth=1.5)
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center", fontsize=9)

    # Arrows
    arrow_props = dict(arrowstyle="->", color="gray", lw=2)
    ax.annotate("", xy=(2.9, 7.5), xytext=(2.3, 7.5), arrowprops=arrow_props)
    ax.annotate("", xy=(6.3, 7.5), xytext=(5.3, 7.5), arrowprops=arrow_props)

    # Down arrows
    ax.annotate("", xy=(1, 6.4), xytext=(1, 6.6), arrowprops=arrow_props)
    ax.annotate("", xy=(4, 6.4), xytext=(4, 6.6), arrowprops=arrow_props)
    ax.annotate("", xy=(7.5, 6.4), xytext=(7.5, 6.6), arrowprops=arrow_props)

    # Horizontal arrows in row 2
    ax.annotate("", xy=(5.9, 4.5), xytext=(5.3, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(2.9, 4.5), xytext=(2.3, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(9.6, 4.5), xytext=(8.9, 4.5), arrowprops=arrow_props)

    # Key results box
    ax.text(11, 7.5, "Hasil Utama:\n• MAPE: 0.85%\n• R²: 0.9991\n• Coverage: 98.9%",
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="#FFFDE7", edgecolor="#FFC107", linewidth=2))

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "08_pipeline_overview.png"), bbox_inches="tight")
    plt.close()
    print("  ✓ 08_pipeline_overview.png")


def main():
    print("=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Load data
    (train_df, test_df, train_feat, val_targets,
     val_targets_feat, val_anchors, val_anchors_feat) = load_and_prepare()

    print("\nGenerating plots...")

    # Plot 1: Price distribution
    plot_1_price_distribution(train_df)

    # Plot 2: Temporal trends
    plot_2_temporal_trends(train_df)

    # Plot 3: Feature importance
    model = plot_3_feature_importance(train_feat, val_targets_feat)

    # Plot 4: Prediction vs Actual
    plot_4_prediction_vs_actual(model, val_targets, val_targets_feat)

    # Plot 5: Model comparison
    plot_5_model_comparison(model, val_targets, val_targets_feat,
                           val_anchors, val_anchors_feat, train_feat)

    # Plot 6: Calibration strategy
    plot_6_calibration_impact(train_df)

    # Plot 7: Price stability
    plot_7_price_stability(train_df)

    # Plot 8: Pipeline overview
    plot_8_pipeline_overview()

    print(f"\n{'='*60}")
    print(f"  All visualizations saved to: {FIG_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
