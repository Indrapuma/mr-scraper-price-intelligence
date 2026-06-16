# Price Intelligence

**MrScraper AI Engineer — Take Home Test**

## Overview

This project builds a price prediction system that reconstructs product prices during scraping outages. Using historical data (306K rows) and a small anchor set (100 manually collected prices per day), the system predicts prices for thousands of products with **MAPE < 1%** and **R² = 0.9991**.

## Approach Summary

### Tier 1 — Global Marketplace Model
A single LightGBM model trained on the entire historical dataset with 56 engineered features. Uses anchor samples for hierarchical bias correction.

- **MAE**: 172,839 | **MAPE**: 0.85% | **R²**: 0.9991
- Best iteration: 1197 (early stopping)
- Top features: `hist_price_mean`, `date_ordinal`, `shop_quality_score`, `item_price_min`

### Tier 2 — Shop/Product Level Model
A multi-level ensemble combining:
1. **Product-level lookup** — Historical price statistics per (shopId, itemId, modelId)
2. **Shop-level LightGBM** — Captures per-shop pricing patterns
3. **Adaptive ensemble** — Weights based on data richness per product

Coverage: 98.9% of test products have historical data.

### Anchor Calibration Strategy
Hierarchical correction using the 100 anchor samples:
- **Shop-level** (80.5% of predictions) — most granular
- **Category-level** (19.4%) — fallback
- **Global-level** (0.1%) — last resort

## Key Findings

1. **Prices are remarkably stable** — majority of products have CV < 0.05, making historical mean extremely predictive
2. **Historical features dominate** — `hist_price_mean` alone provides strong baseline
3. **Anchor calibration has minimal impact on "normal" days** — but becomes critical during platform-wide price shifts (flash sales, promotions)
4. **Ensemble marginally outperforms global model** for products with rich history

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data download, loading & quality filtering
│   ├── feature_engineering.py  # 56 engineered features
│   ├── model_global.py         # Tier 1: Global Marketplace Model
│   ├── model_shop.py           # Tier 2: Shop/Product Level Model
│   ├── anchor_calibration.py   # Hierarchical anchor calibration
│   └── evaluate.py             # Metrics & comparison
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis (12 sections)
│   └── anchor_simulation.py    # Anchor calibration impact simulation
├── figures/                    # Generated visualizations
├── models/                     # Saved model artifacts (.gitkeep)
├── data/                       # CSV files (not committed)
├── main.py                     # Full pipeline: train + validate + predict
├── predict.py                  # Inference-only script (for interview)
├── visualize.py                # Generate all result visualizations
├── pyproject.toml              # uv project config & pinned dependencies
├── .python-version             # Python 3.11 (required for wheel availability)
├── requirements.txt            # Pinned dependencies (pip fallback)
└── README.md
```

## Setup & Installation

### With uv (recommended)

```bash
git clone https://github.com/<your-username>/mrscraper-price-intelligence.git
cd mrscraper-price-intelligence

# Install uv if not already installed
# https://docs.astral.sh/uv/getting-started/installation/

uv sync          # Creates .venv with Python 3.11 + all dependencies
```

Activate the environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### With pip (fallback)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

## Running the Pipeline

### 1. Download Data
```bash
python src/data_loader.py
```
Downloads `train.csv` (306K rows) and `test.csv` from Google Drive into `data/`.

### 2. Full Pipeline (Train + Validate + Predict)
```bash
python main.py
```
This runs:
- Data loading & validation split
- Feature engineering (56 features)
- Tier 1: Global model training + evaluation
- Tier 2: Shop/Product model training + evaluation
- Model comparison
- Anchor calibration
- Final predictions → `predictions.csv`

### 3. Inference Only (for interview)
```bash
python predict.py --test_file data/test_full.csv --output predictions_full.csv
```
Uses pre-trained models from `models/` to predict on new test data.

### 4. Generate Visualizations
```bash
python visualize.py
```
Outputs 8 plots to `figures/`.

## Reproducing Results

- **Random seed**: 42
- **Python**: 3.11 (pinned via `.python-version` for wheel compatibility)
- **Validation**: Time-based split (last 3 days of training data)
- **Anchor simulation**: 100 random samples per validation day

Models are saved to `models/` after running `main.py`. To reproduce from scratch, delete `models/` and re-run `main.py`.

## Validation Methodology

The validation split simulates the real outage scenario:
1. Hold out the last 3 days of training data
2. From each day, randomly select 100 samples as "anchor set"
3. Predict prices for the remaining samples
4. Compare predictions against actual prices

This mirrors the exact test conditions described in the problem statement.

## Discussion: When to Use Each Approach

| Scenario | Recommended Approach |
|----------|---------------------|
| Products with rich history (≥20 observations) | Tier 2 (Shop/Product) |
| Products with sparse history (<10 observations) | Tier 1 (Global) |
| Platform-wide price shift (promo event) | Tier 1 + strong anchor calibration |
| Shop-specific pricing change | Tier 2 with shop-level calibration |
| New/cold-start products | Tier 1 (Global) |
| Production outage (general) | Ensemble of both |

## Limitations & Future Work

1. **No time-series modeling** — Could benefit from LSTM/Transformer for temporal patterns
2. **Anchor selection bias** — Real anchor samples may not be random
3. **Extreme price events** — Flash sales create outliers that are hard to predict
4. **Cross-shop intelligence** — Same product across multiple shops could be leveraged
5. **Online learning** — Incremental model updates as new data arrives daily
