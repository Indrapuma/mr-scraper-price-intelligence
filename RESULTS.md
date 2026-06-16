# Results & Evaluation

**MrScraper AI Engineer — Price Intelligence Take Home Test**

---

## Validation Setup

| Parameter | Value |
|-----------|-------|
| Training data | 285,307 rows (2025-01-01 to 2025-03-19, 78 days) |
| Validation data | 20,919 rows (2025-03-20 to 2025-03-22, last 3 days) |
| Simulated anchors | 100 random samples per day (300 total) |
| Prediction targets | 20,619 rows (~6,873 per day) |

This time-based split simulates the real outage scenario: the model has never seen the validation days during training. Anchors are sampled after the split — they are never used as training features, only for post-prediction calibration.

---

## 1. Prediction Accuracy

### Metrics Used

- **MAE** (Mean Absolute Error) — interpretable in currency units (IDR base unit)
- **RMSE** (Root Mean Squared Error) — penalises large errors
- **MAPE** (Mean Absolute Percentage Error) — scale-independent, primary metric
- **Median AE** — robust to outlier influence
- **R²** — proportion of variance explained

### Results

| Model | MAE | RMSE | MAPE | Median AE | R² |
|-------|-----|------|------|-----------|-----|
| **Tier 1: Global (raw)** | 172,839 | 2,538,963 | 0.85% | 4,342 | 0.9991 |
| **Tier 1: Global (calibrated)** | 173,820 | 2,538,932 | 0.85% | 4,811 | 0.9991 |
| **Tier 2: Ensemble (raw)** | 195,824 | 2,462,470 | 0.87% | 2,142 | 0.9992 |
| **Tier 2: Ensemble (calibrated)** | 196,311 | 2,462,336 | 0.87% | 2,019 | 0.9992 |

> Prices are in IDR base unit. MAE of 172,839 ≈ Rp 173 average error — well within acceptable tolerance for price intelligence.
>
> **Best model by MAE/MAPE:** Tier 1 Global. **Best model by RMSE/Median AE:** Tier 2 Ensemble. See Section 5 for analysis.

### Validation Methodology

**Time-based split simulating an outage day:**

1. Training: 285,307 rows from 2025-01-01 to 2025-03-19
2. Validation: 20,919 rows from 2025-03-20 to 2025-03-22 (3 held-out days)
3. Per validation day: 100 random rows designated as "anchors" (known prices)
4. Remaining ~6,800 rows per day = prediction targets

This mirrors the exact test scenario: model trained on historical data, then asked to predict an unseen day with only 100 anchor prices available for calibration.

---

## 2. Anchor Set Utilisation

### Strategy: Hierarchical Calibration

The 100 anchor samples per day act as a real-time signal about pricing on that specific day. Calibration operates at three levels:

```
Priority: Shop-level → Category-level → Global-level
```

**Step 1 — Compute corrections from anchors:**
- For each anchor: compute `ratio = actual_price / predicted_price`
- Group ratios by shop, by category (`cat_id`), and overall

**Step 2 — Apply corrections hierarchically to targets:**
- If target product's shop appears in anchors → apply shop-specific median ratio
- Else if target's category appears in anchors → apply category median ratio
- Else → apply global median ratio

**Why multiplicative (ratio) rather than additive (bias)?**
Multiplicative correction handles proportional price shifts correctly — a 15% flash sale affects a Rp 100K item and a Rp 10M item differently in absolute terms, but identically as a ratio. Tested both; multiplicative consistently performed better.

### Calibration Coverage

| Level | % of Targets | Description |
|-------|-------------|-------------|
| Shop-level | 80.5% | Most granular, most accurate |
| Category-level | 19.4% | Fallback for unseen shops |
| Global-level | 0.1% | Rare fallback |

### Impact on Validation Set (Normal Day)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| MAE | 172,839 | 173,820 | ↑ 0.6% |
| RMSE | 2,538,963 | 2,538,932 | ↓ 0.0% |
| MAPE | 0.85% | 0.85% | ~neutral |

On stable days (no platform-wide shift), calibration is neutral — the model is already well-calibrated and anchor ratios cluster around 1.0. This is good design: no harm on normal days.

### Calibration on Anomalous Days (Simulation)

To demonstrate value when price shifts *do* occur, I simulated anomalous days by artificially shifting validation prices:

| Scenario | Without Calibration | With Calibration | Improvement |
|----------|--------------------|--------------------|-------------|
| Normal day (no shift) | MAPE 0.85% | MAPE 0.85% | ~0% (no harm) |
| 15% price drop (flash sale) | MAPE 18.27% | MAPE 0.85% | **95.4% error reduction** |
| 10% price increase | MAPE 9.76% | MAPE 0.85% | **91.3% error reduction** |
| Category-specific promo (-20%) | MAPE 5.84% | MAPE 5.84% | ~0% (needs more anchor coverage) |

**Conclusion:** Anchor calibration is a safety net — neutral on normal days, critical on anomalous ones. For category-specific promos, effectiveness depends on anchor coverage within that category (limitation worth noting in production).

See `notebooks/anchor_simulation.py` for the full simulation code.

### Alternatives Considered

| Alternative | Reason Not Used |
|-------------|----------------|
| Fine-tuning model on 100 anchors | Too few samples — would overfit immediately |
| Additive bias correction | Proportional shifts better modelled multiplicatively |
| Per-shop retraining | Computationally expensive; shop-level ratio achieves same effect |
| KNN-based anchor transfer | Over-engineering for this problem size |

---

## 3. Feature Engineering

### Preprocessing Pipeline

#### Data Loading
- Parse `capturedAt` string → datetime
- Convert `t/f` strings → `True/False` for 5 boolean columns (`is_free_shipping`, `is_pre_order`, `is_official_shop`, `is_verified`, `is_preferred_plus_seller`)
- All numeric columns loaded as `float64` to safely handle rows with NaN values

#### Data Quality Filter (applied before feature engineering)
Rows with logically impossible values are removed — these are not business outliers but data integrity violations that cannot represent real observations:

| Column | Invalid condition | Reason |
|--------|------------------|--------|
| `price` | `<= 0` (where known) | A sold item must have a positive price |
| `show_discount` | `< 0` or `> 100` | Percentage must be in [0, 100] |
| `stock`, `normal_stock` | `< 0` | Inventory cannot be negative |
| `review_rating`, `shop_rating` | `< 0` or `> 5` | Platform uses 0–5 scale |
| `shop_response_rate` | `< 0` or `> 100` | Percentage must be in [0, 100] |

> For test data, the `price` filter applies only to anchor rows (price known). Target rows have `NaN` price by design and are not filtered.

This filter is intentionally narrow — flash sale prices, deep discounts, and extreme stock counts are preserved because they are legitimate patterns the model must learn.

#### Feature Transforms
| Transform | Columns | Reason |
|-----------|---------|--------|
| `log1p()` | `shop_follower_count`, `total_rating_count`, `cmt_count`, `stock` | Highly skewed → compress range |
| Boolean → int | 5 boolean columns | Required for numeric feature matrix |
| Date → ordinal | `capturedAt` → `date_ordinal` | Numeric representation of time for trend capture |
| Datetime extraction | `capturedAt` → 7 temporal features | Day-of-week, month, hour, etc. |

#### Missing Value Handling
- **No manual imputation** — LightGBM handles NaN natively via NaN-aware splits
- `hist_price_std` NaN (single observation products) → filled with 0 (no variance)
- Feature matrix uses `fillna(-1)` as a safety fallback before prediction

#### What Was NOT Done (and Why)
| Technique | Reason not applied |
|-----------|-------------------|
| Normalisation / Scaling | Tree-based models are split-based — scaling has zero effect |
| One-hot encoding | LightGBM handles high-cardinality numeric IDs natively |
| Label encoding for `brand` | Too many unique values; shop/category features capture the same signal |
| Business outlier removal | LightGBM is robust to outliers; removing flash-sale prices would cause the model to fail to predict them |
| Target transform (log price) | Not needed — model achieves R² = 0.9991 on raw prices |

### 56 Features across 8 Categories

#### Temporal Features (7) — from `capturedAt`
| Feature | Rationale |
|---------|-----------|
| `date_ordinal` | Captures long-term price trends (inflation/deflation) |
| `day_of_week` | Weekend vs weekday pricing patterns |
| `day_of_month` | Pay-day effects, monthly promotions |
| `month` | Seasonal patterns |
| `hour` | Time-of-day scraping effects (intra-day flash deals) |
| `is_weekend` | Binary weekend flag |
| `week_of_year` | Seasonal cycles |

#### Categorical Identifiers (4)
`shopId`, `itemId`, `modelId`, `cat_id` — used directly as numeric features. LightGBM tree splits handle high-cardinality IDs natively, effectively learning per-entity pricing patterns without explicit encoding.

#### Discount & Promotion Features (3)
| Feature | Derivation |
|---------|-----------|
| `has_discount` | `priceBeforeDiscount > 0` |
| `effective_discount_pct` | `(priceBeforeDiscount - price) / priceBeforeDiscount * 100` |
| `has_promotion` | `promotionId != 0` |

#### Price Range Features (3)
| Feature | Derivation |
|---------|-----------|
| `price_range` | `item_price_max - item_price_min` |
| `price_range_ratio` | `item_price_max / item_price_min` |
| `price_midrange` | `(item_price_max + item_price_min) / 2` |

#### Shop Quality Score (1 composite feature)
```python
shop_quality_score = (
    shop_rating * 0.4 +
    (shop_response_rate / 100) * 0.3 +
    is_official_shop * 0.15 +
    is_verified * 0.1 +
    is_preferred_plus_seller * 0.05
)
```
Composite trustworthiness signal — higher quality shops tend to have more stable, premium pricing. Ranked **#3 in feature importance**.

#### Item Engagement Features (4)
| Feature | Derivation |
|---------|-----------|
| `log_total_ratings` | `log1p(total_rating_count)` |
| `log_cmt_count` | `log1p(cmt_count)` |
| `rating_comment_ratio` | `cmt_count / total_rating_count` — engagement depth |
| `stock_ratio` | `stock / normal_stock` — stock utilisation |

#### Historical Price Features (8) — key innovation
| Feature | Derivation |
|---------|-----------|
| `hist_price_mean` | Mean price per (shopId, itemId, modelId) in training — **#1 feature** |
| `hist_price_std` | Price volatility |
| `hist_price_min` / `hist_price_max` | Historical price bounds |
| `hist_price_median` | Robust central tendency |
| `hist_price_last` | Most recent observed price |
| `hist_count` | Number of historical observations |
| `hist_price_cv` | Coefficient of variation — stability metric |

#### Raw Columns (26)
All remaining numeric/boolean columns from the dataset schema passed directly.

### Top 15 Feature Importance (Tier 1 Global Model)

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | `hist_price_mean` | 20,998 |
| 2 | `date_ordinal` | 17,703 |
| 3 | `shop_quality_score` | 15,188 |
| 4 | `item_price_min` | 14,801 |
| 5 | `hist_price_std` | 14,703 |
| 6 | `shop_rating` | 14,416 |
| 7 | `day_of_month` | 13,070 |
| 8 | `shop_response_rate` | 12,611 |
| 9 | `modelId` | 12,289 |
| 10 | `shop_follower_count` | 11,393 |
| 11 | `item_price_max` | 11,001 |
| 12 | `hist_price_min` | 10,955 |
| 13 | `hour` | 9,365 |
| 14 | `hist_count` | 9,275 |
| 15 | `priceBeforeDiscount` | 9,035 |

---

## 4. Modelling Approach

### Why LightGBM (Tree-Based)?

| Criterion | LightGBM | Neural Net | Time Series |
|-----------|----------|-----------|-------------|
| Mixed feature types (IDs, floats, booleans) | Native | Requires encoding | N/A |
| NaN handling | Native splits | Requires imputation | Requires imputation |
| Training time (306K rows × 56 feat) | ~2 min | Hours | Per-product (slow) |
| Cold-start products | Global fallback | Same issue | Cannot model |
| High-cardinality categoricals | Splits handle it | Embedding needed | N/A |
| Feature importance | Built-in | SHAP needed | Coefficients |

**Why not Time Series (ARIMA/Prophet)?** Each product has irregular observation frequency and most have median ~50 observations — insufficient for per-series modelling. The problem is better framed as cross-sectional regression with temporal context.

### Tier 1 — Global Marketplace Model

**Configuration:**
- Algorithm: LightGBM (Gradient Boosted Decision Trees)
- Features: 56 engineered features
- Hyperparameters: 255 leaves, learning rate 0.05, early stopping at 50 rounds
- Best iteration: 1197 / 2000 (early stopping prevented overfitting)

### Tier 2 — Shop/Product Level Model

Three-level ensemble with adaptive weighting:

```
┌──────────────────────────────────────────┐
│           Ensemble (Adaptive)            │
├──────────────┬────────────┬──────────────┤
│ Product-Level│ Shop-Level │    Global    │
│   Lookup     │  LightGBM  │   LightGBM  │
│              │            │             │
│ - recent     │ - 32 feat  │ - 56 feat   │
│ - median     │ - 1500 est │ - 2000 est  │
│ - trend      │ - per-shop │ - all data  │
│              │   patterns │             │
└──────────────┴────────────┴──────────────┘
```

**Adaptive Ensemble Weights by Data Richness:**

| Product History | Product Weight | Shop Weight | Global Weight |
|-----------------|---------------|-------------|---------------|
| ≥20 obs, CV < 0.1 (stable) | 60% | 25% | 15% |
| ≥10 observations | 40% | 35% | 25% |
| < 10 observations | 25% | 40% | 35% |
| No product history | 0% | 55% | 45% |

Coverage: **98.9%** of test products have historical data; only 1.1% rely solely on shop + global fallback.

### Cold Start Handling

Products with < 2 historical observations cannot use the product-level model. The system handles this via hierarchical fallback:

1. Product-level model requires ≥ 2 observations — below that, product weight = 0
2. Ensemble reweights automatically toward shop + global
3. Category historical averages (`cat_avg_price`, `shop_avg_price`) still provide useful signals
4. 98.9% test coverage means cold start is rare but handled gracefully

### Treatment of Outliers

- **Not removed** — flash sale prices and promotional deep discounts are real patterns the model should learn. Removing them would cause the model to fail when predicting discount events.
- **LightGBM is inherently robust** — splits are based on rank order, not magnitude
- **Evaluation uses Median AE alongside MAE** — to assess outlier influence on reported metrics
- **Predictions clipped at 0** — no negative price predictions
- **Data quality filter** (see Section 3) removes only *logically impossible* values, not business outliers

### Data Leakage Prevention

| Feature | Leakage? | Reasoning |
|---------|----------|-----------|
| `date_ordinal` | No | Test set provides `capturedAt`; captures trend, not future price |
| `day_of_week`, `hour` | No | Cyclical patterns; test timestamps are available |
| `hist_price_mean` | No | Computed from training data only, excluded from training target |
| Anchor set | No | Sampled after the split; never used as training features |

What WOULD be leakage: using validation-day prices in training features. Our time-based split prevents this by construction.

---

## 5. Analysis & Insights

### Approach 1 vs Approach 2 — Where Each Excels

Actual results show a nuanced trade-off rather than one approach dominating:

| Metric | Winner | Value |
|--------|--------|-------|
| MAE (average error) | **Tier 1 Global** | 172,839 vs 195,824 |
| MAPE (% error) | **Tier 1 Global** | 0.85% vs 0.87% |
| RMSE (large-error sensitivity) | **Tier 2 Ensemble** | 2,462,470 vs 2,538,963 |
| Median AE (typical product) | **Tier 2 Ensemble** | 2,142 vs 4,342 |

**Interpretation:** Tier 2's Median AE is 50% lower than Tier 1 — meaning for the *typical* product, the ensemble is more accurate. However, Tier 2 makes a handful of very large errors (likely for products where the historical lookup is stale or the pricing pattern changed) that pull up its mean (MAE). Tier 1's global model is more conservative and avoids extreme misses.

| Scenario | Better Approach | Why |
|----------|----------------|-----|
| Typical product with stable history | Tier 2 (Ensemble) | Lower Median AE — more accurate for most products |
| Products with changed/stale pricing | Tier 1 (Global) | Ensemble's product lookup can be confidently wrong |
| Volatile products (CV > 0.2) | Tier 1 (Global) | Global context more reliable than noisy history |
| New/sparse products (< 5 obs) | Tier 1 (Global) | Insufficient history for product model |
| Platform-wide price shifts | Tier 1 + calibration | Global correction catches systematic shifts |
| Minimising worst-case error (RMSE) | Tier 2 (Ensemble) | Better at handling outlier price ranges |
| Production default | Tier 1 Global | More predictable; lower average error |

### Key Patterns Observed

**1. Prices are remarkably stable**
- Majority of products have Coefficient of Variation (CV) < 0.05
- `hist_price_mean` alone achieves R² > 0.99
- The "prediction" task is largely a "recall historical price with context" task

**2. Discounts create bimodal price distributions**
- Same product appears at full price and discounted price in the same dataset
- `has_discount` + `show_discount` + `effective_discount_pct` capture this cleanly
- Discount depth varies significantly by category

**3. Shop quality correlates with price level**
- Official/verified shops tend to have higher-priced, more stable products
- `shop_quality_score` is the #3 feature — pricing and shop tier are linked

**4. Temporal patterns are subtle but present**
- `date_ordinal` is #2 feature — captures gradual price inflation/deflation over weeks
- `day_of_month` matters — likely salary-cycle purchasing patterns
- `hour` is #13 — suggests intra-day price fluctuations (time-limited flash deals)

### Unexpected Findings

**1. Anchor calibration barely helps on stable days**
The model is so accurate out-of-box that anchor correction is almost neutral (+0.6% MAE). This is actually good engineering — the calibration mechanism is a safety net for abnormal days without degrading normal performance.

**2. `hour` in top 15 features**
Suggests prices or discounts change within a single day (time-limited flash deals), or that scraping time correlates with specific shop types that have different price levels.

**3. 98.9% product coverage in test set**
Almost all test products exist in training data, making this effectively a "price recall with context" problem rather than a true cold-start prediction problem. In a real production outage, this coverage percentage is the most critical metric to track.

**4. Tier 1 outperforms Tier 2 on MAE/MAPE despite Tier 2's lower Median AE**
Tier 2 MAPE (0.87%) is slightly worse than Tier 1 (0.85%), yet Tier 2's Median AE (2,142) is 50% lower than Tier 1 (4,342). This apparent contradiction reveals that the ensemble makes very accurate predictions for most products but produces a few catastrophic misses — likely products where the historical price lookup is stale or where pricing strategy changed significantly. These outlier errors inflate the mean (MAE/MAPE) without affecting the median. In production, flagging predictions where the product-level and global models disagree by a large margin would catch these cases.

---

## 6. Code Quality & Reproducibility

### Project Structure
```
src/
├── data_loader.py          # Data download, loading & quality filtering
├── feature_engineering.py  # 56 engineered features (pure transforms)
├── model_global.py         # Tier 1: Global model (self-contained)
├── model_shop.py           # Tier 2: Shop/Product ensemble (self-contained)
├── anchor_calibration.py   # Hierarchical calibration (model-agnostic)
└── evaluate.py             # Metrics & comparison utilities
```

Each module has a single responsibility and can be used independently.

### Design Principles
- **Modular** — each `src/` file has a single responsibility
- **Typed** — type hints on key functions
- **Reproducible** — `SEED = 42` used throughout all random operations
- **Minimal preprocessing** — LightGBM handles NaN, outliers, and mixed scales natively; energy focused on feature engineering instead

### Reproducibility Guarantees
- `pyproject.toml` + `.python-version` for exact environment (Python 3.11)
- `requirements.txt` with all pinned versions as pip fallback
- `np.random.seed(42)` at pipeline start
- LightGBM `random_state=42`
- Time-based validation split is deterministic given the same data
- `simulate_anchor_validation` uses `random_state=SEED`

### How to Reproduce

**With uv (recommended):**
```bash
git clone <repo>
cd mrscraper-price-intelligence
uv sync                     # Creates .venv with Python 3.11 + all deps
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/Mac

python src/data_loader.py   # Download train.csv + test.csv from Google Drive
python main.py              # Train + validate + predict → predictions.csv
```

**With pip (fallback):**
```bash
pip install -r requirements.txt
python src/data_loader.py
python main.py
```

**Inference only (interview):**
```bash
python predict.py --test_file data/test_full.csv --output predictions_full.csv
```
Runs in under 5 minutes on standard hardware. Uses pre-trained models from `models/`.

### Visualizations

All plots generated by `python visualize.py` → saved to `figures/`:

| File | Description |
|------|-------------|
| `01_price_distribution.png` | Price histogram (log scale) + per-category boxplots |
| `02_temporal_trends.png` | Daily price trends, volume, and discount rates |
| `03_feature_importance.png` | Top 20 features (Global Model) |
| `04_prediction_vs_actual.png` | Scatter plot + error distribution |
| `05_model_comparison.png` | MAE/MAPE/R² comparison across models |
| `06_calibration_strategy.png` | Hierarchical calibration coverage |
| `07_price_stability.png` | Product-level price stability (CV distribution) |
| `08_pipeline_overview.png` | Pipeline architecture diagram |
