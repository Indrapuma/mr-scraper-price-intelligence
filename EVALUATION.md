# Evaluation Criteria — Detailed Responses

---

## 1. Prediction Accuracy

### Metrics Used
- **MAE** (Mean Absolute Error) — interpretable in currency units
- **RMSE** (Root Mean Squared Error) — penalizes large errors
- **MAPE** (Mean Absolute Percentage Error) — scale-independent
- **Median AE** — robust to outliers
- **R²** — proportion of variance explained

### Results per Model

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Tier 1: Global Model | 172,839 | 2,538,963 | 0.85% | 0.9991 |
| Tier 2: Shop/Product Ensemble | ~168,000 | ~2,500,000 | ~0.80% | ~0.9992 |

### Validation Methodology
**Time-based split simulating an outage day:**

1. Training: 285,307 rows from Jan 1 – Mar 19, 2025 (56 days)
2. Validation: 20,919 rows from Mar 20 – Mar 22, 2025 (3 days)
3. Per validation day: 100 random samples designated as "anchors" (known prices)
4. Remaining ~6,800 per day = prediction targets

This mirrors the exact test scenario: model trained on historical data, then asked to predict an unseen day with only 100 anchor prices available for calibration.

---

## 2. Anchor Set Utilisation

### Strategy: Hierarchical Calibration

The 100 anchor samples per day serve as a real-time signal about the price landscape on that specific day. My calibration strategy operates at three levels:

```
Priority: Shop-level → Category-level → Global-level
```

**Step 1 — Compute corrections from anchors:**
- For each anchor sample, compute: `ratio = actual_price / predicted_price`
- Group ratios by shop, category, and overall

**Step 2 — Apply corrections hierarchically:**
- If target product's shop has anchors → apply shop-specific median ratio (80.5%)
- Else if target's category has anchors → apply category median ratio (19.4%)
- Else → apply global median ratio (0.1%)

### Does calibration improve accuracy?

On **normal days** (validation set): minimal improvement (+0.6% MAE) because the model is already well-calibrated (median ratio ≈ 1.0000).

However, the mechanism is designed for **anomalous days**:
- Platform-wide flash sales → global ratio would deviate significantly from 1.0
- Category-specific promotions → category ratios would catch localized shifts
- Shop-level price changes → shop ratios provide the most granular correction

### Baseline comparison (no calibration vs with calibration):
The system is structured so that when price shifts *do* occur, the hierarchical calibration automatically detects and corrects them. On stable days, it gracefully falls back to ~1.0x correction (no harm).

### Alternative strategies considered:
- **Per-category adjustment** — implemented as fallback layer
- **Fine-tuning on anchors** — not pursued (100 samples too few to retrain without overfitting)
- **Global bias correction (additive vs multiplicative)** — tested both; multiplicative performs better for proportional price shifts

### Anchor Calibration on Anomalous Days (Simulation)

To demonstrate the value of anchor calibration when price shifts *do* occur, I simulated an anomalous day by artificially shifting validation prices by 15% (simulating a platform-wide flash sale):

| Scenario | Without Calibration | With Calibration | Improvement |
|----------|--------------------|--------------------|-------------|
| Normal day (no shift) | MAPE 0.85% | MAPE 0.85% | ~0% (no harm) |
| 15% price drop (flash sale) | MAPE 18.27% | MAPE 0.85% | **95.4% error reduction** |
| 10% price increase | MAPE 9.76% | MAPE 0.85% | **91.3% error reduction** |
| Category-specific promo (-20%) | MAPE 5.84% | MAPE 5.84% | ~0% (needs more anchor coverage) |

**Conclusion**: On normal days, calibration is neutral (no harm). On platform-wide shifts, it reduces error by **91-95%** — bringing MAPE from double digits back to sub-1%. For category-specific promos, the hierarchical approach needs sufficient anchor coverage in that category to be effective — a limitation worth noting.

See `notebooks/anchor_simulation.py` for the full simulation code.

---

## 3. Data Preprocessing & Feature Engineering

### Preprocessing Pipeline

#### Data Loading
- **Parse dates**: `capturedAt` string → datetime object
- **Boolean conversion**: `t/f` strings → `True/False` for 5 boolean columns (`is_free_shipping`, `is_pre_order`, `is_official_shop`, `is_verified`, `is_preferred_plus_seller`)
- **Dtype handling**: All numeric columns loaded as `float64` to safely handle NaN values in some rows

#### Missing Value Handling
- **No manual imputation** — LightGBM handles NaN natively via NaN-aware splits (routes missing values to optimal child node during tree construction)
- `hist_price_std` NaN (products with single observation) → filled with 0 (no variance)
- Feature matrix passed with `fillna(-1)` as safety fallback before prediction

#### Feature Transforms Applied
| Transform | Columns | Reason |
|-----------|---------|--------|
| `log1p()` | `shop_follower_count`, `total_rating_count`, `cmt_count`, `stock` | Highly skewed distributions → compress range |
| Boolean → int | All 5 boolean columns | Required for numeric feature matrix |
| Date → ordinal | `capturedAt` → `date_ordinal` | Captures long-term price trend as numeric |
| Datetime extraction | `capturedAt` → 7 temporal features | Day-of-week, month, hour, etc. |

#### What was NOT done (and why)
| Technique | Reason not applied |
|-----------|-------------------|
| Normalization / Scaling | Tree-based models are split-based, not distance-based — scaling has zero effect |
| One-hot encoding | LightGBM handles high-cardinality numeric IDs natively via splits |
| Label encoding for `brand` | Too many unique values; information already captured through shop/category features |
| Outlier removal | LightGBM is robust to outliers; removing them risks losing legitimate flash-sale patterns |
| Target transform (log price) | Not needed — model achieves R² = 0.9991 on raw prices |
| Winsorization | Could mask real price extremes that the model should learn |

#### Design Philosophy
Preprocessing is kept **minimal and intentional** because:
1. LightGBM handles NaN, outliers, and mixed scales natively
2. Heavy preprocessing can introduce information loss
3. Energy is better spent on **feature engineering** (56 derived features) than cleaning already-clean data

---

### 56 Total Features across 8 categories:

#### Temporal Features (from `capturedAt`)
| Feature | Rationale |
|---------|-----------|
| `day_of_week` | Weekend vs weekday pricing patterns |
| `day_of_month` | Pay-day effects, monthly promotions |
| `month` | Seasonal patterns |
| `hour` | Time-of-day scraping effects |
| `is_weekend` | Binary weekend flag |
| `week_of_year` | Seasonal cycles |
| `date_ordinal` | Captures long-term price trends |

#### Categorical Identifiers
- `shopId`, `itemId`, `modelId`, `cat_id` — used directly as numeric features in LightGBM (tree-based models handle high-cardinality categoricals natively via split decisions)
- `brand` — not encoded separately (too many unique values, info captured by shop/category)

#### Discount & Promotion Features
| Feature | Derivation |
|---------|-----------|
| `has_discount` | `priceBeforeDiscount > 0` |
| `effective_discount_pct` | `(priceBeforeDiscount - price) / priceBeforeDiscount * 100` |
| `has_promotion` | `promotionId != 0` |

#### Price Range Features
| Feature | Derivation |
|---------|-----------|
| `price_range` | `item_price_max - item_price_min` |
| `price_range_ratio` | `item_price_max / item_price_min` |
| `price_midrange` | `(item_price_max + item_price_min) / 2` |

#### Shop Engagement Score (Creative/Derived)
```python
shop_quality_score = (
    shop_rating * 0.4 +
    (shop_response_rate / 100) * 0.3 +
    is_official_shop * 0.15 +
    is_verified * 0.1 +
    is_preferred_plus_seller * 0.05
)
```
This composite score captures overall shop trustworthiness — higher quality shops tend to have more stable, premium pricing.

#### Item Engagement Features
| Feature | Derivation |
|---------|-----------|
| `log_total_ratings` | `log1p(total_rating_count)` — handles skew |
| `log_cmt_count` | `log1p(cmt_count)` |
| `rating_comment_ratio` | `cmt_count / total_rating_count` — engagement depth |
| `stock_ratio` | `stock / normal_stock` — stock utilization |

#### Historical Price Features (per product — key innovation)
| Feature | Derivation |
|---------|-----------|
| `hist_price_mean` | Mean price for this (shop, item, model) in training |
| `hist_price_std` | Price volatility |
| `hist_price_min/max` | Historical price bounds |
| `hist_price_median` | Robust central tendency |
| `hist_price_last` | Most recent observed price |
| `hist_count` | Number of historical observations |
| `hist_price_cv` | Coefficient of variation (stability metric) |

#### Price Momentum (implicit via trend)
In Tier 2's product-level model:
```python
price_trend = slope of linear regression on last 10 price observations
```
Positive trend → prices increasing. Used to project forward.

---

## 4. Modelling Approach

### Why LightGBM (Tree-Based)?

1. **Handles mixed feature types** — numeric, categorical, high-cardinality IDs all work natively
2. **Fast training** — 306K rows × 56 features trains in ~2 minutes
3. **Strong out-of-the-box performance** — minimal tuning needed for excellent results
4. **Feature importance built-in** — explainability for free
5. **Handles missing values** — NaN-aware splits (no imputation needed)
6. **Non-linear interactions** — captures price-discount-category interactions automatically

### Why not Deep Learning?
- Dataset size (306K) is in the "sweet spot" for gradient boosting
- Tabular data with many categorical features favors tree-based methods
- Training time: LightGBM = 2 min vs neural net = potentially hours
- Marginal accuracy gains unlikely to justify complexity for this dataset size

### Why not Time Series (ARIMA/Prophet)?
- Each product has irregular observation frequency
- Most products have very few observations (median ~50)
- The problem is more "cross-sectional prediction with temporal context" than pure time series

### Multi-Level Architecture

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

Weighting adapts based on per-product data availability:
- Rich history → trust product-level more (60%)
- Sparse history → trust global more (45%)

### Treatment of Outliers
- **Detection**: Products with CV > 0.5 flagged as potentially anomalous
- **Handling**: LightGBM is inherently robust to outliers (splits don't depend on magnitude)
- **Evaluation**: Using Median AE alongside MAE to assess impact of outliers
- **Price bounds**: Predictions clipped to `max(prediction, 0)` — no negative prices
- **Documentation**: Outliers in this dataset are primarily flash-sale prices and data entry errors. Rather than removing them, the model learns to associate them with contextual signals (`has_promotion`, `show_discount`, low `stock_ratio`). This is intentional — in production, we *want* the model to predict flash-sale prices when the context indicates one is active.

### Treatment of Missing Values
- **Numeric NaN**: Passed directly to LightGBM (handles natively via NaN-aware splits)
- **Feature fallback**: When `hist_price_mean` is NaN (new product), model relies on `item_price_min/max` and category averages
- **Boolean columns**: Explicit True/False mapping from `t/f` strings

### Cold Start Handling
Products or shops with very few historical observations (< 5 rows in training) present a cold-start challenge for Tier 2. The system handles this via **hierarchical fallback**:

1. **Product-level model** requires ≥ 2 historical observations. Products with 0-1 observations get no product-level prediction (NaN).
2. **Ensemble weighting** adapts: sparse-history products receive higher weight on global + shop models (global 35%, shop 40%, product 25%).
3. **If no product model exists at all** (1.1% of test targets): prediction = 55% shop-level + 45% global — skipping product entirely.
4. **Category/shop historical features** (e.g., `cat_avg_price`, `shop_avg_price`) still provide useful signals even for cold-start items.

**Result**: 98.9% of test products have sufficient history for product-level models. The remaining 1.1% are handled gracefully by the fallback mechanism.

### Data Leakage Prevention
The `capturedAt` column is available in both training and test data. Key leakage considerations:

1. **`date_ordinal`** — Derived from `capturedAt`. This is NOT leakage because:
   - The test set provides `capturedAt` for all rows (including targets)
   - It captures long-term price trends (inflation/deflation over weeks)
   - The model doesn't learn "on March 22 prices are X" — it learns "as time passes, prices drift by Y per day"

2. **Temporal features** (`day_of_week`, `hour`, etc.) — Also NOT leakage because:
   - These capture cyclical patterns (weekend pricing, time-of-day effects)
   - The test set explicitly provides these timestamps

3. **What WOULD be leakage**: Using future prices from the training set to predict past dates. Our time-based validation split prevents this — training only uses data *before* the validation period.

4. **Anchor set isolation**: During validation, anchors are randomly sampled but never used as training features — only for post-prediction calibration.

---

## 5. Analysis & Insights

### Approach 1 vs Approach 2 — Where Does Each Excel?

| Segment | Better Approach | Why |
|---------|----------------|-----|
| Stable products (CV < 0.01) | Tier 2 (Product) | Historical mean is nearly perfect |
| Volatile products (CV > 0.2) | Tier 1 (Global) | Needs broader context, product-level is noisy |
| New/sparse products (< 5 obs) | Tier 1 (Global) | Insufficient history for product model |
| Shop-specific promos | Tier 2 (Shop) | Captures per-shop discount patterns |
| Platform-wide events | Tier 1 + calibration | Global correction catches systematic shifts |

### Key Patterns Observed

1. **Price stability is the dominant pattern**
   - 75%+ products have CV < 0.05
   - `hist_price_mean` alone achieves R² > 0.99
   - This means the "prediction" task is largely a "recall historical price" task

2. **Discounts create bimodal price distributions**
   - Same product appears at full price and discounted price
   - `has_discount` + `show_discount` features capture this split

3. **Shop quality correlates with price level**
   - Official/verified shops tend to have higher-priced products
   - `shop_quality_score` is a top-5 feature

4. **Temporal patterns are subtle but present**
   - `date_ordinal` is #2 feature — captures gradual inflation/deflation
   - `day_of_month` matters — likely salary-cycle purchasing patterns

### Unexpected Findings

1. **Calibration barely helps on stable days** — The model is so accurate out-of-box that anchor correction is almost neutral. This is actually good engineering: the calibration mechanism is a safety net for abnormal days without degrading normal performance.

2. **`hour` is in top 15 features** — Suggests prices/discounts may change within a day (time-limited flash deals), or scraping time correlates with specific shop types.

3. **98.9% product coverage** — Almost all test products exist in training data, making this effectively a "price recall with context" problem rather than a cold-start prediction problem.

---

## 6. Code Quality & Reproducibility

### Structure
```
src/
├── data_loader.py          # Clean separation: data I/O
├── feature_engineering.py  # Pure transformation functions
├── model_global.py         # Tier 1 model (self-contained)
├── model_shop.py           # Tier 2 model (self-contained)
├── anchor_calibration.py   # Calibration logic (model-agnostic)
└── evaluate.py             # Metrics & comparison utilities
```

### Design Principles
- **Modular**: Each file has a single responsibility
- **Documented**: Every function has docstrings explaining purpose, args, returns
- **Typed**: Type hints on key functions
- **Reproducible**: `SEED = 42` used throughout; pinned dependencies

### Reproducibility Guarantees
- `requirements.txt` with pinned versions
- `np.random.seed(42)` at pipeline start
- LightGBM `random_state=42`
- Time-based validation split (deterministic given data)
- `simulate_anchor_validation` uses `random_state=SEED`

### How to Reproduce
```bash
git clone <repo>
pip install -r requirements.txt
python src/data_loader.py   # Download data
python main.py              # Full pipeline → predictions.csv
```

### For Interview (inference on new test data):
```bash
python predict.py --test_file data/test_full.csv --output predictions_full.csv
```
Runs in < 5 minutes on standard hardware.
