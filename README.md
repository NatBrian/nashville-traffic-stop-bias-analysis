# Nashville Traffic Stop Bias Analysis

End-to-end analysis of racial bias in Nashville Police Department traffic stops using interpretable machine learning, statistical tests, and fairness audits. This README is a full narrative of what the three notebooks do, so you do not need to re-open them to understand the project.

<i><u><b>Created By: Nathanael Brian</b></u></i>

## Research Question

Can we predict whether a traffic stop results in an arrest, and do prediction patterns reveal demographic disparities?

We answer both parts with:
- A minimal baseline model (6 features) to establish a floor.
- A fully engineered model (32 features) to test whether feature engineering improves performance.
- A fairness audit to quantify disparities in model errors by race.

## Dataset

- Source: Stanford Open Policing Project (Nashville subset)
- File: `tn_nashville_2020_04_01.csv`
- Size: 3,092,351 rows, 42 columns
- Target: `arrest_made` (binary)
- Base rate: 1.62% arrests (60.6:1 class imbalance)

## Environment and Reproducibility

- Python: 3.12.x
- Seed: 42 used across all notebooks
- Key libraries: pandas, numpy, scikit-learn, imbalanced-learn, seaborn, matplotlib

Install:

```bash
pip install -r requirements.txt
```

## Pipeline Overview (Notebook Flow)

1. **Notebook 01: Data Audit, Profiling, Cleaning, Baseline Model**
   - Validate structure, missingness, integrity, and leakage.
   - Build baseline model with minimal features.

2. **Notebook 02: Deep EDA and Feature Engineering**
   - Deep statistical EDA and bias detection.
   - Engineer temporal, spatial, officer, and interaction features.
   - Build and save a full preprocessing pipeline.

3. **Notebook 03: Modeling and Fairness Analysis (v2)**
   - Train interpretable models (Logistic Regression, Decision Tree).
   - Compare baseline vs engineered performance.
   - Audit fairness and test mitigation strategies.

---

# Notebook 01: Data Audit, Profiling, Cleaning, Baseline Model

## 1. Data Loading and First Look
- Load the full CSV (3.09M rows) and confirm schema.
- Focus on reproducibility (seed, package versions, timestamp).

## 2. Data Profiling (Three-Tier Discovery)

### 2.1 Structure Discovery
- Identified low-cardinality columns suitable for categorical types (e.g., `subject_race`, `subject_sex`).
- Noted that `precinct` and `zone` are stored as floats because of missingness.
- Found extreme sparsity (>95%) in `contraband_*` and `search_basis` columns.

### 2.2 Content Discovery
- Inspected numerical distributions and categorical frequencies.
- Confirmed realistic values for `subject_age` (10-99).

### 2.3 Relationship Discovery
- Examined correlations for numeric fields to detect redundant features.

## 3. Missing Value Strategy
Key patterns:
- **Structurally missing (MNAR)**: `contraband_*`, `search_basis` (~96%) only present when a search occurs.
- **Administrative/geographic gaps (MAR)**: `precinct`, `zone`, `lat`, `lng`.
- **Critical features (MCAR)**: `subject_age`, `subject_race`, `subject_sex`, `arrest_made` (<1%).

Strategy:
- Drop rows with missing target or critical demographics (very small impact).
- Impute boolean search fields as `False`.
- Impute age with median.

## 4. Duplicate and Outlier Checks
- Duplicate rows: 0
- Age outlier detection (z-score, IQR): values 10-99 considered valid, so **kept**.

## 5. Integrity Checks
- **Geographic bounds**: ~0.66% of stops outside Nashville boundary (flagged, not removed).
- **Contraband logic**: 0 cases where contraband was present without search.
- **Temporal consistency**: 100% valid times.

## 6. Target Imbalance
- Arrest rate: **1.62%**
- Imbalance ratio: **60.6:1**
- Accuracy alone is misleading; F1 and ROC-AUC are required.

## 7. Leakage Detection
Dropped post-event columns such as:
- `outcome`, `citation_issued`, `warning_issued`, `raw_*_issued`, `raw_search_arrest`

These encode post-decision outcomes and would invalidate model evaluation.

## 8. Data Quality Scorecard
- Completeness: 86.8%
- Validity: 100% (age range)
- Consistency: 100% (contraband logic)
- Uniqueness: 100% (no duplicates)

## 9. Baseline Model (Minimal Features)

**Purpose**: establish a clean, leakage-free baseline.

Features (6):
- `subject_age`, `subject_race`, `subject_sex`
- `type`, `search_conducted`, `frisk_performed`

Model: Logistic Regression (`class_weight='balanced'`)

Baseline results (Notebook 01):
- ROC-AUC: **0.9055**
- F1 (arrest class): **0.4324**
- Precision: 0.30, Recall: 0.77

Key error patterns:
- False positives disproportionately higher for Black and Hispanic drivers.
- False negatives higher for Hispanic and female drivers.

Artifacts saved:
- `artifacts/cleaned_full.parquet`
- `artifacts/baseline_model_seed42.pkl`

---

# Notebook 02: Deep EDA and Feature Engineering

## 1. Data Loading and Split
- Load `artifacts/cleaned_full.parquet`.
- Stratified 80/20 train/test split.
- Arrest rate remains 1.62% in both splits.

## 2. Deep EDA

### 2.1 Univariate Findings
- Class imbalance confirmed (60.6:1).
- `subject_age` mildly right-skewed but valid.
- Cardinality analysis:
  - Low: `subject_sex` (2), `subject_race` (6), `violation_group` (7)
  - High: `officer_id_hash` (2247) -> needs target encoding.

### 2.2 Bivariate Findings
- Arrest rate by race:
  - Hispanic: 3.11%
  - Black: 2.26%
  - White: 1.09%
- Temporal analysis:
  - Night (00:00-04:00) arrest rate > 4.5%
  - Daytime arrest rates < 1.0%

### 2.3 Bias Detection
- Chi-square (race vs arrest): significant (p < 0.001)
- Cramer's V = 0.0532 (small effect size but meaningful due to large sample)
- Sampling bias vs Nashville population:
  - Black drivers overrepresented (1.35x)
  - Hispanic drivers underrepresented (0.53x)

### 2.4 Veil of Darkness Test (modified)
- Night arrest rates significantly higher across all races.
- Hispanic drivers show the largest nighttime increase.

### 2.5 Simpson's Paradox Check
- Overall race disparity stable across 87.5% of precincts.
- Moderate reversal risk in a single precinct.

## 3. Feature Engineering

### 3.1 Temporal Features
- Extracted year, month, day_of_week, hour.
- Cyclic encoding: `sin_hour`, `cos_hour`.
- Binary feature: `is_night`.

### 3.2 Encoding Strategy
- One-Hot: `subject_sex`, `type`, `violation_group`, `race_canonical`.
- Target Encoding (OOF + smoothing m=10): `officer_id_hash`.
- Binary: `is_out_of_state` from plate state.

### 3.3 Scaling
- RobustScaler used for all numeric features due to outlier presence.

### 3.4 Spatial Features
- KMeans clustering (k=5) on `lat/lng`.
- Missing coordinates assigned cluster -1.

### 3.5 Officer Features
- `officer_arrest_rate` (target-encoded)
- `officer_search_rate` (target-encoded)
- `officer_total_stops_log` removed due to multicollinearity (VIF > 10)

### 3.6 Violation Grouping + Interactions
- Regex groups into 6 categories (Traffic, Equipment, Registration, etc.)
- Interaction features:
  - `is_young_male`
  - `is_out_of_state`
  - Race/Sex-specific median age imputation

### 3.7 Multicollinearity (VIF)
- Removed `officer_total_stops_log` (VIF 12.55)
- Retained age and is_night (VIF reduced to acceptable levels)

### 3.8 Bias Mitigation Demonstration
- Reweighing (race x outcome) improved disparate impact ratio:
  - 0.719 -> 0.743 (still below 0.80 threshold)

## 4. Final Pipeline (32 engineered features)

Feature groups:
- Numeric (5): `subject_age_imputed`, `officer_arrest_rate`, `officer_search_rate`, `sin_hour`, `cos_hour`
- Binary (3): `is_night`, `is_young_male`, `is_out_of_state`
- Categorical (4): `race_canonical`, `violation_group`, `type`, `location_cluster`
- Boolean (4): `search_conducted`, `frisk_performed`, `search_person`, `search_vehicle`

After one-hot encoding: **32 features**.

Artifacts saved:
- `artifacts/pipeline.pkl`
- `artifacts/X_train_final.parquet`
- `artifacts/X_test_final.parquet`
- `artifacts/metadata.pkl`

---

# Notebook 03: Modeling and Fairness Analysis (v2)

## 1. Data Loading
- Load `X_train_final.parquet` and `X_test_final.parquet`.
- Extract `sample_weight` (from reweighing) if present.
- Confirm sizes: 2,473,858 train, 618,465 test.

## 2. Baseline vs Prepared Comparison

Baseline (6 features, Notebook 01 recomputed on test split):
- F1: **0.4344**
- ROC-AUC: **0.9059**
- Precision: 0.3016, Recall: 0.7762

Prepared model (32 features + SMOTEENN):
- F1: **0.5082**
- ROC-AUC: **0.9276**
- Precision: 0.4009, Recall: 0.6941

Net effect:
- F1 +17.0%
- ROC-AUC +2.4%
- Precision +32.9%, Recall -10.6%

## 3. Class Imbalance Strategy (SMOTEENN)
- Original imbalance: 60.6:1
- Subsample 20% for feasibility
- SMOTEENN target ratio ~10:1
- Post-resample ratio: **11.8:1**

## 4. Model Training

Models (interpretable only):
- Logistic Regression (GridSearch: C in {0.1, 1.0})
- Decision Tree (max_depth 5/10, min_samples_leaf 50/100)

Best models on test set:
- Logistic Regression: F1 0.4920, ROC-AUC 0.9326
- Decision Tree: F1 **0.5082**, ROC-AUC 0.9276

Selected model: **Decision Tree** (best F1).

## 5. Interpretability

Logistic Regression top signals:
- `search_conducted` (OR ~331x)
- `search_person` (OR ~22x)
- Race and violation features appear, indicating fairness risk.

Decision Tree:
- Root split is `search_person` (importance ~0.92).
- Officer rates and time features are secondary.

Permutation importance confirms search-related features dominate across models.

## 6. Fairness Audit

Decision Tree fairness metrics (FPR by race):
- Black: **2.69%**
- Hispanic: **2.77%**
- White: **0.99%**

Disparity ratios vs White (FPR):
- Black: 2.71x
- Hispanic: 2.80x
- Asian and Unknown: < 0.8x (also flagged by 4/5 rule)

FPR range (max - min):
- Baseline: 0.0303
- LR (32): 0.0283
- DT (32): **0.0210**

Fairness improved in global spread, but disparities for Black/Hispanic vs White remain large.

## 7. Error Analysis

- False positives: 10,415
- False negatives: 3,072

FP rate by race (% of group):
- Hispanic 2.68, Black 2.63, White 0.98, Asian 0.66

## 8. Mitigation Experiments

### Threshold Tuning
- Tested thresholds 0.3 to 0.7
- Best tradeoff at **0.7**:
  - F1 = 0.5228
  - FPR range = 1.44%

### Class Weight Experiment
- `balanced` caused large fairness disparities (FPR range 12%)
- `{0:1, 1:10}` achieved best F1 (0.4861) with moderate fairness (2.93%)

## 9. Conclusions

- **Prediction**: YES. ROC-AUC > 0.92, F1 ~0.50 under extreme imbalance.
- **Disparities**: YES. Black and Hispanic drivers face ~2.7x higher false positive rates.
- **Best Model**: Decision Tree (F1 0.5082).
- **Recommendation**: Do NOT deploy without fairness interventions and policy review.

Artifacts saved:
- `artifacts/metrics_report.csv`
- `artifacts/fairness_report.csv`
- `artifacts/disparity_comparison.csv`
- `artifacts/best_model.pkl`
- `artifacts/fair_model.pkl` (recommended threshold 0.7)

---

# How to Run (Notebook Order)

```bash
# 1) Data audit + baseline
jupyter nbconvert --execute --to html Notebook_01_Data_Audit.ipynb

# 2) EDA + feature engineering
jupyter nbconvert --execute --to html Notebook_02_Deep_EDA_Feature_Engineering.ipynb

# 3) Modeling + fairness audit
jupyter nbconvert --execute --to html Notebook_03_Modeling_Fairness_v2.ipynb
```

---

# Limitations and Ethics

- The dataset reflects historical policing patterns; models may encode those biases.
- False positives have real-world consequences; model outputs should not drive enforcement actions.
- Demographic features help quantify disparities, but their use in deployment is ethically sensitive.
- Missing contextual data (socioeconomic factors, officer discretion) limits causal conclusions.

Recommended next steps:
- Independent policy review with community stakeholders.
- Human-in-the-loop decision processes if any deployment is considered.
- Ongoing fairness audits and threshold calibration.

---

# Key Artifacts

| File | Description |
| --- | --- |
| `artifacts/cleaned_full.parquet` | Cleaned dataset after audit and integrity checks |
| `artifacts/baseline_model_seed42.pkl` | Baseline model from Notebook 01 |
| `artifacts/pipeline.pkl` | Full preprocessing pipeline (Notebook 02) |
| `artifacts/X_train_final.parquet` | Transformed training data (32 features) |
| `artifacts/X_test_final.parquet` | Transformed test data (32 features) |
| `artifacts/metrics_report.csv` | Baseline vs LR vs DT performance table |
| `artifacts/fairness_report.csv` | Fairness metrics by race |
| `artifacts/disparity_comparison.csv` | FPR disparity ratios across models |
| `artifacts/best_model.pkl` | Selected best model (DT) |
| `artifacts/fair_model.pkl` | Best model with fairness-optimized threshold |

---

## Summary (TL;DR)

- Minimal baseline (6 features): F1 0.4344, ROC-AUC 0.9059
- Engineered model (32 features + SMOTEENN): F1 0.5082, ROC-AUC 0.9276
- Fairness audit: Black/Hispanic drivers show ~2.7x higher FPR than White drivers
- Threshold tuning to 0.7 improves both F1 (0.5228) and fairness (FPR range 1.44%)

This project demonstrates both the predictive power and the fairness risks of arrest prediction models in real-world policing data.
