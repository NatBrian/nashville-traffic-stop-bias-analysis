# Nashville Traffic Stops (Stanford Open Policing) — End-to-End Arrest Prediction + Racial Fairness Audit

This is the complete technical narrative of the project executed across **three notebooks**. It includes: integrity checks, missingness taxonomy, leakage rules, exact metrics, feature engineering decisions, modeling choices, fairness audit outputs, mitigation experiments, and the **full artifact map**. 

---

*Created By: Nathanael Brian*

---

## Table of Contents
1. Project Goal and Research Questions  
2. Dataset and Target  
3. Environment and Reproducibility  
4. Pipeline Overview (Phases + Notebook Map)  
5. Phase 1 — Data Audit, Profiling, Cleaning, Leakage Controls, Baseline Model (Notebook 01)  
6. Phase 2 — Deep EDA, Bias Diagnostics, Feature Engineering, Saved Preprocessing Pipeline (Notebook 02)  
7. Phase 3 — Modeling Under Extreme Imbalance, Interpretability, Fairness Audit, Mitigation (Notebook 03)  
8. Evaluation Protocol and Metrics (What/Why)  
9. Fairness Definitions and Audit Protocol  
10. Limitations, Ethics, and Deployment Guidance  
11. Artifact Inventory (All Saved Outputs)  
12. How to Run (Notebook Execution Order)

---

## 1) Project Goal and Research Questions

### Primary Question
**Can we predict whether a traffic stop results in an arrest?**

### Secondary Question
**Do prediction patterns and errors reveal demographic disparities (race/sex), and can we quantify them via fairness audits?**

### Method Strategy (Why this structure)
We intentionally separate the work into three phases:

- **Phase 1 (Trust the data + establish a floor):** verify data integrity, enforce leakage controls, and build a minimal baseline model.  
- **Phase 2 (Explain the signal + build context):** deep EDA and statistical tests for disparity + feature engineering into a reproducible preprocessing pipeline.  
- **Phase 3 (Model + audit):** train interpretable models under extreme class imbalance, measure performance, and run fairness audits (error-rate disparities by race). Demonstrate mitigation levers.

### Core Takeaways (from the full pipeline)
- **Predictability:** Arrests are predictable under severe imbalance (AUC > 0.90 baseline; > 0.92 with engineered model).  
- **Fairness:** Significant error-rate disparities persist: Black and Hispanic drivers have **~2.7–2.8× higher False Positive Rates** than White drivers in the selected model.  
- **Mitigation demo:** Threshold tuning improved both F1 and fairness spread, but does not eliminate disparities.  
- **Use case:** The pipeline is suited for **auditing / measurement**, not automated deployment.

---

## 2) Dataset and Target

- **Source:** Stanford Open Policing Project (Nashville subset)  
- **Input file:** `tn_nashville_2020_04_01.csv`  
- **Size:** 3,092,351 rows × 42 columns  
- **Target:** `arrest_made` (binary)  
- **Base rate:** **1.62% arrests**  
- **Imbalance ratio:** **~60.6:1 (no-arrest : arrest)**

---

## 3) Environment and Reproducibility

- **Python:** 3.12.x  
- **Global random seed:** `42` (used across notebooks)  
- **Key libraries:** pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn  
- **Install:**
```bash
pip install -r requirements.txt
````

Reproducibility notes:

* Seed is set for: train/test split, model initialization, resampling, clustering (KMeans), any randomized procedures.
* Outputs are saved as versioned artifacts in `artifacts/`.

---

## 4) Pipeline Overview (Phases + Notebook Map)

### Notebook 01 — Data Audit → Cleaning → Leakage Controls → Baseline Model

**Goal:** create a trusted dataset, prevent leakage, and produce a minimal model baseline.

Key outputs:

* cleaned dataset (`cleaned_full.parquet`)
* baseline model (`baseline_model_seed42.pkl`)

### Notebook 02 — Deep EDA → Bias Diagnostics → Feature Engineering → Saved Preprocessing Pipeline

**Goal:** quantify disparities and engineer contextual features into a reusable pipeline.

Key outputs:

* preprocessing pipeline (`pipeline.pkl`)
* transformed train/test matrices (`X_train_final.parquet`, `X_test_final.parquet`)
* metadata (`metadata.pkl`)

### Notebook 03 — Modeling → Interpretability → Fairness Audit → Mitigation

**Goal:** train interpretable models under imbalance, compare baseline vs engineered, audit fairness, test mitigation levers.

Key outputs:

* metrics reports (`metrics_report.csv`)
* fairness reports (`fairness_report.csv`, `disparity_comparison.csv`)
* best model (`best_model.pkl`)
* fairness-optimized threshold model (`fair_model.pkl`)

---

## 5) Phase 1 — Data Audit, Profiling, Cleaning, Leakage Controls, Baseline Model (Notebook 01)

### 5.1 Data Loading and Schema Confirmation

* Load full CSV (3.09M rows)
* Confirm:

  * expected columns present
  * data types and memory footprint (noting suboptimal storage like categoricals as object)
  * target existence and base rate

### 5.2 Profiling Framework: Three-Tier Discovery

#### A) Structure Discovery (types, cardinality, sparsity)

Findings and actions:

* Low-cardinality categorical candidates: `subject_race`, `subject_sex`, etc.
* `precinct` and `zone` stored as floats largely due to missingness → noted for imputation/handling.
* Extreme sparsity (>95%) in `contraband_*` and `search_basis` → treated as **structurally missing** (see taxonomy).

#### B) Content Discovery (ranges, distributions, plausibility)

* Validated `subject_age` range: **10–99** (kept; treated as valid domain values given dataset realities).
* Verified boolean fields contain expected values.
* Checked categorical frequency distributions for anomalies (rare categories, unknown labels).

#### C) Relationship Discovery (redundancy, leakage signals, correlations)

* Numeric correlation scans for redundant features.
* Early leakage suspicion for post-stop outcome columns (confirmed and removed in leakage step).

### 5.3 Missingness Taxonomy and Handling Decisions

We classified missingness by mechanism to avoid invalid imputation:

#### (1) Structurally Missing / MNAR (Missing Not At Random)

Meaning: field is only populated when a specific event occurred.
Examples:

* `contraband_*`, `search_basis` (~96% missing): only meaningful when a search occurs.

Handling:

* Boolean search-related fields imputed as `False` when missing (consistent with “no search” case).
* Search-basis fields treated carefully to avoid implying “no basis” equals “innocent”; used as presence/absence where appropriate.

#### (2) Administrative / Geographic Missing / MAR

Meaning: missing due to recording gaps rather than event structure.
Examples:

* `precinct`, `zone`, `lat`, `lng`

Handling:

* Do not drop rows wholesale due to location missingness.
* Preserve missingness as signal when appropriate:

  * spatial clustering assigns missing coords to cluster `-1` later (Notebook 02).

#### (3) Critical Fields for Valid Labeling and Fairness / (effectively) MCAR at low rate

Meaning: small missingness but essential for target/fairness analysis.
Examples (all <1%):

* `arrest_made`, `subject_race`, `subject_sex`, `subject_age`

Handling:

* Drop rows missing target or critical demographics (small impact, high validity gain).
* Age imputed with median (later improved with demographic-aware imputation during feature engineering).

### 5.4 Duplicate and Outlier Checks

* Duplicate rows: **0**
* Age outliers:

  * z-score/IQR checks performed
  * values 10–99 retained as valid domain values, handled via robust scaling later

### 5.5 Integrity Checks (Logic + Geography + Temporal)

**A) Geographic bounds**

* ~**0.66%** of stops outside expected Nashville boundary
* Action: **flagged, not removed** (could represent highways/jurisdiction overlap; removing could bias geospatial patterns)

**B) Contraband logic consistency**

* Verified no cases where contraband present without search:

  * `contraband_* == True` implies `search_conducted == True`
* Result: **0 invalid cases**

**C) Temporal consistency**

* Verified time fields parse cleanly
* Result: **100% valid times** (no malformed time strings after parsing rules)

### 5.6 Leakage Detection Rules (Hard Constraint)

Definition used: any feature that encodes or is downstream of the decision/outcome (arrest/citation/warning) is **not allowed**.

Dropped post-event outcome columns, including:

* `outcome`
* `citation_issued`, `warning_issued`
* `raw_*_issued`
* `raw_search_arrest`
  (And any similar columns that encode decisions made after/at the stop rather than conditions at stop initiation.)

Rationale:

* These variables would allow the model to “cheat” by using the consequence to predict the consequence.

### 5.7 Data Quality Scorecard (Post-cleaning)

* Completeness: **86.8%**
* Validity: **100%** (age range check)
* Consistency: **100%** (contraband/search logic)
* Uniqueness: **100%** (no duplicates)

### 5.8 Baseline Model (Minimal Features)

**Purpose:** establish a clean, leakage-free floor before feature engineering.

* Features (6):

  * `subject_age`, `subject_race`, `subject_sex`
  * `type`, `search_conducted`, `frisk_performed`
* Model: Logistic Regression with `class_weight='balanced'`

Baseline results (Notebook 01):

* ROC-AUC: **0.9055**
* F1 (arrest class): **0.4324**
* Precision: **0.30**
* Recall: **0.77**

Error pattern note (initial audit signal):

* False positives disproportionately higher for Black and Hispanic drivers
* False negatives higher for Hispanic and female drivers

Artifacts saved:

* `artifacts/cleaned_full.parquet`
* `artifacts/baseline_model_seed42.pkl`

---

## 6) Phase 2 — Deep EDA, Bias Diagnostics, Feature Engineering, Saved Pipeline (Notebook 02)

### 6.1 Data Load and Split Protocol

* Load: `artifacts/cleaned_full.parquet`
* Split: **stratified 80/20 train/test**
* Verified arrest base rate remains ~1.62% in both splits (important for comparability)

### 6.2 Deep EDA (What we measured and why)

#### A) Univariate / distribution checks

* Confirmed class imbalance: **60.6:1**
* Age distribution: mildly right-skewed but valid
* Cardinality analysis:

  * Low: `subject_sex` (2), `subject_race` (~6), `violation_group` (7)
  * High: `officer_id_hash` (**2247**) → requires high-cardinality encoding

#### B) Bivariate outcome patterns (arrest rate slices)

Arrest rate by race:

* Hispanic: **3.11%**
* Black: **2.26%**
* White: **1.09%**

Temporal pattern:

* 00:00–04:00 arrest rate **> 4.5%**
* Daytime arrest rate **< 1.0%**

These findings directly motivated:

* explicit `is_night` feature
* cyclic hour encoding (`sin_hour`, `cos_hour`)

#### C) Statistical disparity tests

* Chi-square test (race vs arrest): **p < 0.001**
* Cramer’s V: **0.0532**
  Interpretation:
* Effect size is small, but practically meaningful with 3M observations.

#### D) Sampling bias vs population baseline

* Black drivers overrepresented in stops: **1.35×**
* Hispanic drivers underrepresented: **0.53×**
  This matters because model “fairness” measurements depend on what the dataset represents.

#### E) Veil of Darkness (modified)

* Night arrest rates significantly higher across all races
* Hispanic drivers show the largest nighttime increase

#### F) Simpson’s paradox check (aggregation traps)

* Race disparity stable across **87.5%** of precincts
* Moderate reversal risk in one precinct
  Action: interpret global disparities with stratification awareness.

### 6.3 Feature Engineering (What we built and why)

Goal: move from raw demographics/stop indicators to contextual features (time, place, officer behavior) while preserving interpretability and reproducibility.

#### A) Temporal Features

* Extracted: year, month, day_of_week, hour
* Cyclic encoding: `sin_hour`, `cos_hour`
* `is_night` binary indicator (00:00–04:00 high-risk window)

#### B) Encoding Strategy (by feature type)

* One-hot:

  * `subject_sex`, `type`, `violation_group`, `race_canonical`
* Target encoding (out-of-fold + smoothing `m=10`):

  * `officer_id_hash` (high cardinality, prevents exploding dimension)
* Binary:

  * `is_out_of_state` (plate state)

#### C) Scaling

* RobustScaler for numeric features
  Rationale: robust to outliers (safer than StandardScaler when tails exist).

#### D) Spatial Features

* KMeans clustering on `lat/lng` with k=5
* Missing coordinates assigned cluster `-1`
  Rationale:
* raw lat/lng can be unstable and may not map cleanly to interpretable patterns in trees
* clustering provides coarse “zone” context without encoding exact location

#### E) Officer Behavior Features

* `officer_arrest_rate` (target-encoded)
* `officer_search_rate` (target-encoded)
* Removed `officer_total_stops_log` due to multicollinearity:

  * VIF > 10 (observed ~12.55)

#### F) Violation Grouping + Interactions

* Regex-based grouping into ~6 categories (Traffic, Equipment, Registration, etc.)
* Interaction flags:

  * `is_young_male`
  * `is_out_of_state` (kept as binary; also used as interaction in analysis)
* Race/sex-specific median age imputation (reduces distortion if age missing is not uniform across groups)

#### G) Multicollinearity Controls

* VIF analysis used to remove or revise features
* Retained interpretable context features (age, night indicators) after VIF reduction

#### H) Bias Mitigation Demonstration (not final fix)

* Reweighing by (race × outcome) improved disparate impact ratio:

  * **0.719 → 0.743** (still below the 0.80 “4/5 rule” guideline threshold)

### 6.4 Final Engineered Feature Set

After transformations and one-hot encoding: **32 features** (engineered + encoded).

Saved artifacts:

* `artifacts/pipeline.pkl`
* `artifacts/X_train_final.parquet`
* `artifacts/X_test_final.parquet`
* `artifacts/metadata.pkl`

---

## 7) Phase 3 — Modeling Under Extreme Imbalance, Interpretability, Fairness Audit, Mitigation (Notebook 03)

### 7.1 Data Loading and Sizes

* Load `X_train_final.parquet`, `X_test_final.parquet`
* Extract `sample_weight` from reweighing if present
* Confirmed:

  * Train: **2,473,858**
  * Test: **618,465**

### 7.2 Performance Comparison Setup

We compare:

* baseline (6 features) vs engineered (32 features) under consistent held-out testing.

Baseline recomputed on test split:

* F1: **0.4344**
* ROC-AUC: **0.9059**
* Precision: **0.3016**
* Recall: **0.7762**

Engineered pipeline + resampling:

* F1: **0.5082**
* ROC-AUC: **0.9276**
* Precision: **0.4009**
* Recall: **0.6941**

Net effect:

* F1: **+17.0%**
* ROC-AUC: **+2.4%**
* Precision: **+32.9%**
* Recall: **−10.6%** (precision/fairness improvements required accepting some recall loss)

### 7.3 Imbalance Strategy: SMOTEENN (and why)

Original imbalance: **60.6:1**
Approach:

* Subsample 20% for computational feasibility
* Apply SMOTEENN:

  * SMOTE generates minority samples to increase signal
  * ENN removes ambiguous/noisy majority points near boundary

Result:

* Target ~10:1; achieved post-resample **~11.8:1**

### 7.4 Models Trained (Interpretable Only)

* Logistic Regression:

  * GridSearch on C ∈ {0.1, 1.0}
* Decision Tree:

  * max_depth ∈ {5, 10}
  * min_samples_leaf ∈ {50, 100}

Best models (test):

* Logistic Regression: F1 **0.4920**, ROC-AUC **0.9326**
* Decision Tree: F1 **0.5082**, ROC-AUC **0.9276**

Selected model: **Decision Tree** (best F1 under evaluation criteria)

### 7.5 Interpretability Results (What drives predictions)

Logistic Regression top signals:

* `search_conducted` (OR ~331×)
* `search_person` (OR ~22×)
* race and violation features appear among influential coefficients → fairness risk indicator

Decision Tree:

* Root split: `search_person` (importance ~0.92)
* Secondary: officer rates + temporal features

Permutation importance confirms:

* search-related features dominate across both model families

### 7.6 Fairness Audit (Primary focus: False Positive Rate by race)

Why FPR:

* In this context, a false positive is “predicted arrest risk” when no arrest occurred.
* False positives disproportionately affecting a group represent a concrete harm: more incorrect escalation signals for that group.

Decision Tree fairness metrics (FPR by race):

* Black: **2.69%**
* Hispanic: **2.77%**
* White: **0.99%**

Disparity ratios vs White:

* Black: **2.71×**
* Hispanic: **2.80×**
* Asian and Unknown: < 0.8× (also flagged relative to “4/5 rule” style thresholds)

Fairness spread (FPR range max–min):

* Baseline: **0.0303**
* LR (32): **0.0283**
* DT (32): **0.0210**

Interpretation:

* Engineered models reduced global spread, but **Black/Hispanic vs White disparities remain large**.

### 7.7 Error Analysis (Counts + group rates)

* False positives: **10,415**
* False negatives: **3,072**

FP rate by race (% of group):

* Hispanic: **2.68**
* Black: **2.63**
* White: **0.98**
* Asian: **0.66**

### 7.8 Mitigation Experiments (Demonstration, not “solved”)

#### A) Threshold Tuning

* Tested thresholds from 0.3 to 0.7
* Best tradeoff at **0.7**:

  * F1 = **0.5228**
  * FPR range = **1.44%**
    Interpretation:
* Raising threshold trades recall for better precision and improved fairness spread.

#### B) Class Weight Experiments

* `balanced` increased fairness disparities (FPR range ~12%)
* `{0:1, 1:10}` gave best F1 (0.4861) with moderate fairness (2.93%)

### 7.9 Phase 3 Outputs

Artifacts saved:

* `artifacts/metrics_report.csv`
* `artifacts/fairness_report.csv`
* `artifacts/disparity_comparison.csv`
* `artifacts/best_model.pkl`
* `artifacts/fair_model.pkl` (threshold 0.7)

---

## 8) Evaluation Protocol and Metrics (What/Why)

### Why accuracy is not reported as a primary metric

With 1.62% positives, a trivial “predict no arrest” model yields ~98.38% accuracy. This is not useful.

Primary metrics used:

* **ROC-AUC:** ranking quality under imbalance
* **F1 (positive class):** precision/recall balance for arrests
* **Precision / Recall:** operational tradeoffs
* **Confusion matrix + group-conditioned error rates:** required for fairness audit

---

## 9) Fairness Definitions and Audit Protocol

Fairness is evaluated on **model errors**, not only outcomes:

* Outcome disparities can reflect policing patterns.
* Model disparities show whether a predictive instrument amplifies or mirrors those patterns.

Primary fairness metric:

* **False Positive Rate (FPR) by race**

  * FPR = FP / (FP + TN) computed within each race group

Secondary checks:

* disparity ratios vs a reference group (White)
* fairness spread across groups (max–min)

Note:

* Demographic features are included to measure disparity. Their use in production is ethically sensitive.

---

## 10) Limitations, Ethics, and Deployment Guidance

Limitations:

* This is observational policing data; patterns are not causal explanations.
* Missing contextual covariates (e.g., socioeconomic conditions, officer discretion narratives) constrain interpretation.
* High predictive power may arise from proxies (especially search-related variables), not underlying “risk.”

Ethics:

* False positives have real-world consequences.
* Using model outputs for enforcement decisions risks institutionalizing historical bias.

Deployment guidance:

* **Do not deploy** for automated decision-making in current form.
* Recommended use: auditing, measurement, and policy review support, ideally with community stakeholders and independent oversight.

---

## 11) Artifact Inventory (All Saved Outputs)

| File                                  | Description                                                      |
| ------------------------------------- | ---------------------------------------------------------------- |
| `artifacts/cleaned_full.parquet`      | Cleaned dataset after audit, missingness handling, leakage drops |
| `artifacts/baseline_model_seed42.pkl` | Baseline LR model (6 features)                                   |
| `artifacts/pipeline.pkl`              | Full preprocessing + feature engineering pipeline                |
| `artifacts/X_train_final.parquet`     | Engineered training matrix (post-transform)                      |
| `artifacts/X_test_final.parquet`      | Engineered test matrix (post-transform)                          |
| `artifacts/metadata.pkl`              | Feature metadata (encoders, mappings, etc.)                      |
| `artifacts/metrics_report.csv`        | Baseline vs engineered model performance table                   |
| `artifacts/fairness_report.csv`       | Fairness metrics by race (FPR, etc.)                             |
| `artifacts/disparity_comparison.csv`  | Disparity ratios relative to reference group                     |
| `artifacts/best_model.pkl`            | Selected best model (Decision Tree, best F1)                     |
| `artifacts/fair_model.pkl`            | Threshold-tuned model (threshold = 0.7)                          |

---

## 12) How to Run (Notebook Execution Order)

```
# 1) Data audit + baseline
jupyter nbconvert --execute --to html Notebook_01_Data_Audit.ipynb

# 2) Deep EDA + feature engineering
jupyter nbconvert --execute --to html Notebook_02_Deep_EDA_Feature_Engineering.ipynb

# 3) Modeling + fairness audit
jupyter nbconvert --execute --to html Notebook_03_Modeling_Fairness_v2.ipynb
```