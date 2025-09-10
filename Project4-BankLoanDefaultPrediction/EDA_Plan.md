# EDA Plan — Project 4: Bank Loan Default Prediction

Use this checklist to drive EDA work in `01_EDA.ipynb`. Complete items in order. When a task is done, check it and append “— Finish”.

Notes:
- Data paths: `data/raw/`, `data/processed/`, artifacts saved under `results/EDA/…`
- Notebook: `notebooks/01_EDA.ipynb`

## 0) Context & Assumptions
- [ ] Confirm target column name and positive class label (e.g., `Default` == 1).  
      Output: update in notebook header.
- [ ] Confirm unique ID column (e.g., `ID`/`Loan_ID`) and any date/timestamp columns.  
      Output: note in notebook header.
- [ ] Verify train/test sources match `data/raw/train.csv` and `data/raw/test.csv`.  
      Output: printed confirmation.

## 1) Helper Functions to Add (in 01_EDA under “Helper functions”)
- [ ] `memory_report(df) -> pd.DataFrame` — per-column + total memory usage.
- [ ] `missingness_table(df, sort=True) -> pd.DataFrame` — counts/% missing.
- [ ] `class_balance(y) -> dict[pct, counts, ratio]` — target distribution summary.
- [ ] `nunique_by_col(df) -> pd.DataFrame` — cardinality per column.
- [ ] `numeric_summary(df, target=None) -> pd.DataFrame` — describe + skew/kurtosis; optional by-target.
- [ ] `plot_target_balance(y, save_path=None)` — bar plot of class counts/rates.
- [ ] `plot_num_dists(df, cols, target=None, save_dir=None)` — hist/KDE for numeric columns (overall/by-target).
- [ ] `plot_cat_counts(df, col, target=None, top_n=20, save_path=None)` — bar plot for categorical.
- [ ] `corr_heatmap(df, cols=None, method='spearman', save_path=None)` — heatmap for numeric/selected.
- [ ] `high_corr_pairs(df, threshold=0.95) -> pd.DataFrame` — highly correlated pairs report.
- [ ] `train_test_schema_check(train, test) -> dict[payload]` — dtype/cardinality/unseen categories.
- [ ] `range_check(train, test, num_cols) -> pd.DataFrame` — numeric range overlap/drift flags.

## 2) EDA Steps (execute sequentially)

### 2.1 Dataset Overview
- [ ] Show shapes, columns, dtypes, memory footprint.  
      Output: `results/EDA/overview/overview.md` and `overview.csv`.
- [ ] Peek head/tail (3–5 rows).  
      Output: printed only.

### 2.2 Target Distribution
- [ ] Compute class counts, positive rate, imbalance ratio.  
      Output: `results/EDA/target/target_stats.csv`.
- [ ] Plot class balance (bar).  
      Output: `results/EDA/target/target_balance.png`.

### 2.3 Missingness
- [ ] Column-wise missingness table (count/%), sorted.  
      Output: `results/EDA/missing/missing_table.csv`.
- [ ] Missingness bar (top-N) and heatmap (lightweight).  
      Output: `results/EDA/missing/missing_bar.png`, `missing_heatmap.png`.

### 2.4 Duplicates & Keys
- [ ] Duplicated rows count; near-duplicates if applicable.  
      Output: `results/EDA/quality/duplicates.txt`.
- [ ] Validate uniqueness of ID column (if present).  
      Output: `results/EDA/quality/id_uniqueness.txt`.

### 2.5 Numerical Profiling
- [ ] Summary stats, skew, kurtosis for numeric columns.  
      Output: `results/EDA/numeric/summary.csv`.
- [ ] Distributions (hist/KDE) overall and by target (selected top-N).  
      Output: `results/EDA/numeric/dists/*.png`.
- [ ] Outlier scan (IQR rule); flag extreme columns.  
      Output: `results/EDA/numeric/outliers.csv`.

### 2.6 Categorical Profiling
- [ ] Cardinality table for categorical columns; top-k levels.  
      Output: `results/EDA/categorical/cardinality.csv`.
- [ ] Count plots (top-N) overall and by target for key features.  
      Output: `results/EDA/categorical/*.png`.
- [ ] Rare-category detection (e.g., <1% freq).  
      Output: `results/EDA/categorical/rare_levels.csv`.

### 2.7 Correlations & Redundancy
- [ ] Spearman correlation heatmap for numeric features (or selected).  
      Output: `results/EDA/corr/corr_heatmap.png`.
- [ ] High-correlation pairs report (>|r| >= 0.95).  
      Output: `results/EDA/corr/high_corr_pairs.csv`.

### 2.8 Leakage Checks
- [ ] Identify columns that could encode outcome or post-outcome info (e.g., status after default).  
      Output: `results/EDA/leakage/leakage_report.md`.

### 2.9 Train–Test Alignment
- [ ] Schema check: dtype matches, categorical level overlap, unseen categories.  
      Output: `results/EDA/alignment/schema_check.json`.
- [ ] Range check: numeric min/max overlap between train/test.  
      Output: `results/EDA/alignment/range_check.csv`.

### 2.10 Interactions & Bivariate
- [ ] Selected 2D plots (e.g., loan_amount vs. income by target).  
      Output: `results/EDA/bivariate/*.png`.
- [ ] Mutual information (top-k) between features and target.  
      Output: `results/EDA/bivariate/mutual_information.csv`.

### 2.11 Data Quality Flags
- [ ] Constant/near-constant columns; mixed types; invalid values.  
      Output: `results/EDA/quality/data_quality_flags.csv`.

### 2.12 Consolidated EDA Artifacts
- [ ] Write a short EDA summary with key findings and next steps for ETL.  
      Output: `results/EDA/summary/summary.md`.

## 3) Handoff to ETL (02_ETL.ipynb)
- [ ] Capture actionable ETL plan based on EDA (imputation, encoding, rare-level binning, scaling strategy, feature pruning).  
      Output: `results/EDA/summary/etl_plan.md`.

