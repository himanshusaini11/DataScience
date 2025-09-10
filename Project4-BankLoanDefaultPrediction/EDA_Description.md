# EDA Description — Project 4: Bank Loan Default Prediction

This document summarizes what we have implemented so far in `notebooks/01_EDA.ipynb` and where artifacts are written. We will append to this file as we progress through the project.

**Objectives**
- Build a largely automated, configurable EDA notebook that requires minimal user input.
- Persist a small context/config JSON for reuse across notebooks (ETL, Modeling, Dash).
- Produce reusable EDA artifacts (tables/plots) under `results/EDA/`.

**Notebook Structure (Current)**
- `Project4-BankLoanDefaultPrediction/notebooks/01_EDA.ipynb`
- `Project4-BankLoanDefaultPrediction/EDA_Plan.md` — step checklist we mark as “Finish” as we proceed.

**Context & Assumptions (Step 0)**
- **Config `EDA_USER`**: Optional overrides the user can set (before inference):
  - **target**: label column (for this project: “Loan Status”).
  - **positive_label**: value representing positive class (e.g., “Default”).
  - **id_cols**: identifier columns excluded from features (e.g., `Loan_ID`).
  - **date_cols**: date/time columns. Formats are inferred and persisted.
  - **name_hints**: optional regex tie‑breakers for target guessing.
- **Inference `infer_context(train, test, user)`**:
  - Ranks target candidates via general heuristics: present-only-in-train, low cardinality, not ID‑like, plausible imbalance; `name_hints` are tie‑breakers only.
  - Infers positive label as the minority class by default.
  - Detects ID‑like columns and date-like columns.
  - Prints a preview and prompts to confirm before saving.
- **Artifacts**:
  - Saves to `results/EDA/context/config.json` with keys: `target_col`, `positive_label`, `id_cols`, `date_cols`, and `date_parse` (per‑column datetime parsing strategy).

**Robust Datetime Parsing**
- **Helper `guess_datetime_format`** added and preview logic appended to Cell 14:
  - Tries common fixed formats on a sample; accepts the best format if parse rate ≥ threshold; else falls back to dateutil with warnings suppressed.
  - Prints min/max/nulls/parse rate/mode per date column.
  - Persists chosen strategy in `ctx["date_parse"]` to reuse in ETL.

**Helper Functions (Step 4.0 block)**
- **`describe_df(df, ...)`**: JSON‑friendly dataset summary
  - Per‑column: dtype label, non‑nulls, null %, nunique, sample values, flags (constant, high‑cardinality, potential_id, mixed types).
  - Numeric summary: min/p25/p50/p75/max/mean/std/skew/kurtosis/zeros/negatives.
  - Datetime summary: min/max.
  - Target info: positive rate and imbalance ratio when `target`/`positive_label` provided.
  - Saves overview JSON and a flattened CSV (columns summary) under `results/EDA/overview/`.
  - Fixes applied: corrected sampling `random_state` usage; updated dtype detection to pandas‑2.x‑safe `_type_name` (uses `pd.CategoricalDtype`).
- **Primary key detection (vectorized)**
  - Recommended helper pattern to find PK candidates via `df.notna().all()` and `df.nunique()==len(df)`; optional composite key scan for top high‑cardinality pairs.
- **Cardinality summary**
  - Recommended helper to compute per‑column nunique (train/test), Jaccard overlap for small domains, and save to `results/EDA/categorical/cardinality.csv`.

**What’s Implemented So Far**
- Context inference + preview + confirmation prompt (Cell 13/14), exporting: `TARGET_COL`, `POS_LABEL`, `ID_COLS`, `DATE_COLS`.
- Robust datetime parsing preview appended to Cell 14; attaches `date_parse` into `ctx` before saving.
- EDA helper suite defined (Cell 19), including `describe_df` and improved `_type_name`.
- Diagnostic echo cells for `TARGET_COL` / `POS_LABEL` / `ID_COLS` (Cells 20–22). A consolidated “Dataset Overview” cell is prepared to replace these with a single step that runs `describe_df` and writes artifacts.

**Artifacts (Current/Planned)**
- `Project4-BankLoanDefaultPrediction/results/EDA/context/config.json` — saved after confirmation.
- `Project4-BankLoanDefaultPrediction/results/EDA/overview/overview.json` — dataset overview (once 2.1 runs).
- `Project4-BankLoanDefaultPrediction/results/EDA/overview/columns_summary.csv` — per‑column summary (once 2.1 runs).
- `Project4-BankLoanDefaultPrediction/results/EDA/categorical/cardinality.csv` — categorical cardinality report (optional helper to run).

**Next Steps (per EDA_Plan.md)**
- 2.1 Dataset Overview: run `describe_df` and save `overview.json` and `columns_summary.csv`. — Finish
- 2.2 Target Distribution: compute class stats and plot balance to `results/EDA/target/`. 
- 2.3 Missingness: table + bar/heatmap under `results/EDA/missing/`.
- 2.4 Duplicates & Keys: duplicate rows count and ID uniqueness checks under `results/EDA/quality/`.
- 2.5 Numerical Profiling: stats, distributions, IQR outliers under `results/EDA/numeric/`.
- 2.6 Categorical Profiling: cardinality, top‑k plots, rare levels under `results/EDA/categorical/`.
- 2.7 Correlations & Redundancy: Spearman heatmap + high‑corr pairs under `results/EDA/corr/`.
- 2.8 Leakage Checks: report under `results/EDA/leakage/`.
- 2.9 Train–Test Alignment: schema/range checks under `results/EDA/alignment/`.
- 2.10 Interactions & Bivariate: selected 2D plots + MI under `results/EDA/bivariate/`.
- 2.11 Data Quality Flags: constant/near‑constant/mixed types under `results/EDA/quality/`.
- 2.12 Consolidated Summary: key findings + ETL plan under `results/EDA/summary/`.

**Usage Notes**
- Set `EDA_USER` minimally with `target` and `positive_label`; add `id_cols` and `date_cols` if known.
- Run Cell 13/14 to infer/confirm context and export variables; rerun on schema changes.
- Run the overview cell (2.1) to generate and persist baseline EDA artifacts before deeper analysis.

