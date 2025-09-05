# SECOM Semiconductor Yield Prediction

Predicting which wafer runs are likely to fail final quality checks so fabs can reduce rework, avoid downtime, and focus engineering effort where it matters.

## Executive Summary
- **Business goal:** Early failure detection to save cost and time on the production line.
- **Real-world [SECOM dataset (UCI)](https://archive.ics.uci.edu/dataset/179/secom):** 1,567 production runs (wafer lots), 590 sensors per run; only ~6.6% are “fail” (rare event).
- **What we built:** Cleaned and prepared the data, trained several models, and evaluated them on a held‑out test set. We balanced the trade‑off between missed failures and false alarms, and made the results explainable.
- **Key results (test set):**
  - Best by precision‑recall: Logistic Regression (PR‑AUC ~0.12). A simple averaging ensemble is close (~0.116).
  - Best by ROC‑AUC: Averaging ensemble (~0.749), then Stacking (~0.709); Logistic and Random Forest are mid‑pack (~0.64/~0.62).
  - Operations lens: When missing a failure is much more costly than a false alarm, Random Forest can be preferable because it catches more rare failures at the expense of extra alarms. Otherwise, Logistic is a solid default.
- **Interpretability:** SHAP highlights a small, stable set of impactful sensors (e.g., f033, f059, f460) that consistently influence predictions. This can guide monitoring and process investigations.

## How to Browse the Project
- Read the full report: [ProjectReport.md](ProjectReport.md) (all figures and decisions).
- Notebooks (step‑by‑step):
  - EDA: [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb)
  - ETL: [notebooks/02_ETL.ipynb](notebooks/02_ETL.ipynb)
  - Modeling: [notebooks/03_Modeling.ipynb](notebooks/03_Modeling.ipynb)
  - Interpretability: [notebooks/04_Interpretability.ipynb](notebooks/04_Interpretability.ipynb)
- Key figures:
  - Precision–Recall (test): [results/interpretability/01_pr_curve_test.png](results/interpretability/01_pr_curve_test.png)
  - SHAP summary (RF): [results/interpretability/04_shap_rf_summary_bar.png](results/interpretability/04_shap_rf_summary_bar.png)
  - Cost vs FN:FP sweep: [results/interpretability/08_cost_vs_fn_fp_ratio.png](results/interpretability/08_cost_vs_fn_fp_ratio.png)

## What’s Inside
- **Data & setup:**
  - Chronological train/validation/test split to mimic production (no leakage from the future).
  - Careful handling of missing values, outliers, and redundant sensors.
  - Final modeling feature set: 375 features (list in `data/processed/features_final.txt`).
- **Models evaluated:**
  - Logistic Regression, Random Forest, XGBoost, MLP (neural net), feature‑selection pipelines, simple averaging and stacking ensembles.
  - Thresholds chosen on validation, then evaluated on test; probabilities calibrated.
- **How we measure success:**
  - Precision–Recall AUC (PR‑AUC) for rare‑event detection (primary).
  - ROC‑AUC as a secondary view; cost‑sensitive curves to reflect fab trade‑offs.

## Results at a Glance (Test)
- *PR‑AUC:* Logistic ~0.12 (best), Averaging ensemble ~0.116, Stacking ~0.098; others lower.
- *ROC‑AUC:* Averaging ~0.749, Stacking ~0.709, Logistic ~0.643, Random Forest ~0.617.
- *Cost trade‑offs:* Preferred model/threshold depends on how costly a missed failure is relative to a false alarm.

## Reproducibility and Artifacts
- *Processed datasets:* `data/processed/*.parquet` (includes `label` and `timestamp` for traceability; timestamp is not used as a model feature).
- *Final features:* `data/processed/features_final.txt`.
- *Trained models:* `models/`.
- *Metrics & figures:* `results/`.

## Limitations and Next Steps
- Rare‑event setting means modest absolute precision is expected; cost‑aware thresholds are essential.
- Results can shift with future process changes (prevalence drift). Consider periodic recalibration/retuning.
- Potential additions: reliability plots (calibration/ECE/Brier), feature‑importance stability over time, and an interactive dashboard to explore operating points.

## Contact
- Name: Himanshu Saini
- Email: himanshusaini.rf@gmail.com
- LinkedIn: https://www.linkedin.com/in/sainihimanshu/

## Acknowledgments
This project used AI‑assisted tooling (OpenAI ChatGPT via the Codex CLI) for editorial support, refactoring, and documentation. All analysis, code, and conclusions were implemented, reviewed, and validated by the author, and results are reproducible from the notebooks and artifacts.
