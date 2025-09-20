PROJECT NAME: Semiconductor Wafer Yield Prediction
DATES: March 2025 to Sep 2025
CONTEXT/CLIENT TYPE: Semiconductor industry client (freelance engagement)

PROBLEM (S):
In a semiconductor fab, a small fraction of wafer runs (~6.6%) fail final quality control, yet each miss is costly; rule-based alarms either miss excursions or flood operators with false alerts. The telemetry is high-dimensional (590 sensors), partially missing, correlated, and time-ordered, so naive training leaks future information and accuracy metrics are misleading under severe class imbalance. The problem is to predict—before QC release—which runs are likely to fail, producing calibrated probabilities that enable cost-aware thresholds and a controlled alarm load.

ROLE/TASK (T):
Data Scientist (solo). Build a leakage-safe, cost-sensitive classifier; compare models; calibrate probabilities; select operating thresholds; and deliver a decision-useful analysis report to the client.

ACTIONS (A):
- Data:
  • Client data: proprietary wafer telemetry (590 sensors per run).
  • Public replication: UCI SECOM dataset (1,567 runs, 590 features, ~6.6% fails).
  • ETL (leakage-safe): drop sensors with >70% missing; add missingness indicators for 10–70% missing; median-impute remaining; per-feature outlier handling (log1p for skewed nonnegative, winsorize 1–99% for heavy tails); standardize numeric features; chronological split 60/20/20 (train/val/test).
  • Train-only pruning on sensors: low-variance threshold (1e-8), duplicate-feature removal, correlation pruning (|r|≥0.98). Persist processed parquet and final feature list.
- Modeling:
  • Models compared: Logistic Regression (with/without L1), Random Forest, XGBoost, and MLP (PyTorch).
  • Class imbalance mitigation: class weighting / cost-sensitive emphasis.
  • Primary metric: PR-AUC (with Recall/F1); ROC-AUC tracked as secondary.
  • Probability calibration: isotonic regression (fit on validation, evaluated on test).
  • Threshold selection: sweep decision thresholds under explicit FN:FP cost ratios; report alarm-load frontiers (FP/1k vs recall) and minimum-cost curves.
- Engineering/MLOps:
  • Reproducible pipelines/notebooks; deterministic seeds; artifacts persisted under /models and /results; features saved under /data/processed/*.parquet.
  • Bootstrap (B=1000) confidence intervals for key metrics and alarm-frontier points.
  • Deliverable: written report with figures (leaderboards, PR/ROC curves, cost curves, alarm frontiers, SHAP analyses) and recommendations.
- Tools:
  • Python, pandas, numpy, scikit-learn, XGBoost, PyTorch, SHAP, matplotlib, seaborn.
- Collaboration/Challenges:
  • Individual project. Key challenges: extreme class imbalance, noisy/high-dimensional signals, missingness, and leakage risk. Addressed via time-based split, regularization, calibrated probabilities, and interpretable models. Maintained NDA by publishing a replicated workflow on public data.

RESULTS (R):
- Metrics:
  • Test (public SECOM replication):
    – Logistic Regression: PR‑AUC ≈ 0.136 (95% CI [0.053, 0.254]); ROC‑AUC ≈ 0.640 (95% CI [0.474, 0.791]).
    – Random Forest: PR‑AUC ≈ 0.081 (95% CI [0.041, 0.135]); ROC‑AUC ≈ 0.616 (95% CI [0.509, 0.727]).
    – (Reference) Simple averaging ensemble reached highest ROC‑AUC ≈ 0.749 (PR‑AUC remained below logistic).
  • Calibration improved probability reliability (Brier qualitatively reduced) and enabled cost‑aware thresholding.
- Business/Operational impact:
  • Delivered calibrated operating thresholds matched to fab cost ratios and alarm budgets; analysis shows Random Forest becomes cheaper beyond ≈19:1 FN:FP cost ratio, while Logistic is preferred under lower ratios.
  • Identified a stable top‑k (≈5) sensor subset (via SHAP) associated with failures, guiding targeted monitoring and root‑cause investigation.
- Validation:
  • Chronological 60/20/20 split; thresholds tuned on validation; final metrics on held‑out test only.
  • Bootstrap 95% CIs for PR‑AUC/ROC‑AUC and alarm‑frontier summaries.
  • Cross‑model SHAP agreement (e.g., overlap on top sensors) increases confidence in signals, not artifacts.
- Links/Evidence:
  • Public repo (replicated workflow on SECOM): https://github.com/himanshusaini11/DataScience/tree/master/Project3-SECOMSemiconductorYieldPrediction
  • Notable artifacts (within repo):
    – results/modeling/02_model_leaderboard_pr_auc.png, 02_model_leaderboard_roc_auc.png
    – results/advanced_research/04_alarm_frontier_Logistic.png, 04_alarm_frontier_RandomForest.png
    – results/interpretability/* (SHAP summary/dependency plots)
    – data/processed/*.parquet (final feature sets and splits)
- NDA note (if applicable):
  • Client dataset and exact operational metrics are confidential. The GitHub project reproduces the methodology on public data (SECOM). Model handover/report delivered to client; deployment to be handled by client.
