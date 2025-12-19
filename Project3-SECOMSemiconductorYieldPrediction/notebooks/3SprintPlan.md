# 🔄 Revised 3-Sprint Plan (SECOM Yield Project)


## **Sprint 1 — Foundations & Interpretability**
**Goal:** Establish clean, reproducible baselines and open the “black box.”  

- [x] **Data/ETL**  
  - Missingness filtering, imputation, leakage-safe scaling, train/val/test splits.  

- [x] **Baseline models**  
  - Logistic, RF, XGB, MLP, FeatureSel, Stacking.  

- [x] **Evaluation**  
  - PR/ROC curves, threshold vs Precision/Recall, cost-ratio sweeps.  

- [x] **Interpretability**  
  - SHAP (RF & XGB): top features, overlaps, divergence.  
  - Dependence plots (f059, f033, f417, etc.).  

- [x] **Bootstrap Confidence Intervals**  
  - PR-AUC, ROC-AUC, Precision/Recall stability.  

- [ ] **New tasks**  
  - SHAP *temporal drift check*: are top features stable across train vs test years?  
  - Build a **“top-10 consensus” feature table**: features consistently important across models & bootstraps.  

**Deliverables:**  
- Clean PR/ROC figures.  
- SHAP bar/summary/dependence plots.  
- Consensus feature table.  
- Bootstrap CI forest plots.  


## **Sprint 2 — Calibration & Cost-Sensitive Analysis**
**Goal:** Link model outputs to fab-relevant costs and probabilities.  

- [x] **Isotonic calibration**  
  - Raw vs calibrated, reliability plots.  

- [x] **Decision Curve Analysis (DCA)**  
  - Net benefit curves (raw vs iso).  

- [x] **Cost sweeps**  
  - Logistic baseline → RandomForest takeover (19:1 raw, 13:1 iso).  

- [x] **Experiment A**  
  - Crossover shift table (iso vs raw).  

- [x] **Experiment B**  
  - Bootstrap robustness (crossover medians).  

- [x] **Experiment C**  
  - Prevalence-shift robustness.  

- [x] **Experiment D**  
  - Alarm-load frontiers (FP/1000 vs recall).  

- [ ] **New tasks**  
  - Reliability-constrained net benefit (ECE/Brier ≤ cap).  
  - Panel of **ΔBrier vs ΔPR-AUC vs ΔAUNB** (probability quality vs decision utility).  
  - Stratified calibration by **wafer subgroups** (e.g., early vs late production batches).  

**Deliverables:**  
- Calibration vs discrimination forest plots.  
- ΔAUNB heatmap.  
- Alarm-load frontiers.  
- Crossover shift executive digest.  


## **Sprint 3 — Operational Extensions & Publication-Ready Story**
**Goal:** Translate technical results into fab-actionable insights.  

- [x] **Ops-style frontiers**  
  - FP per 1k wafers vs Recall.  
  - Extend with **alarm budget overlays** (e.g., ≤50 FP/1000 wafers).  

- [x] **Scenario analysis**  
  - Simulate prevalence drift (done in Experiment C).  
  - Overlay alarm budgets to guide fab operating points.  

- [ ] **Interpretability tie-back**  
  - Connect SHAP top features (f059, f033, etc.) to fab monitoring and process variables.  

- [ ] **Novel angles for publication**  
  - Reliability-constrained operating points.  
  - Alarm-load frontiers in wafer units (new operational metric).  
  - Calibration-shifted crossovers (Logistic → RF, 19:1 → 13:1).  

**Deliverables:**  
- Final “Executive Digest” (A–D).  
- Ops-style plots with alarm budgets.  
- Publication-ready narrative:  
  - *“Calibration strengthens RF, weakens Logistic/XGB, changes takeover points, and clarifies operational trade-offs.”*  