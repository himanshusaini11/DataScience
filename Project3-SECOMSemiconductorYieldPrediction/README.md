# **SECOM Semiconductor Yield Prediction**

### Summary

>*Developed an end-to-end predictive yield modeling pipeline on SECOM semiconductor dataset. Built logistic regression, random forests, gradient boosting, deep learning, and online streaming models. Addressed extreme class imbalance with threshold tuning and cost-sensitive evaluation. Demonstrated that model choice depends on false negative vs false positive trade-offs, with crossover ratios as low as 4:1. Produced reproducible analysis in Python with SHAP interpretability and visualization.*

###  Project Overview

This project applies machine learning to the **SECOM semiconductor dataset (UCI repository)** to predict whether a wafer production run will pass or fail quality inspection.
- **Goal:** Predict failures early to reduce costly rework and downtime.
- **Dataset:**
	- 1,567 runs (wafer lots)
	- 590 continuous sensor measurements
	- Labels: -1 = pass, 1 = fail (6.6% fails --> extreme imbalance)

### Challenges
1. Severe class imbalance (~93% pass, ~7% fail).
2. High dimensionality (590 features, many correlated).
3. Missing values in multiple sensors.
4. Weak signal — failures are difficult to separate from passes.
5. Noisy labels — sensor readings may not fully capture defect causes.

### Methodology

1. Data Preparation
	- Removed constant features and handled missing values.
	- Standardized features.
	- Chronological splits (train / validation / test) to respect production drift.

2. Baseline Models
	- Logistic Regression
	- Random Forest
	- XGBoost
	- MLP (`PyTorch`)

3. Advanced Models
	- MLP with Weighted BCE / Focal Loss
	- Autoencoder (unsupervised anomaly detection)
	- Streaming SGD (online logistic regression)

4. Interpretability
	- `SHAP` values identified top sensor features most correlated with failures.

5. Cost-Sensitive Evaluation
	- Beyond PR-AUC/ROC-AUC, evaluated models under false negative (FN) vs false positive (FP) cost trade-offs.
	- Introduced a cost function: 
      $\text{Cost} = C_{fp} \cdot FP + C_{fn} \cdot FN$
	- Swept FN:FP ratios (1:1 to 30:1) to mimic fab economics.

### Key Results

1. Classical Metrics (PR-AUC, ROC-AUC)
	- Logistic Regression (PR-AUC ≈ 0.16) performed best.
	- Other models struggled (PR-AUC < 0.14).

2. Cost-Sensitive Analysis
	- Crossover points (where models become cheaper than Logistic Regression):
	- MLP (Focal): ~4:1
	- XGBoost: ~5:1
	- RF / MLP (Weighted BCE): ~6:1
	- Streaming SGD: dominated (never cheaper up to 30:1)
	- Insight: Even if Logistic looks best by AUC, fabs with higher FN penalties should prefer MLP or tree models.

3. Interpretability
	- `SHAP` analysis identified sensors `f059`, `f460`, `f033` as most impactful in predicting failures.

⸻

### Visuals
- PR / ROC curves
- SHAP summary plot
- Cost vs FN:FP curves (Block 41B — key figure)

### Takeaways
- No single best model: choice depends on fab’s FN:FP trade-off.
- Cost-sensitive analysis is more relevant than AUC for high-stakes manufacturing.
- Streaming/online models are attractive for real-time monitoring, but classical supervised models with tuned thresholds remain more cost-effective here.

### Repository Contents
- notebooks/ – full analysis (step-by-step Jupyter notebooks)
- results/ – figures, CSVs, SHAP outputs
- src/ – scripts for preprocessing and evaluation
- README.md – this file

### Future Work
- Feature engineering with domain knowledge (sensor aggregates, ratios).
- Semi-supervised anomaly detection with pseudo-labels.
- Deploy interactive Streamlit dashboard for FN:FP trade-off exploration.
- Benchmark against GAN-based imputation approaches (Semi-GAN).


## License
This project is licensed under the [MIT License](https://github.com/himanshusaini11/DataScience/LICENSE.md)

## Contact
For any queries or collaborations, feel free to reach out:

Name: Himanshu Saini
Email: himanshusaini.rf@gmail.com
LinkedIn: [LinkedIn](https://www.linkedin.com/in/sainihimanshu/)