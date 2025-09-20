PROJECT NAME: Bank Loan Default Prediction
DATES: Jan 2025 - Mar 2025
CONTEXT/CLIENT TYPE: Financial services (client-based freelance)

PROBLEM (S):
Lenders need early default risk signals to price credit and control losses. Historical loan data is highly imbalanced (defaults are rare) and includes mixed-type features with outliers. The objective was to build a leakage-safe classifier that improves minority-class recall while keeping precision and operational false-alarm rates acceptable.

ROLE/TASK (T):
Freelance Data Scientist (solo). Own end-to-end modeling: data prep, resampling strategy, model comparison, thresholding guidelines, and a reproducible report/notebook handover.

ACTIONS (A):
- Data:
  - Structured loan records with numeric and categorical predictors; heavy class imbalance (defaults ≪ non-defaults).
  - Data prep: outlier handling (e.g., winsorization/clip for heavy tails), missing-value treatment, train/test split with stratification, and feature scaling for linear/SVM models.
  - Categorical encoding where applicable (One-Hot) within sklearn Pipeline.
- Modeling:
  - Compared Logistic Regression, Decision Tree, Random Forest, SVC, and XGBoost.
  - Addressed imbalance with SMOTE (oversampling) and NearMiss (under-sampling); evaluated “Original” (no resampling) as baseline.
  - Hyperparameter tuning via GridSearchCV; metrics: Precision, Recall, F1 (focus on minority class) and Accuracy for reference.
- Engineering/MLOps:
  - Sklearn Pipelines for clean preprocessing + model steps; deterministic seeds; clear separation of train/test; reproducible notebook.
  - Results table aggregating metrics across (model × resampling) variants.
- Tools:
  - Python, pandas, numpy, scikit-learn, imbalanced-learn (SMOTE, NearMiss), XGBoost, matplotlib, seaborn.
- Collaboration/Challenges:
  - Individual project. Key challenges: extreme imbalance and precision–recall trade-offs. Mitigated via resampling strategies, threshold awareness, and model comparison.

RESULTS (R):
- Metrics:
  - From the project notebook (test-set summaries by variant):
    - **Best Recall:** RandomForest–NearMiss **Recall ≈ 0.773**, Precision ≈ 0.094, F1 ≈ 0.168, Accuracy ≈ 0.289.
    - **Best F1 among tested:** XGBoost–NearMiss **F1 ≈ 0.168**, Recall ≈ 0.771, Precision ≈ 0.094, Accuracy ≈ 0.295.
    - **Balanced option (oversampling):** SVC–SMOTE **Recall ≈ 0.510**, Precision ≈ 0.098, F1 ≈ 0.164, Accuracy ≈ 0.520.
    - Logistic–SMOTE: Recall ≈ 0.475, Precision ≈ 0.098, F1 ≈ 0.163, Accuracy ≈ 0.548.
- Business/Operational impact:
  - Delivered trade-off guidance: choose NearMiss + tree/boosting for highest catch-rate (recall) when misses are very costly; prefer SMOTE + linear/SVC when a more balanced precision/recall profile is desired.
  - Provided a comparison table to support threshold and policy decisions (e.g., when to flag loans for manual review).
- Validation:
  - Stratified train/test split; metrics computed on held-out test.
  - Cross-validation via GridSearchCV during model selection; resampling confined to training folds only to avoid leakage.
  - Reported multiple operating points to reflect business tolerance for false positives.
- Links/Evidence:
  - GitHub repo: https://github.com/himanshusaini11/DataScience/tree/master/Project4-BankLoanDefaultPrediction
- NDA note (if applicable):
  - Client specifics are confidential; notebook demonstrates methodology on a public loan dataset analogue.
