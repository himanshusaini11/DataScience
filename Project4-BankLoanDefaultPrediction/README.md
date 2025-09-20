# Bank Loan Default Prediction

## Problem statement
Banks face losses when borrowers default. The goal is to flag risky applications **before** approval using historical data.

- **Dataset Link:** [Kaggle Hackathon](https://www.kaggle.com/datasets/ankitkalauni/bank-loan-defaulter-prediction-hackathon)
- **Data**: `data/raw/train.csv`, `data/raw/test.csv`
- **Target**: `Loan Status` (0 = non-default, 1 = default)
- **Challenge**: **class imbalance** (defaults are rare). Standard accuracy is misleading; recall and F1 matter.

## Methodology
Build an **end‑to‑end** ML pipeline that:
- Handles class imbalance robustly.
- Benchmarks multiple models fairly.
- Produce predtictions and save them into the ```.csv``` file.

**Success criteria**
- Transparent metrics beyond accuracy: **Precision, Recall, F1, ROC AUC**.
- Reproducible pipeline in a single notebook.
- Visual evidence saved in `results/`.

## Results
What I implemented, step by step:
1. **Data audit & EDA**
   - Previewed schema and dtypes in the notebook.
   - Saved overview figures (e.g., [Unique values per column](results/01_UniqueValuesPerColumn.png), [Distributions](results/02_DistributionPlot.png)).
2. **Preprocessing**
   - Basic cleaning and type handling performed inside `notebooks/Project.ipynb`.
3. **Imbalance strategies**
   - **SMOTE** (oversampling) and **NearMiss** (undersampling) variants for each model.
4. **Models evaluated**
   - Logistic Regression, Decision Tree, Random Forest, XGBoost, and SVC.
   - Select models tuned via **GridSearchCV** (see notebook cells).
5. **Evaluation**
   - Reported **Accuracy, Precision, Recall, F1, ROC AUC**; created side‑by‑side comparison plots:
     - [Accuracy](results/03_ModelAccuracyComparison.png)
     - [F1](results/06_ModelF1ScoreComparison.png)
     - [ROC AUC](results/07_Model_ROC_AUC_Comparison.png)
6. **Feature insight**
   - Random Forest feature importance: [figure](results/13_ImportantFeatures_RF.png).
7. **Submission generation**
   - Created predictions in `results/HS_Submission*.csv` for review.

<details>
<summary>Per‑model figures</summary>

- Logistic Regression: [Original](results/1_LogReg%20-%20Original.png) · [SMOTE](results/2_LogReg%20-%20SMOTE.png) · [NearMiss](results/3_LogReg%20-%20NearMiss.png)
- Decision Tree: [Original](results/4_DecisionTree%20-%20Original.png) · [SMOTE](results/5_DecisionTree%20-%20SMOTE.png) · [NearMiss](results/6_DecisionTree%20-%20NearMiss.png)
- Random Forest: [Original](results/7_RandomForest%20-%20Original.png) · [SMOTE](results/8_RandomForest%20-%20SMOTE.png) · [NearMiss](results/9_RandomForest%20-%20NearMiss.png)
- XGBoost: [Original](results/10_XGBoost%20-%20Original.png) · [SMOTE](results/11_XGBoost%20-%20SMOTE.png) · [NearMiss](results/12_XGBoost%20-%20NearMiss.png)

</details>

## Key Result
All numbers are printed by `notebooks/Project.ipynb` and summarized here for quick review.

- **Best F1**: **XGBoost + NearMiss**
  Accuracy: **0.2945** · Precision: **0.0944** · Recall: **0.7708** · F1: **0.1681** · ROC AUC: **0.5092**

- **Best ROC AUC**: **SVC + SMOTE**
  ROC AUC: **≈0.5165** · F1: **≈0.1641**

**Interpretation**
- High **recall** with modest **precision** is consistent with severe imbalance: the model prioritizes catching defaults.
- ROC AUC near ~0.51 suggests signal is present but weak; threshold tuning and cost‑sensitive methods are likely next wins.


## Repo at a glance
```
data/
  raw/               # train.csv, test.csv
  processed/         # created by notebook
notebooks/
  Project.ipynb      # main workflow
results/
  ...                # all figures and HS_Submission*.csv files
```

## Reproduce locally
1. Create a Python 3 environment.
2. Install deps:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter
   ```
3. Run the notebook:
   ```bash
   jupyter notebook notebooks/Project.ipynb
   ```
4. Outputs will appear in `results/` and include `HS_Submission.csv` variants.


## Contact
- Name: Himanshu Saini
- Email: himanshusaini.rf@gmail.com
- LinkedIn: https://www.linkedin.com/in/sainihimanshu/
