# Programming – Hands-On Assignment

## Note
1. This assignment is to be accompanied by the **`sales_pred_case.zip`** dataset. The unzipped contents should be a **single CSV file**.
2. Total time allocated: **24 hours** (you will need ~2–3 dedicated hours to complete the case).
3. The submission should be a **single executable Jupyter notebook**.
4. Assessment criteria:
   - **Code Quality:** Readability, conciseness, CPU/memory/time efficiency.
   - **Use of Standard Libraries:** All libraries must be installable through `pip`.
   - **Case Adherence:** Code must directly address the problem with no filler.
   - **Algorithm Quality:** Suitability of chosen model(s).
   - **Result Quality:** Accuracy on held-out samples.
   - **Scope for Improvement:** Potential to further tune or enhance the model.

---

## Data Description

The file **`sales_pred_case.csv`** contains **~1000 Material–Customer pairs** with **~3 years of weekly sales data**.

### Variable Dictionary

1. **Key**  
   Concatenation of Material & Customer codes.  
   ~970 unique keys. Predictions must be made **per key**.

2. **Material**, **Customer**, **CustomerGroup**, **Category**  
   Label-encoded categorical identifiers.

3. **Columns H–N**  
   Time and holiday features; integer or one-hot encoded.

4. **Columns O–T**  
   Promotion-related variables.  
   - `DiscountedPrice` → float  
   - Others → categorical

5. **Sales**  
   Target variable (float).

6. **YearWeek**  
   Time index; format: **YYYY-WW** (e.g., `2022-46`).

---

## Problem Statement

Predict **weekly sales** for each key for the weeks:

**`2022-46` → `2023-02` (inclusive)**

You may use **any data on or before `2022-45`** for training, validation, or model selection.

You may use **any approach**:
- Time Series  
- Traditional ML  
- Deep Learning  

---

## Accuracy Metric

Weighted MAPE (WMAPE):

Accuracy = 1 − SUM(|Sales − Prediction|) / SUM(Sales)

Where:

- **Absolute Error** = `|Sales − Prediction|`

Bias:
Bias = ( SUM(Sales) / SUM(Prediction) ) − 1

Goal:
- **Maximize Accuracy**
- **Minimize Bias**

---

## Additional Notes

1. Python must be used.
2. Only open-source, pip-installable libraries are allowed.
3. Any ML or DL package may be used (scikit-learn, TensorFlow, PyTorch, etc.).
4. Plots are optional.
5. Feature engineering steps must include **clear comments**.
6. The notebook must include:
   - Choice of model(s)
   - Choice of loss function
   - General observations about the data
   - Suggestions for further improvements