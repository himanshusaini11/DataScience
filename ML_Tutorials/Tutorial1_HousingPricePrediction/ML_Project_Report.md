# Tutorial: End-to-End Machine Learning Project

_Auto-generated on 2025-08-16 16:30._

## Executive Summary

This notebook implements the **Chapter 2** end-to-end ML pipeline on the **California Housing** dataset to predict median house values per district. It follows the book's flow: framing the problem, creating a reliable test set, exploring and visualizing, preparing data with pipelines, selecting and fine-tuning models, and evaluating on a hold-out set.

## Pipeline Overview

- Deterministic **hash-based** train/test split by ID.
- Baseline **random split** for comparison.
- **Stratified** split on engineered `income_cat` to preserve income distribution.
- **Missing values** handled via `SimpleImputer` (median for numeric).
- **Custom transformers** (e.g., `CombinedAttributesAdder`, `DataFrameSelector`).
- **Pipelines** to chain preprocessing steps.
- **Feature combination** via `ColumnTransformer`.
- **Categoricals** encoded with `OneHotEncoder` (text → one-hot).
- **Feature scaling** via `StandardScaler`.

## Modeling & Evaluation

- Tried **LinearRegression** as a baseline.
- Explored **DecisionTreeRegressor**.
- Tuned **RandomForestRegressor** with cross-validation.
- Used **cross-validation** to estimate generalization error.
- Hyperparameter search via `GridSearchCV`.
- Primary metric(s): **RMSE**, **MAE**.

## Notable Features (from feature importances)

- median_income: 0.329
- INLAND: 0.158
- pop_per_hhold: 0.111
- longitude: 0.075
- bedrooms_per_room: 0.072
- latitude: 0.067
- room_per_hhold: 0.063
- housing_median_age: 0.042
- population: 0.019
- total_rooms: 0.017

## Interactive Geospatial View

- Notebook includes **Folium** maps (heatmap / markers) to explore spatial patterns.

## Error Analysis & Next Steps

- Inspect worst residuals and outliers; consider adding/removing features accordingly.
- Validate effects of pruning weak categorical dummies (e.g., `ocean_proximity`) via cross-validation.
- Document assumptions, and monitor data drift post-deployment.
