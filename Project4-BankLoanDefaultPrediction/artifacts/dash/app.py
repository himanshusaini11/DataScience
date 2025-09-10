import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class BankLoanPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.grade_order = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
        self.label_cols = ["Sub Grade", "Batch Enrolled"]
        self.label_encoders = {}
        self.freq_map = None
        self.one_hot_cols = ["Initial List Status", "Employment Duration", "Verification Status"]
        self.one_hot_columns_fitted = None  # to align test data with train

    def fit(self, X, y=None):
        # Fit Label Encoders
        for col in self.label_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le

        # Fit Frequency Encoding
        self.freq_map = X["Loan Title"].value_counts().to_dict()

        # Fit One-Hot Columns
        dummies = pd.get_dummies(X[self.one_hot_cols], drop_first=True)
        self.one_hot_columns_fitted = dummies.columns.tolist()

        return self

    def transform(self, X):
        X = X.copy()

        # Grade Ordinal Mapping
        X["Grade"] = X["Grade"].map(self.grade_order)

        # Label Encoding
        for col in self.label_cols:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # Frequency Encoding
        X["Loan Title"] = X["Loan Title"].map(self.freq_map).fillna(0)

        # One-Hot Encoding (align columns)
        dummies = pd.get_dummies(X[self.one_hot_cols], drop_first=True)
        for col in self.one_hot_columns_fitted:
            if col not in dummies:
                dummies[col] = 0
        dummies = dummies[self.one_hot_columns_fitted]
        X = X.drop(columns=self.one_hot_cols)
        X = pd.concat([X, dummies], axis=1)

        return X

# Define the features used in the model
features = [
    ("Loan Amount", "number", "12000"),
    ("Funded Amount", "number", "11700"),
    ("Funded Amount Investor", "number", "11500"),
    ("Term", "text", "Short Term"),
    ("Batch Enrolled", "text", "BAT2522922"),  # ✅ VALID FORMAT
    ("Interest Rate", "number", "10.65"),
    ("Grade", "text", "B"),
    ("Sub Grade", "text", "B3"),
    ("Home Ownership", "text", "RENT"),
    ("Loan Title", "text", "Debt Consolidation"),
    ("Debit to Income", "number", "13.2"),
    ("Delinquency - two years", "number", "0"),
    ("Inquires - six months", "number", "1"),
    ("Open Account", "number", "5"),
    ("Public Record", "number", "0"),
    ("Revolving Balance", "number", "23400"),
    ("Revolving Utilities", "number", "0.72"),
    ("Total Accounts", "number", "15"),
    ("Total Received Interest", "number", "142.50"),
    ("Total Received Late Fee", "number", "0.00"),
    ("Recoveries", "number", "0.00"),
    ("Collection Recovery Fee", "number", "0.00"),
    ("Last week Pay", "text", "May-2023"),
    ("Total Collection Amount", "number", "0.00"),
    ("Total Current Balance", "number", "28700"),
    ("Total Revolving Credit Limit", "number", "35000"),
    ("Initial List Status_w", "number", "1"),
    ("Employment Duration_OWN", "number", "0"),
    ("Employment Duration_RENT", "number", "1"),
    ("Verification Status_Source Verified", "number", "1"),
    ("Verification Status_Verified", "number", "0")
]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create a grid layout of inputs (5 per row)
input_grid = []
row = []
for i, (name, dtype, example) in enumerate(features):
    input_id = name.replace(" ", "_")
    input_box = dbc.Col([
        html.Label(name),
        dcc.Input(
            id=input_id,
            type=dtype,
            placeholder=f"e.g., {example}",
            value=example if dtype == "number" else example,
            debounce=True,
            style={"width": "100%"}
        ),
        dbc.Tooltip(f"Example: {example}", target=input_id, placement="top")
    ], width=2)

    row.append(input_box)
    if (i + 1) % 5 == 0:
        input_grid.append(dbc.Row(row, className="mb-2"))
        row = []

if row:
    input_grid.append(dbc.Row(row, className="mb-2"))

# Load trained pipeline model (which includes preprocessing)
with open("model/rf_pipeline_model.pkl", "rb") as f:
    model = joblib.load(f)

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(name.replace(" ", "_"), "value") for name, *_ in features]
)
def predict(n_clicks, *values):
    if n_clicks is None:
        return ""
    try:
        input_data = pd.DataFrame([values], columns=[name for name, *_ in features])
        prediction = model.predict(input_data)[0]
        return dbc.Alert(f"Prediction: {'Default' if prediction == 1 else 'No Default'}", color="info")
    except Exception as e:
        return dbc.Alert(f"Error in prediction: {str(e)}", color="danger")

app.layout = dbc.Container([
    html.H2("🔍 Loan Default Predictor"),
    html.Div(input_grid),
    html.Br(),
    dbc.Button("Predict", id="predict-btn", color="primary"),
    html.Br(),
    html.Div(id="prediction-output")
], fluid=True)

if __name__ == "__main__":
    app.run_server(debug=True)