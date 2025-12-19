# Data Science Projects

Welcome to the **Data Science** repository. It houses multiple end‑to‑end projects covering data acquisition, exploratory analysis, modeling, interpretability, and lightweight apps/dashboards. Each project is self‑contained with its own notebooks, results, and artifacts.

## Table of Contents

- [Projects](#projects)
  - [Project 1: Extracting and Visualizing Stock Data](./Project1-ExtractingAndVisualizingStockData)
  - [Project 2: SpaceX Falcon 9 Landing Prediction](./Project2-SpaceX_F9_LandingPrediction)
  - [Project 3: SECOM Semiconductor Yield Prediction](./Project3-SECOMSemiconductorYieldPrediction)
  - [Project 4: Bank Loan Default Prediction](./Project4-BankLoanDefaultPrediction)
  - [Project 5: A/B Test Analysis](./Project5-AB_Testing)
  - [Project 6: Industrial Safety Prediction](./Project6-IndustrialSafetyPrediction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Projects

### Project 1: Extracting and Visualizing Stock Data

This project is a hands-on exercise where we perform web scraping using Python's BeautifulSoup library to extract data from a web page and analyze it using Pandas. The data extracted is focused on financial information, and we demonstrate how to clean, process, and visualize this data.

- **Web Scraping**: Using BeautifulSoup to scrape data from a table on a webpage.
- **Data Cleaning**: Processing and cleaning the scraped data to remove unwanted characters.
- **Data Analysis**: Storing the cleaned data in a Pandas DataFrame for further analysis.
- **Visualization**: Generating visual representations of the data using Matplotlib.

### Project 2: SpaceX Falcon 9 Landing Prediction

This project aims to predict the reusability of the first stage of SpaceX Falcon 9 rockets using data analysis, visualization, and machine learning techniques. Reusability is a critical factor in reducing the cost of space travel.

- **Data Collection**: Gathered data using SpaceX REST API and web scraping (Wikipedia).
- **Data Wrangling**: Cleaned, transformed, and prepared the dataset for analysis and modeling.
- **Exploratory Data Analysis (EDA)**: Performed using SQL and visualization tools to uncover insights.
- **Interactive Visualizations**: Created with Folium and Plotly Dash.
- **Predictive Modeling**: Implemented machine learning models for classification tasks.
- **Interactive Dashboard**: Built with Plotly Dash and deployed on Azure.

### Project 3: SECOM Semiconductor Yield Prediction

Predictive modeling for rare fab failures using the UCI SECOM dataset (1,567 runs, 590 sensors, ≈6.6% fails). Emphasis on imbalanced classification, calibrated probabilities, and cost‑aware thresholding.

- **EDA & ETL**: Missingness indicators, variance/duplicate pruning, standardization.
- **Modeling**: Logistic baseline (best PR‑AUC ≈ 0.12), ensembles (avg ROC‑AUC ≈ 0.75), with calibration.
- **Interpretability**: SHAP, feature importance stability, alarm‑load trade‑offs.

### Project 4: Bank Loan Default Prediction

End‑to‑end pipeline for loan default detection (Kaggle‑style split). The project is organized similarly to Project 3.

- **Structure**: `data/{raw,processed,interim}`, `notebooks`, `results`, `models`, `artifacts`.
- **EDA**: Automated notebook generates key artifacts under `Project4-BankLoanDefaultPrediction/results/EDA/`:
  - Target balance, missingness (bar + heatmap)
  - Numeric correlation heatmap
  - Top‑K numeric distributions by target; categorical default‑rate plots
  - Decile default‑rate curves; train–test drift overlays
- **Modeling**: Notebook split for EDA --> ETL --> Modeling --> Interpretability; model binaries are kept local (large `.pkl` files ignored in Git).

### Project 5: A/B Test Analysis

Design and analysis of controlled experiments, including power analysis, invariant metrics checks, and non‑parametric tests where appropriate. Emphasis on clear assumptions and effect size interpretation.

### Project 6: Industrial Safety Prediction

Classification of incident/safety outcomes with attention to class imbalance, feature engineering, and interpretable reporting for operations stakeholders.

## Installation

- Clone the repository:
  ```bash
  git clone https://github.com/himanshusaini11/DataScience.git
  cd DataScience
  ```
- Create a Python 3.10+ environment and install common deps:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap plotly
  ```
- Some projects have additional, project‑specific dependencies (e.g., Dash, Folium). Install from per‑project notes where applicable.

## Usage

- Navigate to a project folder and open its notebooks under `notebooks/`.
- Project 3 and 4 follow a split notebook pattern: `01_EDA.ipynb` → `02_ETL.ipynb` → `03_Modeling.ipynb` → `04_Interpretability.ipynb`.
- Generated artifacts (tables/plots) are written under each project’s `results/` directory.

## Contributing

We welcome contributions from the community! If you would like to contribute to this repository, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

Please ensure that your contributions adhere to our coding guidelines and standards.

## License

This repository is licensed under the MIT License.

## Contact

For any questions or inquiries, please contact:

- **Himanshu Saini**
- Email: [himanshusaini.rf@gmail.com](mailto:himanshusaini.rf@gmail.com)
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/sainihimanshu/)

We hope you find these projects useful and look forward to your contributions!
