# Predictive Maintenance Modeling for Industrial Machinery

## Overview
This project focuses on developing a predictive maintenance system for industrial machinery using generative sensor data. The goal is to predict equipment faults in advance, allowing maintenance teams to address issues proactively, reduce downtime, and improve operational safety.

The model leverages time-series sensor data to train an LSTM-based predictive model capable of identifying early signs of equipment failure with high recall and minimal false positives.

## Features
- **Synthetic Sensor Data Generation:** Created realistic multi-sensor time-series datasets representing machinery operating conditions and fault patterns.
- **Feature Engineering:** Applied rolling statistical feature extraction to capture temporal trends and early anomaly indicators.
- **Data Preprocessing:** Implemented scaling, train-test splits, and sequence formatting for LSTM model compatibility.
- **Model Development:** Built and trained an LSTM-based deep learning model for fault detection.
- **Imbalance Handling:** Addressed class imbalance through oversampling and careful threshold tuning to maintain <1% false positive rate.
- **Evaluation Metrics:** Used recall, precision, F1 score, and confusion matrix for rigorous performance evaluation.
- **Reproducible Pipeline:** Modular code structure for data generation, preprocessing, model training, and evaluation.

## Technologies Used
- **Languages:** Python (Pandas, NumPy)
- **Machine Learning & Deep Learning:** TensorFlow, Keras, Scikit-learn
- **Visualization:** Matplotlib
- **Feature Engineering:** Rolling statistics, scaling, sequence generation
- **Evaluation:** Confusion Matrix, Classification Report, Precision-Recall curves

## Notebooks and Resources
- [Project Notebook](./notebooks/Project.ipynb)

## Key Results
- **~99% Recall:** Successfully detected almost all faults within the dataset.
- **<1% False Positive Rate:** Minimised unnecessary maintenance interventions.
- **18% Improvement in Early Fault Detection Accuracy:** Achieved through rolling statistical features.
- **Robust Pipeline:** End-to-end reproducible workflow from synthetic data generation to final model evaluation.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/himanshusaini11/DataScience.git
2. Navigate to the project folder:
   ```bash
   cd Project3-IndustrialSafetyPrediction

## License
This project is licensed under the [MIT License](https://github.com/himanshusaini11/DataScience/LICENSE.md)

## Contact
For any queries or collaborations, feel free to reach out:

Name: Himanshu Saini
Email: himanshusaini.rf@gmail.com
LinkedIn: [LinkedIn](https://www.linkedin.com/in/sainihimanshu/)
