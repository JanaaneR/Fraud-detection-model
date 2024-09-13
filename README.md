# Fraud detection model

This project implements both TensorFlow and PyTorch models to predict audit fraud risk based on historical data. The goal is to classify instances as either high or low risk of fraud using machine learning models. The project employs data preprocessing, model training, cross-validation, and evaluation metrics for model comparison.

## Features

- Data Preprocessing: Handle missing values, standardize numerical features, and encode categorical features.
- TensorFlow Model: A simple feedforward neural network using Keras to classify audit fraud risk.
- PyTorch Model: A fully connected neural network for fraud classification, with batch processing using DataLoader.
- Cross-Validation: 5-fold cross-validation for model evaluation and performance tracking.
- Evaluation Metrics: Accuracy, confusion matrix, classification report, ROC-AUC, and Precision-Recall AUC.
- Confusion Matrix Heatmap: Visualization of model predictions vs. actual values using Seaborn.
- Display results and generate a report.

## Requirements

- Python 3.x
- TensorFlow 2.x
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- Seaborn
- Matplotlib


## Installation

1. Clone the repository:

```sh
git clone https://github.com/yourusername/audit-fraud-detection.git
cd audit-fraud-detection

