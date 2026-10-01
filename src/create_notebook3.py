import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Section 1
cells.append(nbf.v4.new_markdown_cell("""# CyberTwin Model Training (Milestone 3)

This notebook covers the training of Baseline models (Logistic Regression, Random Forest) and the advanced deep-learning model (1D-CNN + BiLSTM).
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import json
import joblib
import torch
from IPython.display import Image, display

# 1. Load processed data
X_train = pd.read_csv('../data/processed/X_train.csv')
y_train = pd.read_csv('../data/processed/y_train.csv')['Label_binary']
X_val = pd.read_csv('../data/processed/X_val.csv')
y_val = pd.read_csv('../data/processed/y_val.csv')['Label_binary']
X_test = pd.read_csv('../data/processed/X_test.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')['Label_binary']

# 2. Verify dimensions
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3 & 4. Baseline Models
The baseline models were trained via script (`train_baselines.py`). We will display the hyperparameter selection logs.
"""))

cells.append(nbf.v4.new_code_cell("""selection = pd.read_csv('../results/tables/model_selection.csv')
display(selection)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 5 & 6. Advanced CNN + BiLSTM and Training Curves
The deep learning model was trained using `train_deep_model.py`. The architecture consists of two 1D convolutional layers followed by a BiLSTM and Dropout.
"""))

cells.append(nbf.v4.new_code_cell("""display(Image('../results/figures/deep_model_loss_curve.png'))
display(Image('../results/figures/deep_model_f1_curve.png'))
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7 & 8. Validation Performance and Final Test Evaluation
We evaluate all models on the test set.
"""))

cells.append(nbf.v4.new_code_cell("""final_comp = pd.read_csv('../results/tables/final_model_comparison.csv')
display(final_comp)
display(Image('../results/figures/model_performance_comparison.png'))
"""))

cells.append(nbf.v4.new_markdown_cell("""## 9. Confusion Matrices
"""))
cells.append(nbf.v4.new_code_cell("""display(Image('../results/figures/confusion_matrix_logistic_regression.png'))
display(Image('../results/figures/confusion_matrix_random_forest.png'))
display(Image('../results/figures/confusion_matrix_cnn_bilstm.png'))
"""))

cells.append(nbf.v4.new_markdown_cell("""## 10 & 11. ROC and PR Curves
"""))
cells.append(nbf.v4.new_code_cell("""display(Image('../results/figures/roc_curve_comparison.png'))
display(Image('../results/figures/pr_curve_comparison.png'))
"""))

cells.append(nbf.v4.new_markdown_cell("""## 12. Model Comparison
The CNN+BiLSTM model offers more robust feature interaction modeling. As shown in the PR-AUC and F1-score, it handles the network traffic slightly better than Random Forest and significantly better than Logistic Regression.
"""))

cells.append(nbf.v4.new_markdown_cell("""## 13. Preliminary error analysis
"""))
cells.append(nbf.v4.new_code_cell("""errors = pd.read_csv('../results/tables/error_analysis.csv')
print(f"Total Errors: {len(errors)}")
display(errors.head())
"""))

cells.append(nbf.v4.new_markdown_cell("""## 14. Feature Importance
"""))
cells.append(nbf.v4.new_code_cell("""display(Image('../results/figures/random_forest_feature_importance.png'))
display(Image('../results/figures/logistic_regression_feature_coefficients.png'))
"""))

cells.append(nbf.v4.new_markdown_cell("""## 15. Findings and Limitations
The baselines and deep model are fully implemented. Next steps would include embedding this into a Digital Twin simulation logic.
"""))

nb['cells'] = cells

with open('notebooks/03_model_training.ipynb', 'w') as f:
    nbf.write(nb, f)
