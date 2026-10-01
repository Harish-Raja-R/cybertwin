import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Section 1
cell1 = nbf.v4.new_markdown_cell("""# CyberTwin Preprocessing and Feature Engineering (Milestone 2)

This notebook implements the data preprocessing and feature engineering pipeline for the CyberTwin project.

## 1. Load raw dataset
We begin by loading the raw network traffic dataset.
""")
cell2 = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import sys

plt.style.use('seaborn-v0_8-darkgrid')
df = pd.read_csv('../data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
df.columns = df.columns.str.strip()
print(f"Raw shape: {df.shape}")
""")

# Section 2
cell3 = nbf.v4.new_markdown_cell("""## 2. Verify Milestone 1 findings
We confirm the class imbalance, presence of missing values, and identifying column structures as documented in `EDA_SUMMARY.md`.
""")
cell4 = nbf.v4.new_code_cell("""print(df['Label'].value_counts())
print(f"Total Missing: {df.isna().sum().sum()}")
print(f"Total Infinite: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")
""")

# Section 3
cell5 = nbf.v4.new_markdown_cell("""## 3. Feature classification
We classify features into:
- Target (Label)
- Identifiers / Leakage risks (Destination Port, Source IP, etc.)
- Behavioral Traffic features (Flow Duration, Packets/s, etc.)
""")
cell6 = nbf.v4.new_code_cell("""# See DATASET_REPORT.md for full breakdown. 
target = 'Label'
identifiers = ['Destination Port', 'Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
""")

# Section 4
cell7 = nbf.v4.new_markdown_cell("""## 4. Leakage-control decisions
We must remove identifiers because a Digital Twin or a generalizable NIDS should not memorize IP addresses or specific ports. It should learn the *behavior* of the traffic.
""")
cell8 = nbf.v4.new_code_cell("""drop_cols = [c for c in identifiers if c in df.columns]
print(f"Dropping leakage features: {drop_cols}")
df = df.drop(columns=drop_cols)
""")

# Section 5
cell9 = nbf.v4.new_markdown_cell("""## 5. Duplicate handling
Network flows can have exact duplicates. Removing them ensures the model doesn't overfit to repeated identical flows.
""")
cell10 = nbf.v4.new_code_cell("""n_dupes = df.duplicated().sum()
print(f"Duplicates found: {n_dupes}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after duplicate removal: {df.shape}")
""")

# Section 6
cell11 = nbf.v4.new_markdown_cell("""## 6. Missing/infinite value handling
We replace infinite values with NaN so they can be imputed properly later in the pipeline.
""")
cell12 = nbf.v4.new_code_cell("""df = df.replace([np.inf, -np.inf], np.nan)
""")

# Section 7
cell13 = nbf.v4.new_markdown_cell("""## 7. Feature engineering
We create behavioral features to help the model learn ratios and totals. 
""")
cell14 = nbf.v4.new_code_cell("""df['Total_Packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']
df['Fwd_Bwd_Packet_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1e-6)
df['Fwd_Bwd_Byte_Ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1e-6)
""")

# Label prep
cell_label_md = nbf.v4.new_markdown_cell("## 8. Prepare Labels\nWe preserve multiclass labels and create a binary detection target.")
cell_label_code = nbf.v4.new_code_cell("""df['Label_multiclass'] = df['Label']
df['Label_binary'] = (df['Label'] != 'BENIGN').astype(int)
""")

# Constant drop
cell_const_md = nbf.v4.new_markdown_cell("## 9. Drop Constant Features\nConstant features provide no predictive power.")
cell_const_code = nbf.v4.new_code_cell("""numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in ['Label_binary']]
constant_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]
print(f"Dropping constant cols: {constant_cols}")
df = df.drop(columns=constant_cols)
""")

# Section 8
cell15 = nbf.v4.new_markdown_cell("""## 10. Train/validation/test split
We split 70/15/15 using stratification on the binary label to maintain class balance across splits.
""")
cell16 = nbf.v4.new_code_cell("""X = df.drop(columns=['Label', 'Label_multiclass', 'Label_binary'])
y = df[['Label_multiclass', 'Label_binary']]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y['Label_binary'])
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp['Label_binary'])
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
""")

# Section 9
cell17 = nbf.v4.new_markdown_cell("""## 11. Class distribution
Visualizing the class distribution across splits.
""")
cell18 = nbf.v4.new_code_cell("""def plot_dist(y_df, name):
    plt.figure(figsize=(6,4))
    sns.countplot(data=y_df, x='Label_binary')
    plt.title(f'Class Distribution - {name}')
    plt.savefig(f'../results/figures/class_distribution_{name.lower()}.png')
    plt.close()

plot_dist(y_train, 'Train')
plot_dist(y_val, 'Val')
plot_dist(y_test, 'Test')
print("Distributions plotted.")
""")

# Section 10
cell19 = nbf.v4.new_markdown_cell("""## 12. Fit preprocessing pipeline
We use `SimpleImputer(median)` and `RobustScaler()`. RobustScaler is chosen because EDA revealed extreme outliers in network flow metrics (like duration).
""")
cell20 = nbf.v4.new_code_cell("""preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Fit ONLY on train
preprocessor.fit(X_train)
""")

# Section 11
cell21 = nbf.v4.new_markdown_cell("""## 13. Transform datasets
Transform all splits using the fitted preprocessor.
""")
cell22 = nbf.v4.new_code_cell("""X_train_scaled = pd.DataFrame(preprocessor.transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(preprocessor.transform(X_val), columns=X_val.columns)
X_test_scaled = pd.DataFrame(preprocessor.transform(X_test), columns=X_test.columns)

# Feature distributions plot
plt.figure(figsize=(10,4))
sns.histplot(X_train['Flow Duration'].dropna(), bins=50)
plt.title('Flow Duration BEFORE scaling')
plt.savefig('../results/figures/feature_distribution_before_scaling.png')
plt.close()

plt.figure(figsize=(10,4))
sns.histplot(X_train_scaled['Flow Duration'].dropna(), bins=50)
plt.title('Flow Duration AFTER RobustScaler')
plt.savefig('../results/figures/feature_distribution_after_scaling.png')
plt.close()
""")

# Section 12
cell23 = nbf.v4.new_markdown_cell("""## 14. Validate transformed data
Check for NaN or Inf.
""")
cell24 = nbf.v4.new_code_cell("""print(f"NaN in train: {X_train_scaled.isna().sum().sum()}")
print(f"Inf in train: {np.isinf(X_train_scaled).sum().sum()}")
""")

# Section 13
cell25 = nbf.v4.new_markdown_cell("""## 15. Save processed datasets
""")
cell26 = nbf.v4.new_code_cell("""# Data is saved via the preprocessing.py script, this notebook is for interactive exploration.
print("Finished!")
""")

nb['cells'] = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9, cell10, 
               cell11, cell12, cell13, cell14, cell_label_md, cell_label_code, cell_const_md, cell_const_code,
               cell15, cell16, cell17, cell18, cell19, cell20, cell21, cell22, cell23, cell24, cell25, cell26]

with open('notebooks/02_preprocessing.ipynb', 'w') as f:
    nbf.write(nb, f)
