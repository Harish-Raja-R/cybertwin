import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Section 1
cells.append(nbf.v4.new_markdown_cell("""# CyberTwin: Digital Twin-Based Cyberattack Detection Using Deep Learning

## 1. Project Introduction
CyberTwin aims to simulate a Digital Twin of a network environment and use deep learning to detect cyberattacks from network traffic. 
Machine learning is appropriate because modern cyberattacks involve complex, high-dimensional patterns in network traffic that traditional signature-based systems often fail to detect. A deep learning approach can learn behavioral features of normal vs. malicious traffic, making it capable of generalizing to novel attacks in a simulated digital twin.
"""))

cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
"""))

# Section 2
cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Overview
Use the CIC-IDS2017 dataset. We load the sample and display its basic structure."""))

cells.append(nbf.v4.new_code_cell("""data_files = glob.glob('../data/raw/*.csv')
dfs = []
for f in data_files:
    df_part = pd.read_csv(f)
    dfs.append(df_part)
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()

print(f"Total samples: {len(df)}")
print(f"Total features: {len(df.columns)}")

# Identify Target
target_col = None
for col in df.columns:
    if col.lower() == 'label':
        target_col = col
        break

print(f"Target variable: {target_col}")
print("Attack categories:")
print(df[target_col].unique())

df.head()
"""))

# Section 3
cells.append(nbf.v4.new_markdown_cell("""## 3. Data Quality
Check for missing values, infinite values, and data types."""))

cells.append(nbf.v4.new_code_cell("""missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0])

numeric_cols = df.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df[numeric_cols]).sum()
print("\\nInfinite values per column:")
print(inf_counts[inf_counts > 0])

print("\\nDuplicate rows:", df.duplicated().sum())

print("\\nConstant Features (Zero variance):")
nunique = df.nunique()
print(nunique[nunique <= 1].index.tolist())
"""))

# Section 4
cells.append(nbf.v4.new_markdown_cell("""## 4. Class Distribution
Visualize the balance between normal traffic and various attack types."""))

cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))
class_counts = df[target_col].value_counts()
ax = sns.barplot(x=class_counts.values, y=class_counts.index, palette='viridis')
plt.title('Class Distribution (Counts)', fontsize=14)
plt.xlabel('Number of Samples')
plt.ylabel('Attack Type')
for p in ax.patches:
    ax.annotate(f"{int(p.get_width())}", (p.get_width(), p.get_y() + p.get_height() / 2),
                xytext=(5, 0), textcoords='offset points', ha='left', va='center')
plt.tight_layout()
os.makedirs('../results/figures', exist_ok=True)
plt.savefig('../results/figures/class_distribution.png', dpi=300)
plt.show()

# Percentages
plt.figure(figsize=(10, 6))
class_pct = class_counts / len(df) * 100
ax2 = sns.barplot(x=class_pct.values, y=class_pct.index, palette='magma')
plt.title('Class Distribution (Percentages)', fontsize=14)
plt.xlabel('Percentage (%)')
plt.ylabel('Attack Type')
for p in ax2.patches:
    ax2.annotate(f"{p.get_width():.2f}%", (p.get_width(), p.get_y() + p.get_height() / 2),
                 xytext=(5, 0), textcoords='offset points', ha='left', va='center')
plt.tight_layout()
plt.savefig('../results/figures/class_distribution_percentage.png', dpi=300)
plt.show()
"""))

# Section 5
cells.append(nbf.v4.new_markdown_cell("""## 5. Numerical Feature Distribution
Plot a few representative numerical features."""))

cells.append(nbf.v4.new_code_cell("""# Select some important features based on literature (e.g., Flow Duration, Packet Lengths)
plot_features = [col for col in df.columns if 'Duration' in col or 'Packet Length' in col][:4]

plt.figure(figsize=(14, 10))
for i, col in enumerate(plot_features, 1):
    plt.subplot(2, 2, i)
    # Use log1p transformation for visualization due to extreme skew in network data
    sns.histplot(np.log1p(df[col].dropna()), bins=50, kde=True, color='teal')
    plt.title(f'Log1p Distribution of {col}')
    plt.xlabel('Log1p Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('../results/figures/feature_distributions.png', dpi=300)
plt.show()
"""))

# Section 6
cells.append(nbf.v4.new_markdown_cell("""## 6. Correlation Analysis
Identify highly correlated features. We select the top 20 most variable features for readability."""))

cells.append(nbf.v4.new_code_cell("""# Filter constant and target cols
valid_num = [c for c in numeric_cols if c not in nunique[nunique <= 1].index.tolist() and c != target_col]
# Fill na/inf for correlation
temp_df = df[valid_num].replace([np.inf, -np.inf], np.nan).fillna(0)

# Select top 20 by variance
top_vars = temp_df.var().sort_values(ascending=False).head(20).index
corr = temp_df[top_vars].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True, linewidths=.5)
plt.title('Correlation Heatmap (Top 20 Features by Variance)', fontsize=14)
plt.tight_layout()
plt.savefig('../results/figures/feature_correlation.png', dpi=300)
plt.show()
"""))

# Section 7
cells.append(nbf.v4.new_markdown_cell("""## 7. Outlier Analysis
Network traffic naturally contains outliers (e.g., an attack might generate an abnormally large packet). Thus, outliers should not necessarily be deleted without justification."""))

cells.append(nbf.v4.new_code_cell("""# Boxplots to visualize outliers for selected features
plt.figure(figsize=(12, 6))
sns.boxplot(data=temp_df[top_vars[:5]], orient='h', palette='Set2')
plt.title('Outlier Analysis for Top 5 Features (Raw Values)', fontsize=14)
plt.xscale('log')
plt.xlabel('Value (Log Scale)')
plt.tight_layout()
plt.savefig('../results/figures/outlier_analysis.png', dpi=300)
plt.show()
"""))

# Section 8
cells.append(nbf.v4.new_markdown_cell("""## 8. Feature Relationships
Explore how features relate to the attack labels."""))

cells.append(nbf.v4.new_code_cell("""if len(plot_features) >= 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.log1p(df[plot_features[0]]), y=np.log1p(df[plot_features[1]]), 
                    hue=df[target_col], palette='tab10', alpha=0.6)
    plt.title(f'Relationship: {plot_features[0]} vs {plot_features[1]}', fontsize=14)
    plt.xlabel(f'Log1p {plot_features[0]}')
    plt.ylabel(f'Log1p {plot_features[1]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../results/figures/feature_relationships.png', dpi=300)
    plt.show()
"""))

# Section 9
cells.append(nbf.v4.new_markdown_cell("""## 9. Dataset Findings

### Analytical Summary
1. **Dataset complexity**: The dataset has high dimensionality (> 75 features) and requires extensive preprocessing.
2. **Class imbalance**: Expected heavy imbalance, with benign traffic vastly outnumbering specific attack classes.
3. **Data quality**: The dataset contains some missing values (NaN) and infinite values (inf) generated during feature extraction (e.g., division by zero in Flow Bytes/s).
4. **Important features**: Features related to Flow Duration, Packet Length, and TCP Flags generally exhibit high variance.
5. **Potential challenges**: Heavy skewness, multicollinearity, and infinite values require robust cleaning strategies like robust scaling and missing value imputation.
6. **Potential leakage risks**: Features like IPs, Ports, or Timestamps must be excluded before modeling to prevent the model from memorizing the network topology instead of learning behavioral patterns.
7. **Implications for modeling**: Deep learning models (CNN, LSTM) will require normalized inputs. We need to handle class imbalance (e.g., using SMOTE) during Milestone 2.
"""))

nb['cells'] = cells
with open('notebooks/01_dataset_eda.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook created successfully.")
