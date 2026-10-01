import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
os.makedirs('results/figures', exist_ok=True)

print("Loading data...")
data_files = glob.glob('data/raw/*.csv')
dfs = []
for f in data_files:
    df_part = pd.read_csv(f)
    dfs.append(df_part)
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()

print(f"Total samples: {len(df)}")
target_col = None
for col in df.columns:
    if col.lower() == 'label':
        target_col = col
        break

if target_col:
    # 1. Class Distribution
    print("Generating class distribution...")
    plt.figure(figsize=(10, 6))
    class_counts = df[target_col].value_counts()
    ax = sns.barplot(x=class_counts.values, y=class_counts.index, palette='viridis')
    plt.title('Class Distribution (Counts)', fontsize=14)
    plt.xlabel('Number of Samples')
    plt.ylabel('Attack Type')
    plt.tight_layout()
    plt.savefig('results/figures/class_distribution.png', dpi=300)
    plt.close()

    # Percentages
    plt.figure(figsize=(10, 6))
    class_pct = class_counts / len(df) * 100
    ax2 = sns.barplot(x=class_pct.values, y=class_pct.index, palette='magma')
    plt.title('Class Distribution (Percentages)', fontsize=14)
    plt.xlabel('Percentage (%)')
    plt.ylabel('Attack Type')
    plt.tight_layout()
    plt.savefig('results/figures/class_distribution_percentage.png', dpi=300)
    plt.close()

# 2. Distributions
print("Generating feature distributions...")
plot_features = [col for col in df.columns if 'Duration' in col or 'Packet Length' in col][:4]
if plot_features:
    plt.figure(figsize=(14, 10))
    for i, col in enumerate(plot_features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(np.log1p(df[col].dropna()), bins=50, kde=True, color='teal')
        plt.title(f'Log1p Distribution of {col}')
        plt.xlabel('Log1p Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('results/figures/feature_distributions.png', dpi=300)
    plt.close()

# 3. Correlation
print("Generating correlation...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
nunique = df.nunique()
valid_num = [c for c in numeric_cols if c not in nunique[nunique <= 1].index.tolist() and c != target_col]
temp_df = df[valid_num].replace([np.inf, -np.inf], np.nan).fillna(0)
top_vars = temp_df.var().sort_values(ascending=False).head(20).index
corr = temp_df[top_vars].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, square=True, linewidths=.5)
plt.title('Correlation Heatmap (Top 20 Features by Variance)', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/feature_correlation.png', dpi=300)
plt.close()

# 4. Outliers
print("Generating outliers...")
plt.figure(figsize=(12, 6))
sns.boxplot(data=temp_df[top_vars[:5]], orient='h', palette='Set2')
plt.title('Outlier Analysis for Top 5 Features (Raw Values)', fontsize=14)
plt.xscale('log')
plt.xlabel('Value (Log Scale)')
plt.tight_layout()
plt.savefig('results/figures/outlier_analysis.png', dpi=300)
plt.close()

# 5. Relationships
if len(plot_features) >= 2 and target_col:
    print("Generating relationships...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=np.log1p(df[plot_features[0]]), y=np.log1p(df[plot_features[1]]), 
                    hue=df[target_col], palette='tab10', alpha=0.6)
    plt.title(f'Relationship: {plot_features[0]} vs {plot_features[1]}', fontsize=14)
    plt.xlabel(f'Log1p {plot_features[0]}')
    plt.ylabel(f'Log1p {plot_features[1]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('results/figures/feature_relationships.png', dpi=300)
    plt.close()

print("All plots generated!")
