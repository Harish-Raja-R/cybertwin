import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import skew
import warnings
warnings.filterwarnings('ignore')

def audit_dataset(data_dir="data/raw"):
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not files:
        print("No CSV files found in data/raw!")
        return

    report = []
    report.append("# CyberTwin Dataset Audit Report\n")
    report.append("## 1. Dataset Source")
    report.append("Canadian Institute for Cybersecurity (CIC) - Intrusion Detection Evaluation Dataset (CIC-IDS2017).\n")
    report.append("## 2. Dataset Description")
    report.append("The dataset contains network flow features extracted using CICFlowMeter, representing normal traffic and various cyberattacks.\n")
    
    print("Loading data...")
    total_size = sum(os.path.getsize(f) for f in files)
    
    dfs = []
    for f in files:
        df_part = pd.read_csv(f)
        df_part.columns = df_part.columns.str.strip()
        dfs.append(df_part)
        
    df = pd.concat(dfs, ignore_index=True)
    
    report.append("## 3. Dataset Size")
    report.append(f"- Number of files: {len(files)}")
    report.append(f"- Total rows (samples): {len(df)}")
    report.append(f"- Total columns (features): {len(df.columns)}")
    report.append(f"- Total file size: {total_size / (1024*1024):.2f} MB")
    report.append(f"- Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB\n")
    
    report.append("## 4. Feature Description")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    report.append(f"- Numerical columns: {len(numeric_cols)}")
    report.append(f"- Categorical columns: {len(categorical_cols)}\n")
    
    target_col = None
    for col in df.columns:
        if col.lower() == 'label':
            target_col = col
            break
            
    report.append("## 5. Target Classes")
    if target_col:
        report.append(f"- Identified Target Column: `{target_col}`")
        class_counts = df[target_col].value_counts()
        report.append(f"- Number of classes: {len(class_counts)}\n")
        
        report.append("## 6. Class Distribution")
        for label, count in class_counts.items():
            pct = (count / len(df)) * 100
            report.append(f"- {label}: {count} ({pct:.2f}%)")
            
        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
        report.append(f"\nClass imbalance ratio (Max/Min): {imbalance_ratio:.2f}\n")
    else:
        report.append("Could not identify target column ('Label' not found).\n")
        
    report.append("## 7. Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_cols = missing[missing > 0]
    report.append(f"- Columns with missing values: {len(missing_cols)}")
    for col, count in missing_cols.items():
        report.append(f"  - `{col}`: {count} ({missing_pct[col]:.2f}%)")
    report.append("\n")
        
    report.append("## 8. Infinite Values")
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0]
    report.append(f"- Columns with infinite values: {len(inf_cols)}")
    for col, count in inf_cols.items():
        report.append(f"  - `{col}`: {count} ({(count/len(df))*100:.2f}%)")
    report.append("\n")
        
    report.append("## 9. Duplicate Analysis")
    dup_rows = df.duplicated().sum()
    report.append(f"- Duplicate rows: {dup_rows} ({(dup_rows/len(df))*100:.2f}%)\n")
    
    report.append("## 10. Constant/Low-Variance Features")
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    report.append(f"- Constant columns (variance=0): {len(constant_cols)}")
    if constant_cols:
        report.append(f"  - {constant_cols}")
    low_variance_cols = nunique[(nunique > 1) & (nunique < 5)].index.tolist()
    report.append(f"- Near-zero variance / categorical-like features (< 5 unique values): {len(low_variance_cols)}\n")
    
    report.append("## 11. Outlier Analysis")
    valid_numeric = [c for c in numeric_cols if c not in constant_cols and c != target_col]
    temp_df = df[valid_numeric].replace([np.inf, -np.inf], np.nan)
    try:
        skewness = temp_df.skew()
        highly_skewed = skewness[abs(skewness) > 3].index.tolist()
        report.append(f"- Highly skewed features (|skew| > 3): {len(highly_skewed)}")
        report.append("Many network traffic features naturally exhibit extreme outliers (e.g., packet length, flow duration). These represent valid extreme cases (e.g. large file transfer or DDoS attack) and should generally be scaled using robust techniques rather than deleted.\n")
    except Exception:
        report.append("- Could not calculate skewness.\n")
        
    report.append("## 12. Correlation Analysis")
    try:
        sample_df = temp_df.sample(min(10000, len(temp_df)), random_state=42)
        corr_matrix = sample_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
        report.append(f"- Highly correlated features (>0.95): {len(high_corr)}")
        if len(high_corr) > 0:
             report.append(f"  - Example highly correlated features: {high_corr[:5]}")
        report.append("\n")
    except Exception:
        report.append("- Could not calculate correlation.\n")
        
    report.append("## 13. Potential Data Leakage")
    suspicious_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if 'time' in col_lower or 'date' in col_lower:
            suspicious_cols.append((col, "Timestamp/Date - Often causes leakage if attack patterns depend on time of day in the synthetic dataset."))
        elif 'ip' in col_lower or 'port' in col_lower or 'mac' in col_lower:
            suspicious_cols.append((col, "Identifier - Models may overfit to specific attacker/victim IPs rather than learning behavior."))
        elif col_lower == 'id':
            suspicious_cols.append((col, "Identifier - Direct ID column, provides no behavioral info."))
            
    report.append(f"- Identified {len(suspicious_cols)} potentially suspicious columns:")
    for col, reason in suspicious_cols:
        report.append(f"  - `{col}`: {reason}")
    report.append("\n")
        
    report.append("## 14. Dataset Challenges")
    report.append("- **Missing/Infinite values**: Flow Bytes/s and Packets/s can contain Inf due to division by zero duration.")
    report.append("- **Multicollinearity**: Several flow characteristics are highly correlated.")
    report.append("- **Class Imbalance**: Attack traffic is vastly outnumbered by normal traffic, requiring oversampling techniques like SMOTE.")
    report.append("- **Scale**: Values range from fractions to tens of millions (e.g., flow duration in microseconds).\n")
    
    report.append("## 15. Implications for CyberTwin")
    report.append("The goal is to map this dataset into a Digital Twin simulation. The dataset provides realistic network flows, but we must exclude leakage features (IPs, Ports, timestamps) so the Digital Twin focuses purely on flow behavior (e.g., inter-arrival times, packet sizes, flag counts) regardless of network topology.\n")
    
    report.append("## 16. Recommended Preprocessing for Milestone 2")
    report.append("1. **Cleaning**: Replace Inf values with NaN, then impute NaN with the median of the respective column.")
    report.append("2. **Feature Selection**: Drop identifiers (IP, Port, Timestamp, Flow ID) to prevent target leakage.")
    report.append("3. **Scaling**: Apply `RobustScaler` or `MinMaxScaler` after `log1p` transformation to handle extreme skewness.")
    report.append("4. **Imbalance Handling**: Apply SMOTE to the training set only (after splitting) to balance the minority attack classes.")
    
    os.makedirs('docs', exist_ok=True)
    with open('docs/DATASET_REPORT.md', 'w') as f:
        f.write("\n".join(report))
        
    print("Dataset audit complete. Report written to docs/DATASET_REPORT.md")

if __name__ == "__main__":
    audit_dataset()
