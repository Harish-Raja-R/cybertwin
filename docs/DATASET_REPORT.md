# CyberTwin Dataset Audit Report

## 1. Dataset Source
Canadian Institute for Cybersecurity (CIC) - Intrusion Detection Evaluation Dataset (CIC-IDS2017).

## 2. Dataset Description
The dataset contains network flow features extracted using CICFlowMeter, representing normal traffic and various cyberattacks.

## 3. Dataset Size
- Number of files: 1
- Total rows (samples): 286467
- Total columns (features): 79
- Total file size: 73.34 MB
- Memory usage: 185.80 MB

## 4. Feature Description
- Numerical columns: 78
- Categorical columns: 1

## 5. Target Classes
- Identified Target Column: `Label`
- Number of classes: 2

## 6. Class Distribution
- PortScan: 158930 (55.48%)
- BENIGN: 127537 (44.52%)

Class imbalance ratio (Max/Min): 1.25

## 7. Missing Values
- Columns with missing values: 1
  - `Flow Bytes/s`: 15 (0.01%)


## 8. Infinite Values
- Columns with infinite values: 2
  - `Flow Bytes/s`: 356 (0.12%)
  - `Flow Packets/s`: 371 (0.13%)


## 9. Duplicate Analysis
- Duplicate rows: 72353 (25.26%)

## 10. Constant/Low-Variance Features
- Constant columns (variance=0): 10
  - ['Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'CWE Flag Count', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate']
- Near-zero variance / categorical-like features (< 5 unique values): 9

## 11. Outlier Analysis
- Highly skewed features (|skew| > 3): 62
Many network traffic features naturally exhibit extreme outliers (e.g., packet length, flow duration). These represent valid extreme cases (e.g. large file transfer or DDoS attack) and should generally be scaled using robust techniques rather than deleted.

## 12. Correlation Analysis
- Highly correlated features (>0.95): 28
  - Example highly correlated features: ['Total Backward Packets', 'Bwd Packet Length Std', 'Flow IAT Std', 'Fwd IAT Total', 'Fwd IAT Mean']


## 13. Potential Data Leakage
- Identified 1 potentially suspicious columns:
  - `Destination Port`: Identifier - Models may overfit to specific attacker/victim IPs rather than learning behavior.


## 14. Dataset Challenges
- **Missing/Infinite values**: Flow Bytes/s and Packets/s can contain Inf due to division by zero duration.
- **Multicollinearity**: Several flow characteristics are highly correlated.
- **Class Imbalance**: Attack traffic is vastly outnumbered by normal traffic, requiring oversampling techniques like SMOTE.
- **Scale**: Values range from fractions to tens of millions (e.g., flow duration in microseconds).

## 15. Implications for CyberTwin
The goal is to map this dataset into a Digital Twin simulation. The dataset provides realistic network flows, but we must exclude leakage features (IPs, Ports, timestamps) so the Digital Twin focuses purely on flow behavior (e.g., inter-arrival times, packet sizes, flag counts) regardless of network topology.

## 16. Recommended Preprocessing for Milestone 2
1. **Cleaning**: Replace Inf values with NaN, then impute NaN with the median of the respective column.
2. **Feature Selection**: Drop identifiers (IP, Port, Timestamp, Flow ID) to prevent target leakage.
3. **Scaling**: Apply `RobustScaler` or `MinMaxScaler` after `log1p` transformation to handle extreme skewness.
4. **Imbalance Handling**: Apply SMOTE to the training set only (after splitting) to balance the minority attack classes.