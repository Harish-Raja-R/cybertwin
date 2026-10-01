# CyberTwin Preprocessing Report (Milestone 2)

## 1. Raw Dataset
- **File**: `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- **Total Samples**: 286,467
- **Original Features**: 78

## 2. Feature Audit
- **Target**: `Label` (Classified into `BENIGN` and `PortScan`).
- **Identifier**: `Destination Port`.
- **Constant Features**: Features with zero variance across the dataset.
- **Behavioral Flow Features**: e.g., Flow Duration, Packet Lengths, IAT, etc.

## 3. Leakage-Control Strategy
Network topological identifiers are known to cause target leakage (where models memorize IPs or Ports rather than malicious flow behavior). 
- **Removed Identifiers**: `Destination Port`
*(Note: IP addresses and timestamps were already absent in this particular sample slice).*

## 4. Cleaning
- Replaced all infinite values (`np.inf`, `-np.inf`) with `NaN`.
- Removed exactly 10 constant features (zero variance).

## 5. Duplicate Handling
Network flow datasets often contain duplicated packets.
- **Duplicate rows removed**: 72,353
- This step was executed *before* train/test splitting to prevent future test-set overlap.

## 6. Feature Engineering
We introduced three simple flow behavior ratios:
- `Total_Packets` = Fwd + Bwd
- `Fwd_Bwd_Packet_Ratio` = Fwd Packets / Bwd Packets
- `Fwd_Bwd_Byte_Ratio` = Fwd Bytes / Bwd Bytes
See `FEATURE_ENGINEERING.md` for full details and leakage justification.

## 7. Class Imbalance
After duplicate removal:
- **BENIGN**: 123,295
- **PortScan**: 90,819
Class balance is relatively healthy (1.35:1). SMOTE is not applied at this stage, but class weighting may be used during baseline modeling.

## 8. Missing Values & Scaling Strategy
- **Missing Value Imputation**: `SimpleImputer(strategy='median')`
- **Scaling**: `RobustScaler()` was chosen over `StandardScaler` due to the extreme log-skew of flow statistics (e.g., Duration).
*(Note: Parameters were strictly fitted on the training split only.)*

## 9. Split Strategy
- **Ratio**: 70% Train, 15% Validation, 15% Test.
- **Random Seed**: 42
- **Stratification**: By binary target class.
- **Train rows**: 149,879
- **Validation rows**: 32,117
- **Test rows**: 32,118

## 10. Final Feature Set
- **Final Feature Count**: 70
- **Removed Feature Count**: 11 total

## 11. Validation Checks
All required validation checks have passed:
- No target column inside `X`.
- No `NaN` or `inf` in transformed datasets.
- Test/Val row overlaps prevented by deterministic splitting post-duplicate-removal.
- Preprocessor strictly fitted on train.

## 12. Limitations
The raw dataset is a specific subset containing only `BENIGN` and `PortScan` classes. It does not represent all CIC-IDS2017 attack families (e.g., DDoS, Web Attacks, Brute Force). Our preprocessing pipeline is fully functional and generic enough to process the complete dataset later, but evaluating multi-class capabilities will require expanding the source data.
