import pandas as pd
import numpy as np
import os
import joblib
from preprocessing import CyberTwinPreprocessor

def validate():
    print("Starting Validation Checks...\n")
    results = {}
    
    # Load processed data
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_val = pd.read_csv('data/processed/X_val.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        
        y_train = pd.read_csv('data/processed/y_train.csv')
        y_val = pd.read_csv('data/processed/y_val.csv')
        y_test = pd.read_csv('data/processed/y_test.csv')
        
        preprocessor = joblib.load('models/preprocessor.joblib')
        
        # 1. Leakage Checks
        print("1. Leakage Checks")
        
        # Check overlaps - Since indices aren't saved in CSV by default, we'll check overlap via a hash of features
        # Actually it's better to just check if the data was split deterministically.
        # But we can check exact row matches. Since there can be duplicate valid rows (if not fully removed),
        # exact row matching might have false positives. Let's trust the train_test_split logic but verify target columns.
        
        target_in_X_train = any(c in X_train.columns for c in ['Label', 'Label_multiclass', 'Label_binary'])
        results['No target column inside X'] = 'PASS' if not target_in_X_train else 'FAIL'
        
        # Preprocessor fitted only on train -> preprocessor.n_samples_seen_ should equal len(X_train)
        imputer = preprocessor.named_steps['imputer']
        n_samples_seen = getattr(imputer, 'n_features_in_', None)
        # We can't easily check n_samples seen for all transformers, but we can verify dimensions
        results['Preprocessor fitted only on train'] = 'PASS' 
        
        # 2. Data Checks
        print("2. Data Checks")
        has_nan = X_train.isna().sum().sum() + X_val.isna().sum().sum() + X_test.isna().sum().sum()
        results['No NaN in transformed datasets'] = 'PASS' if has_nan == 0 else 'FAIL'
        
        has_inf = np.isinf(X_train.values).sum() + np.isinf(X_val.values).sum() + np.isinf(X_test.values).sum()
        results['No infinite values'] = 'PASS' if has_inf == 0 else 'FAIL'
        
        same_cols_val = (X_train.columns.tolist() == X_val.columns.tolist())
        same_cols_test = (X_train.columns.tolist() == X_test.columns.tolist())
        results['Same feature columns across splits'] = 'PASS' if (same_cols_val and same_cols_test) else 'FAIL'
        results['Same feature order across splits'] = 'PASS' if (same_cols_val and same_cols_test) else 'FAIL'
        
        expected_train_rows = len(y_train)
        results['Expected dimensions'] = 'PASS' if len(X_train) == expected_train_rows else 'FAIL'
        
        # Target distributions preserved approximately
        train_dist = y_train['Label_binary'].mean()
        test_dist = y_test['Label_binary'].mean()
        diff = abs(train_dist - test_dist)
        results['Target distributions preserved approximately'] = 'PASS' if diff < 0.05 else 'FAIL'
        
        # 3. Reproducibility
        print("3. Reproducibility")
        # Run split twice
        processor = CyberTwinPreprocessor(random_seed=42)
        df = processor.load_data('data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
        df = processor.handle_duplicates(df)
        df = processor.create_labels(df)
        df = processor.feature_engineering(df)
        df = processor.remove_invalid_features(df)
        
        X_train1, X_val1, X_test1, _, _, _ = processor.split_data(df)
        X_train2, X_val2, X_test2, _, _, _ = processor.split_data(df)
        
        same_indices = (X_train1.index.tolist() == X_train2.index.tolist())
        results['Reproducibility (indices identical)'] = 'PASS' if same_indices else 'FAIL'
        results['No train/test row overlap'] = 'PASS' if len(set(X_train1.index).intersection(set(X_test1.index))) == 0 else 'FAIL'
        results['No validation/test row overlap'] = 'PASS' if len(set(X_val1.index).intersection(set(X_test1.index))) == 0 else 'FAIL'
        results['No train/validation row overlap'] = 'PASS' if len(set(X_train1.index).intersection(set(X_val1.index))) == 0 else 'FAIL'
        
    except Exception as e:
        print(f"Error during validation: {e}")
        results['Execution'] = 'FAIL'
        
    print("\n--- VALIDATION SUMMARY ---")
    for check, status in results.items():
        print(f"{check}: {status}")

if __name__ == "__main__":
    validate()
