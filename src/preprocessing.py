import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import joblib

class CyberTwinPreprocessor:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        self.removal_log = []
        self.preprocessor = None
        self.manifest = {}
        
    def load_data(self, filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()
        self.manifest['source_dataset'] = filepath
        self.manifest['raw_file_name'] = os.path.basename(filepath)
        self.manifest['total_rows'] = len(df)
        self.manifest['original_feature_count'] = len(df.columns) - 1 # excluding target
        return df
        
    def handle_duplicates(self, df):
        n_duplicates = df.duplicated().sum()
        self.manifest['duplicate_rows_removed'] = int(n_duplicates)
        print(f"Removing {n_duplicates} duplicate rows...")
        df = df.drop_duplicates().reset_index(drop=True)
        return df
        
    def create_labels(self, df, target_col='Label'):
        df['Label_multiclass'] = df[target_col]
        df['Label_binary'] = (df[target_col] != 'BENIGN').astype(int)
        
        self.manifest['multiclass_distribution'] = df['Label_multiclass'].value_counts().to_dict()
        self.manifest['binary_class_distribution'] = df['Label_binary'].value_counts().to_dict()
        return df
        
    def feature_engineering(self, df):
        print("Engineering features...")
        df['Total_Packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']
        df['Fwd_Bwd_Packet_Ratio'] = df['Total Fwd Packets'] / (df['Total Backward Packets'] + 1e-6)
        df['Fwd_Bwd_Byte_Ratio'] = df['Total Length of Fwd Packets'] / (df['Total Length of Bwd Packets'] + 1e-6)
        self.manifest['engineered_features'] = ['Total_Packets', 'Fwd_Bwd_Packet_Ratio', 'Fwd_Bwd_Byte_Ratio']
        return df
        
    def remove_invalid_features(self, df):
        print("Identifying features to remove...")
        features_to_drop = []
        
        # 1. Identifiers/Leakage
        if 'Destination Port' in df.columns:
            features_to_drop.append(('Destination Port', 'Identifier / Leakage Risk'))
        if 'Flow ID' in df.columns:
            features_to_drop.append(('Flow ID', 'Identifier'))
        if 'Source IP' in df.columns:
            features_to_drop.append(('Source IP', 'Identifier'))
        if 'Destination IP' in df.columns:
            features_to_drop.append(('Destination IP', 'Identifier'))
        if 'Timestamp' in df.columns:
            features_to_drop.append(('Timestamp', 'Identifier'))
            
        # Replace inf with nan for variance calculation
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Constant features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c not in ['Label_binary']]
        
        for col in numeric_cols:
            if df[col].nunique(dropna=True) <= 1:
                features_to_drop.append((col, 'Constant column or zero variance'))
                
        # Drop them
        drop_cols = [f[0] for f in features_to_drop]
        
        # Ensure we don't drop target columns by accident
        safe_drop_cols = [c for c in drop_cols if c in df.columns and c not in ['Label', 'Label_multiclass', 'Label_binary']]
        df = df.drop(columns=safe_drop_cols)
        
        # Update log
        for col, reason in features_to_drop:
            if col in safe_drop_cols:
                self.removal_log.append({'feature': col, 'reason': reason, 'action': 'REMOVED'})
                
        # Save removal log
        os.makedirs('results/tables', exist_ok=True)
        pd.DataFrame(self.removal_log).to_csv('results/tables/feature_removal_log.csv', index=False)
        
        self.manifest['removed_features'] = safe_drop_cols
        self.manifest['removed_feature_count'] = len(safe_drop_cols)
        self.manifest['final_feature_count'] = len(df.columns) - 3 # excluding original, multiclass, binary labels
        
        return df

    def split_data(self, df):
        print("Splitting data (70/15/15)...")
        # Ensure no inf
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Features and targets
        X = df.drop(columns=['Label', 'Label_multiclass', 'Label_binary'])
        y = df[['Label_multiclass', 'Label_binary']]
        
        # Train (70%) and Temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=self.random_seed, stratify=y['Label_binary']
        )
        
        # Val (15%) and Test (15%) -> Split Temp in half
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.random_seed, stratify=y_temp['Label_binary']
        )
        
        self.manifest['random_seed'] = self.random_seed
        self.manifest['split_ratio'] = '70/15/15'
        self.manifest['train_rows'] = len(X_train)
        self.manifest['validation_rows'] = len(X_val)
        self.manifest['test_rows'] = len(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def fit_transform_pipeline(self, X_train, X_val, X_test):
        print("Fitting preprocessing pipeline...")
        
        # We know from EDA that data is highly skewed, so RobustScaler is chosen
        self.preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        # Fit on train ONLY
        X_train_scaled = self.preprocessor.fit_transform(X_train)
        
        # Transform val and test
        X_val_scaled = self.preprocessor.transform(X_val)
        X_test_scaled = self.preprocessor.transform(X_test)
        
        # Convert back to DataFrame
        feature_names = X_train.columns
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names, index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_names, index=X_val.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index)
        
        self.manifest['preprocessing_methods'] = ['SimpleImputer(median)', 'RobustScaler']
        
        return X_train_scaled, X_val_scaled, X_test_scaled
        
    def save_artifacts(self, X_train, X_val, X_test, y_train, y_val, y_test):
        print("Saving artifacts...")
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Save datasets
        X_train.to_csv('data/processed/X_train.csv', index=False)
        X_val.to_csv('data/processed/X_val.csv', index=False)
        X_test.to_csv('data/processed/X_test.csv', index=False)
        
        y_train.to_csv('data/processed/y_train.csv', index=False)
        y_val.to_csv('data/processed/y_val.csv', index=False)
        y_test.to_csv('data/processed/y_test.csv', index=False)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'models/preprocessor.joblib')
        
        # Save manifest
        with open('results/tables/dataset_manifest.json', 'w') as f:
            json.dump(self.manifest, f, indent=4)
            
        print("Preprocessing complete!")

if __name__ == "__main__":
    processor = CyberTwinPreprocessor(random_seed=42)
    raw_path = 'data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    
    # 1. Load Data
    df = processor.load_data(raw_path)
    
    # 2. Duplicate Handling
    df = processor.handle_duplicates(df)
    
    # 3. Label Definition
    df = processor.create_labels(df)
    
    # 4. Feature Engineering
    df = processor.feature_engineering(df)
    
    # 5. Remove Invalid Features
    df = processor.remove_invalid_features(df)
    
    # 6. Train/Val/Test Split
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(df)
    
    # 7. Imputation and Scaling
    X_train_scaled, X_val_scaled, X_test_scaled = processor.fit_transform_pipeline(X_train, X_val, X_test)
    
    # 8. Save
    processor.save_artifacts(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
