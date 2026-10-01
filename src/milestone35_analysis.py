import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return acc, prec, rec, f1, roc, pr, spec, fpr, fnr

def main():
    # 1. Load Data
    print("Loading data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['Label_binary']
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv')['Label_binary']
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Label_binary']
    
    # 2. Duplicate Analysis (Feature duplicates across splits)
    print("Running duplicate analysis...")
    # Check if there are exact rows in X_test that also appear in X_train
    # Pandas merge/isin is best
    train_hashes = pd.util.hash_pandas_object(X_train)
    val_hashes = pd.util.hash_pandas_object(X_val)
    test_hashes = pd.util.hash_pandas_object(X_test)
    
    train_test_dups = test_hashes.isin(train_hashes).sum()
    val_test_dups = test_hashes.isin(val_hashes).sum()
    train_val_dups = val_hashes.isin(train_hashes).sum()
    
    dup_results = [
        {'comparison': 'train_test', 'duplicate_count': int(train_test_dups), 'percent': float(train_test_dups/len(X_test)*100)},
        {'comparison': 'val_test', 'duplicate_count': int(val_test_dups), 'percent': float(val_test_dups/len(X_test)*100)},
        {'comparison': 'train_val', 'duplicate_count': int(train_val_dups), 'percent': float(train_val_dups/len(X_val)*100)}
    ]
    pd.DataFrame(dup_results).to_csv('results/tables/duplicate_stress_test.csv', index=False)
    
    # 3. Train Base RF
    print("Training base RF...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 4. Feature Importance
    print("Saving feature importances...")
    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'feature': X_train.columns, 'importance': importances}).sort_values('importance', ascending=False)
    feat_imp.to_csv('results/tables/rf_top_features.csv', index=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp.head(20), x='importance', y='feature')
    plt.title("RF Top 20 Features")
    plt.tight_layout()
    plt.savefig('results/figures/rf_top_features_stress_test.png')
    plt.close()
    
    # 5. Probability Analysis
    print("Saving probability distributions...")
    probs = rf.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(8, 6))
    sns.histplot(probs[y_test == 0], color='blue', alpha=0.5, label='BENIGN', bins=50)
    sns.histplot(probs[y_test == 1], color='red', alpha=0.5, label='PortScan', bins=50)
    plt.legend()
    plt.title("RF Probability Distribution on Test Set")
    plt.tight_layout()
    plt.savefig('results/figures/rf_probability_distribution.png')
    plt.close()
    
    # 6. Confusion Matrix Values
    preds = rf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    cm_df = pd.DataFrame([{'True Negatives': tn, 'False Positives': fp, 'True Positives': tp, 'False Negatives': fn}])
    cm_df.to_csv('results/tables/rf_confusion_matrix_values.csv', index=False)
    
    # 7. Distribution Shift (KS Test)
    print("Calculating distribution shifts...")
    ks_results = []
    # Test top 20 features
    for col in feat_imp.head(20)['feature']:
        stat, pval = ks_2samp(X_train[col], X_test[col])
        ks_results.append({
            'feature': col,
            'train_mean': X_train[col].mean(),
            'test_mean': X_test[col].mean(),
            'train_std': X_train[col].std(),
            'test_std': X_test[col].std(),
            'ks_stat': stat,
            'p_value': pval
        })
    pd.DataFrame(ks_results).to_csv('results/tables/train_test_distribution_shift.csv', index=False)
    
    # 8. Feature Ablation
    print("Running Feature Ablation...")
    ablation_results = []
    
    def evaluate_ablation(name, features_to_keep):
        model = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=1, random_state=42, n_jobs=-1)
        model.fit(X_train[features_to_keep], y_train)
        pred = model.predict(X_test[features_to_keep])
        prob = model.predict_proba(X_test[features_to_keep])[:, 1]
        metrics = compute_metrics(y_test, pred, prob)
        return {
            'experiment': name,
            'features': len(features_to_keep),
            'accuracy': metrics[0],
            'precision': metrics[1],
            'recall': metrics[2],
            'f1': metrics[3],
            'roc_auc': metrics[4],
            'pr_auc': metrics[5],
            'specificity': metrics[6],
            'fpr': metrics[7],
            'fnr': metrics[8],
        }

    # A: Current
    ablation_results.append(evaluate_ablation('Original Random Forest', X_train.columns))
    
    # B: No engineered
    eng_feats = ['Total_Packets', 'Fwd_Bwd_Packet_Ratio', 'Fwd_Bwd_Byte_Ratio']
    keep_b = [c for c in X_train.columns if c not in eng_feats]
    ablation_results.append(evaluate_ablation('RF without engineered features', keep_b))
    
    # D: Core behavioral only (Just an example subset: standard deviation, means, packet lengths)
    core_feats = [c for c in X_train.columns if 'Length' in c or 'IAT' in c]
    if len(core_feats) > 0:
        ablation_results.append(evaluate_ablation('RF behavioral-only (Length/IAT)', core_feats))
        
    # C: Drop top 3 features (extreme stress test)
    top3 = feat_imp['feature'].head(3).tolist()
    keep_c = [c for c in X_train.columns if c not in top3]
    ablation_results.append(evaluate_ablation('RF without top 3 features', keep_c))

    pd.DataFrame(ablation_results).to_csv('results/tables/rf_ablation_results.csv', index=False)
    
    # 9. Cross Validation
    print("Running Cross Validation...")
    # Stratified 5-fold CV on Train Set only
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def run_cv(model_name, model):
        scores = cross_validate(model, X_train, y_train, cv=cv, 
                                scoring=['accuracy', 'f1', 'roc_auc', 'average_precision'],
                                n_jobs=-1)
        return {
            'experiment': f'{model_name} cross-validation',
            'mean_accuracy': scores['test_accuracy'].mean(),
            'std_accuracy': scores['test_accuracy'].std(),
            'mean_f1': scores['test_f1'].mean(),
            'std_f1': scores['test_f1'].std(),
            'mean_roc_auc': scores['test_roc_auc'].mean(),
            'std_roc_auc': scores['test_roc_auc'].std(),
            'mean_pr_auc': scores['test_average_precision'].mean(),
            'std_pr_auc': scores['test_average_precision'].std(),
        }
    
    cv_rf = run_cv('Random Forest', RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=1, random_state=42))
    cv_lr = run_cv('Logistic Regression', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
    
    pd.DataFrame([cv_rf, cv_lr]).to_csv('results/tables/rf_cross_validation.csv', index=False)
    
    # 10. Summary Table
    print("Creating Summary Table...")
    # We load the final_model_comparison to pull the originals
    try:
        orig = pd.read_csv('results/tables/final_model_comparison.csv')
        orig['experiment'] = 'Original ' + orig['model']
        orig['split'] = 'Test (Random 15%)'
        orig['features'] = 70
        
        orig_summary = orig[['experiment', 'split', 'features', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'specificity', 'false_positive_rate', 'false_negative_rate']]
        orig_summary.rename(columns={'false_positive_rate': 'fpr', 'false_negative_rate': 'fnr'}, inplace=True)
    except:
        orig_summary = pd.DataFrame()
        
    ablation_df = pd.DataFrame(ablation_results)
    ablation_df['split'] = 'Test (Random 15%)'
    
    # Merge for final summary table
    summary_cols = ['experiment', 'split', 'features', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'specificity', 'fpr', 'fnr']
    final_summary = pd.concat([orig_summary, ablation_df], ignore_index=True)
    final_summary['notes'] = ''
    # Manually append CV results as notes or just leave them in their own file. We will add a row for CV
    cv_row = pd.DataFrame([{
        'experiment': 'RF cross-validation',
        'split': 'Train (5-Fold CV)',
        'features': 70,
        'accuracy': cv_rf['mean_accuracy'],
        'precision': None, # We didn't calc precision explicitly in CV to save time, it's fine
        'recall': None,
        'f1': cv_rf['mean_f1'],
        'roc_auc': cv_rf['mean_roc_auc'],
        'pr_auc': cv_rf['mean_pr_auc'],
        'specificity': None,
        'fpr': None,
        'fnr': None,
        'notes': f"std_f1={cv_rf['std_f1']:.6f}"
    }])
    final_summary = pd.concat([final_summary, cv_row], ignore_index=True)
    final_summary.to_csv('results/tables/milestone_3_5_summary.csv', index=False)
    
    print("Milestone 3.5 analysis complete.")

if __name__ == "__main__":
    main()
