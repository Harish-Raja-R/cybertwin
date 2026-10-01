import pandas as pd
import os
import json
import time
from baseline_models import train_logistic_regression, train_random_forest

def main():
    print("Loading data...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['Label_binary']
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv')['Label_binary']
    
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    selection_results = []
    
    # LR
    start = time.time()
    lr_model, lr_c, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
    lr_time = time.time() - start
    
    selection_results.append({
        'model': 'LogisticRegression',
        'best_params': f"C={lr_c}",
        'val_accuracy': lr_metrics['accuracy'],
        'val_precision': lr_metrics['precision'],
        'val_recall': lr_metrics['recall'],
        'val_f1': lr_metrics['f1'],
        'val_roc_auc': lr_metrics['roc_auc'],
        'training_time_seconds': lr_time
    })
    
    # RF
    start = time.time()
    rf_model, rf_params, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    rf_time = time.time() - start
    
    selection_results.append({
        'model': 'RandomForest',
        'best_params': str(rf_params),
        'val_accuracy': rf_metrics['accuracy'],
        'val_precision': rf_metrics['precision'],
        'val_recall': rf_metrics['recall'],
        'val_f1': rf_metrics['f1'],
        'val_roc_auc': rf_metrics['roc_auc'],
        'training_time_seconds': rf_time
    })
    
    # Save
    pd.DataFrame(selection_results).to_csv('results/tables/model_selection.csv', index=False)
    print("Baseline training and selection complete. Saved to results/tables/model_selection.csv")

if __name__ == "__main__":
    main()
