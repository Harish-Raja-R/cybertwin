import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    try:
        y_prob = model.predict_proba(X)[:, 1]
        roc = roc_auc_score(y, y_prob)
        pr = average_precision_score(y, y_prob)
    except:
        roc = np.nan
        pr = np.nan
        
    return {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc,
        'pr_auc': pr
    }

def train_logistic_regression(X_train, y_train, X_val, y_val):
    print("Training Logistic Regression with hyperparameter search...")
    C_values = [0.01, 0.1, 1, 10]
    best_f1 = -1
    best_model = None
    best_c = None
    best_metrics = None
    
    for c in C_values:
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        metrics = evaluate_model(model, X_val, y_val)
        print(f"C={c} -> Val F1: {metrics['f1']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = model
            best_c = c
            best_metrics = metrics
            
    print(f"Best LR configuration: C={best_c} with Val F1={best_f1:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/logistic_regression.joblib')
    
    return best_model, best_c, best_metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    print("Training Random Forest with hyperparameter search...")
    n_estimators_list = [50, 100]  # Reduced for CPU training speed
    max_depth_list = [10, 20]
    min_samples_leaf_list = [1, 5]
    
    best_f1 = -1
    best_model = None
    best_params = {}
    best_metrics = None
    
    for n in n_estimators_list:
        for d in max_depth_list:
            for l in min_samples_leaf_list:
                model = RandomForestClassifier(
                    n_estimators=n, max_depth=d, min_samples_leaf=l,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                model.fit(X_train, y_train)
                metrics = evaluate_model(model, X_val, y_val)
                print(f"n={n}, d={d}, l={l} -> Val F1: {metrics['f1']:.4f}")
                
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_model = model
                    best_params = {'n_estimators': n, 'max_depth': d, 'min_samples_leaf': l}
                    best_metrics = metrics
                    
    print(f"Best RF configuration: {best_params} with Val F1={best_f1:.4f}")
    joblib.dump(best_model, 'models/random_forest.joblib')
    
    return best_model, best_params, best_metrics
