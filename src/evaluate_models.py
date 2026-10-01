import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve)
from baseline_models import evaluate_model
from deep_model import CNN_BiLSTM

def calc_advanced_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)
    
    try:
        roc = roc_auc_score(y_true, y_prob)
        pr = average_precision_score(y_true, y_prob)
    except:
        roc = np.nan
        pr = np.nan
        
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc,
        'pr_auc': pr,
        'specificity': specificity,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr
    }

def main():
    print("Evaluating models on TEST set...")
    
    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')['Label_binary']
    
    # Selection info
    selection = pd.read_csv('results/tables/model_selection.csv')
    lr_time = selection[selection['model']=='LogisticRegression']['training_time_seconds'].values[0]
    rf_time = selection[selection['model']=='RandomForest']['training_time_seconds'].values[0]
    
    # 1. LR
    lr_model = joblib.load('models/logistic_regression.joblib')
    lr_prob = lr_model.predict_proba(X_test)[:, 1]
    lr_pred = lr_model.predict(X_test)
    lr_metrics = calc_advanced_metrics(y_test, lr_pred, lr_prob)
    lr_metrics['model'] = 'LogisticRegression'
    lr_metrics['training_time_seconds'] = lr_time
    
    # 2. RF
    rf_model = joblib.load('models/random_forest.joblib')
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    rf_pred = rf_model.predict(X_test)
    rf_metrics = calc_advanced_metrics(y_test, rf_pred, rf_prob)
    rf_metrics['model'] = 'RandomForest'
    rf_metrics['training_time_seconds'] = rf_time
    
    # 3. Deep Model
    num_features = X_test.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    deep_model = CNN_BiLSTM(num_features).to(device)
    deep_model.load_state_dict(torch.load('models/cybertwin_cnn_bilstm_best.pth'))
    deep_model.eval()
    
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        deep_out = deep_model(X_test_tensor)
        deep_prob = torch.sigmoid(deep_out).cpu().numpy().flatten()
        deep_pred = (deep_prob >= 0.5).astype(int)
        
    deep_metrics = calc_advanced_metrics(y_test, deep_pred, deep_prob)
    deep_metrics['model'] = '1D-CNN + BiLSTM'
    deep_train_history = pd.read_csv('results/metrics/deep_model_training_history.csv')
    deep_metrics['training_time_seconds'] = deep_train_history['duration'].sum()
    
    # Save final comparison
    final_df = pd.DataFrame([lr_metrics, rf_metrics, deep_metrics])
    final_df = final_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 
                         'specificity', 'false_positive_rate', 'false_negative_rate', 'training_time_seconds']]
    final_df.to_csv('results/tables/final_model_comparison.csv', index=False)
    
    # Confusion Matrices
    def plot_cm(y_t, y_p, title, filename):
        plt.figure(figsize=(6,5))
        cm = confusion_matrix(y_t, y_p)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'PortScan'], yticklabels=['BENIGN', 'PortScan'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
    plot_cm(y_test, lr_pred, 'Logistic Regression CM', 'results/figures/confusion_matrix_logistic_regression.png')
    plot_cm(y_test, rf_pred, 'Random Forest CM', 'results/figures/confusion_matrix_random_forest.png')
    plot_cm(y_test, deep_pred, 'CNN+BiLSTM CM', 'results/figures/confusion_matrix_cnn_bilstm.png')
    
    # ROC Curves
    plt.figure(figsize=(8,6))
    for name, prob in [('Logistic Regression', lr_prob), ('Random Forest', rf_prob), ('CNN+BiLSTM', deep_prob)]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f'{name}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('ROC Curve Comparison')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('results/figures/roc_curve_comparison.png')
    plt.close()
    
    # PR Curves
    plt.figure(figsize=(8,6))
    for name, prob in [('Logistic Regression', lr_prob), ('Random Forest', rf_prob), ('CNN+BiLSTM', deep_prob)]:
        precision, recall, _ = precision_recall_curve(y_test, prob)
        plt.plot(recall, precision, label=f'{name}')
    plt.title('Precision-Recall Curve Comparison')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('results/figures/pr_curve_comparison.png')
    plt.close()
    
    # Performance comparison bar chart
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    plot_df = final_df.melt(id_vars='model', value_vars=metrics_to_plot, var_name='metric', value_name='score')
    plt.figure(figsize=(12,6))
    sns.barplot(data=plot_df, x='metric', y='score', hue='model')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('results/figures/model_performance_comparison.png')
    plt.close()
    
    # Error Analysis for Deep Model
    errors = y_test != deep_pred
    error_indices = np.where(errors)[0]
    error_types = ['False Positive' if p == 1 else 'False Negative' for p in deep_pred[errors]]
    
    error_df = pd.DataFrame({
        'sample_index': error_indices,
        'true_label': y_test.iloc[error_indices].values,
        'predicted_label': deep_pred[error_indices],
        'predicted_probability': deep_prob[error_indices],
        'error_type': error_types
    })
    error_df.to_csv('results/tables/error_analysis.csv', index=False)
    
    # Feature Importance (RF)
    rf_importances = pd.DataFrame({
        'feature': X_test.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    rf_importances.to_csv('results/tables/random_forest_feature_importance.csv', index=False)
    
    plt.figure(figsize=(10,8))
    sns.barplot(data=rf_importances.head(20), x='importance', y='feature', palette='viridis')
    plt.title('Top 20 Random Forest Feature Importances')
    plt.tight_layout()
    plt.savefig('results/figures/random_forest_feature_importance.png')
    plt.close()
    
    # Feature Coefficients (LR)
    lr_coefs = pd.DataFrame({
        'feature': X_test.columns,
        'coefficient': lr_model.coef_[0],
        'abs_coef': np.abs(lr_model.coef_[0])
    }).sort_values('abs_coef', ascending=False)
    lr_coefs.to_csv('results/tables/logistic_regression_coefficients.csv', index=False)
    
    plt.figure(figsize=(10,8))
    sns.barplot(data=lr_coefs.head(20), x='coefficient', y='feature', palette='coolwarm')
    plt.title('Top 20 Logistic Regression Coefficients (Magnitude)')
    plt.tight_layout()
    plt.savefig('results/figures/logistic_regression_feature_coefficients.png')
    plt.close()
    
    # Deep Model Curves
    plt.figure(figsize=(8,5))
    plt.plot(deep_train_history['epoch'], deep_train_history['train_loss'], label='Train Loss')
    plt.plot(deep_train_history['epoch'], deep_train_history['val_loss'], label='Val Loss')
    plt.title('Deep Model Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/figures/deep_model_loss_curve.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    plt.plot(deep_train_history['epoch'], deep_train_history['train_accuracy'], label='Train Acc')
    plt.plot(deep_train_history['epoch'], deep_train_history['val_accuracy'], label='Val Acc')
    plt.title('Deep Model Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/figures/deep_model_accuracy_curve.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    plt.plot(deep_train_history['epoch'], deep_train_history['val_f1'], label='Val F1', color='green')
    plt.title('Deep Model F1 Curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig('results/figures/deep_model_f1_curve.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    plt.plot(deep_train_history['epoch'], deep_train_history['learning_rate'], label='LR', color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig('results/figures/learning_rate_curve.png')
    plt.close()

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
