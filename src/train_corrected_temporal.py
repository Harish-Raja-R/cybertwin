import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import pandas as pd
import numpy as np
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.temporal_bilstm import TemporalBiLSTM
from src.temporal_dataset import TemporalDataset
from src.corrected_temporal_simulator import load_and_simulate_corrected

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except:
        pr_auc = 0.5
        
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'roc_auc': roc_auc, 'pr_auc': pr_auc,
        'specificity': specificity, 'fpr': fpr, 'fnr': fnr,
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }

def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_data, val_data, test_data, manifest = load_and_simulate_corrected(window_size=20)
    
    train_loader = DataLoader(TemporalDataset(train_data[0], train_data[1]), batch_size=128, shuffle=True)
    val_loader = DataLoader(TemporalDataset(val_data[0], val_data[1]), batch_size=128, shuffle=False)
    test_loader = DataLoader(TemporalDataset(test_data[0], test_data[1]), batch_size=128, shuffle=False)
    
    model = TemporalBiLSTM(input_dim=70, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = 30
    patience = 5
    best_val_f1 = -1
    epochs_no_improve = 0
    
    history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets, train_probs = [], [], []
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_probs.extend(probs)
            train_preds.extend((probs >= 0.5).astype(int))
            train_targets.extend(batch_y.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_metrics = compute_metrics(train_targets, train_preds, train_probs)
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets, val_probs = [], [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend((probs >= 0.5).astype(int))
                val_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_metrics = compute_metrics(val_targets, val_preds, val_probs)
        
        epoch_time = time.time() - start_time
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_acc': train_metrics['accuracy'], 'val_acc': val_metrics['accuracy'],
            'train_f1': train_metrics['f1'], 'val_f1': val_metrics['f1'],
            'lr': optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s - "
              f"Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Train F1: {train_metrics['f1']:.4f} - Val F1: {val_metrics['f1']:.4f}")
              
        scheduler.step(val_metrics['f1'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'results/milestone_4_1/temporal_model_best.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    pd.DataFrame(history).to_csv('results/milestone_4_1/metrics/temporal_training_history.csv', index=False)
    
    # Test Evaluation
    model.load_state_dict(torch.load('results/milestone_4_1/temporal_model_best.pth', weights_only=True))
    model.eval()
    
    test_preds, test_targets, test_probs = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend((probs >= 0.5).astype(int))
            test_targets.extend(batch_y.cpu().numpy())
            
    test_metrics = compute_metrics(test_targets, test_preds, test_probs)
    print("\nTest Metrics (Corrected Balanced Dataset):")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
    pd.DataFrame([test_metrics]).to_csv('results/milestone_4_1/tables/corrected_model_comparison.csv', index=False)
    
    # Save Confusion Matrix Figure & CSV
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1])
    pd.DataFrame(cm, index=['True Benign', 'True Attack'], columns=['Pred Benign', 'Pred Attack']).to_csv('results/milestone_4_1/tables/temporal_confusion_matrix.csv')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix (Corrected Temporal BiLSTM)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('results/milestone_4_1/figures/temporal_confusion_matrix.png')
    plt.close()
    
    # Evaluate by Scenario
    audit_test = pd.read_csv('results/milestone_4_1/tables/sequence_source_audit.csv')
    audit_test = audit_test[audit_test['source_partition'] == 'TEST']
    
    scenario_metrics = []
    scenarios = ['NORMAL', 'ATTACK_ONSET', 'SUSTAINED_ATTACK', 'ATTACK_RECOVERY']
    
    # We must match indices. Since test_data[0] was built sequentially based on audit log,
    # the predictions are exactly 1-to-1 with the test rows in the audit log.
    audit_test['pred'] = test_preds
    audit_test['prob'] = test_probs
    audit_test.to_csv('results/milestone_4_1/tables/test_predictions.csv', index=False)
    
    for scen in scenarios:
        sub = audit_test[audit_test['scenario'] == scen]
        if len(sub) == 0:
            continue
        acc = accuracy_score(sub['label'], sub['pred'])
        avg_prob = np.mean(sub['prob'])
        scenario_metrics.append({
            'scenario': scen,
            'count': len(sub),
            'true_label': sub['label'].iloc[0],
            'accuracy': acc,
            'mean_predicted_prob': avg_prob
        })
        
    pd.DataFrame(scenario_metrics).to_csv('results/milestone_4_1/tables/scenario_performance.csv', index=False)
    print("\nScenario Performance saved to results/milestone_4_1/tables/scenario_performance.csv")
    print(pd.DataFrame(scenario_metrics))

if __name__ == "__main__":
    train_and_evaluate()
