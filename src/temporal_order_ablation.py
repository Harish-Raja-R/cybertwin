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

def train_model(train_data, val_data, test_data, experiment_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device} for {experiment_name}")
    
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
    
    train_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds, val_targets, val_probs = [], [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_preds.extend((probs >= 0.5).astype(int))
                val_targets.extend(batch_y.cpu().numpy())
                
        val_metrics = compute_metrics(val_targets, val_preds, val_probs)
        scheduler.step(val_metrics['f1'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), f'results/milestone_4_1/temporal_model_best_{experiment_name}.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    train_time = time.time() - train_start
    
    model.load_state_dict(torch.load(f'results/milestone_4_1/temporal_model_best_{experiment_name}.pth', weights_only=True))
    model.eval()
    
    test_preds, test_targets, test_probs = [], [], []
    infer_start = time.time()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend((probs >= 0.5).astype(int))
            test_targets.extend(batch_y.cpu().numpy())
    infer_time = (time.time() - infer_start) * 1000 # ms
            
    test_metrics = compute_metrics(test_targets, test_preds, test_probs)
    test_metrics['training_time_sec'] = train_time
    test_metrics['inference_time_ms'] = infer_time
    test_metrics['experiment'] = experiment_name
    
    cm = confusion_matrix(test_targets, test_preds, labels=[0, 1])
    
    return test_metrics, cm, test_preds, test_probs

def shuffle_sequences(X, seed=42):
    rng = np.random.default_rng(seed)
    X_shuffled = np.zeros_like(X)
    
    changed_count = 0
    total_positions_changed = 0
    
    for i in range(X.shape[0]):
        seq = X[i]
        seq_len = seq.shape[0]
        permutation = rng.permutation(seq_len)
        X_shuffled[i] = seq[permutation]
        
        positions_changed = np.sum(permutation != np.arange(seq_len))
        total_positions_changed += positions_changed
        if positions_changed > 0:
            changed_count += 1
            
    return X_shuffled, changed_count, total_positions_changed

def main():
    print("Loading Original Sequences...")
    train_data, val_data, test_data, manifest = load_and_simulate_corrected(window_size=20)
    
    print("Shuffling Sequences...")
    X_train_shuffled, train_changed, train_pos = shuffle_sequences(train_data[0])
    X_val_shuffled, val_changed, val_pos = shuffle_sequences(val_data[0])
    X_test_shuffled, test_changed, test_pos = shuffle_sequences(test_data[0])
    
    integrity_records = [
        {'split': 'Train', 'num_sequences': train_data[0].shape[0], 'sequence_length': train_data[0].shape[1], 
         'changed_sequences': train_changed, 'unchanged_sequences': train_data[0].shape[0] - train_changed, 
         'mean_positions_changed': train_pos / train_data[0].shape[0] if train_data[0].shape[0] > 0 else 0},
        {'split': 'Val', 'num_sequences': val_data[0].shape[0], 'sequence_length': val_data[0].shape[1], 
         'changed_sequences': val_changed, 'unchanged_sequences': val_data[0].shape[0] - val_changed, 
         'mean_positions_changed': val_pos / val_data[0].shape[0] if val_data[0].shape[0] > 0 else 0},
        {'split': 'Test', 'num_sequences': test_data[0].shape[0], 'sequence_length': test_data[0].shape[1], 
         'changed_sequences': test_changed, 'unchanged_sequences': test_data[0].shape[0] - test_changed, 
         'mean_positions_changed': test_pos / test_data[0].shape[0] if test_data[0].shape[0] > 0 else 0}
    ]
    pd.DataFrame(integrity_records).to_csv('results/milestone_4_1/tables/order_shuffle_integrity.csv', index=False)
    print("Saved Integrity Records.")
    
    train_data_shuffled = (X_train_shuffled, train_data[1])
    val_data_shuffled = (X_val_shuffled, val_data[1])
    test_data_shuffled = (X_test_shuffled, test_data[1])
    
    # Train Ordered
    ordered_metrics, ordered_cm, ordered_preds, ordered_probs = train_model(train_data, val_data, test_data, 'Ordered')
    
    # Train Shuffled
    shuffled_metrics, shuffled_cm, shuffled_preds, shuffled_probs = train_model(train_data_shuffled, val_data_shuffled, test_data_shuffled, 'Shuffled')
    
    # Save Confusion Matrices
    pd.DataFrame(ordered_cm, index=['True Benign', 'True Attack'], columns=['Pred Benign', 'Pred Attack']).to_csv('results/milestone_4_1/tables/confusion_ordered.csv')
    pd.DataFrame(shuffled_cm, index=['True Benign', 'True Attack'], columns=['Pred Benign', 'Pred Attack']).to_csv('results/milestone_4_1/tables/confusion_shuffled.csv')
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(ordered_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix (Ordered)')
    plt.subplot(1, 2, 2)
    sns.heatmap(shuffled_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
    plt.title('Confusion Matrix (Shuffled)')
    plt.tight_layout()
    plt.savefig('results/milestone_4_1/figures/confusion_ordered_vs_shuffled.png')
    plt.close()
    
    # Save Metrics
    ordered_metrics['ordering'] = 'Ordered'
    shuffled_metrics['ordering'] = 'Order-Shuffled'
    
    results_df = pd.DataFrame([ordered_metrics, shuffled_metrics])
    cols = ['experiment', 'ordering', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'specificity', 'fpr', 'fnr', 'training_time_sec', 'inference_time_ms']
    results_df = results_df[cols]
    results_df.to_csv('results/milestone_4_1/tables/temporal_order_ablation.csv', index=False)
    print("Saved Ablation Metrics.")
    
    # Delta
    delta = {}
    for col in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'specificity', 'fpr', 'fnr']:
        delta[f'delta_{col}'] = ordered_metrics[col] - shuffled_metrics[col]
    pd.DataFrame([delta]).to_csv('results/milestone_4_1/tables/temporal_order_ablation_delta.csv', index=False)
    
    # Plot Metrics
    metrics_to_plot = ['accuracy', 'f1', 'roc_auc', 'pr_auc', 'specificity']
    plot_data = pd.melt(results_df, id_vars=['ordering'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_data, x='Metric', y='Score', hue='ordering')
    plt.title('Temporal Order Ablation: Ordered vs Shuffled Metrics')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig('results/milestone_4_1/figures/temporal_order_ablation_metrics.png')
    plt.close()
    
    # Scenario Specific Ablation
    audit_test = pd.read_csv('results/milestone_4_1/tables/sequence_source_audit.csv')
    audit_test = audit_test[audit_test['source_partition'] == 'TEST'].reset_index(drop=True)
    
    scenarios = ['NORMAL', 'ATTACK_ONSET', 'SUSTAINED_ATTACK', 'ATTACK_RECOVERY']
    scenario_metrics = []
    
    audit_test['pred_ordered'] = ordered_preds
    audit_test['prob_ordered'] = ordered_probs
    audit_test['pred_shuffled'] = shuffled_preds
    audit_test['prob_shuffled'] = shuffled_probs
    
    for scen in scenarios:
        sub = audit_test[audit_test['scenario'] == scen]
        if len(sub) == 0:
            continue
        acc_ordered = accuracy_score(sub['label'], sub['pred_ordered'])
        acc_shuffled = accuracy_score(sub['label'], sub['pred_shuffled'])
        
        scenario_metrics.append({
            'scenario': scen,
            'ordering': 'Ordered',
            'accuracy': acc_ordered,
            'mean_predicted_prob': np.mean(sub['prob_ordered'])
        })
        scenario_metrics.append({
            'scenario': scen,
            'ordering': 'Shuffled',
            'accuracy': acc_shuffled,
            'mean_predicted_prob': np.mean(sub['prob_shuffled'])
        })
        
    pd.DataFrame(scenario_metrics).to_csv('results/milestone_4_1/tables/scenario_order_ablation.csv', index=False)
    
    # Example Visualization
    sample_idx = 0
    # Find an onset sequence
    onset_idx = audit_test[audit_test['scenario'] == 'ATTACK_ONSET'].index[0]
    
    orig_seq = test_data[0][onset_idx]
    shuf_seq = test_data_shuffled[0][onset_idx]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    sns.heatmap(orig_seq.T, cmap='viridis', cbar=False)
    plt.title('Original Sequence (Attack Onset)')
    plt.ylabel('Features')
    
    plt.subplot(2, 1, 2)
    sns.heatmap(shuf_seq.T, cmap='viridis', cbar=False)
    plt.title('Shuffled Sequence (Attack Onset)')
    plt.xlabel('Time Step')
    plt.ylabel('Features')
    
    plt.tight_layout()
    plt.savefig('results/milestone_4_1/figures/temporal_order_example.png')
    plt.close()

if __name__ == "__main__":
    main()
