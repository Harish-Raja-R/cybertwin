import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt

from src.temporal_simulator import load_and_simulate
from src.temporal_dataset import get_dataloaders
from src.models.temporal_bilstm import TemporalBiLSTM, TemporalCNNBiLSTM

def train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5, device='cuda'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    best_model_state = None
    early_stop_counter = 0
    
    history = []
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_train_preds.extend(preds.cpu().numpy())
            all_train_targets.extend(batch_y.cpu().numpy())
            
        train_loss = np.mean(train_losses)
        train_f1 = f1_score(all_train_targets, all_train_preds, zero_division=0)
        
        # Validation
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_losses.append(loss.item())
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(batch_y.cpu().numpy())
                
        val_loss = np.mean(val_losses)
        val_f1 = f1_score(all_val_targets, all_val_preds, zero_division=0)
        val_acc = accuracy_score(all_val_targets, all_val_preds)
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'val_acc': val_acc,
            'learning_rate': current_lr,
            'epoch_time': epoch_time
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s - "
              f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
              f"Train F1: {train_f1:.4f} - Val F1: {val_f1:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            torch.save(best_model_state, 'results/temporal_model_best.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Load best model
    model.load_state_dict(torch.load('results/temporal_model_best.pth', weights_only=True))
    
    # Save history
    pd.DataFrame(history).to_csv('results/metrics/temporal_model_training_history.csv', index=False)
    
    return model, pd.DataFrame(history)

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(batch_y.numpy())
            
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
    
    metrics = {
        'Accuracy': accuracy_score(all_targets, all_preds),
        'Precision': precision_score(all_targets, all_preds, zero_division=0),
        'Recall': recall_score(all_targets, all_preds, zero_division=0),
        'F1': f1_score(all_targets, all_preds, zero_division=0),
        'ROC-AUC': roc_auc_score(all_targets, all_probs),
        'PR-AUC': average_precision_score(all_targets, all_probs),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0
    }
    
    return metrics, all_probs, all_preds, all_targets

def plot_history(history_df, output_dir='results/figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Temporal Model Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/temporal_loss_curve.png')
    plt.close()
    
    # F1 curve
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_f1'], label='Train F1')
    plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Temporal Model F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/temporal_f1_curve.png')
    plt.close()
    
    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Temporal Model Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/temporal_accuracy_curve.png')
    plt.close()

if __name__ == '__main__':
    os.makedirs('results/metrics', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading simulated temporal sequences (window_size=20)...")
    train_data, val_data, test_data, manifest = load_and_simulate(window_size=20, strategy='any')
    
    train_loader, val_loader, test_loader = get_dataloaders(
        train_data[0], train_data[1],
        val_data[0], val_data[1],
        test_data[0], test_data[1],
        batch_size=128
    )
    
    print("Initializing Temporal BiLSTM...")
    model = TemporalBiLSTM(input_dim=70, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    
    print("Training Temporal BiLSTM...")
    model, history = train_model(model, train_loader, val_loader, epochs=30, lr=1e-3, patience=5, device=device)
    
    plot_history(history)
    
    print("Evaluating Temporal BiLSTM on test set...")
    metrics, probs, preds, targets = evaluate_model(model, test_loader, device=device)
    
    print("\nTest Metrics (Temporal Sequence-Level):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    metrics['Model'] = 'Milestone 4 BiLSTM (Sequence-Level)'
    metrics_df = pd.DataFrame([metrics])
    
    # Append to comparison if exists
    comp_file = 'results/tables/cybertwin_model_comparison.csv'
    if os.path.exists(comp_file):
        try:
            existing = pd.read_csv(comp_file)
            metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
        except Exception as e:
            print("Could not merge with existing comparison table.")
    metrics_df.to_csv(comp_file, index=False)
    print(f"Saved evaluation metrics to {comp_file}")
    
    # Save test predictions for error analysis
    test_results = pd.DataFrame({
        'true_label': targets,
        'pred_prob': probs,
        'pred_label': preds
    })
    test_results.to_csv('results/tables/temporal_test_predictions.csv', index=False)
