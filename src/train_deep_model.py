import pandas as pd
import numpy as np
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deep_model import CyberTwinDataset, CNN_BiLSTM

def train_deep_model():
    print("Loading data for Deep Model...")
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')['Label_binary']
    X_val = pd.read_csv('data/processed/X_val.csv')
    y_val = pd.read_csv('data/processed/y_val.csv')['Label_binary']
    
    # Calculate positive weight based on TRAIN SET ONLY
    pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
    print(f"Calculated pos_weight: {pos_weight:.4f}")
    
    num_features = X_train.shape[1]
    
    train_dataset = CyberTwinDataset(X_train, y_train)
    val_dataset = CyberTwinDataset(X_val, y_val)
    
    # CPU specific optimizations: larger batch sizes
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = CNN_BiLSTM(num_features=num_features).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    epochs = 30
    patience = 5
    best_f1 = -1
    epochs_no_improve = 0
    
    history = []
    
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            preds = torch.sigmoid(outputs) >= 0.5
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(y_batch.cpu().numpy())
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_true, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.sigmoid(outputs) >= 0.5
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_true, val_preds)
        val_prec = precision_score(val_true, val_preds, zero_division=0)
        val_rec = recall_score(val_true, val_preds, zero_division=0)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        
        epoch_dur = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'val_loss': val_loss,
            'train_accuracy': train_acc, 'val_accuracy': val_acc,
            'val_precision': val_prec, 'val_recall': val_rec,
            'val_f1': val_f1, 'learning_rate': current_lr, 'duration': epoch_dur
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        scheduler.step(val_f1)
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'models/cybertwin_cnn_bilstm_best.pth')
            # Save metadata
            meta = {
                'architecture': '1D-CNN + BiLSTM',
                'features': num_features,
                'class_mapping': {'0': 'BENIGN', '1': 'PortScan'},
                'preprocessing': 'RobustScaler',
                'best_val_f1': best_f1,
                'seed': 42
            }
            with open('models/cybertwin_model_metadata.json', 'w') as f:
                json.dump(meta, f)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
    # Save history
    pd.DataFrame(history).to_csv('results/metrics/deep_model_training_history.csv', index=False)
    
    # Plotting is handled in evaluate script or notebook to avoid GUI issues here
    print("Deep learning training complete.")

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    train_deep_model()
