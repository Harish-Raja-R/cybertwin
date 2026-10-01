import torch
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from src.models.temporal_bilstm import TemporalBiLSTM

def temporal_explainability(model, test_loader, feature_names, device='cuda', output_dir='results/figures'):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    # We will use DeepExplainer or GradientExplainer on a small subset
    # because SHAP on LSTMs can be very slow.
    
    background_data = []
    test_data = []
    
    # Collect a small background and test set for SHAP
    with torch.no_grad():
        for i, (batch_X, _) in enumerate(test_loader):
            if i == 0:
                background_data.append(batch_X[:50]) # 50 samples for background
                test_data.append(batch_X[50:100])    # 50 samples for testing
                break
                
    background = torch.cat(background_data).to(device)
    test_set = torch.cat(test_data).to(device)
    
    print(f"Running GradientExplainer with background shape {background.shape} and test shape {test_set.shape}")
    
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(test_set)
        
        # shap_values is typically a list or an array. For binary classification with BCEWithLogitsLoss,
        # shape is (num_test_samples, seq_len, num_features).
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        print(f"SHAP values computed. Shape: {np.array(shap_values).shape}")
        
        # We want to aggregate the SHAP values over the sequence length to get feature importance
        # Average absolute SHAP value across the sequence length and samples
        mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 1))
        
        # Save feature importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        importance_df.to_csv('results/tables/shap_feature_importance.csv', index=False)
        print("Saved SHAP feature importance to results/tables/shap_feature_importance.csv")
        
        # Plot Top 20 features
        top_20 = importance_df.head(20)
        plt.figure(figsize=(12, 8))
        plt.barh(top_20['feature'][::-1], top_20['mean_abs_shap'][::-1], color='dodgerblue')
        plt.xlabel('Mean |SHAP Value| (Aggregated over time)')
        plt.title('Top 20 Features Driving Temporal CyberTwin Predictions')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_temporal_bar.png')
        plt.close()
        
    except Exception as e:
        print(f"SHAP Explainer failed: {e}")
        print("Note: SHAP support for complex PyTorch RNNs is sometimes unreliable. Documenting limitation.")
        with open('results/tables/shap_feature_importance.csv', 'w') as f:
            f.write("feature,mean_abs_shap\n")
            f.write("LIMITATION,0\n")

if __name__ == '__main__':
    from src.temporal_dataset import get_dataloaders
    from src.temporal_simulator import load_and_simulate
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_data, val_data, test_data, manifest = load_and_simulate(window_size=20, strategy='any')
    
    _, _, test_loader = get_dataloaders(
        train_data[0], train_data[1],
        val_data[0], val_data[1],
        test_data[0], test_data[1],
        batch_size=128
    )
    
    model = TemporalBiLSTM(input_dim=70, hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load('results/temporal_model_best.pth', weights_only=True))
    
    # Load original features to get names
    sample_df = pd.read_csv('data/processed/X_train.csv', nrows=1)
    feature_names = sample_df.columns.tolist()
        
    temporal_explainability(model, test_loader, feature_names, device=device)
