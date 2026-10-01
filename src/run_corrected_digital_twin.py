import pandas as pd
import numpy as np
import json
import torch
import shap
import joblib
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.models.temporal_bilstm import TemporalBiLSTM
from src.temporal_dataset import TemporalDataset
from torch.utils.data import DataLoader

def run_shap_random_forest():
    print("Running SHAP on Random Forest...")
    # Load RF and X_test
    try:
        rf_model = joblib.load('models/random_forest.joblib')
        X_test_df = pd.read_csv('data/processed/X_test.csv')
        
        # Take a subset to be fast (e.g. 1000 rows)
        X_sample = X_test_df.sample(n=1000, random_state=42)
        
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_sample)
        
        os.makedirs('results/milestone_4_1/figures', exist_ok=True)
        os.makedirs('results/milestone_4_1/tables', exist_ok=True)
        
        # In newer shap versions for binary classification, shap_values might be a list or Explanation
        if hasattr(shap_values, 'values'): # Explanation object
            sv = shap_values.values
            if sv.ndim == 3: # (n_samples, n_features, n_classes)
                sv = sv[:, :, 1]
        elif isinstance(shap_values, list):
            sv = shap_values[1] # Class 1
        else:
            sv = shap_values
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, show=False)
        plt.tight_layout()
        plt.savefig('results/milestone_4_1/figures/shap_rf_summary.png', dpi=150)
        plt.close()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig('results/milestone_4_1/figures/shap_rf_bar.png', dpi=150)
        plt.close()
        
        # Save exact values (mean absolute shap value per feature)
        mean_shap = np.abs(sv).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': X_sample.columns,
            'mean_absolute_shap': mean_shap
        }).sort_values('mean_absolute_shap', ascending=False)
        
        shap_df.to_csv('results/milestone_4_1/tables/shap_rf_importance.csv', index=False)
        print("SHAP analysis completed.")
    except Exception as e:
        print(f"Error during SHAP generation: {e}")

def compare_models():
    print("Generating Model Comparisons...")
    
    # Let's generate old vs corrected
    comparison_df = pd.DataFrame([
        {
            'Experiment': 'Initial M4',
            'Sequence Construction': 'Random 20-flow',
            'Labeling': 'Any attack',
            'Main Result': 'F1=0.9978, Specificity=0.0',
            'Interpretation': 'Invalid/biased due to mathematical near-certainty of attack flows.'
        },
        {
            'Experiment': 'Corrected M4.1',
            'Sequence Construction': 'Controlled scenarios',
            'Labeling': 'Scenario state',
            'Main Result': 'Actual Metrics (See corrected_model_comparison.csv)',
            'Interpretation': 'Primary temporal experiment with balanced, structured scenarios.'
        }
    ])
    comparison_df.to_csv('results/milestone_4_1/tables/old_vs_corrected_temporal.csv', index=False)

def simulate_digital_twin_log():
    # Load corrected test predictions
    audit_test = pd.read_csv('results/milestone_4_1/tables/test_predictions.csv')
    
    # We will simulate risk scores over time. 
    # State transitions: 
    # NORMAL -> SUSPICIOUS (prob > 0.5) -> UNDER_ATTACK (prob > 0.8) -> RECOVERING (prob < 0.3) -> NORMAL
    
    log = []
    
    # Group by scenario to show transitions sequentially (Normal -> Onset -> Sustained -> Recovery)
    scenarios_ordered = ['NORMAL', 'ATTACK_ONSET', 'SUSTAINED_ATTACK', 'ATTACK_RECOVERY']
    current_state = 'NORMAL'
    risk_score = 0.0
    
    step = 0
    for scen in scenarios_ordered:
        sub = audit_test[audit_test['scenario'] == scen]
        for _, row in sub.iterrows():
            prob = row['prob']
            true_scen = row['scenario']
            seq_id = row['sequence_id']
            
            # Risk formula: alpha * prob + beta * past_risk
            risk_score = 0.7 * prob + 0.3 * risk_score
            
            # State machine logic
            if current_state == 'NORMAL':
                if prob > 0.8:
                    current_state = 'UNDER_ATTACK'
                elif prob > 0.5:
                    current_state = 'SUSPICIOUS'
            elif current_state == 'SUSPICIOUS':
                if prob > 0.8:
                    current_state = 'UNDER_ATTACK'
                elif prob < 0.3:
                    current_state = 'NORMAL'
            elif current_state == 'UNDER_ATTACK':
                if prob < 0.5:
                    current_state = 'RECOVERING'
            elif current_state == 'RECOVERING':
                if prob > 0.7:
                    current_state = 'UNDER_ATTACK'
                elif prob < 0.2:
                    current_state = 'NORMAL'
                    
            log.append({
                'simulation_step': step,
                'sequence_id': seq_id,
                'scenario': true_scen,
                'predicted_probability': prob,
                'predicted_state': current_state,
                'risk_score': risk_score,
                'true_scenario': true_scen
            })
            step += 1
            
    log_df = pd.DataFrame(log)
    log_df.to_csv('results/milestone_4_1/tables/digital_twin_state_log.csv', index=False)
    print("Digital Twin log generated.")

if __name__ == "__main__":
    run_shap_random_forest()
    compare_models()
    simulate_digital_twin_log()
