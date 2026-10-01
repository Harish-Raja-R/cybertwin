import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.corrected_temporal_simulator import load_and_simulate_corrected

def create_deployment_artifact():
    ROOT = Path(__file__).resolve().parents[1]
    
    # We run the simulation builder to get the 'TEST' partition
    # which Streamlit uses for demoing.
    train_data, val_data, test_data, manifest = load_and_simulate_corrected(
        data_dir=str(ROOT / 'data' / 'processed'),
        window_size=20,
        output_dir=str(ROOT / 'results' / 'milestone_4_1' / 'tables')
    )
    
    X_seq_test, y_seq_test = test_data
    
    # Also grab a small snippet of X_val for the "Sample CIC-IDS2017 data" mode in Streamlit
    x_val = pd.read_csv(ROOT / 'data' / 'processed' / 'X_val.csv', nrows=100)
    y_val = pd.read_csv(ROOT / 'data' / 'processed' / 'y_val.csv', nrows=100)
    
    deploy_dir = ROOT / 'results' / 'deployment'
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the deployment artifacts
    np.savez_compressed(
        deploy_dir / 'temporal_simulation_data.npz',
        X_seq_test=X_seq_test,
        y_seq_test=y_seq_test,
        X_val_sample=x_val.values,
        y_val_sample=y_val.values
    )
    
    # Save the audit test manifest which is needed to know the scenario types
    audit_df = pd.read_csv(ROOT / 'results' / 'milestone_4_1' / 'tables' / 'sequence_source_audit.csv')
    audit_test = audit_df[audit_df['source_partition'] == 'TEST'].reset_index(drop=True)
    audit_test.to_csv(deploy_dir / 'audit_test_manifest.csv', index=False)
    
    print(f"Deployment artifacts created in {deploy_dir}")

if __name__ == '__main__':
    create_deployment_artifact()
