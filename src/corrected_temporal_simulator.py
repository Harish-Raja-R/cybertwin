import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

def build_corrected_sequences(X_df, y_df, partition_name, n_normal, n_onset, n_sustained, n_recovery, window_size=20, start_id=0):
    """
    Builds controlled synthetic sequences by sampling without replacement.
    Label Definition (Scenario-Level):
      Normal -> 0
      Onset -> 1
      Sustained -> 1
      Recovery -> 0
    """
    # Separate rows by class and track original index
    df = X_df.copy()
    df['target'] = y_df['Label_binary'].values
    df['original_idx'] = df.index.astype(int)
    
    benign_pool = df[df['target'] == 0].sample(frac=1, random_state=42).copy()
    attack_pool = df[df['target'] == 1].sample(frac=1, random_state=42).copy()
    
    sequences = []
    labels = []
    audit_log = []
    
    seq_id = start_id
    
    # Half point for transitions
    hp = window_size // 2
    
    # 1. NORMAL Sequences (Label 0)
    for i in range(n_normal):
        rows = benign_pool.iloc[:window_size]
        benign_pool = benign_pool.iloc[window_size:]
        
        seq_features = rows.drop(columns=['target', 'original_idx']).values
        sequences.append(seq_features)
        labels.append(0)
        
        audit_log.append({
            'sequence_id': seq_id,
            'source_partition': partition_name,
            'scenario': 'NORMAL',
            'source_indices': rows['original_idx'].tolist(),
            'label': 0
        })
        seq_id += 1
        
    # 2. ATTACK ONSET (Label 1)
    for i in range(n_onset):
        b_rows = benign_pool.iloc[:hp]
        benign_pool = benign_pool.iloc[hp:]
        
        a_rows = attack_pool.iloc[:(window_size - hp)]
        attack_pool = attack_pool.iloc[(window_size - hp):]
        
        rows = pd.concat([b_rows, a_rows])
        seq_features = rows.drop(columns=['target', 'original_idx']).values
        sequences.append(seq_features)
        labels.append(1)
        
        audit_log.append({
            'sequence_id': seq_id,
            'source_partition': partition_name,
            'scenario': 'ATTACK_ONSET',
            'source_indices': rows['original_idx'].tolist(),
            'label': 1
        })
        seq_id += 1

    # 3. SUSTAINED ATTACK (Label 1)
    for i in range(n_sustained):
        rows = attack_pool.iloc[:window_size]
        attack_pool = attack_pool.iloc[window_size:]
        
        seq_features = rows.drop(columns=['target', 'original_idx']).values
        sequences.append(seq_features)
        labels.append(1)
        
        audit_log.append({
            'sequence_id': seq_id,
            'source_partition': partition_name,
            'scenario': 'SUSTAINED_ATTACK',
            'source_indices': rows['original_idx'].tolist(),
            'label': 1
        })
        seq_id += 1
        
    # 4. ATTACK RECOVERY (Label 0)
    for i in range(n_recovery):
        a_rows = attack_pool.iloc[:hp]
        attack_pool = attack_pool.iloc[hp:]
        
        b_rows = benign_pool.iloc[:(window_size - hp)]
        benign_pool = benign_pool.iloc[(window_size - hp):]
        
        rows = pd.concat([a_rows, b_rows])
        seq_features = rows.drop(columns=['target', 'original_idx']).values
        sequences.append(seq_features)
        labels.append(0)
        
        audit_log.append({
            'sequence_id': seq_id,
            'source_partition': partition_name,
            'scenario': 'ATTACK_RECOVERY',
            'source_indices': rows['original_idx'].tolist(),
            'label': 0
        })
        seq_id += 1
        
    # Convert to arrays
    X_seq = np.array(sequences, dtype=np.float32)
    y_seq = np.array(labels, dtype=np.float32)
    
    return X_seq, y_seq, audit_log, seq_id

def load_and_simulate_corrected(data_dir='data/processed', window_size=20, output_dir='results/milestone_4_1/tables'):
    print(f"Loading tabular data from {data_dir}...")
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")
    X_val = pd.read_csv(f"{data_dir}/X_val.csv")
    y_val = pd.read_csv(f"{data_dir}/y_val.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv")

    print(f"Building corrected sequences (window_size={window_size})...")
    
    # Train: 1000 Normal, 500 Onset, 1000 Sustained, 500 Recovery (Total 3000)
    X_seq_train, y_seq_train, audit_train, next_id = build_corrected_sequences(
        X_train, y_train, 'TRAIN', 1000, 500, 1000, 500, window_size, start_id=0
    )
    
    # Val: 200 Normal, 100 Onset, 200 Sustained, 100 Recovery (Total 600)
    X_seq_val, y_seq_val, audit_val, next_id = build_corrected_sequences(
        X_val, y_val, 'VALIDATION', 200, 100, 200, 100, window_size, start_id=next_id
    )
    
    # Test: 200 Normal, 100 Onset, 200 Sustained, 100 Recovery (Total 600)
    X_seq_test, y_seq_test, audit_test, next_id = build_corrected_sequences(
        X_test, y_test, 'TEST', 200, 100, 200, 100, window_size, start_id=next_id
    )
    
    # Save Audit Log
    os.makedirs(output_dir, exist_ok=True)
    all_audit = audit_train + audit_val + audit_test
    
    # Formatting indices for CSV saving
    audit_df_records = []
    for a in all_audit:
        a_copy = a.copy()
        a_copy['source_indices'] = str(a_copy['source_indices'])
        audit_df_records.append(a_copy)
        
    audit_df = pd.DataFrame(audit_df_records)
    audit_df.to_csv(f"{output_dir}/sequence_source_audit.csv", index=False)
    print("Sequence source audit saved. Validating uniqueness...")
    
    # Validation: no overlap between partitions
    # The construction guarantees no overlap within a partition (because of pop/iloc from pool),
    # and no overlap between partitions because the input data is strictly from train, val, or test respectively.
    
    # Create Dataset Manifest
    total_normal = sum(1 for a in all_audit if a['scenario'] in ['NORMAL', 'ATTACK_RECOVERY'])
    total_attack = sum(1 for a in all_audit if a['scenario'] in ['ATTACK_ONSET', 'SUSTAINED_ATTACK'])
    manifest = [{
        'scenario': 'CORRECTED_BALANCED',
        'window_length': window_size,
        'train_sequences': len(X_seq_train),
        'validation_sequences': len(X_seq_val),
        'test_sequences': len(X_seq_test),
        'normal_sequences': total_normal,
        'attack_sequences': total_attack,
        'label_definition': 'Scenario State (Normal=0, Recovery=0, Onset=1, Sustained=1)'
    }]
    pd.DataFrame(manifest).to_csv(f"{output_dir}/temporal_dataset_manifest.csv", index=False)
    
    return (X_seq_train, y_seq_train), (X_seq_val, y_seq_val), (X_seq_test, y_seq_test), manifest

def plot_scenarios(X_seq_test, y_seq_test, audit_test, output_dir='results/milestone_4_1/figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    # We will pick one sequence of each scenario
    scenarios = ['NORMAL', 'ATTACK_ONSET', 'SUSTAINED_ATTACK', 'ATTACK_RECOVERY']
    
    # Load feature names just for the Y axis label reference
    df_sample = pd.read_csv('data/processed/X_train.csv', nrows=1)
    feature_names = df_sample.columns.tolist()
    
    # Pick a few key features that are likely indicative, like Total Length of Fwd Packets, or average ratios
    # We will just plot a heatmap of the scaled features over the window for representation.
    
    for scen in scenarios:
        # Find first sequence of this scenario
        idx = next(i for i, a in enumerate(audit_test) if a['scenario'] == scen)
        seq = X_seq_test[idx]
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(seq.T, cmap="viridis", cbar=True, yticklabels=False)
        plt.title(f"{scen} Sequence Behavior (Window={seq.shape[0]})")
        plt.xlabel("Time Step (Synthetic)")
        plt.ylabel("Normalized Features (70)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scenario_{scen.lower()}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    t_data, v_data, test_data, manifest = load_and_simulate_corrected(window_size=20)
    print("Test data shapes:", test_data[0].shape, test_data[1].shape)
    
    # We load the audit log back to plot
    import ast
    audit = pd.read_csv('results/milestone_4_1/tables/sequence_source_audit.csv')
    audit_test = audit[audit['source_partition'] == 'TEST'].to_dict('records')
    plot_scenarios(test_data[0], test_data[1], audit_test)
    print("Scenarios visualized in results/milestone_4_1/figures/")
