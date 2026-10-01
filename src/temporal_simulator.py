import numpy as np
import pandas as pd
import json
import os
from sklearn.preprocessing import RobustScaler

class TemporalSimulator:
    def __init__(self, window_size=20, sequence_label_strategy='any'):
        self.window_size = window_size
        self.sequence_label_strategy = sequence_label_strategy
        
    def create_sequences(self, X, y):
        """
        Create fixed-length sequences from tabular data using a sliding window.
        Stride = window_size // 2 to ensure overlap.
        """
        n_samples = len(X)
        stride = max(1, self.window_size // 2)
        
        sequences = []
        labels = []
        
        for i in range(0, n_samples - self.window_size + 1, stride):
            window_X = X[i:i + self.window_size]
            window_y = y[i:i + self.window_size]
            
            sequences.append(window_X)
            
            if self.sequence_label_strategy == 'any':
                seq_label = 1 if np.any(window_y == 1) else 0
            elif self.sequence_label_strategy == 'majority':
                seq_label = 1 if np.mean(window_y) >= 0.5 else 0
            elif self.sequence_label_strategy == 'last':
                seq_label = window_y[-1]
            else:
                raise ValueError(f"Unknown strategy {self.sequence_label_strategy}")
                
            labels.append(seq_label)
            
        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)

def load_and_simulate(data_dir='data/processed', window_size=20, strategy='any', output_dir='results/tables'):
    """
    Loads train/val/test, generates sequences, and saves manifest.
    """
    X_train = pd.read_csv(f"{data_dir}/X_train.csv").values
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").values.ravel()
    
    X_val = pd.read_csv(f"{data_dir}/X_val.csv").values
    y_val = pd.read_csv(f"{data_dir}/y_val.csv").values.ravel()
    
    X_test = pd.read_csv(f"{data_dir}/X_test.csv").values
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").values.ravel()
    
    simulator = TemporalSimulator(window_size, strategy)
    
    seq_X_train, seq_y_train = simulator.create_sequences(X_train, y_train)
    seq_X_val, seq_y_val = simulator.create_sequences(X_val, y_val)
    seq_X_test, seq_y_test = simulator.create_sequences(X_test, y_test)
    
    os.makedirs(output_dir, exist_ok=True)
    manifest = {
        'sequence_length': window_size,
        'feature_count': X_train.shape[1],
        'train_sequences': len(seq_X_train),
        'validation_sequences': len(seq_X_val),
        'test_sequences': len(seq_X_test),
        'sequence_label_definition': strategy,
        'random_seed': 42,
        'simulation_method': 'Sliding window over random stratified splits (synthetic chronology)',
        'stride': max(1, window_size // 2)
    }
    
    with open(f"{output_dir}/temporal_dataset_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=4)
        
    return (seq_X_train, seq_y_train), (seq_X_val, seq_y_val), (seq_X_test, seq_y_test), manifest

if __name__ == '__main__':
    train_data, val_data, test_data, manifest = load_and_simulate()
    print("Simulated dataset manifest:")
    print(json.dumps(manifest, indent=2))
