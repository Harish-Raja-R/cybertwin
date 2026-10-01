import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TemporalDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        Args:
            sequences (np.ndarray): Shape (num_sequences, window_size, num_features)
            labels (np.ndarray): Shape (num_sequences,)
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def get_dataloaders(seq_X_train, seq_y_train, seq_X_val, seq_y_val, seq_X_test, seq_y_test, batch_size=128):
    train_dataset = TemporalDataset(seq_X_train, seq_y_train)
    val_dataset = TemporalDataset(seq_X_val, seq_y_val)
    test_dataset = TemporalDataset(seq_X_test, seq_y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
