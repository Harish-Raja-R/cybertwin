import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CyberTwinDataset(Dataset):
    def __init__(self, X, y):
        # Expected X to be numpy array or pandas DataFrame
        # Shape: (N, features)
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        self.X = torch.tensor(X, dtype=torch.float32)
        # Reshape X for CNN: (N, channels=1, features)
        self.X = self.X.unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNN_BiLSTM(nn.Module):
    def __init__(self, num_features):
        super(CNN_BiLSTM, self).__init__()
        
        # 1D Convolution block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 1D Convolution block 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate sequence length after pooling
        # Formula: L_out = floor((L_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
        # Or simpler for MaxPool: L_out = L_in // 2
        seq_len = num_features // 2 // 2
        
        # BiLSTM
        # Input to LSTM should be (batch, seq, features)
        # Our conv output is (batch, channels, seq)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
                            
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected
        self.fc = nn.Linear(64 * 2 * seq_len, 1) # * 2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch, 1, features)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Reshape for LSTM: (batch, seq, channels)
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        return x
