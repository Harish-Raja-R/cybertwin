import torch
import torch.nn as nn

class TemporalBiLSTM(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=128, num_layers=2, dropout=0.3):
        super(TemporalBiLSTM, self).__init__()
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # We use the last hidden state of both directions
        # Output dim = hidden_dim * 2 (because bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        projected = self.projection(x)
        
        # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        # hidden[0] shape: (num_layers * 2, batch_size, hidden_dim)
        lstm_out, (hidden, cell) = self.bilstm(projected)
        
        # Get the last hidden state from the last layer (both forward and backward)
        # forward: hidden[-2], backward: hidden[-1]
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # predictions
        out = self.fc(last_hidden)
        
        return out.squeeze(1)

# Optional CNN + BiLSTM
class TemporalCNNBiLSTM(nn.Module):
    def __init__(self, input_dim=70, hidden_dim=128, num_layers=1, dropout=0.3):
        super(TemporalCNNBiLSTM, self).__init__()
        
        # 1D Temporal Convolution (takes features as channels)
        # Expected input to Conv1d: (batch_size, channels, seq_len)
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Swap axes for Conv1d -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Swap axes back for LSTM -> (batch_size, seq_len', hidden_dim)
        x = x.transpose(1, 2)
        
        lstm_out, (hidden, cell) = self.bilstm(x)
        
        last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        out = self.fc(last_hidden)
        return out.squeeze(1)
