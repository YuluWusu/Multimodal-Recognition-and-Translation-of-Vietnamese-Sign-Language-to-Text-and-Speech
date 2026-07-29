import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, num_joints=76, coords=3, hidden_size=256, num_classes=400):
        super(BiLSTM, self).__init__()
        input_size = num_joints * coords
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        
        if len(x.shape) == 4:
            batch_size, seq_len, joints, coords = x.shape
            x = x.view(batch_size, seq_len, joints * coords)
        elif len(x.shape) == 3:
            
            pass
        
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        return self.fc(out)
