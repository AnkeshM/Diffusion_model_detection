import torch
import torch.nn as nn

class DetectorNet(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for detecting AI-generated images
    using FFT-based features (147 dimensions).
    """
    def __init__(self, input_dim=147, hidden_dim=128):
        super(DetectorNet, self).__init__()
        self.net = nn.Sequential(
            # Input normalization — critical for FFT features with varying scales
            nn.BatchNorm1d(input_dim),

            # Layer 1 (147 -> 128)
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2 (128 -> 64)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3 (64 -> 32)
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output Layer (32 -> 1)
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, x):
        return self.net(x)
