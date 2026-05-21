import torch.nn as nn

class Encoder(nn.Module):
    """Small ConvNet that maps a (1, 28, 28) image patch to a 128-dim embedding."""
    def __init__(self, embed_dim=128, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 2, embed_dim),
        )
    def forward(self, x):
        return self.net(x)
    
class Predictor(nn.Module):
    """MLP that maps a context embedding to a predicted target embedding."""
    def __init__(self, embed_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
    def forward(self, x):
        return self.net(x)