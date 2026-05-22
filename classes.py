import torch.nn as nn
from torchvision.models import resnet18


class Encoder(nn.Module):
    """ResNet-18 backbone adapted for 1-channel 28×28 MNIST patches.

    Changes vs stock ResNet-18:
      - conv1: 7×7 stride-2 → 3×3 stride-1  (keeps spatial resolution on tiny images)
      - maxpool replaced with Identity         (avoids collapsing 28px input too early)
      - fc replaced with a linear projection   (maps 512-d pool → embed_dim)
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = resnet18()
        backbone.conv1   = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc      = nn.Linear(512, embed_dim)
        self.net = backbone

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """MLP that maps a context embedding to a predicted target embedding."""
    def __init__(self, embed_dim=256, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
    def forward(self, x):
        return self.net(x)
