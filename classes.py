import torch
import torch.nn as nn
from torchvision.models import resnet18

PATCH_SIZE = 7   # each patch is 7×7 pixels
GRID_SIZE  = 4   # 4×4 grid → 16 patches cover the full 28×28 image


class Encoder(nn.Module):
    """ResNet-18 backbone with two output modes.

    global_embed(x)  — masked image → (B, embed_dim) context vector
    patch_embeds(x)  — full image   → (B, 16, embed_dim) per-patch vectors

    The 4×4 spatial feature map produced by layer4 aligns perfectly with
    the 4×4 patch grid (each feature position covers one 7×7 patch).
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        backbone = resnet18()
        backbone.conv1   = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

        self.stem   = nn.Sequential(backbone.conv1, backbone.bn1,
                                    backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # 28×28 × 64
        self.layer2 = backbone.layer2   # 14×14 × 128
        self.layer3 = backbone.layer3   #  7×7  × 256
        self.layer4 = backbone.layer4   #  4×4  × 512

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(512, embed_dim)
        self.patch_proj  = nn.Linear(512, embed_dim)

    def _features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x                                          # (B, 512, 4, 4)

    def global_embed(self, x):
        """Masked image → (B, embed_dim) context embedding."""
        feat   = self._features(x)
        pooled = self.global_pool(feat).flatten(1)        # (B, 512)
        return self.global_proj(pooled)                   # (B, embed_dim)

    def patch_embeds(self, x):
        """Full image → (B, 16, embed_dim) per-patch embeddings."""
        feat = self._features(x)                          # (B, 512, 4, 4)
        B    = feat.shape[0]
        feat = feat.permute(0, 2, 3, 1).reshape(B, GRID_SIZE * GRID_SIZE, 512)
        return self.patch_proj(feat)                      # (B, 16, embed_dim)

    def forward(self, x):
        return self.global_embed(x)


class Predictor(nn.Module):
    """MLP conditioned on context embedding + learnable positional query.

    For each target patch the predictor receives:
        [ctx_embed ‖ pos_embed[patch_idx]]  (2 × embed_dim → embed_dim)
    """
    def __init__(self, embed_dim=256, hidden=1024):
        super().__init__()
        self.pos_embed = nn.Embedding(GRID_SIZE * GRID_SIZE, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, ctx_embed, patch_indices):
        """
        ctx_embed:     (B, embed_dim)
        patch_indices: (B, N_tgt)  — long tensor of target patch indices
        Returns:       (B, N_tgt, embed_dim)
        """
        N   = patch_indices.shape[1]
        pos = self.pos_embed(patch_indices)                    # (B, N, embed_dim)
        ctx = ctx_embed.unsqueeze(1).expand(-1, N, -1)         # (B, N, embed_dim)
        return self.net(torch.cat([ctx, pos], dim=-1))         # (B, N, embed_dim)
