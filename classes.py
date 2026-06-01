import torch
import torch.nn as nn
from torchvision.models import resnet18

PATCH_SIZE = 7   # each patch is 7×7 pixels
GRID_SIZE  = 4   # 4×4 grid → 16 patches cover the full 28×28 image


class Encoder(nn.Module):
    """ResNet-18 backbone with three output modes.

    global_embed(x)   — image → (B, embed_dim) global vector
    patch_embeds(x)   — image → (B, 16, embed_dim) per-patch vectors
    embed_context(x)  — single forward pass returning both of the above
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
        return x                                               # (B, 512, 4, 4)

    def embed_context(self, x):
        """Single pass → global (B, D) and all patch embeddings (B, 16, D)."""
        feat   = self._features(x)                            # (B, 512, 4, 4)
        B      = feat.shape[0]
        global_emb = self.global_proj(self.global_pool(feat).flatten(1))
        patch_emb  = self.patch_proj(
            feat.permute(0, 2, 3, 1).reshape(B, GRID_SIZE * GRID_SIZE, 512)
        )
        return global_emb, patch_emb                          # (B,D), (B,16,D)

    def global_embed(self, x):
        feat   = self._features(x)
        return self.global_proj(self.global_pool(feat).flatten(1))

    def patch_embeds(self, x):
        feat = self._features(x)
        B    = feat.shape[0]
        feat = feat.permute(0, 2, 3, 1).reshape(B, GRID_SIZE * GRID_SIZE, 512)
        return self.patch_proj(feat)

    def forward(self, x):
        return self.global_embed(x)


class Predictor(nn.Module):
    """Transformer predictor conditioned on per-patch context embeddings.

    Context tokens  = context patch embeddings + positional embeddings
    Target tokens   = positional embeddings only (no content — these are the queries)

    Both are concatenated and passed through a small transformer.
    The target-position outputs are projected to produce predicted embeddings.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super().__init__()
        self.pos_embed = nn.Embedding(GRID_SIZE * GRID_SIZE, embed_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, ctx_embeds, ctx_indices, tgt_indices):
        """
        ctx_embeds:  (B, N_ctx, embed_dim)
        ctx_indices: (B, N_ctx) long  — positions of visible patches
        tgt_indices: (B, N_tgt) long  — positions of target patches
        Returns:     (B, N_tgt, embed_dim)
        """
        ctx_tokens = ctx_embeds + self.pos_embed(ctx_indices)  # (B, N_ctx, D)
        tgt_tokens = self.pos_embed(tgt_indices)               # (B, N_tgt, D)
        tokens     = torch.cat([ctx_tokens, tgt_tokens], dim=1)  # (B, N_ctx+N_tgt, D)
        out        = self.transformer(tokens)
        return self.proj(out[:, ctx_tokens.shape[1]:, :])      # (B, N_tgt, D)
