import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy
import random

from classes import Encoder, Predictor, PATCH_SIZE, GRID_SIZE

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

N_PATCHES = GRID_SIZE * GRID_SIZE   # 16
N_CONTEXT = 10                       # patches kept visible
N_TARGET  = 4                        # patches to predict each step


# ---------------------------------------------------------------------------
# patch helpers
# ---------------------------------------------------------------------------

def sample_patches():
    """Randomly sample non-overlapping context and target patch indices."""
    idx = list(range(N_PATCHES))
    random.shuffle(idx)
    return idx[:N_CONTEXT], idx[N_CONTEXT:N_CONTEXT + N_TARGET]


def mask_image(images, target_indices):
    """Zero out target patches in a batch of images.
    images:         (B, 1, 28, 28)
    target_indices: list of int patch indices to blank
    """
    masked = images.clone()
    for i in target_indices:
        r, c = divmod(i, GRID_SIZE)
        masked[:, :, r*PATCH_SIZE:(r+1)*PATCH_SIZE,
                     c*PATCH_SIZE:(c+1)*PATCH_SIZE] = 0.0
    return masked


# ---------------------------------------------------------------------------
# SIGReg
# ---------------------------------------------------------------------------

def sigreg(z, num_projections=64, beta=1.0, lam=1.0):
    """Sketched Isotropic Gaussian Regularizer (LeCun et al., arXiv 2603.19312).

    Projects embeddings onto random unit-norm directions, then applies the
    Epps-Pulley test statistic to each 1-D projection.  T → 0 under N(0,I).

      T = (1/n)·Σ_{j,k} exp(-β²/2·(Y_j-Y_k)²)
          - (2/√(1+β²))·Σ_j exp(-β²·Y_j²/(2(1+β²)))
          + n/√(1+2β²)
    """
    B, D = z.shape
    b2 = beta ** 2
    W  = F.normalize(torch.randn(D, num_projections, device=z.device), dim=0)
    y  = z @ W                                                   # (B, M)

    diff_sq    = (y.unsqueeze(0) - y.unsqueeze(1)).pow(2)        # (B, B, M)
    pair_term  = torch.exp(-b2 / 2 * diff_sq).sum(dim=[0,1]) / B
    indiv_term = (2 / (1 + b2) ** 0.5) * \
                 torch.exp(-b2 / (2*(1+b2)) * y.pow(2)).sum(0)
    const_term = B / (1 + 2*b2) ** 0.5
    # Normalise by B so the statistic is batch-size invariant (T → 0 under N(0,1))
    T = (pair_term - indiv_term + const_term) / B

    return lam * T.mean()


# ---------------------------------------------------------------------------
# EMA update
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_target_encoder(encoder, target_encoder, momentum=0.996):
    for p, pt in zip(encoder.parameters(), target_encoder.parameters()):
        pt.data.mul_(momentum).add_(p.data, alpha=1 - momentum)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train():
    loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=512, shuffle=True,
    )
    val_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=512, shuffle=False,
    )

    encoder        = Encoder(embed_dim=256).to(device)
    predictor      = Predictor(embed_dim=256, num_heads=4, num_layers=2).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=3e-4,
    )
    warmup    = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=5
    )
    cosine    = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=95, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[5]
    )

    for epoch in range(100):
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)                           # (B, 1, 28, 28)
            B = images.shape[0]

            # Sample a new random context/target split each step
            ctx_idx, tgt_idx = sample_patches()
            ctx_tensor = torch.tensor(ctx_idx, device=device).unsqueeze(0).expand(B, -1)
            tgt_tensor = torch.tensor(tgt_idx, device=device).unsqueeze(0).expand(B, -1)

            # Mask target patches before encoding context
            masked = mask_image(images, tgt_idx)

            # Online encoder: global embed (for SIGReg) + per-patch context embeds
            ctx_global, all_ctx_embeds = encoder.embed_context(masked)
            ctx_patch_embeds = all_ctx_embeds[:, ctx_idx, :]     # (B, N_ctx, 256)

            # Transformer predictor: attends over context patches to predict targets
            pred_embeds = predictor(ctx_patch_embeds, ctx_tensor, tgt_tensor)  # (B, N_tgt, 256)

            # Target encoder: per-patch embeddings from full image
            with torch.no_grad():
                tgt_embeds = target_encoder.patch_embeds(images)[:, tgt_idx, :]  # (B, N_tgt, 256)

            mse  = F.mse_loss(pred_embeds, tgt_embeds)
            reg  = sigreg(ctx_global, num_projections=16, lam=0.1)
            loss = mse + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
            )
            optimizer.step()
            update_target_encoder(encoder, target_encoder)

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    embed_std = ctx_global.std(dim=0).mean().item()
                print(f"Epoch {epoch} Batch {batch_idx} | "
                      f"MSE: {mse.item():.4f} | "
                      f"SIGReg: {reg.item():.4f} | "
                      f"Embed std: {embed_std:.4f}", flush=True)

        # Validation MSE at end of each epoch (no mode switch — preserves batch norm state)
        val_mse_total, val_batches = 0.0, 0
        with torch.no_grad():
            for val_images, _ in val_loader:
                val_images = val_images.to(device)
                Bv = val_images.shape[0]
                ctx_idx_v, tgt_idx_v = sample_patches()
                ctx_tensor_v = torch.tensor(ctx_idx_v, device=device).unsqueeze(0).expand(Bv, -1)
                tgt_tensor_v = torch.tensor(tgt_idx_v, device=device).unsqueeze(0).expand(Bv, -1)
                masked_v = mask_image(val_images, tgt_idx_v)
                _, all_ctx_v = encoder.embed_context(masked_v)
                ctx_patch_v = all_ctx_v[:, ctx_idx_v, :]
                pred_v = predictor(ctx_patch_v, ctx_tensor_v, tgt_tensor_v)
                tgt_v  = target_encoder.patch_embeds(val_images)[:, tgt_idx_v, :]
                val_mse_total += F.mse_loss(pred_v, tgt_v).item()
                val_batches += 1
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch} | Val MSE: {val_mse_total / val_batches:.4f} | LR: {lr:.2e}", flush=True)
        scheduler.step()

    return encoder


if __name__ == "__main__":
    encoder = train()
    torch.save(encoder.state_dict(), "tiny_jepa_encoder.pt")
    print("Saved encoder to tiny_jepa_encoder.pt")
