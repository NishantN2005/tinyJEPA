# tiny_jepa.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import copy

from classes import Encoder, Predictor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def split_context_target(images):
    """
    Split each 28x28 image into a context half (top) and target half (bottom).
    Returns (context, target), each shape (B, 1, 14, 28).
    """
    context = images[:, :, :14, :]
    target = images[:, :, 14:, :]
    return context, target


@torch.no_grad()
def update_target_encoder(encoder, target_encoder, momentum=0.996):
    for p_main, p_target in zip(encoder.parameters(), target_encoder.parameters()):
        p_target.data.mul_(momentum).add_(p_main.data, alpha=1 - momentum)


def sigreg(z, num_projections=64, beta=1.0, lam=1.0):
    """Sketched Isotropic Gaussian Regularizer (LeCun et al., arXiv 2603.19312).

    Projects embeddings onto random unit-norm directions, then applies the
    Epps-Pulley test statistic to each 1-D projection.  The statistic T → 0
    when the projected distribution is N(0,1) and T > 0 otherwise, so
    minimising it forces the full embedding distribution toward N(0, I).

    Closed-form Epps-Pulley statistic (standardised samples Y, bandwidth β):
      T = (1/n)·Σ_{j,k} exp(-β²/2·(Y_j-Y_k)²)
          - (2/√(1+β²))·Σ_j exp(-β²·Y_j²/(2(1+β²)))
          + n/√(1+2β²)
    """
    B, D = z.shape
    b2 = beta ** 2

    W = F.normalize(torch.randn(D, num_projections, device=z.device), dim=0)
    y = z @ W                                          # (B, M)

    diff_sq   = (y.unsqueeze(0) - y.unsqueeze(1)).pow(2)          # (B, B, M)
    pair_term = torch.exp(-b2 / 2 * diff_sq).sum(dim=[0, 1]) / B  # (M,)
    indiv_term = (2 / (1 + b2) ** 0.5) * \
                 torch.exp(-b2 / (2 * (1 + b2)) * y.pow(2)).sum(0) # (M,)
    const_term = B / (1 + 2 * b2) ** 0.5

    T = pair_term - indiv_term + const_term            # (M,), → 0 under N(0,1)
    return lam * T.mean()


def train():
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(train_set, batch_size=128, shuffle=True)

    encoder = Encoder().to(device)
    predictor = Predictor().to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3
    )

    for epoch in range(15):
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            context, target = split_context_target(images)

            # Encode context (with gradients) and target (no gradients)
            ctx_embed = encoder(context)
            pred_embed = predictor(ctx_embed)
            with torch.no_grad():
                tgt_embed = target_encoder(target)

            mse  = F.mse_loss(pred_embed, tgt_embed)
            reg  = sigreg(ctx_embed, lam=0.1)
            loss = mse + reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_target_encoder(encoder, target_encoder)

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    embed_std = ctx_embed.std(dim=0).mean().item()
                print(f"Epoch {epoch} Batch {batch_idx} | MSE: {mse.item():.4f} | SIGReg: {reg.item():.4f} | Embed std: {embed_std:.4f}")

    return encoder

if __name__ == "__main__":
    encoder = train()
    torch.save(encoder.state_dict(), "tiny_jepa_encoder.pt")
    print("Saved encoder to tiny_jepa_encoder.pt")