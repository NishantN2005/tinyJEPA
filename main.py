# tiny_jepa.py
import torch
import torch.nn as nn
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

    for epoch in range(5):
        for batch_idx, (images, _) in enumerate(loader):
            images = images.to(device)
            context, target = split_context_target(images)

            # Encode context (with gradients) and target (no gradients)
            ctx_embed = encoder(context)
            pred_embed = predictor(ctx_embed)
            with torch.no_grad():
                tgt_embed = target_encoder(target)

            loss = F.mse_loss(pred_embed, tgt_embed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_target_encoder(encoder, target_encoder)

            if batch_idx % 100 == 0:
                # Track embedding variance — collapse detector!
                with torch.no_grad():
                    embed_std = ctx_embed.std(dim=0).mean().item()
                print(f"Epoch {epoch} Batch {batch_idx} | Loss: {loss.item():.4f} | Embed std: {embed_std:.4f}")

    return encoder

if __name__ == "__main__":
    encoder = train()
    torch.save(encoder.state_dict(), "tiny_jepa_encoder.pt")
    print("Saved encoder to tiny_jepa_encoder.pt")