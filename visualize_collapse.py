"""
Side-by-side t-SNE comparison of two SIGReg runs:
  Left  — predictor hidden=256, 15 epochs
  Right — predictor hidden=512, 30 epochs
Saves to assets/day1/predictor_comparison.png
"""
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE

from classes import Encoder, Predictor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

COLORS = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def split_context_target(images):
    return images[:, :, :14, :], images[:, :, 14:, :]


@torch.no_grad()
def ema_update(enc, target, momentum=0.996):
    for p, pt in zip(enc.parameters(), target.parameters()):
        pt.data.mul_(momentum).add_(p.data, alpha=1 - momentum)


def sigreg(z, num_projections=64, beta=1.0, lam=1.0):
    B, D = z.shape
    b2 = beta ** 2
    W = F.normalize(torch.randn(D, num_projections, device=z.device), dim=0)
    y = z @ W
    diff_sq    = (y.unsqueeze(0) - y.unsqueeze(1)).pow(2)
    pair_term  = torch.exp(-b2 / 2 * diff_sq).sum(dim=[0, 1]) / B
    indiv_term = (2 / (1 + b2) ** 0.5) * \
                 torch.exp(-b2 / (2 * (1 + b2)) * y.pow(2)).sum(0)
    const_term = B / (1 + 2 * b2) ** 0.5
    T = pair_term - indiv_term + const_term
    return lam * T.mean()


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------

def train(predictor_hidden: int, epochs: int):
    loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=128, shuffle=True,
    )

    encoder        = Encoder().to(device)
    predictor      = Predictor(hidden=predictor_hidden).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
    )

    label = f"hidden={predictor_hidden}"
    for epoch in range(epochs):
        for images, _ in loader:
            images = images.to(device)
            ctx, tgt = split_context_target(images)

            ctx_embed  = encoder(ctx)
            pred_embed = predictor(ctx_embed)
            with torch.no_grad():
                tgt_embed = target_encoder(tgt)

            loss = F.mse_loss(pred_embed, tgt_embed) + sigreg(ctx_embed, lam=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            ema_update(encoder, target_encoder)

        print(f"  [{label}] epoch {epoch + 1}/{epochs} done")

    return encoder


# ---------------------------------------------------------------------------
# linear probe
# ---------------------------------------------------------------------------

def linear_probe(encoder):
    import torch.nn as nn
    encoder.eval()
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True,  transform=transforms.ToTensor()),
        batch_size=256, shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=256,
    )

    clf = nn.Linear(128, 10).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for _ in range(5):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                emb = encoder(images[:, :, :14, :])
            loss = nn.functional.cross_entropy(clf(emb), labels)
            opt.zero_grad(); loss.backward(); opt.step()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = clf(encoder(images[:, :, :14, :])).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_embeddings(encoder, n=2000):
    loader = DataLoader(
        datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=256,
    )
    embeds, labels = [], []
    encoder.eval()
    for images, lbls in loader:
        embeds.append(encoder(images[:, :, :14, :].to(device)).cpu())
        labels.append(lbls)
        if sum(len(e) for e in embeds) >= n:
            break
    return torch.cat(embeds)[:n].numpy(), torch.cat(labels)[:n].numpy()


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def tsne_scatter(ax, embeds, labels, title, accuracy):
    proj = TSNE(n_components=2, perplexity=40, random_state=0).fit_transform(embeds)

    for cls in range(10):
        mask = labels == cls
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=COLORS[cls], s=8, alpha=0.7, linewidths=0)

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=10)
    ax.set_xticks([]); ax.set_yticks([])

    std_val = embeds.std(axis=0).mean()
    for text, (x, va) in zip(
        [f"embed std = {std_val:.3f}", f"linear probe = {accuracy:.1%}"],
        [(0.03, "top"), (0.03, "bottom")],
    ):
        t = ax.text(x, 0.97 if va == "top" else 0.03, text,
                    transform=ax.transAxes, fontsize=10,
                    va=va, ha="left", color="white", fontweight="bold")
        t.set_path_effects([pe.withStroke(linewidth=2, foreground="black")])


def main():
    runs = [
        (256,  15),
        (512,  30),
        (1024, 40),
    ]

    encoders, accuracies = [], []
    for hidden, epochs in runs:
        print(f"\n=== Training: predictor hidden={hidden}, {epochs} epochs ===")
        enc = train(predictor_hidden=hidden, epochs=epochs)
        print("  Running linear probe …")
        acc = linear_probe(enc)
        print(f"  Accuracy: {acc:.1%}")
        encoders.append(enc)
        accuracies.append(acc)

    print("\nCollecting embeddings …")
    embeddings = [collect_embeddings(enc) for enc in encoders]

    print("Running t-SNE …")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor="#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")

    for ax, (emb, lbl), (hidden, epochs), acc in zip(axes, embeddings, runs, accuracies):
        tsne_scatter(ax, emb, lbl,
                     f"Predictor  hidden={hidden}  ·  {epochs} epochs", acc)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COLORS[i], markersize=8, label=str(i))
        for i in range(10)
    ]
    fig.legend(handles=handles, title="Digit", ncol=10,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               framealpha=0.2, labelcolor="white",
               title_fontsize=10, fontsize=9)
    fig.suptitle(
        "tinyJEPA  ·  SIGReg  ·  Effect of predictor size & training length",
        fontsize=15, fontweight="bold", color="white", y=1.01,
    )
    plt.tight_layout()

    out = "assets/day1/predictor_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
