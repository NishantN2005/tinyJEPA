import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from classes import Encoder

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

encoder = Encoder(embed_dim=256).to(device)
encoder.load_state_dict(torch.load("tiny_jepa_encoder.pt"))
encoder.eval()

# Quick sanity check — full image at inference (no masking)
test_set = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())
image, label = test_set[0]
image = image.unsqueeze(0).to(device)
with torch.no_grad():
    embedding = encoder(image)
print(f"Label: {label}, Embedding shape: {embedding.shape}, "
      f"First few values: {embedding[0, :5]}")


def linear_probe(encoder):
    encoder.eval()
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True,  transform=transforms.ToTensor()),
        batch_size=256, shuffle=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, transform=transforms.ToTensor()),
        batch_size=256,
    )

    classifier = nn.Linear(256, 10).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    for _ in range(5):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                emb = encoder(images)          # full image, no masking
            loss = nn.functional.cross_entropy(classifier(emb), labels)
            opt.zero_grad(); loss.backward(); opt.step()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            pred = classifier(encoder(images)).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
    print(f"Linear probe test accuracy (full image): {correct/total:.4f}")
    return classifier


classifier = linear_probe(encoder)
