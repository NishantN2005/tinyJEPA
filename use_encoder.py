import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from classes import Encoder

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("tiny_jepa_encoder.pt"))
encoder.eval()

test_set = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())
image, label = test_set[0]
image = image.unsqueeze(0).to(device)  # add batch dim
context = image[:, :, :14, :]  # top half — what the encoder was trained on
with torch.no_grad():
    embedding = encoder(context)
print(f"Label: {label}, Embedding shape: {embedding.shape}, First few values: {embedding[0, :5]}")

def linear_probe(encoder):
    encoder.eval()
    train_set = datasets.MNIST("./data", train=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256)

    classifier = nn.Linear(128, 10).to(device)
    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    for epoch in range(5):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            context = images[:, :, :14, :]
            with torch.no_grad():
                emb = encoder(context)
            logits = classifier(emb)
            loss = nn.functional.cross_entropy(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()

    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            context = images[:, :, :14, :]
            pred = classifier(encoder(context)).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"Linear probe test accuracy (top half only): {correct/total:.4f}")
    return classifier

classifier = linear_probe(encoder)