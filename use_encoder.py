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