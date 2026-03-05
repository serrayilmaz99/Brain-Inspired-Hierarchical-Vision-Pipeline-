import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch.optim as optim
import matplotlib.pyplot as plt


from flat_model import *


class CIFAROnly(Dataset):
    def __init__(self, train=True):
        self.data = datasets.CIFAR10(root='./data', train=train, download=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return self.transform(image), label




batch_size = 64
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 dataset (only RGB image and label used)
train_dataset_flat = CIFAROnly(train=True)
test_dataset_flat = CIFAROnly(train=False)
train_loader = DataLoader(train_dataset_flat, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset_flat, batch_size=batch_size, shuffle=False)

# Flat CNN model
model_flat = FlatCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_flat.parameters(), lr=1e-3)

train_losses = []
test_accuracies = []


# Training loop
for epoch in range(num_epochs):
    model_flat.train()
    running_loss = 0.0

    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model_flat(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"[FlatCNN] Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model_flat.eval()
    total_correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_flat(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    accuracy = total_correct / len(test_dataset_flat)
    test_accuracies.append(accuracy)
    print(f"FlatCNN Test Accuracy: {accuracy * 100:.2f}%")



plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("FlatCNN Training Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), [a * 100 for a in test_accuracies], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("FlatCNN Test Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()