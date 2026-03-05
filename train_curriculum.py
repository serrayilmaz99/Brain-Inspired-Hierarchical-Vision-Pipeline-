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

from curriculum_learner import *




class CIFARWithEdges(Dataset):
    def __init__(self, train=True):
        self.cifar = datasets.CIFAR10(root="./data", train=train, download=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        image, label = self.cifar[idx]
        image_tensor = self.transform(image)

        gray = np.array(image.convert("L"))

        # Edge map using Canny
        edge = cv2.Canny(gray, 100, 200)
        edge = torch.tensor(edge / 255.0, dtype=torch.float32).unsqueeze(0)

        # Corner map using Harris
        gray_float = np.float32(gray)
        harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
        harris = cv2.dilate(harris, None)
        corner = (harris > 0.01 * harris.max()).astype(np.float32)
        corner = torch.tensor(corner, dtype=torch.float32).unsqueeze(0)

        # Contour map using binary threshold + findContours
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)
        contour = torch.tensor(contour_mask / 255.0, dtype=torch.float32).unsqueeze(0)

        # Saliency map approximation using edge+corner weighted sum
        saliency = np.clip(edge.squeeze().numpy() * 0.5 + corner.squeeze().numpy() * 0.5, 0, 1)
        saliency = torch.tensor(saliency, dtype=torch.float32).unsqueeze(0)

        return image_tensor, edge, corner, contour, saliency, label



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
train_dataset = CIFARWithEdges(train=True)
test_dataset = CIFARWithEdges(train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



num_epochs = 20
num_epochs_2 = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. EdgeNet
edgenet = EdgeNet().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(edgenet.parameters(), lr=1e-3)

for epoch in range(num_epochs_2):
    edgenet.train()
    for images, edge_gt, _, _, _, _ in train_loader:
        images, edge_gt = images.to(device), edge_gt.to(device)
        _, edge_pred = edgenet(images)
        loss = criterion(edge_pred, edge_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[EdgeNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 2. CornerNet
cornernet = CornerNet().to(device)
optimizer = optim.Adam(cornernet.parameters(), lr=1e-3)

for epoch in range(num_epochs_2):
    cornernet.train()
    for images, _, corner_gt, _, _, _ in train_loader:
        images, corner_gt = images.to(device), corner_gt.to(device)
        with torch.no_grad():
            _, edge_pred = edgenet(images)
        edge_input = torch.cat([images, edge_pred], dim=1)
        _, corner_pred = cornernet(edge_input)
        loss = criterion(corner_pred, corner_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[CornerNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 3. ContourNet
contournet = ContourNet().to(device)
optimizer = optim.Adam(contournet.parameters(), lr=1e-3)

for epoch in range(num_epochs_2):
    contournet.train()
    for images, _, corner_gt, contour_gt, _, _ in train_loader:
        images, contour_gt = images.to(device), contour_gt.to(device)
        with torch.no_grad():
            _, edge_pred = edgenet(images)
            _, corner_pred = cornernet(torch.cat([images, edge_pred], dim=1))
        contour_input = torch.cat([edge_pred, corner_pred], dim=1)
        _, contour_pred = contournet(contour_input)
        loss = criterion(contour_pred, contour_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[ContourNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4. SaliencyNet
saliencynet = SaliencyNet().to(device)
optimizer = optim.Adam(saliencynet.parameters(), lr=1e-3)

for epoch in range(num_epochs_2):
    saliencynet.train()
    for images, _, corner_gt, contour_gt, saliency_gt, _ in train_loader:
        images, saliency_gt = images.to(device), saliency_gt.to(device)
        with torch.no_grad():
            _, edge_pred = edgenet(images)
            _, corner_pred = cornernet(torch.cat([images, edge_pred], dim=1))
            _, contour_pred = contournet(torch.cat([edge_pred, corner_pred], dim=1))
        sal_input = torch.cat([corner_pred, contour_pred], dim=1)
        _, sal_pred = saliencynet(sal_input)
        loss = criterion(sal_pred, saliency_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"[SaliencyNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")





# 5. RecognitionNet
num_epochs = 20

recognitionnet = RecognitionNet(num_classes=10).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(recognitionnet.parameters(), lr=1e-3)

train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    recognitionnet.train()
    running_loss = 0.0
    for images, _, _, _, _, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            _, edge_pred = edgenet(images)
            _, corner_pred = cornernet(torch.cat([images, edge_pred], dim=1))
            _, contour_pred = contournet(torch.cat([edge_pred, corner_pred], dim=1))
            _, sal_pred = saliencynet(torch.cat([corner_pred, contour_pred], dim=1))

        stacked_input = torch.cat([images, edge_pred, corner_pred, contour_pred, sal_pred], dim=1)
        logits = recognitionnet(stacked_input)
        loss = criterion_cls(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"[RecognitionNet] Epoch {epoch+1}, Loss: {loss.item():.4f}")

    recognitionnet.eval()
    total_correct = 0
    with torch.no_grad():
        for images, _, _, _, _, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, edge_pred = edgenet(images)
            _, corner_pred = cornernet(torch.cat([images, edge_pred], dim=1))
            _, contour_pred = contournet(torch.cat([edge_pred, corner_pred], dim=1))
            _, sal_pred = saliencynet(torch.cat([corner_pred, contour_pred], dim=1))

            stacked_input = torch.cat([images, edge_pred, corner_pred, contour_pred, sal_pred], dim=1)
            logits = recognitionnet(stacked_input)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()

    accuracy = total_correct / len(test_dataset)
    test_accuracies.append(accuracy)
    print(f"[RecognitionNet] Test Accuracy: {accuracy * 100:.2f}%")



# Final Evaluation
recognitionnet.eval()
total_correct = 0
with torch.no_grad():
  for images, _, _, _, _, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    _, edge_pred = edgenet(images)
    _, corner_pred = cornernet(torch.cat([images, edge_pred], dim=1))
    _, contour_pred = contournet(torch.cat([edge_pred, corner_pred], dim=1))
    _, sal_pred = saliencynet(torch.cat([corner_pred, contour_pred], dim=1))

    stacked_input = torch.cat([images, edge_pred, corner_pred, contour_pred, sal_pred], dim=1)
    logits = recognitionnet(stacked_input)
    preds = logits.argmax(dim=1)
    total_correct += (preds == labels).sum().item()

accuracy = total_correct / len(test_dataset)
print(f"[RecognitionNet] Final Test Accuracy: {accuracy * 100:.2f}%")


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), [acc * 100 for acc in test_accuracies], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()