import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch.optim as optim


class EdgeNet(nn.Module):
    def __init__(self):
        super(EdgeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feat = F.relu(self.conv3(x))
        edge_pred = torch.sigmoid(self.out(feat))
        return feat, edge_pred

class CornerNet(nn.Module):
    def __init__(self):
        super(CornerNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(x))
        corner_pred = torch.sigmoid(self.out(feat))
        return feat, corner_pred

class ContourNet(nn.Module):
    def __init__(self):
        super(ContourNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(x))
        contour_pred = torch.sigmoid(self.out(feat))
        return feat, contour_pred

class SaliencyNet(nn.Module):
    def __init__(self):
        super(SaliencyNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        feat = F.relu(self.conv2(x))
        sal_pred = torch.sigmoid(self.out(feat))
        return feat, sal_pred

class RecognitionNet(nn.Module):
    def __init__(self, num_classes):
        super(RecognitionNet, self).__init__()
        self.conv1 = nn.Conv2d(7, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
