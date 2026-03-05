import time
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from ptflops import get_model_complexity_info 

from flat_model import *
from curriculum_learner import *

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_m(model):
    return count_params(model) / 1e6


class FullPipelineForFlops(nn.Module):
    def __init__(self, edgenet, cornernet, contournet, saliencynet, recnet):
        super().__init__()
        self.edgenet = edgenet
        self.cornernet = cornernet
        self.contournet = contournet
        self.saliencynet = saliencynet
        self.recnet = recnet

    def forward(self, x):
        _, e = self.edgenet(x)
        _, c = self.cornernet(torch.cat([x, e], dim=1))
        _, t = self.contournet(torch.cat([e, c], dim=1))
        _, s = self.saliencynet(torch.cat([c, t], dim=1))
        stacked = torch.cat([x, e, c, t, s], dim=1)  # 7ch
        return self.recnet(stacked)



class CIFAROnly(Dataset):
    def __init__(self, train=True, root="./data"):
        self.data = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, y = self.data[idx]
        return self.transform(img), int(y)


class CIFARWithMaps(Dataset):
    """
    Returns: rgb, edge, corner, contour, saliency, label
    Shapes : [3,32,32], [1,32,32]...
    """
    def __init__(self, train=True, root="./data"):
        self.data = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, y = self.data[idx]
        rgb = self.transform(img)

        gray = np.array(img.convert("L"))

        # Edge (Canny)
        edge = cv2.Canny(gray, 100, 200).astype(np.float32) / 255.0
        edge = torch.from_numpy(edge).unsqueeze(0)

        # Corner (Harris)
        gray_f = np.float32(gray)
        harris = cv2.cornerHarris(gray_f, 2, 3, 0.04)
        harris = cv2.dilate(harris, None)
        corner = (harris > 0.01 * harris.max()).astype(np.float32)
        corner = torch.from_numpy(corner).unsqueeze(0)

        # Contour (threshold + contours)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, contours, -1, 255, thickness=1)
        contour = (contour_mask.astype(np.float32) / 255.0)
        contour = torch.from_numpy(contour).unsqueeze(0)

        # Saliency (simple proxy)
        sal = np.clip(0.5 * edge.squeeze(0).numpy() + 0.5 * corner.squeeze(0).numpy(), 0, 1).astype(np.float32)
        saliency = torch.from_numpy(sal).unsqueeze(0)

        return rgb, edge, corner, contour, saliency, int(y)


def make_subset_loader(ds_cls, fraction, batch_size, train=True):
    ds = ds_cls(train=train)
    n = len(ds)
    k = max(int(n * fraction), 1)
    sub, _ = random_split(ds, [k, n - k])
    return DataLoader(sub, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)


# -----------------------------
@torch.no_grad()
def eval_acc_flat(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def eval_acc_curriculum(recnet, edgenet, cornernet, contournet, saliencynet, loader, device):
    recnet.eval()
    correct, total = 0, 0
    for rgb, *_maps, y in loader:
        rgb, y = rgb.to(device), y.to(device)

        _, e = edgenet(rgb)
        _, c = cornernet(torch.cat([rgb, e], dim=1))
        _, t = contournet(torch.cat([e, c], dim=1))
        _, s = saliencynet(torch.cat([c, t], dim=1))

        x = torch.cat([rgb, e, c, t, s], dim=1)  # 7-ch
        pred = recnet(x).argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


# -----------------------------
# Train: Flat
# -----------------------------
def train_flat(train_loader, test_loader, device, epochs=10, lr=1e-3):
    model = FlatCNN(num_classes=10).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

    return model, eval_acc_flat(model, test_loader, device)


# -----------------------------
# Train: Curriculum pipeline
# -----------------------------
def train_curriculum(train_loader, test_loader, device, epochs_stage=5, epochs_rec=10, lr=1e-3):
    bce = nn.BCELoss()
    mse = nn.MSELoss()
    xe = nn.CrossEntropyLoss()

    edgenet = EdgeNet().to(device)
    cornernet = CornerNet().to(device)
    contournet = ContourNet().to(device)
    saliencynet = SaliencyNet().to(device)
    recnet = RecognitionNet(num_classes=10).to(device)

    opt_e = optim.Adam(edgenet.parameters(), lr=lr)
    opt_c = optim.Adam(cornernet.parameters(), lr=lr)
    opt_t = optim.Adam(contournet.parameters(), lr=lr)
    opt_s = optim.Adam(saliencynet.parameters(), lr=lr)
    opt_r = optim.Adam(recnet.parameters(), lr=lr)

    # 1) EdgeNet
    for _ in range(epochs_stage):
        edgenet.train()
        for rgb, edge_gt, *_ in train_loader:
            rgb, edge_gt = rgb.to(device), edge_gt.to(device)
            _, e = edgenet(rgb)
            loss = bce(e, edge_gt)
            opt_e.zero_grad(); loss.backward(); opt_e.step()

    # 2) CornerNet (RGB + edge)
    for _ in range(epochs_stage):
        cornernet.train()
        for rgb, _, corner_gt, *_ in train_loader:
            rgb, corner_gt = rgb.to(device), corner_gt.to(device)
            with torch.no_grad():
                _, e = edgenet(rgb)
            _, c = cornernet(torch.cat([rgb, e], dim=1))
            loss = bce(c, corner_gt)
            opt_c.zero_grad(); loss.backward(); opt_c.step()

    # 3) ContourNet (edge + corner)
    for _ in range(epochs_stage):
        contournet.train()
        for rgb, _, _, contour_gt, *_ in train_loader:
            rgb, contour_gt = rgb.to(device), contour_gt.to(device)
            with torch.no_grad():
                _, e = edgenet(rgb)
                _, c = cornernet(torch.cat([rgb, e], dim=1))
            _, t = contournet(torch.cat([e, c], dim=1))
            loss = bce(t, contour_gt)
            opt_t.zero_grad(); loss.backward(); opt_t.step()

    # 4) SaliencyNet (corner + contour) — regression is more sensible here
    for _ in range(epochs_stage):
        saliencynet.train()
        for rgb, *_rest in train_loader:
            # rgb, edge, corner, contour, saliency, y
            rgb, sal_gt = rgb.to(device), _rest[3].to(device)  # saliency is index 3 in _rest
            with torch.no_grad():
                _, e = edgenet(rgb)
                _, c = cornernet(torch.cat([rgb, e], dim=1))
                _, t = contournet(torch.cat([e, c], dim=1))
            _, s = saliencynet(torch.cat([c, t], dim=1))
            loss = mse(s, sal_gt)
            opt_s.zero_grad(); loss.backward(); opt_s.step()

    # 5) RecognitionNet (RGB + all maps)
    for _ in range(epochs_rec):
        recnet.train()
        for rgb, *_rest in train_loader:
            rgb, y = rgb.to(device), _rest[4].to(device)  # label is last (index 4 in _rest)
            with torch.no_grad():
                _, e = edgenet(rgb)
                _, c = cornernet(torch.cat([rgb, e], dim=1))
                _, t = contournet(torch.cat([e, c], dim=1))
                _, s = saliencynet(torch.cat([c, t], dim=1))
            x = torch.cat([rgb, e, c, t, s], dim=1)
            loss = xe(recnet(x), y)
            opt_r.zero_grad(); loss.backward(); opt_r.step()

    acc = eval_acc_curriculum(recnet, edgenet, cornernet, contournet, saliencynet, test_loader, device)
    return (recnet, edgenet, cornernet, contournet, saliencynet), acc


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    fractions = [0.10, 0.25, 0.50, 1.00]
    batch_size = 64

    # Fixed test loaders
    test_flat_loader = DataLoader(CIFAROnly(train=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_pipe_loader = DataLoader(CIFARWithMaps(train=False), batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ---- Params summary (one-time) ----
    flat_tmp = FlatCNN(10)
    e_tmp, c_tmp, t_tmp, s_tmp, r_tmp = EdgeNet(), CornerNet(), ContourNet(), SaliencyNet(), RecognitionNet(10)

    print("\nParameter counts (Millions):")
    print(f"  FlatCNN:        {count_m(flat_tmp):.3f} M")
    print(f"  EdgeNet:        {count_m(e_tmp):.3f} M")
    print(f"  CornerNet:      {count_m(c_tmp):.3f} M")
    print(f"  ContourNet:     {count_m(t_tmp):.3f} M")
    print(f"  SaliencyNet:    {count_m(s_tmp):.3f} M")
    print(f"  RecognitionNet: {count_m(r_tmp):.3f} M")
    print(f"  Pipeline total: {count_m(e_tmp)+count_m(c_tmp)+count_m(t_tmp)+count_m(s_tmp)+count_m(r_tmp):.3f} M")

    # ---- FLOPs summary ----

    print("\nFLOPs / MACs (ptflops):")
    flat_model = FlatCNN(10).to(device)
    pipeline_model = FullPipelineForFlops(EdgeNet(), CornerNet(), ContourNet(), SaliencyNet(), RecognitionNet(10)).to(device)

    macs_f, params_f = get_model_complexity_info(flat_model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)
    macs_p, params_p = get_model_complexity_info(pipeline_model, (3, 32, 32), as_strings=False, print_per_layer_stat=False)

    print(f"  FlatCNN:   {macs_f/1e9:.3f} GMACs,  {params_f/1e6:.3f} M params")
    print(f"  Pipeline:  {macs_p/1e9:.3f} GMACs,  {params_p/1e6:.3f} M params")

    # ---- Data-efficiency experiment ----
    flat_acc, pipe_acc = [], []
    flat_time, pipe_time = [], []

    for frac in fractions:
        print(f"\n=== Fraction: {frac:.2f} ({int(frac*100)}%) ===")

        # Flat
        train_flat_loader = make_subset_loader(CIFAROnly, frac, batch_size, train=True)
        t0 = time.time()
        _flat_model, acc_f = train_flat(train_flat_loader, test_flat_loader, device, epochs=10)
        dt_f = time.time() - t0
        print(f"FlatCNN      Acc={acc_f*100:.2f}%   Time={dt_f:.1f}s")
        flat_acc.append(acc_f)
        flat_time.append(dt_f)

        # Curriculum
        train_pipe_loader = make_subset_loader(CIFARWithMaps, frac, batch_size, train=True)
        t0 = time.time()
        _pipe_models, acc_p = train_curriculum(train_pipe_loader, test_pipe_loader, device, epochs_stage=5, epochs_rec=10)
        dt_p = time.time() - t0
        print(f"Curriculum   Acc={acc_p*100:.2f}%   Time={dt_p:.1f}s")
        pipe_acc.append(acc_p)
        pipe_time.append(dt_p)

    # ---- Plot accuracy ----
    xs = [f * 100 for f in fractions]
    plt.figure()
    plt.plot(xs, [a * 100 for a in flat_acc], marker="o", label="Flat CNN")
    plt.plot(xs, [a * 100 for a in pipe_acc], marker="o", label="Curriculum")
    plt.xlabel("Percent of Training Data")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot time ----
    plt.figure()
    plt.plot(xs, flat_time, marker="o", label="Flat CNN time (s)")
    plt.plot(xs, pipe_time, marker="o", label="Curriculum time (s)")
    plt.xlabel("Percent of Training Data")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()