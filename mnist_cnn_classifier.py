"""
Handwritten Digit Classifier 2026
==================================
Tools  : Python, PyTorch, matplotlib
Dataset: MNIST (70,000 grayscale 28×28 images, 10 classes)
Model  : Convolutional Neural Network (CNN)
Target : ≥ 92% test accuracy
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – saves PNGs without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# HYPER-PARAMETERS
# ─────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 12
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "./mnist_data"

print("=" * 60)
print("  HANDWRITTEN DIGIT CLASSIFIER 2026  (MNIST CNN)")
print("=" * 60)
print(f"\n  Device  : {DEVICE}")
print(f"  Epochs  : {EPOCHS}")
print(f"  Batch   : {BATCH_SIZE}")
print(f"  LR      : {LR}")

# ─────────────────────────────────────────────
# 1. DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
print("\n[1] Loading MNIST dataset …")

train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=8),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=train_transform)
test_dataset  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=test_transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"    Train samples : {len(train_dataset):,}")
print(f"    Test  samples : {len(test_dataset):,}")

# ─────────────────────────────────────────────
# 2. VISUALISE SAMPLE DIGITS
# ─────────────────────────────────────────────
print("\n[2] Visualising sample digits …")

images, labels = next(iter(DataLoader(
    datasets.MNIST(DATA_DIR, train=True, download=False,
                   transform=transforms.ToTensor()), batch_size=25, shuffle=True
)))

fig, axes = plt.subplots(5, 5, figsize=(7, 7))
fig.suptitle("MNIST Sample Digits", fontsize=14, fontweight="bold")
for ax, img, lbl in zip(axes.flat, images, labels):
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(str(lbl.item()), fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig("mnist_sample_digits.png", dpi=150)
# plt.show()  # disabled: Agg backend
print("    [OK]  Saved: mnist_sample_digits.png")

# ─────────────────────────────────────────────
# 3. CNN ARCHITECTURE
# ─────────────────────────────────────────────
class DigitCNN(nn.Module):
    """
    CNN architecture:
      Conv-Block 1 : Conv(1→32, 3×3) → BN → ReLU → Conv(32→32, 3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25)
      Conv-Block 2 : Conv(32→64, 3×3) → BN → ReLU → Conv(64→64, 3×3) → BN → ReLU → MaxPool(2) → Dropout(0.25)
      Classifier   : Flatten → FC(1600→256) → BN → ReLU → Dropout(0.5) → FC(256→10)
    """
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1a = nn.Conv2d(1,  32, 3, padding=1)
        self.bn1a   = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1b   = nn.BatchNorm2d(32)
        self.pool1  = nn.MaxPool2d(2)
        self.drop1  = nn.Dropout2d(0.25)

        # Block 2
        self.conv2a = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2a   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.pool2  = nn.MaxPool2d(2)
        self.drop2  = nn.Dropout2d(0.25)

        # Classifier
        self.fc1  = nn.Linear(64 * 7 * 7, 256)
        self.bn3  = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.5)
        self.fc2  = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.drop1(self.pool1(x))

        # Block 2
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.drop2(self.pool2(x))

        # Classifier
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.drop3(x)
        return self.fc2(x)          # raw logits (CrossEntropyLoss expects logits)


model = DigitCNN().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n[3] Model built — Trainable params: {total_params:,}")
print(model)

# ─────────────────────────────────────────────
# 4. LOSS, OPTIMISER & SCHEDULER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR * 10,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=0.3
)

# ─────────────────────────────────────────────
# 5. TRAIN & EVALUATE LOOP
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total * 100


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total * 100, np.array(all_preds), np.array(all_labels)


print("\n[4] Training …")
print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("-" * 55)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
    vl_loss, vl_acc, _, _ = evaluate(model, test_loader, criterion, DEVICE)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)

    print(f"{epoch:5d}  {tr_loss:10.4f}  {tr_acc:8.2f}%  {vl_loss:8.4f}  {vl_acc:6.2f}%")

# Final evaluation
_, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion, DEVICE)
print(f"\n  [OK]  Final Test Accuracy: {final_acc:.2f}%")

# ─────────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────────
torch.save(model.state_dict(), "mnist_cnn.pth")
print("  [OK]  Model saved: mnist_cnn.pth")

# ─────────────────────────────────────────────
# 7. TRAINING CURVES
# ─────────────────────────────────────────────
print("\n[5] Plotting training curves …")

epochs_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("MNIST CNN — Training Curves", fontsize=15, fontweight="bold")

# Loss
axes[0].plot(epochs_range, history["train_loss"], "o-", color="#E07B54", label="Train Loss")
axes[0].plot(epochs_range, history["val_loss"],   "s-", color="#4C72B0", label="Val Loss")
axes[0].set_title("Loss vs Epoch")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_xticks(list(epochs_range))

# Accuracy
axes[1].plot(epochs_range, history["train_acc"], "o-", color="#E07B54", label="Train Acc")
axes[1].plot(epochs_range, history["val_acc"],   "s-", color="#4C72B0", label="Val Acc")
axes[1].axhline(92, color="green", lw=1.5, linestyle="--", label="92% target")
axes[1].set_title("Accuracy vs Epoch")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
axes[1].legend(); axes[1].grid(alpha=0.3)
axes[1].set_xticks(list(epochs_range))
axes[1].set_ylim(85, 101)
axes[1].annotate(f"Final: {final_acc:.2f}%",
                 xy=(EPOCHS, final_acc),
                 xytext=(EPOCHS - 3, final_acc - 4),
                 arrowprops=dict(arrowstyle="->", color="black"),
                 fontsize=10, color="#333333")

plt.tight_layout()
plt.savefig("mnist_training_curves.png", dpi=150)
# plt.show()  # disabled: Agg backend
print("    [OK]  Saved: mnist_training_curves.png")

# ─────────────────────────────────────────────
# 8. CONFUSION MATRIX
# ─────────────────────────────────────────────
print("\n[6] Confusion Matrix …")
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(all_labels, all_preds)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("MNIST CNN — Confusion Matrix", fontsize=14, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=9,
                color="white" if cm[i, j] > cm.max() * 0.5 else "black")
plt.tight_layout()
plt.savefig("mnist_confusion_matrix.png", dpi=150)
# plt.show()  # disabled: Agg backend
print("    [OK]  Saved: mnist_confusion_matrix.png")

print("\n Classification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=[str(i) for i in range(10)]))

# ─────────────────────────────────────────────
# 9. VISUALISE PREDICTIONS
# ─────────────────────────────────────────────
print("\n[7] Visualising predictions …")

sample_images, sample_labels = next(iter(
    DataLoader(test_dataset, batch_size=25, shuffle=True)
))
model.eval()
with torch.no_grad():
    logits  = model(sample_images.to(DEVICE))
    sample_preds = logits.argmax(dim=1).cpu()

fig, axes = plt.subplots(5, 5, figsize=(9, 9))
fig.suptitle(f"MNIST CNN — Predictions  (Accuracy: {final_acc:.2f}%)",
             fontsize=13, fontweight="bold")
for ax, img, true, pred in zip(axes.flat, sample_images, sample_labels, sample_preds):
    ax.imshow(img.squeeze(), cmap="gray")
    color = "green" if true.item() == pred.item() else "red"
    ax.set_title(f"T:{true.item()} P:{pred.item()}", fontsize=9, color=color)
    ax.axis("off")
plt.tight_layout()
plt.savefig("mnist_predictions.png", dpi=150)
# plt.show()  # disabled: Agg backend
print("    [OK]  Saved: mnist_predictions.png")

# ─────────────────────────────────────────────
# 10. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Model           : DigitCNN (CNN with BatchNorm & Dropout)")
print(f"  Trainable Params: {total_params:,}")
print(f"  Epochs Trained  : {EPOCHS}")
print(f"  Final Test Acc  : {final_acc:.2f}%   {'[OK] Target met!' if final_acc >= 92 else '[WARN] Below 92%'}")
print(f"  Best Val Loss   : {min(history['val_loss']):.4f}")
print(f"  Saved plots     : mnist_sample_digits.png, mnist_training_curves.png,")
print(f"                    mnist_confusion_matrix.png, mnist_predictions.png")
print(f"  Saved model     : mnist_cnn.pth")
print("=" * 60)
