"""
Plant Leaf Disease Recognition using Transfer Learning (ResNet50)
Dataset : PlantVillage  — https://www.kaggle.com/datasets/emmarex/plantdisease
Author  : ---
Requires: torch torchvision scikit-learn matplotlib seaborn tqdm

Setup:
    pip install torch torchvision scikit-learn matplotlib seaborn tqdm
    Place the PlantVillage folder next to this file, then:
    python plant_classifier.py
"""

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ──────────────────────────────────────────────────────────
#  Settings
# ──────────────────────────────────────────────────────────
DATA_PATH    = "./PlantVillage"
SAVE_DIR     = "./results"
RANDOM_STATE = 7

IMG_H, IMG_W = 224, 224
BATCH        = 32
EPOCHS       = 10
LR           = 0.01
MOMENTUM     = 0.9
WD           = 5e-4
STEP_SIZE    = 4       # StepLR: decay every N epochs
GAMMA        = 0.1     # StepLR: multiply LR by this

TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# test = remaining 0.10

os.makedirs(SAVE_DIR, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")


# ──────────────────────────────────────────────────────────
#  Data
# ──────────────────────────────────────────────────────────
class PlantDataset(torch.utils.data.Dataset):
    """Thin wrapper that applies a transform to a Subset."""

    def __init__(self, subset, transform=None):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((IMG_H, IMG_W)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf


def build_loaders(data_path):
    raw        = datasets.ImageFolder(root=data_path)
    labels     = [s[1] for s in raw.samples]
    indices    = list(range(len(raw)))

    # First cut: train vs (val + test)
    idx_train, idx_temp, _, lbl_temp = train_test_split(
        indices, labels,
        test_size=1 - TRAIN_RATIO,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    # Second cut: val vs test (equal halves of the remaining)
    val_frac = VAL_RATIO / (1 - TRAIN_RATIO)
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=1 - val_frac,
        stratify=lbl_temp,
        random_state=RANDOM_STATE,
    )

    train_tf, eval_tf = get_transforms()

    train_ds = PlantDataset(Subset(raw, idx_train), train_tf)
    val_ds   = PlantDataset(Subset(raw, idx_val),   eval_tf)
    test_ds  = PlantDataset(Subset(raw, idx_test),  eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

    print(f"Classes : {len(raw.classes)}")
    print(f"Train / Val / Test : {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    return train_loader, val_loader, test_loader, raw.classes


# ──────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────
class ResNetClassifier(nn.Module):
    """ResNet50 with a custom classification head."""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Freeze all layers first
        for p in backbone.parameters():
            p.requires_grad = False

        # Replace the final FC layer
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        self.net = backbone

    def unfreeze_layer4(self):
        """Unfreeze layer4 + fc for deeper fine-tuning."""
        for name, p in self.net.named_parameters():
            if "layer4" in name or "fc" in name:
                p.requires_grad = True

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, train_loader, val_loader, num_classes):
        self.model        = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = nn.CrossEntropyLoss()
        self.num_classes  = num_classes

        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR, momentum=MOMENTUM, weight_decay=WD
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=STEP_SIZE, gamma=GAMMA
        )

        self.history    = {"train_loss": [], "train_acc": [],
                           "val_loss":   [], "val_acc":   []}
        self.best_acc   = 0.0
        self.best_wts   = None

    def _run_phase(self, loader, training: bool):
        self.model.train() if training else self.model.eval()
        total_loss, correct = 0.0, 0

        with torch.set_grad_enabled(training):
            for imgs, lbls in tqdm(loader, leave=False):
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)

                if training:
                    self.optimizer.zero_grad()

                out  = self.model(imgs)
                loss = self.criterion(out, lbls)

                if training:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                correct    += (out.argmax(1) == lbls).sum().item()

        n = len(loader.dataset)
        return total_loss / n, correct / n

    def fit(self):
        # Phase 1 — only train the new head (epochs 1–3)
        # Phase 2 — unfreeze layer4 as well (epoch 4 onward)
        print("\n── Training ───────────────────────────────────────")
        for epoch in range(1, EPOCHS + 1):

            if epoch == 4:
                self.model.unfreeze_layer4()
                # Rebuild optimizer with the newly unfrozen params
                self.optimizer = torch.optim.SGD(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=LR * GAMMA, momentum=MOMENTUM, weight_decay=WD
                )
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=STEP_SIZE, gamma=GAMMA
                )
                print("  [epoch 4] layer4 unfrozen — LR reset to", LR * GAMMA)

            t0 = time.time()
            tr_loss, tr_acc = self._run_phase(self.train_loader, training=True)
            vl_loss, vl_acc = self._run_phase(self.val_loader,   training=False)
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(vl_loss)
            self.history["val_acc"].append(vl_acc)

            print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
                  f"loss: {tr_loss:.4f} / {vl_loss:.4f}  "
                  f"acc: {tr_acc:.4f} / {vl_acc:.4f}  "
                  f"({time.time()-t0:.1f}s)")

            if vl_acc > self.best_acc:
                self.best_acc  = vl_acc
                self.best_wts  = {k: v.clone() for k, v in self.model.state_dict().items()}
                torch.save(self.best_wts, os.path.join(SAVE_DIR, "resnet50_best.pth"))
                print(f"  → saved  (val_acc={self.best_acc:.4f})")

        print(f"\nBest val accuracy : {self.best_acc:.4f}")
        self.model.load_state_dict(self.best_wts)


# ──────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────
def evaluate(model, loader, class_names):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Evaluating"):
            preds = model(imgs.to(DEVICE)).argmax(1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(lbls.tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_true))
    print(f"\nTest accuracy : {acc:.4f}\n")
    print(classification_report(all_true, all_preds, target_names=class_names, digits=4))
    return all_true, all_preds


# ──────────────────────────────────────────────────────────
#  Plots
# ──────────────────────────────────────────────────────────
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ep = range(1, len(history["train_loss"]) + 1)

    ax1.plot(ep, history["train_loss"], marker="o", label="train")
    ax1.plot(ep, history["val_loss"],   marker="o", label="val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Cross-Entropy Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(ep, history["train_acc"], marker="o", label="train")
    ax2.plot(ep, history["val_acc"],   marker="o", label="val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "history.png")
    plt.savefig(path, dpi=150); print(f"Saved {path}"); plt.show()


def plot_confusion(true, pred, class_names):
    cm  = confusion_matrix(true, pred, normalize="true")
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax)
    ax.set_title("Confusion Matrix — Test Set (normalised)", fontsize=13)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); print(f"Saved {path}"); plt.show()


# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = build_loaders(DATA_PATH)

    model   = ResNetClassifier(num_classes=len(class_names))
    trainer = Trainer(model, train_loader, val_loader, len(class_names))
    trainer.fit()

    true_labels, pred_labels = evaluate(model, test_loader, class_names)

    plot_history(trainer.history)
    plot_confusion(true_labels, pred_labels, class_names)

    print("All done. Results saved to:", SAVE_DIR)