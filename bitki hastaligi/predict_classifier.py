"""
Plant Leaf Disease Predictor — ResNet50
Companion inference script for plant_classifier.py

Usage:
    # Single image
    python predict_classifier.py --img path/to/leaf.jpg

    # Multiple images
    python predict_classifier.py --img img1.jpg img2.jpg img3.jpg

    # Entire folder
    python predict_classifier.py --folder path/to/folder
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image

# ──────────────────────────────────────────────────────────
#  Settings
# ──────────────────────────────────────────────────────────
DATA_PATH  = "./PlantVillage"          # used only to read class names
MODEL_PATH = "./results/resnet50_best.pth"
SAVE_DIR   = "./results"
TOP_K      = 3
IMG_H      = 224
IMG_W      = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {DEVICE}")

# ──────────────────────────────────────────────────────────
#  Class names
# ──────────────────────────────────────────────────────────
raw         = datasets.ImageFolder(root=DATA_PATH)
class_names = raw.classes
n_classes   = len(class_names)
print(f"Classes : {n_classes}")

# ──────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────
class ResNetClassifier(nn.Module):
    """ResNet50 with a custom classification head."""

    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet50(weights=None)

        in_dim = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        self.net = backbone

    def forward(self, x):
        return self.net(x)


def load_model():
    model = ResNetClassifier(num_classes=n_classes)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    return model

# ──────────────────────────────────────────────────────────
#  Transform  (same as eval_tf in plant_classifier.py)
# ──────────────────────────────────────────────────────────
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

eval_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# ──────────────────────────────────────────────────────────
#  Predict
# ──────────────────────────────────────────────────────────
def predict(model, img_path):
    img    = Image.open(img_path).convert("RGB")
    tensor = eval_tf(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top_probs, top_idx = probs.topk(TOP_K)
    results = [(class_names[i], p.item()) for i, p in zip(top_idx, top_probs)]
    return img, results

# ──────────────────────────────────────────────────────────
#  Display
# ──────────────────────────────────────────────────────────
def show_results(entries):
    n   = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))

    if n == 1:
        axes = [axes]

    for row, (img_path, img, preds) in enumerate(entries):
        ax_img, ax_bar = axes[row]

        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(os.path.basename(img_path), fontsize=9)

        labels = [p[0].replace("_", " ") for p in preds]
        values = [p[1] * 100               for p in preds]
        colors = ["#e74c3c" if i == 0 else "#bdc3c7" for i in range(len(preds))]

        ax_bar.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.5)
        ax_bar.set_xlim(0, 105)
        ax_bar.set_xlabel("Confidence (%)")
        ax_bar.set_title(f"Top-{TOP_K} Predictions", fontsize=9)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.3)

        for i, (bar, val) in enumerate(zip(ax_bar.patches, values[::-1])):
            ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=9)

        print(f"\n📷  {os.path.basename(img_path)}")
        for rank, (cls, prob) in enumerate(preds, 1):
            print(f"  #{rank}  {cls:<50}  {prob*100:.2f}%")

    plt.tight_layout()
    out_path = os.path.join(SAVE_DIR, "predictions.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved {out_path}")
    plt.show()

# ──────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img",    nargs="+", default=[], help="Path(s) to image file(s)")
    parser.add_argument("--folder", default=None,          help="Path to a folder of images")
    args = parser.parse_args()

    img_paths = list(args.img)
    if args.folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for f in sorted(os.listdir(args.folder)):
            if os.path.splitext(f)[1].lower() in exts:
                img_paths.append(os.path.join(args.folder, f))

    if not img_paths:
        print("No images provided. Use --img or --folder.")
        exit()

    print(f"Model  : {MODEL_PATH}")
    print(f"Images : {len(img_paths)}\n")

    model   = load_model()
    entries = []
    for path in img_paths:
        img, preds = predict(model, path)
        entries.append((path, img, preds))

    show_results(entries)