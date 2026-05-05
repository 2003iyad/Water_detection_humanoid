"""
water_detection_2.py — Segmentation-based water level detector.
Author: Iyad Laphir

Instead of predicting 4 box coordinates, the model predicts a pixel-level
mask of where the water is (white on black). Accuracy is measured as the
% of pixel overlap (IoU) between the predicted mask and the ground truth mask.

The ground truth mask is built by filling the COCO bounding box white —
this works because the annotations already tightly bound the water region,
not the whole glass.

Architecture:
    ResNet18 encoder (pretrained) + U-Net style decoder → binary mask
    + small classification head                         → level 0-4

Run:
    python water_detection_2.py

Outputs (saved to segmentation_checkpoints/):
    best.pth              — best model weights
    latest.pth            — most recent epoch weights
    training_log.csv      — per-epoch metrics
    training_curves.png   — loss + IoU chart
"""

import csv
import os
import json
import random
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader

from model import WaterSegNet, NUM_CLASSES, CLASS_NAMES

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data")
TRAIN_DIR      = os.path.join(DATA_DIR, "train")
VALID_DIR      = os.path.join(DATA_DIR, "valid")
TEST_DIR       = os.path.join(DATA_DIR, "test")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "segmentation_checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

IMG_SIZE = 416
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

BATCH_SIZE    = 8
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4

LAMBDA_SEG = 1.0
LAMBDA_CLS = 0.5

MASK_THRESHOLD = 0.5

DEVICE      = "cuda"
NUM_WORKERS = 4
SEED        = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

class WaterSegDataset(Dataset):
    """
    Loads images and builds binary segmentation masks from COCO bounding boxes.

    Ground truth mask: pixels inside the bounding box = 1 (water), rest = 0.

    Each item returns:
        image : FloatTensor [3, IMG_SIZE, IMG_SIZE]
        mask  : FloatTensor [1, IMG_SIZE, IMG_SIZE]  values 0.0 or 1.0
        label : int  (0-4, the liquid level)
    """

    def __init__(self, split_dir: str, augment: bool = False):
        self.split_dir = split_dir
        self.augment   = augment

        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        with open(ann_path) as f:
            coco = json.load(f)

        self.id_to_info = {img["id"]: img for img in coco["images"]}

        self.samples = []
        grouped = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            if ann["category_id"] == 0:
                continue
            grouped[ann["image_id"]].append(ann)

        for img_id, anns in grouped.items():
            if anns:
                best = max(anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
                self.samples.append((img_id, best))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, ann = self.samples[idx]
        info     = self.id_to_info[img_id]
        img_path = os.path.join(self.split_dir, info["file_name"])

        image   = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        x, y, bw, bh = ann["bbox"]
        mask_pil = Image.new("L", (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(mask_pil)
        draw.rectangle([x, y, x + bw, y + bh], fill=255)

        image    = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask_pil = mask_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        if self.augment:
            image = T.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2
            )(image)
            if random.random() < 0.05:
                image = TF.to_grayscale(image, num_output_channels=3)

        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=MEAN, std=STD)

        mask_tensor = TF.to_tensor(mask_pil)

        label = ann["category_id"] - 1

        return image_tensor, mask_tensor, label


def get_dataloader(split_dir, batch_size, shuffle, augment=False):
    dataset = WaterSegDataset(split_dir, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOSS + METRICS
# ══════════════════════════════════════════════════════════════════════════════

def segmentation_loss(mask_logits, cls_logits, masks, labels):
    """
    Combined loss:
      - BCEWithLogitsLoss on the pixel mask
      - CrossEntropyLoss  on the level classification
    """
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()

    loss_seg = bce(mask_logits, masks)
    loss_cls = ce(cls_logits, labels)

    total = LAMBDA_SEG * loss_seg + LAMBDA_CLS * loss_cls
    return {"total": total, "seg": loss_seg, "cls": loss_cls}


def pixel_iou(pred_mask: torch.Tensor, true_mask: torch.Tensor) -> float:
    pred = pred_mask.bool()
    true = true_mask.bool()

    intersection = (pred & true).float().sum(dim=(1, 2, 3))
    union        = (pred | true).float().sum(dim=(1, 2, 3))

    return (intersection / (union + 1e-6)).mean().item()


def cls_accuracy(cls_logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = cls_logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    totals = {"total": 0.0, "seg": 0.0, "cls": 0.0}

    for images, masks, labels in loader:
        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        mask_logits, cls_logits = model(images)
        losses = segmentation_loss(mask_logits, cls_logits, masks, labels)
        losses["total"].backward()
        optimizer.step()

        for k in totals:
            totals[k] += losses[k].item()

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals  = {"total": 0.0, "seg": 0.0, "cls": 0.0}
    iou_sum = 0.0
    acc_sum = 0.0

    for images, masks, labels in loader:
        images = images.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        mask_logits, cls_logits = model(images)
        losses = segmentation_loss(mask_logits, cls_logits, masks, labels)

        for k in totals:
            totals[k] += losses[k].item()

        pred_masks = (torch.sigmoid(mask_logits) > MASK_THRESHOLD).float()
        iou_sum   += pixel_iou(pred_masks, masks)
        acc_sum   += cls_accuracy(cls_logits, labels)

    n = len(loader)
    avg = {k: v / n for k, v in totals.items()}
    avg["pixel_iou"] = iou_sum / n
    avg["cls_acc"]   = acc_sum / n
    return avg


# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════

def save_training_chart(history: dict, save_path: str):
    epochs = history["epoch"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True,
                             gridspec_kw={"hspace": 0.4})
    fig.suptitle("Training Progress — Water Segmentation Model",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#E74C3C", linewidth=2,
            marker="o", markersize=3, label="Train loss")
    ax.plot(epochs, history["val_loss"],   color="#3498DB", linewidth=2,
            marker="o", markersize=3, label="Val loss")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ax.plot(epochs, history["pixel_iou"], color="#2ECC71", linewidth=2,
            marker="o", markersize=3, label="Val pixel IoU")
    ax.fill_between(epochs, history["pixel_iou"], alpha=0.15, color="#2ECC71")
    best_iou_epoch = epochs[history["pixel_iou"].index(max(history["pixel_iou"]))]
    best_iou_val   = max(history["pixel_iou"])
    ax.axvline(best_iou_epoch, color="#2ECC71", linestyle="--", alpha=0.4)
    ax.annotate(
        f"best IoU\nepoch {best_iou_epoch}\n{best_iou_val:.3f}",
        xy=(best_iou_epoch, best_iou_val),
        xytext=(best_iou_epoch + max(1, len(epochs) * 0.04), best_iou_val * 0.92),
        fontsize=8, color="#2ECC71",
        arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=1),
    )
    ax.set_ylabel("Pixel IoU")
    ax.set_title("Validation Pixel IoU  (% overlap between predicted and true water region)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    ax = axes[2]
    ax.plot(epochs, history["cls_acc"], color="#9B59B6", linewidth=2,
            marker="o", markersize=3, label="Val level accuracy")
    ax.fill_between(epochs, history["cls_acc"], alpha=0.15, color="#9B59B6")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Level Classification Accuracy  (0-4 correct)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = WaterSegNet().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader = get_dataloader(TRAIN_DIR, BATCH_SIZE, shuffle=True,  augment=True)
    valid_loader = get_dataloader(VALID_DIR, BATCH_SIZE, shuffle=False)

    log_path      = os.path.join(CHECKPOINT_DIR, "training_log.csv")
    best_val_loss = float("inf")
    history       = {"epoch": [], "train_loss": [], "val_loss": [],
                     "pixel_iou": [], "cls_acc": []}

    with open(log_path, "w", newline="") as csvfile:
        fieldnames = ["epoch", "train_loss", "train_seg", "train_cls",
                      "val_loss", "val_seg", "val_cls", "pixel_iou", "cls_acc"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, NUM_EPOCHS + 1):
            train_m = train_one_epoch(model, train_loader, optimizer, device)
            val_m   = validate(model, valid_loader, device)

            print(
                f"Epoch {epoch:03d}/{NUM_EPOCHS}  "
                f"train_loss={train_m['total']:.4f}  "
                f"val_loss={val_m['total']:.4f}  "
                f"pixel_iou={val_m['pixel_iou']:.4f}  "
                f"cls_acc={val_m['cls_acc']:.4f}"
            )

            ckpt = {
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_m["total"],
                "pixel_iou":   val_m["pixel_iou"],
            }
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "latest.pth"))
            if val_m["total"] < best_val_loss:
                best_val_loss = val_m["total"]
                torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pth"))
                print(f"  >> New best model (val_loss={best_val_loss:.4f})")

            history["epoch"].append(epoch)
            history["train_loss"].append(train_m["total"])
            history["val_loss"].append(val_m["total"])
            history["pixel_iou"].append(val_m["pixel_iou"])
            history["cls_acc"].append(val_m["cls_acc"])

            writer.writerow({
                "epoch":      epoch,
                "train_loss": train_m["total"],
                "train_seg":  train_m["seg"],
                "train_cls":  train_m["cls"],
                "val_loss":   val_m["total"],
                "val_seg":    val_m["seg"],
                "val_cls":    val_m["cls"],
                "pixel_iou":  val_m["pixel_iou"],
                "cls_acc":    val_m["cls_acc"],
            })
            csvfile.flush()

    chart_path = os.path.join(CHECKPOINT_DIR, "training_curves.png")
    save_training_chart(history, chart_path)
    print(f"\nTraining complete.")
    print(f"Logs  -> {log_path}")
    print(f"Chart -> {chart_path}")


if __name__ == "__main__":
    main()
