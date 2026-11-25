"""
SonoAssist AI - Attention U-Net Segmentation Training Pipeline
Implements:
- Combined BCE + Dice Loss
- Adam optimizer with learning rate scheduler
- Patient-level cross validation
- Checkpoint saving (best validation Dice score)
- Training & validation loss / metric curves export
"""

import os
import sys
import argparse
import time

# Ensure parent directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.attention_unet import AttentionUNet, BCEDiceLoss, calculate_metrics
from training.dataset import MMOTUDataset, split_dataset_by_patient, generate_sample_ultrasound_data


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs, return_sigmoid=True)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        metrics = calculate_metrics(preds, masks)
        total_dice += metrics["dice"] * imgs.size(0)
        total_iou += metrics["iou"] * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, total_dice / n, total_iou / n


def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs, return_sigmoid=True)
            loss = criterion(preds, masks)

            running_loss += loss.item() * imgs.size(0)
            metrics = calculate_metrics(preds, masks)
            total_dice += metrics["dice"] * imgs.size(0)
            total_iou += metrics["iou"] * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, total_dice / n, total_iou / n


def run_training(
    data_dir: str = "dataset",
    output_dir: str = "models/trained_weights",
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cpu"
):
    print("=" * 60)
    print("SonoAssist AI - Attention U-Net Segmentation Training")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size} | LR: {lr}")
    print("=" * 60)

    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    # If no data exists, generate demo samples
    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        print("[Dataset] No local dataset found. Generating demonstration samples...")
        generate_sample_ultrasound_data(data_dir, num_samples=10)

    split_imgs, split_msks = split_dataset_by_patient(images_dir, masks_dir, train_ratio=0.7, val_ratio=0.3, test_ratio=0.0)

    if len(split_imgs["train"]) == 0:
        # Fallback if single patient or all in one folder
        all_imgs = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))]
        all_msks = [os.path.join(masks_dir, os.path.basename(f)) for f in all_imgs]
        split_imgs["train"] = all_imgs[:max(1, int(len(all_imgs) * 0.7))]
        split_msks["train"] = all_msks[:max(1, int(len(all_imgs) * 0.7))]
        split_imgs["val"] = all_imgs[max(1, int(len(all_imgs) * 0.7)):] or split_imgs["train"]
        split_msks["val"] = all_msks[max(1, int(len(all_msks) * 0.7)):] or split_msks["train"]

    train_ds = MMOTUDataset(split_imgs["train"], split_msks["train"], is_train=True)
    val_ds = MMOTUDataset(split_imgs["val"], split_msks["val"], is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    os.makedirs(output_dir, exist_ok=True)
    best_weights_path = os.path.join(output_dir, "attention_unet_best.pth")

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": []
    }
    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, tr_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_dice, vl_iou = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_dice)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_dice)
        history["val_dice"].append(vl_dice)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(vl_iou)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - Loss: {tr_loss:.4f}/{vl_loss:.4f} | Dice: {tr_dice:.4f}/{vl_dice:.4f} | IoU: {tr_iou:.4f}/{vl_iou:.4f}")

        # Save best model
        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": vl_dice,
                "val_iou": vl_iou,
                "model_mode": "TRAINED_MODEL"
            }, best_weights_path)
            print(f"  --> Saved new best model checkpoint (Val Dice: {vl_dice:.4f})")

    # Plot training curves
    plot_training_history(history, os.path.join(output_dir, "segmentation_training_curve.png"))
    print(f"\nTraining completed. Best checkpoint saved to: {best_weights_path}")
    return history


def plot_training_history(history: dict, save_path: str):
    """Save convergence plots for Loss, Dice score, and IoU."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss", color="royalblue")
    axes[0].plot(history["val_loss"], label="Val Loss", color="crimson", linestyle="--")
    axes[0].set_title("Combined Loss (BCE + Dice)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice
    axes[1].plot(history["train_dice"], label="Train Dice", color="royalblue")
    axes[1].plot(history["val_dice"], label="Val Dice", color="green", linestyle="--")
    axes[1].set_title("Dice Similarity Coefficient")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # IoU
    axes[2].plot(history["train_iou"], label="Train IoU", color="royalblue")
    axes[2].plot(history["val_iou"], label="Val IoU", color="orange", linestyle="--")
    axes[2].set_title("Intersection over Union (IoU)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Plot] Training loss curve saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SonoAssist AI Attention U-Net")
    parser.add_argument("--data_dir", type=str, default="SonoAssistAI/dataset")
    parser.add_argument("--output_dir", type=str, default="SonoAssistAI/models/trained_weights")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_training(args.data_dir, args.output_dir, args.epochs, args.batch_size, args.lr, device)
