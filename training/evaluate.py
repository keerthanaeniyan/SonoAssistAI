"""
SonoAssist AI - Evaluation & Benchmarking Module
Computes clinical segmentation and classification metrics:
- Mean Dice Similarity Coefficient (DSC)
- Mean Intersection over Union (mIoU)
- Precision, Recall, Sensitivity, Specificity
- Area measurement error
"""

import os
import sys

# Ensure parent directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, List

from models.attention_unet import AttentionUNet, calculate_metrics
from models.efficientnet import OvarianClassifier, MMOTU_CATEGORIES
from training.dataset import MMOTUDataset


def evaluate_segmentation_model(
    model_path: str,
    images: List[str],
    masks: List[str],
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Evaluate Attention U-Net model on test dataset.
    """
    if not os.path.exists(model_path):
        return {
            "evaluated": False,
            "error": f"Model checkpoint not found at {model_path}",
            "mean_dice": 0.0,
            "mean_iou": 0.0
        }

    dataset = MMOTUDataset(images, masks, is_train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = AttentionUNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    all_dice, all_iou, all_prec, all_rec = [], [], [], []

    with torch.no_grad():
        for imgs, true_masks, _ in loader:
            imgs = imgs.to(device)
            preds = model(imgs, return_sigmoid=True)
            m = calculate_metrics(preds, true_masks.to(device))
            all_dice.append(m["dice"])
            all_iou.append(m["iou"])
            all_prec.append(m["precision"])
            all_rec.append(m["recall"])

    return {
        "evaluated": True,
        "sample_count": len(dataset),
        "mean_dice": round(float(np.mean(all_dice)), 4),
        "std_dice": round(float(np.std(all_dice)), 4),
        "mean_iou": round(float(np.mean(all_iou)), 4),
        "std_iou": round(float(np.std(all_iou)), 4),
        "mean_precision": round(float(np.mean(all_prec)), 4),
        "mean_recall": round(float(np.mean(all_rec)), 4)
    }


def print_evaluation_report(results: Dict[str, Any]) -> str:
    """Format evaluation results into a clean markdown table."""
    if not results.get("evaluated"):
        return f"Evaluation not conducted: {results.get('error', 'Unknown error')}"

    report = f"""
### SonoAssist AI Model Evaluation Report
- **Evaluated Test Scans**: {results['sample_count']}
- **Mean Dice Score (DSC)**: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}
- **Mean IoU (Jaccard Index)**: {results['mean_iou']:.4f} ± {results['std_iou']:.4f}
- **Precision**: {results['mean_precision']:.4f}
- **Recall / Sensitivity**: {results['mean_recall']:.4f}
"""
    return report
