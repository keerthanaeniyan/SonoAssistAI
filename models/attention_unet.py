"""
SonoAssist AI - Attention U-Net & Standard U-Net Architecture
Implements:
1. Standard U-Net
2. Attention U-Net with additive attention gates
3. Combined BCE + Dice Loss
4. Dice and IoU evaluation metrics
Tailored for 256x256 ovarian ultrasound cyst segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Double convolution block: [Conv2d -> BatchNorm -> ReLU] x 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Additive Attention Gate:
    Suppresses background speckle and false positives in non-ovarian acoustic shadows
    while highlighting boundaries of hypoechoic cyst lesions.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        # F_g: gating signal channels from deeper layer
        # F_l: lower-level skip connection channels
        # F_int: intermediate channel dimension
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Resize g to match x spatial dimensions if needed
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """
    Attention U-Net for Ovarian Cyst Segmentation.
    Input: [B, 1, 256, 256] (Grayscale ultrasound) or [B, in_channels, H, W]
    Output: [B, 1, 256, 256] (Sigmoid probabilities or raw logits)
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, features: Tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.inc = ConvBlock(in_channels, features[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(features[0], features[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(features[1], features[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(features[2], features[3]))

        # Bottleneck
        bottleneck_features = features[3] * 2
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(features[3], bottleneck_features)
        )

        # Attention Gates
        self.ag4 = AttentionGate(F_g=bottleneck_features, F_l=features[3], F_int=features[3] // 2)
        self.ag3 = AttentionGate(F_g=features[3], F_l=features[2], F_int=features[2] // 2)
        self.ag2 = AttentionGate(F_g=features[2], F_l=features[1], F_int=features[1] // 2)
        self.ag1 = AttentionGate(F_g=features[1], F_l=features[0], F_int=features[0] // 2)

        # Decoder Upsampling & Convolutions
        self.up4 = nn.ConvTranspose2d(bottleneck_features, features[3], kernel_size=2, stride=2)
        self.dec4 = ConvBlock(bottleneck_features, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec3 = ConvBlock(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec2 = ConvBlock(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec1 = ConvBlock(features[0] * 2, features[0])

        # Final 1x1 convolution
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, return_sigmoid: bool = True) -> torch.Tensor:
        # Encoder forward pass
        x1 = self.inc(x)         # [B, 32, 256, 256]
        x2 = self.down1(x1)      # [B, 64, 128, 128]
        x3 = self.down2(x2)      # [B, 128, 64, 64]
        x4 = self.down3(x3)      # [B, 256, 32, 32]
        b = self.bottleneck(x4)  # [B, 512, 16, 16]

        # Decoder forward pass with Attention Filtering
        d4 = self.up4(b)
        x4_att = self.ag4(g=b, x=x4)
        d4 = torch.cat([x4_att, d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x3_att = self.ag3(g=d4, x=x3)
        d3 = torch.cat([x3_att, d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x2_att = self.ag2(g=d3, x=x2)
        d2 = torch.cat([x2_att, d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x1_att = self.ag1(g=d2, x=x1)
        d1 = torch.cat([x1_att, d1], dim=1)
        d1 = self.dec1(d1)

        logits = self.outc(d1)
        if return_sigmoid:
            return torch.sigmoid(logits)
        return logits


class DiceLoss(nn.Module):
    """Differentiable Soft Dice Loss for binary segmentation."""
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds expected in [0, 1] range
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (preds * targets).sum()
        total = preds.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice Loss."""
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCELoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> dict:
    """
    Calculate Dice Score, IoU (Jaccard Index), Precision, and Recall.
    Inputs are torch tensors in [0, 1].
    """
    with torch.no_grad():
        preds_bin = (preds > threshold).float().view(-1)
        targets_bin = (targets > 0.5).float().view(-1)

        intersection = (preds_bin * targets_bin).sum().item()
        total_pred = preds_bin.sum().item()
        total_target = targets_bin.sum().item()
        union = total_pred + total_target - intersection

        dice = (2.0 * intersection + smooth) / (total_pred + total_target + smooth)
        iou = (intersection + smooth) / (union + smooth)
        precision = (intersection + smooth) / (total_pred + smooth)
        recall = (intersection + smooth) / (total_target + smooth)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall)
    }
