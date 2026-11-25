"""
SonoAssist AI - Ovarian Lesion Classification Module (EfficientNet-B0)
Implements an EfficientNet-B0 / Medical CNN classifier for the 8 standard MMOTU ovarian classes:
1. Simple cyst
2. Chocolate cyst
3. Serous cystadenoma
4. Mucinous cystadenoma
5. Teratoma
6. Theca cell tumor
7. High-grade serous
8. Normal ovary

Follows clinical decision support guidelines:
- Distinguishes model confidence from disease probability
- Refrains from autonomous benign/malignant diagnosis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import torchvision.models as models

MMOTU_CATEGORIES = [
    "Simple cyst",
    "Chocolate cyst",
    "Serous cystadenoma",
    "Mucinous cystadenoma",
    "Teratoma",
    "Theca cell tumor",
    "High-grade serous",
    "Normal ovary"
]


class OvarianClassifier(nn.Module):
    """
    EfficientNet-B0 based Classifier adapted for 1-channel or 3-channel ultrasound scans.
    """
    def __init__(self, num_classes: int = len(MMOTU_CATEGORIES), in_channels: int = 1, pretrained: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.categories = MMOTU_CATEGORIES

        try:
            # Try loading torchvision efficientnet_b0 architecture
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b0(weights=weights)

            # Adapt first layer if 1-channel
            if in_channels != 3:
                orig_conv = base_model.features[0][0]
                base_model.features[0][0] = nn.Conv2d(
                    in_channels, orig_conv.out_channels,
                    kernel_size=orig_conv.kernel_size,
                    stride=orig_conv.stride,
                    padding=orig_conv.padding,
                    bias=False
                )

            # Extract feature extractor
            self.features = base_model.features
            self.avgpool = base_model.avgpool

            # Classification head
            in_features = base_model.classifier[1].in_features
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=False),
                nn.Dropout(p=0.2, inplace=False),
                nn.Linear(256, num_classes)
            )
            self._using_backbone = "efficientnet_b0"
        except Exception:
            # Fallback lightweight CNN backbone if torchvision download/weights fail offline
            self._using_backbone = "custom_cnn"
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=False),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=False),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.3, inplace=False),
                nn.Linear(256, 128),
                nn.ReLU(inplace=False),
                nn.Linear(128, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


def predict_category(
    model: nn.Module,
    tensor_img: torch.Tensor,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run inference on a single tensor input [1, C, 256, 256].
    Returns predicted label, confidence, and complete probability distribution.
    """
    model.eval()
    with torch.no_grad():
        if tensor_img.dim() == 3:
            tensor_img = tensor_img.unsqueeze(0)
        tensor_img = tensor_img.to(device)

        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(probs.argmax())
    predicted_label = MMOTU_CATEGORIES[top_idx]
    confidence = float(probs[top_idx])

    all_scores = {MMOTU_CATEGORIES[i]: round(float(probs[i]), 4) for i in range(len(MMOTU_CATEGORIES))}

    return {
        "predicted_category": predicted_label,
        "confidence": round(confidence, 4),
        "class_probabilities": all_scores,
        "clinical_disclaimer": "This prediction reflects statistical model alignment with MMOTU categories and is NOT a definitive histology diagnosis."
    }
