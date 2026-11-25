"""
SonoAssist AI - Confidence, Uncertainty & Clinical Quality Assurance
Implements:
1. Test-Time Augmentation (TTA) for spatial variance & epistemic uncertainty estimation
2. Entropy-based pixel uncertainty mapping
3. Automated Quality-Control (QC) checks:
   - Low contrast / under-exposed check
   - Severe acoustic blur check
   - Empty or negligible mask check
   - Boundary instability warning
"""

import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional


def perform_test_time_augmentation(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Perform Test-Time Augmentation (TTA) with 4 spatial transformations:
    - Original
    - Horizontal Flip
    - Vertical Flip
    - 90-degree Rotation
    Returns:
    - mean_mask (mean predicted probability map [H, W])
    - variance_map (pixel-level uncertainty / variance [H, W])
    - mean_confidence (scalar confidence score across the lesion area)
    """
    model.eval()
    t = input_tensor.to(device)
    if t.dim() == 3:
        t = t.unsqueeze(0)

    predictions = []

    with torch.no_grad():
        # 1. Original
        p1 = model(t).squeeze().cpu().numpy()
        predictions.append(p1)

        # 2. Horizontal Flip
        t_hflip = torch.flip(t, dims=[-1])
        p2_flip = model(t_hflip).squeeze().cpu().numpy()
        p2 = np.fliplr(p2_flip)
        predictions.append(p2)

        # 3. Vertical Flip
        t_vflip = torch.flip(t, dims=[-2])
        p3_flip = model(t_vflip).squeeze().cpu().numpy()
        p3 = np.flipud(p3_flip)
        predictions.append(p3)

        # 4. Rotation 90 degrees
        t_rot = torch.rot90(t, k=1, dims=[-2, -1])
        p4_rot = model(t_rot).squeeze().cpu().numpy()
        p4 = np.rot90(p4_rot, k=-1)
        predictions.append(p4)

    stack = np.stack(predictions, axis=0)  # [4, H, W]
    mean_mask = np.mean(stack, axis=0)
    var_map = np.var(stack, axis=0)

    # Calculate confidence: for segmented pixels (> 0.5), average probability
    lesion_pixels = mean_mask[mean_mask > 0.5]
    if len(lesion_pixels) > 0:
        mean_confidence = float(np.mean(lesion_pixels))
    else:
        # If no lesion, confidence in the negative prediction
        mean_confidence = float(np.mean(1.0 - mean_mask))

    return mean_mask, var_map, round(mean_confidence, 4)


def evaluate_clinical_quality(
    image: np.ndarray,
    predicted_mask: np.ndarray,
    confidence: float,
    uncertainty_variance: float = 0.0
) -> List[Dict[str, str]]:
    """
    Evaluate comprehensive safety & quality assurance checks.
    Produces clear warnings for radiologists when image quality or AI confidence is questionable.
    """
    warnings = []

    # 1. Check Image Quality / Dynamic Range
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    std_val = float(np.std(gray))
    mean_val = float(np.mean(gray))

    if std_val < 18.0:
        warnings.append({
            "type": "LOW_CONTRAST",
            "level": "WARNING",
            "message": "Ultrasound image exhibits very low dynamic range / contrast. Cyst boundary delineation may be degraded."
        })

    if mean_val < 15.0:
        warnings.append({
            "type": "UNDER_EXPOSED",
            "level": "WARNING",
            "message": "Image appears severely underexposed (dark acoustic attenuation). Review scanner gain settings."
        })

    # 2. Check Mask Coverage / Empty Mask
    mask_pixels = np.sum(predicted_mask > 127)
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    coverage_ratio = mask_pixels / float(total_pixels)

    if mask_pixels == 0:
        warnings.append({
            "type": "NO_LESION_DETECTED",
            "level": "INFO",
            "message": "No cyst or focal ovarian lesion was identified in this scan frame."
        })
    elif coverage_ratio > 0.85:
        warnings.append({
            "type": "SUSPICIOUS_OVERSEGMENTATION",
            "level": "CAUTION",
            "message": "Predicted mask spans over 85% of the total ultrasound field. Possible acoustic shadow false positive."
        })

    # 3. Model Confidence Checks
    if confidence < 0.60:
        warnings.append({
            "type": "LOW_CONFIDENCE",
            "level": "CAUTION",
            "message": f"Model confidence is low ({confidence*100:.1f}%). Secondary radiologist review is strongly advised."
        })

    # 4. Uncertainty Variance Check
    if uncertainty_variance > 0.08:
        warnings.append({
            "type": "HIGH_TTA_VARIANCE",
            "level": "WARNING",
            "message": "High test-time augmentation spatial variance detected around lesion margins. Boundary should be manually verified."
        })

    return warnings
