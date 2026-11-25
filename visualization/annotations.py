"""
SonoAssist AI - Visualization & Annotation Module
Generates high-contrast, clinically readable overlays, bounding boxes,
and contour visualizers on ultrasound images.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


def create_segmentation_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),  # BGR Green by default
    alpha: float = 0.35,
    draw_contour: bool = True,
    contour_thickness: int = 2
) -> np.ndarray:
    """
    Produce an alpha-blended overlay of the predicted cyst mask over the ultrasound image.
    Preserves underlying speckle and acoustic shadow texture.
    """
    if image is None or mask is None:
        raise ValueError("Image or mask is None")

    # Match dimensions
    if len(image.shape) == 2:
        base_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        base_img = image.copy()

    h, w = base_img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Binary mask
    bin_mask = (mask > 127).astype(np.uint8)

    # Color overlay
    overlay = base_img.copy()
    colored_mask = np.zeros_like(base_img, dtype=np.uint8)
    colored_mask[bin_mask == 1] = color

    cv2.addWeighted(colored_mask, alpha, overlay, 1.0, 0, overlay)

    # Draw sharp boundary
    if draw_contour:
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, contour_thickness)

    return overlay


def annotate_image_with_findings(
    image: np.ndarray,
    mask: np.ndarray,
    morph_features: Optional[Dict[str, Any]] = None,
    predicted_class: Optional[str] = None,
    confidence: Optional[float] = None,
    model_mode: str = "Trained"
) -> np.ndarray:
    """
    Generate the comprehensive annotated diagnostic review image with:
    - Cyst contour highlight
    - Bounding box
    - Callout text with Area, Circularity, Class, and Confidence
    - Demonstration warning badge if running in demo/untrained mode
    """
    if len(image.shape) == 2:
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        annotated = image.copy()

    h, w = annotated.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    bin_mask = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Highlight regions
    for idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < 25:
            continue

        # Draw contour in bright cyan/teal
        cv2.drawContours(annotated, [cnt], -1, (255, 200, 0), 2)

        # Bounding box in light blue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 215, 255), 1)

        # Label badge above bounding box
        label_text = f"Cyst #{idx+1}: {int(cv2.contourArea(cnt))} px"
        label_y = max(15, y - 6)
        cv2.putText(annotated, label_text, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    # Header Overlay Banner with Scan Summary
    banner_height = 42
    banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
    # Dark blue banner (40, 20, 20)
    banner[:] = (45, 30, 25)
    
    cv2.putText(banner, "SonoAssist AI Decision Support", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
    status_str = f"Class: {predicted_class or 'N/A'}"
    if confidence is not None:
        status_str += f" | Conf: {confidence*100:.1f}%"
    if model_mode and "DEMO" in model_mode.upper():
        status_str += " [DEMO MODE]"

    cv2.putText(banner, status_str, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 230, 255), 1, cv2.LINE_AA)

    # Stack banner on top
    final_output = np.vstack([banner, annotated])
    return final_output
