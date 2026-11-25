"""
SonoAssist AI - Comparison Visualizations
Provides side-by-side and multi-panel comparison grids for clinicians.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def create_side_by_side_comparison(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: str = "Original Ultrasound",
    label2: str = "Enhanced (Denoised + CLAHE)"
) -> np.ndarray:
    """
    Concatenate two images side-by-side with clear clinical title banners.
    """
    # Convert grayscale to BGR if necessary
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Resize img2 to match img1 height if different
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        img2 = cv2.resize(img2, (int(w2 * (h1 / h2)), h1))

    # Add header text ribbons
    header_h = 30
    banner1 = np.full((header_h, img1.shape[1], 3), 35, dtype=np.uint8)
    banner2 = np.full((header_h, img2.shape[1], 3), 35, dtype=np.uint8)

    cv2.putText(banner1, label1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(banner2, label2, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 1, cv2.LINE_AA)

    panel1 = np.vstack([banner1, img1])
    panel2 = np.vstack([banner2, img2])

    # Join horizontally with a 4-pixel separator
    sep = np.full((panel1.shape[0], 4, 3), 100, dtype=np.uint8)
    combined = np.hstack([panel1, sep, panel2])
    return combined


def create_four_panel_display(
    original: np.ndarray,
    enhanced: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray
) -> np.ndarray:
    """
    Build a 2x2 clinical review grid:
    [ (1) Original ]  [ (2) Enhanced ]
    [ (3) Binary Mask ] [ (4) Annotated Overlay ]
    """
    # Standardize to 256x256 each for clean 2x2 layout
    size = (300, 300)

    def prep_bgr(im):
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return cv2.resize(im, size)

    p1 = prep_bgr(original)
    p2 = prep_bgr(enhanced)
    p3 = prep_bgr(mask)
    p4 = prep_bgr(overlay)

    # Add labels
    def add_label(img, txt):
        banner = np.full((24, img.shape[1], 3), 30, dtype=np.uint8)
        cv2.putText(banner, txt, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([banner, img])

    row1 = np.hstack([add_label(p1, "1. Original"), add_label(p2, "2. Enhanced")])
    row2 = np.hstack([add_label(p3, "3. Predicted Mask"), add_label(p4, "4. Diagnostic Overlay")])

    grid = np.vstack([row1, row2])
    return grid
