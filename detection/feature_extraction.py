"""
SonoAssist AI - Advanced Segmentation & Morphological Feature Extraction
Extracts quantitative geometric and acoustic descriptor metrics from the predicted cyst mask:
- Area in pixels
- Perimeter in pixels
- Circularity / Isoperimetric quotient
- Aspect ratio (Major / Minor axis & Bounding Box)
- Solidity (Area / Convex Hull Area)
- Boundary roughness and compactness
- Mean and standard deviation of internal pixel intensity
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple


def extract_morphological_features(
    mask: np.ndarray,
    original_gray: Optional[np.ndarray] = None,
    pixel_spacing_mm: Optional[float] = None
) -> Dict[str, Any]:
    """
    Extract descriptive morphological features from a binary segmentation mask.
    All measurements are strictly in pixel units unless calibrated physical pixel spacing
    (mm/pixel) is provided.
    """
    if mask is None:
        raise ValueError("Mask cannot be None")

    # Ensure binary uint8 mask (0 or 255)
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Empty mask case
    if len(contours) == 0:
        return {
            "has_detection": False,
            "cyst_count": 0,
            "primary_cyst": None,
            "all_cysts": [],
            "summary_notes": "No cyst or fluid-filled lesion detected in the segmentation mask."
        }

    # Filter very small noisy contours (< 25 pixels)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 25]
    if len(valid_contours) == 0:
        return {
            "has_detection": False,
            "cyst_count": 0,
            "primary_cyst": None,
            "all_cysts": [],
            "summary_notes": "Only negligible micro-artifacts detected (< 25 pixels); no clinical cyst identified."
        }

    # Sort contours by area descending
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    cysts_data = []

    for idx, cnt in enumerate(valid_contours):
        area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, closed=True))

        # Circularity (Isoperimetric Quotient): 4 * pi * Area / Perimeter^2
        # Pure circle has circularity = 1.0; irregular shapes < 0.6
        if perimeter > 0:
            circularity = float((4.0 * np.pi * area) / (perimeter ** 2))
        else:
            circularity = 0.0
        circularity = min(1.0, max(0.0, circularity))

        # Convex Hull and Solidity
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(area / hull_area) if hull_area > 0 else 1.0
        solidity = min(1.0, max(0.0, solidity))

        # Bounding Box and Aspect Ratio
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / float(h) if h > 0 else 1.0

        # Equivalent Diameter
        equiv_diameter_px = float(np.sqrt(4.0 * area / np.pi))

        # Fit Ellipse if contour has at least 5 points
        ellipse_axes = None
        orientation_deg = 0.0
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                (center_x, center_y), (d1, d2), angle = ellipse
                major_axis = max(d1, d2)
                minor_axis = min(d1, d2)
                ellipse_axes = (round(major_axis, 2), round(minor_axis, 2))
                orientation_deg = round(angle, 1)
            except Exception:
                pass

        # Internal pixel intensity analysis if grayscale image provided
        internal_intensity_mean = None
        internal_intensity_std = None
        if original_gray is not None and original_gray.shape[:2] == mask.shape[:2]:
            c_mask = np.zeros_like(mask)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            pixels = original_gray[c_mask == 255]
            if len(pixels) > 0:
                internal_intensity_mean = round(float(np.mean(pixels)), 2)
                internal_intensity_std = round(float(np.std(pixels)), 2)

        # Boundary Roughness / Compactness index
        # Ratio of perimeter to convex hull perimeter
        hull_perimeter = float(cv2.arcLength(hull, closed=True))
        roughness_index = round(float(perimeter / hull_perimeter), 3) if hull_perimeter > 0 else 1.0

        # Calibrated real-world units if pixel spacing is given
        calibrated_info = {}
        if pixel_spacing_mm and pixel_spacing_mm > 0:
            calibrated_area_cm2 = round((area * (pixel_spacing_mm ** 2)) / 100.0, 3)
            calibrated_diameter_mm = round(equiv_diameter_px * pixel_spacing_mm, 2)
            calibrated_info = {
                "pixel_spacing_mm": pixel_spacing_mm,
                "area_cm2": calibrated_area_cm2,
                "equivalent_diameter_mm": calibrated_diameter_mm
            }

        cyst_info = {
            "cyst_id": idx + 1,
            "area_pixels": round(area, 1),
            "perimeter_pixels": round(perimeter, 1),
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "solidity": round(solidity, 3),
            "roughness_index": roughness_index,
            "equivalent_diameter_pixels": round(equiv_diameter_px, 1),
            "bounding_box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "fitted_axes_pixels": ellipse_axes,
            "orientation_deg": orientation_deg,
            "internal_mean_intensity": internal_intensity_mean,
            "internal_std_intensity": internal_intensity_std,
            "calibrated_measurements": calibrated_info
        }
        cysts_data.append(cyst_info)

    return {
        "has_detection": True,
        "cyst_count": len(cysts_data),
        "primary_cyst": cysts_data[0],
        "all_cysts": cysts_data,
        "summary_notes": f"Detected {len(cysts_data)} distinct region(s). Primary region area: {cysts_data[0]['area_pixels']} px, Circularity: {cysts_data[0]['circularity']}."
    }
