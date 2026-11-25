"""
SonoAssist AI - Denoising Module
Provides dedicated speckle reduction, median filtering, and Non-Local Means (NLM)
denoising tailored for ovarian ultrasound imaging.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def reduce_speckle_noise(image: np.ndarray, diameter: int = 7, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering to suppress ultrasound speckle noise
    without blurring anatomical cyst boundaries.
    """
    if image is None:
        raise ValueError("Input image is None")
    
    # Ensure 8-bit single channel or 3 channel
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
        
    return cv2.bilateralFilter(image, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply median filter to remove salt-and-pepper/impulse acoustic artifacts.
    Kernel size must be an odd positive integer (e.g. 3, 5, 7).
    """
    if image is None:
        raise ValueError("Input image is None")
        
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, kernel_size)
    
    return cv2.medianBlur(image, kernel_size)


def apply_nlm_denoising(image: np.ndarray, h: float = 10.0, template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising.
    Explores patch redundancies across the ultrasound frame to average noise while
    preserving subtle cyst wall textures.
    """
    if image is None:
        raise ValueError("Input image is None")
        
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, None, h=h, templateWindowSize=template_window_size, searchWindowSize=search_window_size)
    elif len(image.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(image, None, h=h, hColor=h, templateWindowSize=template_window_size, searchWindowSize=search_window_size)
    return image
