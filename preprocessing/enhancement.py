"""
SonoAssist AI - Ultrasound Image Enhancement Pipeline
Performs multi-stage enhancement:
1. Format validation & grayscale normalization
2. Speckle noise attenuation (Bilateral / NLM / Median)
3. Contrast-Limited Adaptive Histogram Equalization (CLAHE)
4. Edge-aware gentle unsharp masking
5. Metric computation (contrast, sharpness, SNR estimation)
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from preprocessing.denoising import reduce_speckle_noise, apply_median_filter, apply_nlm_denoising


def load_ultrasound_image(source: Any) -> np.ndarray:
    """
    Safely load an ultrasound image from a file path, byte buffer, or existing array.
    Returns uint8 BGR image.
    """
    if source is None:
        raise ValueError("Image source is None.")

    if isinstance(source, np.ndarray):
        img = source.copy()
    elif isinstance(source, str):
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Could not load image from path: {source}")
    elif hasattr(source, "read"):
        file_bytes = np.asarray(bytearray(source.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from provided byte stream.")
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def apply_clahe(gray_img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance
    contrast in hypoechoic cyst interiors and echogenic stroma without over-amplifying noise.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_img)


def apply_unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.2) -> np.ndarray:
    """
    Sharpen cyst boundaries using unsharp masking:
    enhanced = original + strength * (original - gaussian_blurred)
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def compute_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Compute non-reference image quality metrics:
    - Mean intensity
    - Standard deviation (contrast indicator)
    - Tenengrad sharpness (Sobel gradient magnitude)
    - Estimated signal-to-noise ratio (SNR) in dB
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Tenengrad gradient sharpness
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sharpness = float(np.mean(gx**2 + gy**2))

    # Approximate SNR: 20 * log10(mean / (std + 1e-6))
    snr = 20 * np.log10((mean_val + 1e-5) / (std_val + 1e-5)) if std_val > 0 else 0.0

    return {
        "mean_intensity": round(mean_val, 2),
        "contrast_std": round(std_val, 2),
        "sharpness_tenengrad": round(sharpness, 2),
        "estimated_snr_db": round(float(snr), 2)
    }


def enhance_ultrasound_image(
    image: np.ndarray,
    enable_enhancement: bool = True,
    denoise_method: str = "bilateral",  # 'bilateral', 'nlm', 'median', 'none'
    bilateral_d: int = 7,
    bilateral_sigma: float = 45.0,
    nlm_h: float = 8.0,
    median_k: int = 3,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    sharpen_strength: float = 0.8
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full ultrasound enhancement pipeline.
    Preserves original image without overwriting.
    """
    if not enable_enhancement:
        metrics_orig = compute_image_quality_metrics(image)
        return image.copy(), {
            "enabled": False,
            "metrics_original": metrics_orig,
            "metrics_enhanced": metrics_orig,
            "notes": "Enhancement disabled by user request."
        }

    # Convert to grayscale for clinical ultrasound processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    orig_gray = gray.copy()
    metrics_original = compute_image_quality_metrics(orig_gray)

    # 1. Speckle / noise reduction
    if denoise_method == "bilateral":
        denoised = reduce_speckle_noise(gray, diameter=bilateral_d, sigma_color=bilateral_sigma, sigma_space=bilateral_sigma)
    elif denoise_method == "nlm":
        denoised = apply_nlm_denoising(gray, h=nlm_h)
    elif denoise_method == "median":
        denoised = apply_median_filter(gray, kernel_size=median_k)
    else:
        denoised = gray

    # 2. Contrast Enhancement via CLAHE
    if clahe_clip_limit > 0:
        enhanced = apply_clahe(denoised, clip_limit=clahe_clip_limit, tile_grid_size=(clahe_tile_size, clahe_tile_size))
    else:
        enhanced = denoised

    # 3. Controlled boundary sharpening
    if sharpen_strength > 0:
        sharpened = apply_unsharp_mask(enhanced, sigma=1.0, strength=sharpen_strength)
    else:
        sharpened = enhanced

    # Reconstruct 3-channel BGR for consistent downstream visualization & models
    enhanced_bgr = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    metrics_enhanced = compute_image_quality_metrics(sharpened)

    return enhanced_bgr, {
        "enabled": True,
        "denoise_method": denoise_method,
        "clahe_clip_limit": clahe_clip_limit,
        "sharpen_strength": sharpen_strength,
        "metrics_original": metrics_original,
        "metrics_enhanced": metrics_enhanced
    }
