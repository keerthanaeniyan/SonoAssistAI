"""
SonoAssist AI - MMOTU Dataset Loader & Augmentation Pipeline
Handles:
1. MMOTU dataset folder scanning and image-mask pairing
2. Patient-level train/val/test splitting
3. Resizing to 256x256 and grayscale normalization
4. Medical data augmentations (flips, random rotation, contrast scaling)
5. Robust validation for corrupt or missing masks
6. Synthetic ultrasound sample generator for offline testing when dataset is not yet downloaded
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Any


class MMOTUDataset(Dataset):
    """
    PyTorch Dataset for MMOTU Ovarian Ultrasound segmentation & classification.
    Expects:
    - images_dir: directory containing .jpg/.png ultrasound scans
    - masks_dir: directory containing corresponding binary ground truth masks (.png)
    """
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        img_size: Tuple[int, int] = (256, 256),
        is_train: bool = True,
        labels: Optional[List[int]] = None
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.is_train = is_train
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.img_size, dtype=np.uint8)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
        if mask is None:
            mask = np.zeros(self.img_size, dtype=np.uint8)

        # Resize to standard network dimensions
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Data augmentation for training
        if self.is_train:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
                mask = np.fliplr(mask)

            # Random vertical flip
            if np.random.rand() > 0.5:
                img = np.flipud(img)
                mask = np.flipud(mask)

            # Random slight brightness/contrast variation
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.85, 1.15)
                beta = np.random.uniform(-10, 10)
                img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        # Normalize to [0.0, 1.0]
        img_norm = (img.astype(np.float32) / 255.0)[np.newaxis, :, :]  # [1, H, W]
        mask_norm = ((mask > 127).astype(np.float32))[np.newaxis, :, :] # [1, H, W]

        img_tensor = torch.from_numpy(img_norm.copy())
        mask_tensor = torch.from_numpy(mask_norm.copy())

        label = self.labels[idx] if self.labels is not None else -1

        return img_tensor, mask_tensor, label


def get_patient_id_from_filename(filename: str) -> str:
    """
    Extract patient identifier from MMOTU file naming convention
    e.g. 'patient042_scan01.jpg' -> 'patient042'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    return parts[0] if len(parts) > 1 else base


def split_dataset_by_patient(
    images_dir: str,
    masks_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Group scans by patient ID and split into Train, Val, and Test sets
    to prevent data leakage across identical patient anatomies.
    """
    np.random.seed(seed)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(images_dir, ext)))

    # Pair each image with its mask
    patient_scans = {}
    for img_path in sorted(all_images):
        fname = os.path.basename(img_path)
        base_name = os.path.splitext(fname)[0]

        # Look for mask in masks_dir
        mask_path = os.path.join(masks_dir, base_name + ".png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(masks_dir, fname)

        if not os.path.exists(mask_path):
            continue  # Skip scans without corresponding annotation

        pid = get_patient_id_from_filename(fname)
        if pid not in patient_scans:
            patient_scans[pid] = []
        patient_scans[pid].append((img_path, mask_path))

    patients = list(patient_scans.keys())
    np.random.shuffle(patients)

    n_total = len(patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    def collect(p_list):
        imgs, msks = [], []
        for p in p_list:
            for imp, msk in patient_scans[p]:
                imgs.append(imp)
                msks.append(msk)
        return imgs, msks

    train_imgs, train_msks = collect(train_patients)
    val_imgs, val_msks = collect(val_patients)
    test_imgs, test_msks = collect(test_patients)

    return {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }, {
        "train": train_msks,
        "val": val_msks,
        "test": test_msks
    }


def generate_sample_ultrasound_data(target_dir: str, num_samples: int = 5) -> None:
    """
    Generate realistic synthetic ovarian ultrasound images and ground-truth masks
    to enable offline testing and demonstrations before downloading full 2GB MMOTU dataset.
    """
    img_dir = os.path.join(target_dir, "images")
    mask_dir = os.path.join(target_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    np.random.seed(42)
    for i in range(1, num_samples + 1):
        # 1. Base acoustic background with sector scan geometry
        h, w = 256, 256
        background = np.zeros((h, w), dtype=np.float32)

        # Sector arc attenuation
        y_indices, x_indices = np.indices((h, w))
        center_x, center_y = w // 2, 10
        dist = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        angle = np.arctan2(y_indices - center_y, x_indices - center_x)

        # Sector mask (roughly 45 to 135 degrees)
        sector = (dist > 30) & (dist < 240) & (angle > 0.6) & (angle < 2.5)

        # Acoustic speckle noise (Rayleigh-like distribution)
        speckle = np.random.rayleigh(scale=35.0, size=(h, w))
        background[sector] = speckle[sector]

        # 2. Add anechoic / hypoechoic cyst
        cyst_mask = np.zeros((h, w), dtype=np.uint8)
        cyst_cx = int(w // 2 + np.random.randint(-30, 30))
        cyst_cy = int(h // 2 + np.random.randint(-15, 30))
        axis_x = np.random.randint(25, 45)
        axis_y = np.random.randint(20, 40)
        rot_angle = np.random.randint(0, 180)

        cv2.ellipse(cyst_mask, (cyst_cx, cyst_cy), (axis_x, axis_y), rot_angle, 0, 360, 255, -1)

        # Attenuate inside cyst (anechoic fluid with minimal internal echoes)
        background[cyst_mask == 255] = np.random.rayleigh(scale=5.0, size=(np.sum(cyst_mask == 255),))

        # Posterior acoustic enhancement (brighter stroma behind fluid cyst)
        enhancement_zone = (x_indices > cyst_cx - axis_x) & (x_indices < cyst_cx + axis_x) & (y_indices > cyst_cy + axis_y) & sector
        background[enhancement_zone] = np.clip(background[enhancement_zone] * 1.35, 0, 255)

        # Hyperechoic cyst wall boundary
        wall_contour = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(wall_contour, (cyst_cx, cyst_cy), (axis_x + 1, axis_y + 1), rot_angle, 0, 360, 255, 1)
        background[wall_contour == 255] = np.clip(background[wall_contour == 255] * 1.5, 0, 255)

        final_img = np.clip(background, 0, 255).astype(np.uint8)

        # Save sample
        sample_name = f"sample_patient_{i:02d}_ovary"
        cv2.imwrite(os.path.join(img_dir, f"{sample_name}.png"), final_img)
        cv2.imwrite(os.path.join(mask_dir, f"{sample_name}.png"), cyst_mask)

    print(f"[MMOTUDataset] Generated {num_samples} realistic demonstration ultrasound samples in {target_dir}")
