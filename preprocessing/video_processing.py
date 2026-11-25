"""
SonoAssist AI - Ultrasound Video Processing Module
Provides streaming, memory-efficient video processing for ovarian ultrasound cine-loops:
- Frame-by-frame processing without loading entire video into RAM
- Speckle reduction and CLAHE enhancement per frame
- Segmentation inference with temporal smoothing between consecutive frames
- High-contrast visual overlays
- Progress callback support for Streamlit UI
- Export to MP4/AVI for clinical review
"""

import os
import cv2
import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple
from preprocessing.enhancement import enhance_ultrasound_image
from visualization.annotations import create_segmentation_overlay, annotate_image_with_findings


def smooth_temporal_mask(current_mask: np.ndarray, previous_mask: Optional[np.ndarray], alpha: float = 0.7) -> np.ndarray:
    """
    Apply temporal exponential smoothing to reduce inter-frame boundary flickering
    in ultrasound video cine-loops.
    """
    if previous_mask is None or previous_mask.shape != current_mask.shape:
        return current_mask

    # Exponential weighted moving average
    smoothed = cv2.addWeighted(current_mask.astype(np.float32), alpha, previous_mask.astype(np.float32), 1.0 - alpha, 0)
    return np.clip(smoothed, 0, 255).astype(np.uint8)


def process_ultrasound_video(
    video_path: str,
    output_path: str,
    segmentation_engine: Any,
    apply_enhancement: bool = True,
    sample_rate: int = 1,  # process every Nth frame (1 = every frame)
    max_frames: Optional[int] = 300,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Stream and process an ovarian ultrasound video file.
    Writes the annotated video to output_path.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fallback bounds
    if total_frames <= 0:
        total_frames = 100
    if max_frames:
        total_frames = min(total_frames, max_frames)

    # Video Writer setup - Use mp4v or XVID
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height + 42))

    frame_idx = 0
    processed_count = 0
    previous_mask = None
    cysts_detected_frames = 0

    try:
        while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % sample_rate != 0:
                continue

            # 1. Image Enhancement
            if apply_enhancement:
                enhanced_frame, _ = enhance_ultrasound_image(frame, enable_enhancement=True, clahe_clip_limit=1.8)
            else:
                enhanced_frame = frame.copy()

            # 2. Segmentation
            seg_result = segmentation_engine.predict(enhanced_frame, use_tta=False)
            curr_mask = seg_result["mask"]

            # 3. Temporal smoothing
            smooth_mask = smooth_temporal_mask(curr_mask, previous_mask, alpha=0.75)
            previous_mask = smooth_mask.copy()

            # 4. Check detection
            has_cyst = np.sum(smooth_mask > 127) > 100
            if has_cyst:
                cysts_detected_frames += 1

            # 5. Overlay and Annotation
            annotated_frame = annotate_image_with_findings(
                enhanced_frame,
                smooth_mask,
                morph_features=seg_result.get("morphology"),
                predicted_class="Cyst Region" if has_cyst else "No Focal Lesion",
                confidence=seg_result.get("confidence"),
                model_mode=seg_result.get("model_mode", "DEMO")
            )

            # Ensure dimensions match writer
            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != (height + 42):
                annotated_frame = cv2.resize(annotated_frame, (width, height + 42))

            out_writer.write(annotated_frame)
            processed_count += 1

            if progress_callback:
                progress_callback(frame_idx, total_frames, f"Processing frame {frame_idx}/{total_frames}")

    finally:
        cap.release()
        out_writer.release()

    return {
        "total_frames_read": frame_idx,
        "frames_processed": processed_count,
        "cysts_detected_frames": cysts_detected_frames,
        "detection_rate_pct": round((cysts_detected_frames / max(1, processed_count)) * 100, 1),
        "output_path": output_path
    }
