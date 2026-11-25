"""
SonoAssist AI - Comprehensive Pipeline Verification Script
Tests all 8 modules offline end-to-end:
1. Database initialization and CRUD
2. Image enhancement & quality metrics
3. Attention U-Net model forward pass
4. Morphological feature extraction
5. Classification forward pass
6. Clinical PDF report compilation
7. Video processing with simulated cine-loop
"""

import os
import sys
import numpy as np
import cv2

# Add SonoAssistAI to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from storage.database import init_db, insert_analysis, get_all_analyses
from preprocessing.enhancement import enhance_ultrasound_image, compute_image_quality_metrics
from models.attention_unet import AttentionUNet
from detection.segmentation import SegmentationEngine
from detection.classification import ClassificationEngine
from detection.feature_extraction import extract_morphological_features
from reports.generate_report import generate_pdf_report
from preprocessing.video_processing import process_ultrasound_video


def run_all_tests():
    print("=" * 60)
    print("STARTING SONOASSIST AI END-TO-END VERIFICATION")
    print("=" * 60)

    # 1. Test Database
    print("\n[TEST 1] Testing SQLite Database...")
    init_db()
    test_record = {
        "scan_id": "TEST-SCAN-001",
        "patient_id": "PT-TEST",
        "predicted_class": "Simple cyst",
        "confidence": 0.88,
        "area_pixels": 1250.0,
        "perimeter": 130.0,
        "circularity": 0.85,
        "aspect_ratio": 1.1,
        "solidity": 0.95,
        "model_mode": "TEST_MODE"
    }
    inserted = insert_analysis(test_record)
    assert inserted, "Failed to insert test analysis into database"
    records = get_all_analyses()
    assert len(records) > 0, "Database returned 0 records"
    print("  --> SQLite Database test PASSED.")

    # 2. Test Enhancement Pipeline
    print("\n[TEST 2] Testing Image Enhancement & Denoising...")
    sample_path = os.path.join(BASE_DIR, "dataset", "images", "sample_patient_01_ovary.png")
    if not os.path.exists(sample_path):
        from training.dataset import generate_sample_ultrasound_data
        generate_sample_ultrasound_data(os.path.join(BASE_DIR, "dataset"), 4)

    test_img = cv2.imread(sample_path)
    assert test_img is not None, "Failed to load sample ultrasound image"

    enhanced, meta = enhance_ultrasound_image(test_img, enable_enhancement=True)
    assert enhanced.shape == test_img.shape, "Enhanced shape does not match original"
    assert "metrics_original" in meta, "Missing metrics in enhancement output"
    print(f"  --> Original Contrast Std: {meta['metrics_original']['contrast_std']} -> Enhanced: {meta['metrics_enhanced']['contrast_std']}")
    print("  --> Image Enhancement test PASSED.")

    # 3. Test Attention U-Net & Segmentation Engine
    print("\n[TEST 3] Testing Attention U-Net Model & Segmentation Engine...")
    seg_engine = SegmentationEngine()
    seg_result = seg_engine.predict(test_img, use_tta=True)
    assert "mask" in seg_result, "Missing mask in segmentation result"
    assert "confidence" in seg_result, "Missing confidence in segmentation result"
    assert "morphology" in seg_result, "Missing morphology in segmentation result"
    print(f"  --> Segmentation Mode: {seg_result['model_mode']} | Confidence: {seg_result['confidence']*100:.1f}%")
    print("  --> Segmentation Engine test PASSED.")

    # 4. Test Morphological Feature Extraction
    print("\n[TEST 4] Testing Morphological Feature Extraction...")
    mask = seg_result["mask"]
    morph = extract_morphological_features(mask, original_gray=cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY))
    assert "has_detection" in morph, "Missing detection status"
    if morph["has_detection"]:
        p = morph["primary_cyst"]
        print(f"  --> Area: {p['area_pixels']} px, Circularity: {p['circularity']}, Solidity: {p['solidity']}")
    print("  --> Morphological Feature Extraction test PASSED.")

    # 5. Test Classification Engine
    print("\n[TEST 5] Testing Classification Engine...")
    clf_engine = ClassificationEngine()
    clf_result = clf_engine.predict(test_img, morphological_context=morph)
    assert "predicted_category" in clf_result, "Missing predicted_category"
    assert "class_probabilities" in clf_result, "Missing class_probabilities"
    print(f"  --> Predicted Category: {clf_result['predicted_category']} (Confidence: {clf_result['confidence']*100:.1f}%)")
    print("  --> Classification Engine test PASSED.")

    # 6. Test PDF Report Generation
    print("\n[TEST 6] Testing Clinical PDF Report Generation (ReportLab)...")
    out_pdf = os.path.join(BASE_DIR, "outputs", "reports", "TEST_REPORT.pdf")
    temp_orig = os.path.join(BASE_DIR, "outputs", "enhanced", "temp_orig.png")
    temp_enh = os.path.join(BASE_DIR, "outputs", "enhanced", "temp_enh.png")
    temp_overlay = os.path.join(BASE_DIR, "outputs", "segmented", "temp_overlay.png")

    cv2.imwrite(temp_orig, test_img)
    cv2.imwrite(temp_enh, enhanced)
    cv2.imwrite(temp_overlay, enhanced)

    pdf_path = generate_pdf_report(
        scan_id="VERIFY-001",
        patient_id="PT-TEST",
        timestamp="2026-11-25 10:00:00",
        original_img_path=temp_orig,
        enhanced_img_path=temp_enh,
        overlay_img_path=temp_overlay,
        predicted_class=clf_result["predicted_category"],
        confidence=seg_result["confidence"],
        morph_data=morph,
        warnings=seg_result.get("warnings", []),
        output_pdf_path=out_pdf,
        radiologist_notes="Verification test note.",
        radiologist_name="Dr. Test, MD",
        model_mode="TEST_DEMO"
    )
    assert os.path.exists(pdf_path), "PDF file was not created"
    assert os.path.getsize(pdf_path) > 1000, "PDF file is suspiciously small"
    print(f"  --> PDF generated successfully: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
    print("  --> PDF Report Generation test PASSED.")

    # 7. Test Video Processing Cine-Loop Simulation
    print("\n[TEST 7] Testing Video Cine-Loop Processing...")
    test_video_path = os.path.join(BASE_DIR, "outputs", "videos", "synthetic_test_cineloop.mp4")
    out_video_path = os.path.join(BASE_DIR, "outputs", "videos", "synthetic_test_annotated.mp4")
    os.makedirs(os.path.dirname(test_video_path), exist_ok=True)

    # Generate a brief 10-frame synthetic ultrasound video
    writer = cv2.VideoWriter(test_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (256, 256))
    for f in range(10):
        # Slightly jitter image to simulate probe motion
        M = np.float32([[1, 0, np.random.randint(-2, 3)], [0, 1, np.random.randint(-2, 3)]])
        jittered = cv2.warpAffine(test_img, M, (256, 256))
        writer.write(jittered)
    writer.release()

    vid_stats = process_ultrasound_video(
        video_path=test_video_path,
        output_path=out_video_path,
        segmentation_engine=seg_engine,
        sample_rate=1,
        max_frames=10
    )
    assert os.path.exists(out_video_path), "Output annotated video was not created"
    print(f"  --> Processed {vid_stats['frames_processed']} frames. Output video: {out_video_path}")
    print("  --> Video Processing test PASSED.")

    print("\n" + "=" * 60)
    print("ALL MODULE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
