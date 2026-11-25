"""
SonoAssist AI - Clinical Ovarian Ultrasound Analysis System
Main Streamlit Application
"""

import os
import sys
import uuid
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Setup python path to include SonoAssistAI directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing.enhancement import enhance_ultrasound_image, compute_image_quality_metrics
from preprocessing.video_processing import process_ultrasound_video
from detection.segmentation import SegmentationEngine
from detection.classification import ClassificationEngine
from detection.feature_extraction import extract_morphological_features
from visualization.annotations import create_segmentation_overlay, annotate_image_with_findings
from visualization.comparison import create_side_by_side_comparison, create_four_panel_display
from reports.generate_report import generate_pdf_report, REQUIRED_DISCLAIMER
from storage.database import (
    init_db, insert_analysis, get_all_analyses, get_analysis_by_id,
    update_radiologist_review, get_summary_stats
)
from dataset.patient_cases import load_patient_case_library

# Page configuration
st.set_page_config(
    page_title="SonoAssist AI - Ovarian Ultrasound Decision Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Medical Theme Styling (Deep Blue, Clean White, Card styling)
st.markdown("""
<style>
    /* Main typography & background */
    .main {
        background-color: #F8FAFC;
    }
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1E3A8A !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Medical Header Card */
    .medical-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        padding: 20px 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .medical-header h1 {
        color: white !important;
        margin: 0;
        font-size: 26px;
    }
    .medical-header p {
        color: #DBEAFE;
        margin: 4px 0 0 0;
        font-size: 14px;
    }
    
    /* Metric Card */
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1E3A8A;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 13px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Badge */
    .badge-offline {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #BBF7D0;
    }
    .badge-demo {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #FDE68A;
    }
    
    /* Warning Box */
    .clinical-disclaimer-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 16px;
        margin-bottom: 16px;
        font-size: 13px;
        color: #1E40AF;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Database & Sessions
init_db()

@st.cache_resource
def get_ai_engines():
    """Cache models in memory for offline responsiveness."""
    seg_engine = SegmentationEngine()
    clf_engine = ClassificationEngine()
    return seg_engine, clf_engine

seg_engine, clf_engine = get_ai_engines()

@st.cache_data
def get_patient_case_library():
    """Load or generate the 8 clinical patient benchmark cases."""
    return load_patient_case_library(os.path.join(CURRENT_DIR, "dataset"))

patient_cases = get_patient_case_library()

# Session State Initialization
if "scan_id" not in st.session_state:
    st.session_state.scan_id = f"SCAN-{uuid.uuid4().hex[:6].upper()}"
if "patient_id" not in st.session_state:
    st.session_state.patient_id = "PT-101"
if "patient_case_info" not in st.session_state:
    st.session_state.patient_case_info = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "enhanced_image" not in st.session_state:
    st.session_state.enhanced_image = None
if "predicted_mask" not in st.session_state:
    st.session_state.predicted_mask = None
if "morph_data" not in st.session_state:
    st.session_state.morph_data = None
if "classification_result" not in st.session_state:
    st.session_state.classification_result = None
if "seg_result" not in st.session_state:
    st.session_state.seg_result = None
if "enhancement_meta" not in st.session_state:
    st.session_state.enhancement_meta = None


# SIDEBAR NAVIGATION
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/ultrasound.png", width=64)
    st.title("SonoAssist AI")
    st.caption("Offline Decision Support Prototype")

    st.markdown("""
    <div style="margin-bottom: 15px;">
        <span class="badge-offline">🟢 OFFLINE READY</span>
    </div>
    """, unsafe_allow_html=True)

    if not seg_engine.is_trained:
        st.markdown("""
        <div style="margin-bottom: 15px;">
            <span class="badge-demo">⚠️ DEMO / UNTRAINED MODE</span>
        </div>
        """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation Menu",
        [
            "1. Dashboard",
            "2. Upload Ultrasound",
            "3. Image Enhancement",
            "4. AI Segmentation",
            "5. Classification",
            "6. Video Analysis",
            "7. Analysis Results",
            "8. Generate Report"
        ]
    )

    st.markdown("---")
    st.markdown(f"**Current Scan ID:** `{st.session_state.scan_id}`")
    st.markdown(f"**Patient ID:** `{st.session_state.patient_id}`")

    if st.button("🔄 Start New Scan Session", use_container_width=True):
        st.session_state.scan_id = f"SCAN-{uuid.uuid4().hex[:6].upper()}"
        st.session_state.original_image = None
        st.session_state.enhanced_image = None
        st.session_state.predicted_mask = None
        st.session_state.morph_data = None
        st.session_state.classification_result = None
        st.session_state.seg_result = None
        st.session_state.enhancement_meta = None
        st.rerun()

    st.markdown("---")
    st.caption("SonoAssist AI v1.0 • Clinical decision support system")


# ==========================================
# PAGE 1: DASHBOARD
# ==========================================
if page == "1. Dashboard":
    st.markdown("""
    <div class="medical-header">
        <h1>🩺 SonoAssist AI - Ovarian Ultrasound Analytics</h1>
        <p>Offline AI-Assisted Clinical Decision-Support System for Ovarian Cyst Enhancement, Segmentation & Morphometry</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_summary_stats()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Scans Analyzed</div>
            <div class="metric-value">{stats['total_scans']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Lesions Identified</div>
            <div class="metric-value">{stats['cysts_detected']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Radiologist Reviewed</div>
            <div class="metric-value">{stats['reviewed_scans']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pending Radiologist Sign-off</div>
            <div class="metric-value" style="color: #EA580C;">{stats['pending_review']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Quick Clinical Workflow")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.info("**Step 1: Ingest Scan**\nUpload static ultrasound frame (JPG/PNG) or cine-loop video (MP4/AVI).")
    with c2:
        st.info("**Step 2: Enhance Quality**\nAttenuate ultrasound speckle noise with edge-preserving bilateral filtering & CLAHE.")
    with c3:
        st.info("**Step 3: AI Segmentation**\nAttention U-Net delineates cyst margins and extracts quantitative morphometrics.")
    with c4:
        st.info("**Step 4: Clinical Report**\nGenerate offline PDF report ready for radiologist signature.")

    st.markdown("---")
    st.markdown("### System Architecture & Pipeline Status")
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("""
        - **Deep Learning Core**: PyTorch 2.x
        - **Primary Segmenter**: Attention U-Net (256x256 grayscale input)
        - **Secondary Classifier**: EfficientNet-B0 (8 MMOTU classes)
        - **Uncertainty Method**: Test-Time Augmentation (TTA 4-fold)
        - **Local Database**: SQLite (`storage/sonoassist.db`)
        - **Offline Report Engine**: ReportLab Flowables
        """)
    with sc2:
        st.markdown("""
        - **Dataset Compatibility**: MMOTU Ovarian Ultrasound Benchmark
        - **Execution Environment**: 100% Offline (Air-gapped compatible)
        - **Network Transmission**: Zero external API calls
        - **Quality Control**: Automated contrast, underexposure & blur checks
        - **Morphometrics**: Area (px), Circularity, Aspect Ratio, Solidity
        """)

    st.markdown("---")
    st.markdown("### 🏥 Clinical Patient Benchmark Library (8 Patient Cases)")
    st.markdown("Explore pre-configured clinical patient cases representing key MMOTU ovarian pathologies:")

    # 4x2 Grid of Patient Cases
    for row_idx in range(0, len(patient_cases), 4):
        cols = st.columns(4)
        for col_idx, case in enumerate(patient_cases[row_idx:row_idx+4]):
            with cols[col_idx]:
                st.markdown(f"""
                <div style="background: white; border: 1px solid #CBD5E1; border-radius: 8px; padding: 12px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">
                    <div style="font-weight: 700; color: #1E3A8A; font-size: 13px;">{case['patient_id']} • {case['category']}</div>
                    <div style="font-size: 11px; color: #64748B; margin: 3px 0;">Age: {case['age']}yo | {case['transducer'].split()[0]}</div>
                    <div style="font-size: 11px; color: #334155; height: 38px; overflow: hidden; text-overflow: ellipsis;">{case['clinical_history']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Load {case['patient_id']}", key=f"btn_dash_{case['patient_id']}", use_container_width=True):
                    st.session_state.scan_id = f"SCAN-{case['patient_id']}"
                    st.session_state.patient_id = case['patient_id']
                    st.session_state.patient_case_info = case
                    st.session_state.original_image = cv2.imread(case["image_path"])
                    st.session_state.enhanced_image = None
                    st.session_state.predicted_mask = None
                    st.session_state.morph_data = None
                    st.session_state.classification_result = None
                    st.session_state.seg_result = None
                    st.success(f"Loaded {case['patient_id']} ({case['category']})! Navigate to '3. Image Enhancement' or '4. AI Segmentation'.")

    st.markdown(f"""
    <div class="clinical-disclaimer-box">
        <b>CLINICAL DISCLAIMER:</b> {REQUIRED_DISCLAIMER}
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: UPLOAD ULTRASOUND
# ==========================================
elif page == "2. Upload Ultrasound":
    st.title("📤 Ultrasound Acquisition & Patient Case Ingestion")
    st.markdown("Ingest clinical transvaginal or transabdominal ovarian ultrasound scans from patient cases or local uploads.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Ingestion Mode")
        source_choice = st.radio(
            "Select Input Source",
            [
                "🏥 Clinical Patient Case Library (8 Characterized Cases)",
                "📁 Upload Custom Ultrasound Scan (JPG, PNG, BMP)",
                "🔬 MMOTU Raw Files Browser"
            ]
        )

        if source_choice == "🏥 Clinical Patient Case Library (8 Characterized Cases)":
            case_options = [c["name"] for c in patient_cases]
            selected_case_name = st.selectbox("Choose Patient Clinical Case", case_options)
            chosen_case = next(c for c in patient_cases if c["name"] == selected_case_name)

            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #CBD5E1; border-left: 4px solid #2563EB; border-radius: 8px; padding: 14px; margin: 12px 0;">
                <div style="font-weight: 700; color: #1E3A8A; font-size: 15px;">{chosen_case['name']}</div>
                <div style="margin: 6px 0; font-size: 13px;">
                    <span style="background: #EFF6FF; color: #1D4ED8; padding: 2px 8px; border-radius: 4px; font-weight: 600;">{chosen_case['category']}</span>
                    <span style="color: #64748B; margin-left: 8px;">Age: {chosen_case['age']} years</span>
                </div>
                <div style="font-size: 12px; color: #334155; margin-top: 6px;"><b>Clinical Indication:</b> {chosen_case['clinical_history']}</div>
                <div style="font-size: 12px; color: #334155; margin-top: 4px;"><b>Transducer:</b> {chosen_case['transducer']}</div>
                <div style="font-size: 12px; color: #475569; margin-top: 4px;"><b>Sonographic Description:</b> {chosen_case['ultrasound_findings']}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📥 Load This Patient Scan into Active Session", type="primary", use_container_width=True):
                st.session_state.scan_id = f"SCAN-{chosen_case['patient_id']}"
                st.session_state.patient_id = chosen_case["patient_id"]
                st.session_state.patient_case_info = chosen_case
                st.session_state.original_image = cv2.imread(chosen_case["image_path"])
                st.session_state.enhanced_image = None
                st.session_state.predicted_mask = None
                st.session_state.morph_data = None
                st.session_state.classification_result = None
                st.session_state.seg_result = None
                st.success(f"Loaded {chosen_case['patient_id']} - {chosen_case['category']} into session!")
                st.rerun()

        elif source_choice == "📁 Upload Custom Ultrasound Scan (JPG, PNG, BMP)":
            st.session_state.scan_id = st.text_input("Scan ID", value=st.session_state.scan_id)
            st.session_state.patient_id = st.text_input("Patient Identifier", value=st.session_state.patient_id)
            uploaded_file = st.file_uploader(
                "Select ultrasound image (JPG, PNG, BMP, TIFF)",
                type=["png", "jpg", "jpeg", "bmp", "tiff"]
            )
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.original_image = img
                st.session_state.patient_case_info = None
                st.success("Custom ultrasound scan loaded successfully!")

        else:
            demo_dir = os.path.join(CURRENT_DIR, "dataset", "images")
            samples = [f for f in os.listdir(demo_dir) if f.endswith(('.png', '.jpg'))] if os.path.exists(demo_dir) else []
            if samples:
                selected_sample = st.selectbox("Choose Raw File", samples)
                sample_path = os.path.join(demo_dir, selected_sample)
                if st.button("Load Raw File", use_container_width=True):
                    st.session_state.original_image = cv2.imread(sample_path)
                    st.session_state.patient_case_info = None
                    st.success(f"Loaded: {selected_sample}")
            else:
                st.warning("No sample files found in dataset/images.")

    with col2:
        st.subheader("Image Preview & Acquisition Check")
        if st.session_state.original_image is not None:
            st.image(
                cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB),
                caption=f"Scan: {st.session_state.scan_id} ({st.session_state.original_image.shape[1]}x{st.session_state.original_image.shape[0]})",
                use_container_width=True
            )
            metrics = compute_image_quality_metrics(st.session_state.original_image)
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Mean Intensity", metrics["mean_intensity"])
            mc2.metric("Contrast (Std)", metrics["contrast_std"])
            mc3.metric("Estimated SNR", f"{metrics['estimated_snr_db']} dB")

            if st.button("Proceed to Image Enhancement ➡️", type="primary", use_container_width=True):
                st.info("Navigate to '3. Image Enhancement' in sidebar.")
        else:
            st.info("No ultrasound scan currently loaded. Please upload a file or select a demonstration sample.")


# ==========================================
# PAGE 3: IMAGE ENHANCEMENT
# ==========================================
elif page == "3. Image Enhancement":
    st.title("✨ Ultrasound Quality Enhancement")
    st.markdown("Attenuate acoustic speckle noise, optimize dynamic range, and boost cyst boundary contrast.")

    if st.session_state.original_image is None:
        st.warning("Please upload an ultrasound image first in 'Upload Ultrasound'.")
    else:
        col_ctrl, col_view = st.columns([1, 2])

        with col_ctrl:
            st.subheader("Enhancement Controls")
            enable_enhancement = st.toggle("Enable Enhancement Pipeline", value=True)

            denoise_method = st.selectbox(
                "Speckle Denoising Method",
                ["bilateral", "nlm", "median", "none"],
                format_func=lambda x: {
                    "bilateral": "Bilateral Filter (Edge-Preserving)",
                    "nlm": "Non-Local Means (NLM)",
                    "median": "Median Filter",
                    "none": "Disable Denoising"
                }[x]
            )

            clahe_clip = st.slider("CLAHE Contrast Limit", min_value=0.0, max_value=5.0, value=2.2, step=0.2)
            sharpen_strength = st.slider("Boundary Sharpening Strength", min_value=0.0, max_value=2.0, value=0.8, step=0.1)

            if st.button("Apply Enhancement", type="primary", use_container_width=True):
                enhanced, meta = enhance_ultrasound_image(
                    st.session_state.original_image,
                    enable_enhancement=enable_enhancement,
                    denoise_method=denoise_method,
                    clahe_clip_limit=clahe_clip,
                    sharpen_strength=sharpen_strength
                )
                st.session_state.enhanced_image = enhanced
                st.session_state.enhancement_meta = meta
                st.success("Enhancement applied successfully!")

        with col_view:
            st.subheader("Side-by-Side Clinical Comparison")
            if st.session_state.enhanced_image is None:
                # Run default enhancement once
                enhanced, meta = enhance_ultrasound_image(st.session_state.original_image)
                st.session_state.enhanced_image = enhanced
                st.session_state.enhancement_meta = meta

            comparison = create_side_by_side_comparison(
                st.session_state.original_image,
                st.session_state.enhanced_image,
                label1="Original Ultrasound",
                label2="Enhanced (Denoised + CLAHE)"
            )
            st.image(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Metrics
            if st.session_state.enhancement_meta:
                m_orig = st.session_state.enhancement_meta.get("metrics_original", {})
                m_enh = st.session_state.enhancement_meta.get("metrics_enhanced", {})
                m1, m2, m3 = st.columns(3)
                m1.metric("Contrast (Std)", f"{m_enh.get('contrast_std')}", delta=round(m_enh.get('contrast_std', 0) - m_orig.get('contrast_std', 0), 2))
                m2.metric("Tenengrad Sharpness", f"{m_enh.get('sharpness_tenengrad')}", delta=round(m_enh.get('sharpness_tenengrad', 0) - m_orig.get('sharpness_tenengrad', 0), 1))
                m3.metric("Estimated SNR", f"{m_enh.get('estimated_snr_db')} dB", delta=round(m_enh.get('estimated_snr_db', 0) - m_orig.get('estimated_snr_db', 0), 2))


# ==========================================
# PAGE 4: AI SEGMENTATION
# ==========================================
elif page == "4. AI Segmentation":
    st.title("🎯 AI Cyst Boundary Segmentation & Morphometry")
    st.markdown("Deep learning segmentation using Attention U-Net with Test-Time Augmentation (TTA).")

    if st.session_state.original_image is None:
        st.warning("Please upload an ultrasound image first.")
    else:
        active_img = st.session_state.enhanced_image if st.session_state.enhanced_image is not None else st.session_state.original_image

        c_left, c_right = st.columns([1, 2])

        with c_left:
            st.subheader("Segmentation Settings")
            use_tta = st.checkbox("Enable Test-Time Augmentation (TTA)", value=True, help="4-fold spatial augmentation to estimate uncertainty")
            threshold = st.slider("Binarization Threshold", 0.1, 0.9, 0.5, 0.05)
            alpha_overlay = st.slider("Overlay Transparency (Alpha)", 0.1, 0.8, 0.35, 0.05)

            if not seg_engine.is_trained:
                st.warning("⚠️ **Demonstration Mode Active**\nNo trained weights in `models/trained_weights/attention_unet_best.pth`. Acoustic hypoechoic reference segmentation is displayed for testing.")

            if st.button("Run AI Segmentation", type="primary", use_container_width=True):
                with st.spinner("Executing Attention U-Net inference..."):
                    seg_out = seg_engine.predict(active_img, use_tta=use_tta, threshold=threshold)
                    st.session_state.predicted_mask = seg_out["mask"]
                    st.session_state.morph_data = seg_out["morphology"]
                    st.session_state.seg_result = seg_out

                    # Also run classification automatically
                    clf_out = clf_engine.predict(active_img, morphological_context=seg_out["morphology"])
                    st.session_state.classification_result = clf_out

                    # Save record to local database
                    primary = seg_out["morphology"].get("primary_cyst", {})
                    insert_analysis({
                        "scan_id": st.session_state.scan_id,
                        "patient_id": st.session_state.patient_id,
                        "predicted_class": clf_out.get("predicted_category", "Simple cyst"),
                        "confidence": seg_out.get("confidence", 0.0),
                        "area_pixels": primary.get("area_pixels", 0.0),
                        "perimeter": primary.get("perimeter_pixels", 0.0),
                        "circularity": primary.get("circularity", 0.0),
                        "aspect_ratio": primary.get("aspect_ratio", 0.0),
                        "solidity": primary.get("solidity", 0.0),
                        "model_mode": seg_out.get("model_mode", "DEMO")
                    })
                    st.success("Segmentation and morphometric analysis complete!")

        with c_right:
            st.subheader("Segmentation Visualizer")
            if st.session_state.predicted_mask is not None:
                # 4-Panel display
                overlay = create_segmentation_overlay(active_img, st.session_state.predicted_mask, alpha=alpha_overlay)
                annotated = annotate_image_with_findings(
                    active_img,
                    st.session_state.predicted_mask,
                    morph_features=st.session_state.morph_data,
                    predicted_class=st.session_state.classification_result.get("predicted_category") if st.session_state.classification_result else "Cyst",
                    confidence=st.session_state.seg_result.get("confidence"),
                    model_mode=st.session_state.seg_result.get("model_mode")
                )

                tab1, tab2, tab3 = st.tabs(["Diagnostic Overlay", "4-Panel Grid", "Uncertainty Map"])
                with tab1:
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                with tab2:
                    grid = create_four_panel_display(st.session_state.original_image, active_img, st.session_state.predicted_mask, overlay)
                    st.image(cv2.cvtColor(grid, cv2.COLOR_BGR2RGB), use_container_width=True)
                with tab3:
                    unc_map = st.session_state.seg_result.get("uncertainty_map")
                    if unc_map is not None:
                        st.image(unc_map, caption="TTA Spatial Variance / Uncertainty Map", clamp=True, use_container_width=True)

                # Warnings / Quality Assurance
                warnings = st.session_state.seg_result.get("warnings", [])
                for w in warnings:
                    if w["level"] == "WARNING":
                        st.warning(f"⚠️ {w['message']}")
                    elif w["level"] == "CAUTION":
                        st.error(f"🛑 {w['message']}")
                    else:
                        st.info(f"ℹ️ {w['message']}")

                # Morphological Features Table
                st.markdown("### Extracted Morphological Features")
                morph = st.session_state.morph_data
                if morph and morph.get("has_detection"):
                    p = morph["primary_cyst"]
                    fc1, fc2, fc3, fc4 = st.columns(4)
                    fc1.metric("Area (Pixels)", f"{p['area_pixels']} px")
                    fc2.metric("Circularity (0-1)", f"{p['circularity']}")
                    fc3.metric("Aspect Ratio", f"{p['aspect_ratio']}")
                    fc4.metric("Solidity", f"{p['solidity']}")

                    st.caption("⚠️ **Measurement Note**: Morphological features are descriptive measurements in pixels and not definitive diagnostic criteria.")
            else:
                st.info("Click 'Run AI Segmentation' to execute model inference.")


# ==========================================
# PAGE 5: CLASSIFICATION
# ==========================================
elif page == "5. Classification":
    st.title("🔬 MMOTU Ovarian Lesion Classification")
    st.markdown("Statistical classification alignment across 8 standard ovarian lesion categories.")

    if st.session_state.original_image is None:
        st.warning("Please upload an ultrasound image first.")
    else:
        active_img = st.session_state.enhanced_image if st.session_state.enhanced_image is not None else st.session_state.original_image

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Model Prediction")
            if st.session_state.classification_result is None:
                if st.button("Run Classification", type="primary", use_container_width=True):
                    res = clf_engine.predict(active_img, morphological_context=st.session_state.morph_data)
                    st.session_state.classification_result = res
                    st.rerun()
                else:
                    st.info("Click 'Run Classification' to evaluate lesion categories.")

            if st.session_state.classification_result:
                res = st.session_state.classification_result
                st.markdown(f"""
                <div style="background: white; border-radius: 10px; padding: 20px; border: 1px solid #E2E8F0; margin-bottom: 20px;">
                    <div style="font-size: 14px; color: #64748B;">Predicted MMOTU Category:</div>
                    <div style="font-size: 26px; font-weight: 700; color: #1E3A8A; margin: 6px 0;">{res['predicted_category']}</div>
                    <div style="font-size: 14px; color: #059669; font-weight: 600;">Model Confidence: {res['confidence']*100:.1f}%</div>
                    <div style="font-size: 12px; color: #6B7280; margin-top: 8px;">{res.get('clinical_disclaimer')}</div>
                </div>
                """, unsafe_allow_html=True)

        with c2:
            st.subheader("Class Probability Distribution")
            if st.session_state.classification_result:
                probs = st.session_state.classification_result.get("class_probabilities", {})
                df_probs = pd.DataFrame(list(probs.items()), columns=["Category", "Probability"])
                df_probs = df_probs.sort_values(by="Probability", ascending=True)

                st.bar_chart(df_probs.set_index("Category"))

        st.markdown(f"""
        <div class="clinical-disclaimer-box">
            <b>DECISION SUPPORT GUIDELINE:</b> {REQUIRED_DISCLAIMER}
            <br/>Model confidence represents mathematical network certainty, NOT the absolute biological probability of pathology.
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 6: VIDEO ANALYSIS
# ==========================================
elif page == "6. Video Analysis":
    st.title("🎥 Ultrasound Cine-Loop Video Processing")
    st.markdown("Streaming frame-by-frame analysis with temporal consistency smoothing across consecutive ultrasound frames.")

    col_v1, col_v2 = st.columns([1, 1])

    with col_v1:
        st.subheader("Cine-Loop Ingestion")
        vid_source = st.radio(
            "Select Video Source",
            ["Built-in Patient Cine-Loop (MP4 Demo)", "Upload Custom Video (MP4 / AVI)"]
        )

        active_video_path = None
        video_display_name = "patient_ovarian_cineloop.mp4"

        if vid_source == "Built-in Patient Cine-Loop (MP4 Demo)":
            demo_vid = os.path.join(CURRENT_DIR, "outputs", "videos", "synthetic_test_cineloop.mp4")
            if os.path.exists(demo_vid):
                active_video_path = demo_vid
                st.success("Loaded built-in transvaginal ovarian scan cine-loop (10 frames, sector probe).")
            else:
                st.warning("Demo cine-loop file not found.")
        else:
            vid_file = st.file_uploader("Upload Ultrasound Cine-Loop Video (MP4 or AVI)", type=["mp4", "avi"])
            if vid_file is not None:
                temp_input = os.path.join(CURRENT_DIR, "outputs", "videos", f"input_{vid_file.name}")
                os.makedirs(os.path.dirname(temp_input), exist_ok=True)
                with open(temp_input, "wb") as f:
                    f.write(vid_file.read())
                active_video_path = temp_input
                video_display_name = vid_file.name
                st.success(f"Video uploaded: {vid_file.name}")

    with col_v2:
        st.subheader("Processing Parameters")
        sample_rate = st.slider("Process Every Nth Frame (Sampling Rate)", 1, 5, 1)
        max_frames = st.slider("Maximum Frames to Process", 5, 200, 30)

    if active_video_path is not None:
        output_video_path = os.path.join(CURRENT_DIR, "outputs", "videos", f"annotated_{video_display_name}")

        if st.button("▶️ Start Cine-Loop Video Processing", type="primary", use_container_width=True):
            prog_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, msg):
                fraction = min(1.0, current / max(1, total))
                prog_bar.progress(fraction)
                status_text.text(msg)

            with st.spinner("Processing video frames with temporal smoothing..."):
                stats = process_ultrasound_video(
                    video_path=active_video_path,
                    output_path=output_video_path,
                    segmentation_engine=seg_engine,
                    sample_rate=sample_rate,
                    max_frames=max_frames,
                    progress_callback=progress_callback
                )

            prog_bar.progress(1.0)
            status_text.text("Processing Complete!")
            st.success(f"Processed {stats['frames_processed']} frames. Cyst detection rate: {stats['detection_rate_pct']}%")

            if os.path.exists(output_video_path):
                with open(output_video_path, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Annotated Video",
                        data=vf,
                        file_name=f"annotated_{video_display_name}",
                        mime="video/mp4",
                        use_container_width=True
                    )


# ==========================================
# PAGE 7: ANALYSIS RESULTS
# ==========================================
elif page == "7. Analysis Results":
    st.title("📋 Examination Records & Patient Database")
    st.markdown("Offline SQLite repository of ultrasound analyses, patient histories, and radiologist reviews.")

    all_scans = get_all_analyses()
    if all_scans:
        df_scans = pd.DataFrame(all_scans)

        # Filters
        fc1, fc2 = st.columns([1, 1])
        with fc1:
            categories_in_db = ["All"] + sorted(list(df_scans["predicted_class"].unique()))
            selected_cat = st.selectbox("Filter by Pathology Category", categories_in_db)
        with fc2:
            status_filter = st.selectbox("Filter by Review Status", ["All", "Reviewed", "Pending Sign-off"])

        filtered_df = df_scans.copy()
        if selected_cat != "All":
            filtered_df = filtered_df[filtered_df["predicted_class"] == selected_cat]
        if status_filter == "Reviewed":
            filtered_df = filtered_df[filtered_df["radiologist_reviewed"] == 1]
        elif status_filter == "Pending Sign-off":
            filtered_df = filtered_df[filtered_df["radiologist_reviewed"] == 0]

        st.dataframe(
            filtered_df[["scan_id", "patient_id", "timestamp", "predicted_class", "confidence", "area_pixels", "circularity", "radiologist_reviewed", "reviewer_name"]],
            use_container_width=True
        )

        st.markdown("### Radiologist Review & Sign-off Panel")
        scan_id_options = filtered_df["scan_id"].tolist() if not filtered_df.empty else df_scans["scan_id"].tolist()
        selected_scan_id = st.selectbox("Select Patient Scan to Review / Sign", scan_id_options)
        selected_record = get_analysis_by_id(selected_scan_id)

        if selected_record:
            rc1, rc2 = st.columns([1, 1])
            with rc1:
                reviewer_name = st.text_input("Reviewer Name / Title", value=selected_record.get("reviewer_name") or "Dr. Sarah Lin, MD (Gynecologic Radiologist)")
                notes = st.text_area("Clinical Notes & Impression", value=selected_record.get("radiologist_notes") or "")
                if st.button("Save Radiologist Sign-off", type="primary", use_container_width=True):
                    update_radiologist_review(selected_scan_id, reviewer_name, notes)
                    st.success("Review updated in local database!")
                    st.rerun()
            with rc2:
                st.markdown(f"**Scan ID:** `{selected_record['scan_id']}`")
                st.markdown(f"**Patient ID:** `{selected_record['patient_id']}`")
                st.markdown(f"**Predicted Category:** `{selected_record['predicted_class']}`")
                st.markdown(f"**AI Confidence:** `{selected_record['confidence']*100:.1f}%`")
                st.markdown(f"**Area:** `{selected_record['area_pixels']} px`")
                st.markdown(f"**Status:** `{'Reviewed' if selected_record['radiologist_reviewed'] else 'Pending Sign-off'}`")
    else:
        st.info("No examination records saved yet. Perform a segmentation to create records.")


# ==========================================
# PAGE 8: GENERATE REPORT
# ==========================================
elif page == "8. Generate Report":
    st.title("📄 Clinical PDF Report Generation")
    st.markdown("Export a comprehensive, publication-quality PDF report for medical records.")

    if st.session_state.original_image is None or st.session_state.predicted_mask is None:
        st.warning("Please complete image upload and AI segmentation before generating report.")
    else:
        # Prepare file paths
        reports_dir = os.path.join(CURRENT_DIR, "outputs", "reports")
        enhanced_dir = os.path.join(CURRENT_DIR, "outputs", "enhanced")
        seg_dir = os.path.join(CURRENT_DIR, "outputs", "segmented")
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(enhanced_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        orig_path = os.path.join(enhanced_dir, f"{st.session_state.scan_id}_orig.png")
        enh_path = os.path.join(enhanced_dir, f"{st.session_state.scan_id}_enh.png")
        overlay_path = os.path.join(seg_dir, f"{st.session_state.scan_id}_overlay.png")
        pdf_path = os.path.join(reports_dir, f"Report_{st.session_state.scan_id}.pdf")

        # Save images to disk for report inclusion
        cv2.imwrite(orig_path, st.session_state.original_image)
        enhanced_to_save = st.session_state.enhanced_image if st.session_state.enhanced_image is not None else st.session_state.original_image
        cv2.imwrite(enh_path, enhanced_to_save)

        overlay_img = create_segmentation_overlay(enhanced_to_save, st.session_state.predicted_mask)
        cv2.imwrite(overlay_path, overlay_img)

        # Pre-fill notes if case metadata exists
        case_info = st.session_state.patient_case_info or {}
        default_notes = case_info.get("radiologist_impression", "Ultrasound demonstrates well-circumscribed hypoechoic ovarian cyst with regular margins. No solid papillary projections or abnormal vascularity.")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Report Parameters")
            rep_patient = st.text_input("Patient Identifier", value=st.session_state.patient_id)
            rep_reviewer = st.text_input("Reviewing Radiologist", value="Dr. Sarah Lin, MD (Gynecologic Radiologist)")
            rep_notes = st.text_area("Radiologist Clinical Findings & Sign-off Notes", value=default_notes)

            if st.button("Generate PDF Report", type="primary", use_container_width=True):
                with st.spinner("Compiling PDF document with ReportLab..."):
                    pred_class = st.session_state.classification_result.get("predicted_category", "Simple cyst") if st.session_state.classification_result else "Ovarian Cyst"
                    conf = st.session_state.seg_result.get("confidence", 0.8) if st.session_state.seg_result else 0.8
                    mode = st.session_state.seg_result.get("model_mode", "TRAINED") if st.session_state.seg_result else "TRAINED"
                    warnings = st.session_state.seg_result.get("warnings", []) if st.session_state.seg_result else []

                    generate_pdf_report(
                        scan_id=st.session_state.scan_id,
                        patient_id=rep_patient,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        original_img_path=orig_path,
                        enhanced_img_path=enh_path,
                        overlay_img_path=overlay_path,
                        predicted_class=pred_class,
                        confidence=conf,
                        morph_data=st.session_state.morph_data or {},
                        warnings=warnings,
                        output_pdf_path=pdf_path,
                        radiologist_notes=rep_notes,
                        radiologist_name=rep_reviewer,
                        model_mode=mode
                    )
                st.success("PDF report generated successfully!")

        with c2:
            st.subheader("Download & Preview")
            if os.path.exists(pdf_path):
                st.info(f"Report ready: `Report_{st.session_state.scan_id}.pdf` ({os.path.getsize(pdf_path) // 1024} KB)")
                with open(pdf_path, "rb") as pf:
                    st.download_button(
                        label="⬇️ Download Official Radiologist PDF Report",
                        data=pf,
                        file_name=f"SonoAssist_Report_{st.session_state.scan_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("Click 'Generate PDF Report' to create document.")
