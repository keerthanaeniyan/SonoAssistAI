"""
SonoAssist AI - Clinical PDF Report Generator
Uses ReportLab to generate a clean, publication-grade, offline PDF report for radiologist sign-off.
Includes:
- Scan & Patient Identifiers
- Analysis Timestamp & Offline status
- Original, Enhanced, and Overlay image exhibits
- Model findings and confidence score
- Morphological measurements (pixel area, perimeter, circularity, etc.)
- Clinical safety warnings & model limitations
- Radiologist sign-off & review notes section
- Mandatory clinical decision-support disclaimer
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

REQUIRED_DISCLAIMER = (
    "AI-generated findings are for research and decision-support purposes only. "
    "They are not a medical diagnosis. All findings require review by a qualified radiologist."
)


def generate_pdf_report(
    scan_id: str,
    patient_id: str,
    timestamp: str,
    original_img_path: str,
    enhanced_img_path: str,
    overlay_img_path: str,
    predicted_class: str,
    confidence: float,
    morph_data: Dict[str, Any],
    warnings: list,
    output_pdf_path: str,
    radiologist_notes: str = "",
    radiologist_name: str = "",
    model_mode: str = "Trained"
) -> str:
    """
    Generate an offline PDF clinical report.
    Returns the path to the generated PDF.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styling
    primary_color = colors.HexColor("#1A365D")  # Deep Clinical Blue
    secondary_color = colors.HexColor("#2B6CB0")
    light_bg = colors.HexColor("#EDF2F7")
    accent_warning = colors.HexColor("#C53030")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#4A5568")
    )

    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#2D3748")
    )

    body_bold = ParagraphStyle(
        'BodyDarkBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=primary_color
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#4A5568"),
        alignment=1  # Centered
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>SonoAssist AI</b> | Ovarian Ultrasound Decision Support", title_style),
            Paragraph("<b>STATUS: OFFLINE SYSTEM</b><br/>Confidential Medical Analysis", subtitle_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[4.2 * inch, 2.8 * inch])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT')
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=8))

    # 2. Patient & Examination Metadata Table
    meta_data = [
        [
            Paragraph("<b>Scan Identification:</b>", body_style),
            Paragraph(scan_id, body_bold),
            Paragraph("<b>Analysis Date & Time:</b>", body_style),
            Paragraph(timestamp, body_style)
        ],
        [
            Paragraph("<b>Patient Identifier:</b>", body_style),
            Paragraph(patient_id, body_style),
            Paragraph("<b>Analysis Engine:</b>", body_style),
            Paragraph(f"Attention U-Net [{model_mode}]", body_style)
        ]
    ]
    t_meta = Table(meta_data, colWidths=[1.7 * inch, 1.8 * inch, 1.7 * inch, 1.8 * inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # 3. Image Exhibit Panel (Original | Enhanced | Diagnostic Overlay)
    story.append(Paragraph("<b>Image Exhibits (Original vs. Enhanced vs. AI Segmentation Overlay)</b>", section_header))

    img_w, img_h = 2.2 * inch, 2.2 * inch
    img_cells = []
    
    # Check if files exist
    def get_flowable_img(p):
        if os.path.exists(p):
            return Image(p, width=img_w, height=img_h)
        return Paragraph(f"Image not found:<br/>{os.path.basename(p)}", body_style)

    img_row = [
        get_flowable_img(original_img_path),
        get_flowable_img(enhanced_img_path),
        get_flowable_img(overlay_img_path)
    ]
    lbl_row = [
        Paragraph("<para align=center><b>1. Original Ultrasound</b></para>", body_style),
        Paragraph("<para align=center><b>2. Enhanced (Denoised + CLAHE)</b></para>", body_style),
        Paragraph("<para align=center><b>3. Segmentation Overlay</b></para>", body_style)
    ]
    t_imgs = Table([img_row, lbl_row], colWidths=[2.33 * inch, 2.33 * inch, 2.33 * inch])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 2)
    ]))
    story.append(t_imgs)
    story.append(Spacer(1, 10))

    # 4. Diagnostic Summary & Morphological Findings
    story.append(Paragraph("<b>AI Model Findings & Morphological Measurements</b>", section_header))

    primary_cyst = morph_data.get("primary_cyst") if morph_data else None
    area_px = primary_cyst.get("area_pixels", 0.0) if primary_cyst else 0.0
    peri_px = primary_cyst.get("perimeter_pixels", 0.0) if primary_cyst else 0.0
    circ = primary_cyst.get("circularity", 0.0) if primary_cyst else 0.0
    aspect = primary_cyst.get("aspect_ratio", 0.0) if primary_cyst else 0.0
    solidity = primary_cyst.get("solidity", 0.0) if primary_cyst else 0.0
    roughness = primary_cyst.get("roughness_index", 0.0) if primary_cyst else 0.0

    findings_data = [
        [
            Paragraph("<b>Predicted Classification:</b>", body_style),
            Paragraph(f"<b>{predicted_class}</b>", body_bold),
            Paragraph("<b>Area (Pixels):</b>", body_style),
            Paragraph(f"{area_px} px", body_style)
        ],
        [
            Paragraph("<b>Model Confidence:</b>", body_style),
            Paragraph(f"{confidence*100:.1f}%", body_style),
            Paragraph("<b>Perimeter (Pixels):</b>", body_style),
            Paragraph(f"{peri_px} px", body_style)
        ],
        [
            Paragraph("<b>Circularity Index (0-1):</b>", body_style),
            Paragraph(f"{circ:.3f}", body_style),
            Paragraph("<b>Aspect Ratio (W/H):</b>", body_style),
            Paragraph(f"{aspect:.3f}", body_style)
        ],
        [
            Paragraph("<b>Solidity (Convex Hull):</b>", body_style),
            Paragraph(f"{solidity:.3f}", body_style),
            Paragraph("<b>Boundary Roughness:</b>", body_style),
            Paragraph(f"{roughness:.3f}", body_style)
        ]
    ]

    t_findings = Table(findings_data, colWidths=[1.8 * inch, 1.7 * inch, 1.8 * inch, 1.7 * inch])
    t_findings.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F7FAFC")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    story.append(t_findings)
    story.append(Spacer(1, 8))

    # 5. Quality Assurance & Clinical Warnings
    if warnings:
        warn_paras = [Paragraph("<b>Quality Control / Model Limitations:</b>", section_header)]
        for w in warnings:
            prefix = "[WARNING]" if w.get("level") == "WARNING" else "[INFO]"
            warn_paras.append(Paragraph(f"• <b>{prefix}</b> {w.get('message', '')}", body_style))
        story.extend(warn_paras)
        story.append(Spacer(1, 8))

    # 6. Radiologist Sign-off and Notes Section
    story.append(Paragraph("<b>Radiologist Review & Diagnostic Confirmation</b>", section_header))
    review_content = radiologist_notes if radiologist_notes else "[ Radiologist has not yet entered notes. Pending review. ]"
    reviewer = radiologist_name if radiologist_name else "___________________________, MD"

    review_table_data = [
        [Paragraph("<b>Clinical Impression & Notes:</b>", body_style)],
        [Paragraph(review_content, body_style)],
        [
            Paragraph(f"<b>Reviewing Radiologist:</b> {reviewer}", body_style),
            Paragraph("<b>Signature:</b> ____________________   <b>Date:</b> ___________", body_style)
        ]
    ]
    t_review = Table(review_table_data, colWidths=[7.0 * inch])
    t_review.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E0")),
        ('BACKGROUND', (0, 0), (-1, 0), light_bg),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(t_review)
    story.append(Spacer(1, 14))

    # 7. Mandatory Clinical Disclaimer Banner
    disclaimer_box = Table([[Paragraph(f"<b>CLINICAL NOTICE:</b> {REQUIRED_DISCLAIMER}", disclaimer_style)]], colWidths=[7.0 * inch])
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFF5F5")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#FEB2B2")),
        ('PADDING', (0, 0), (-1, -1), 6)
    ]))
    story.append(disclaimer_box)

    # Build document
    doc.build(story)
    return output_pdf_path
