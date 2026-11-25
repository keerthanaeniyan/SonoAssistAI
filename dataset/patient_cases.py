"""
SonoAssist AI - Clinical Patient Cases & High-Fidelity Ultrasound Synthesizer
Generates 8 distinct, clinically characterized patient cases matching the MMOTU classification
and acoustic ultrasound physics:
1. Case PT-101: Simple Follicular Cyst (Anechoic, thin-walled, posterior enhancement)
2. Case PT-102: Endometrioma / Chocolate Cyst (Diffuse low-level ground-glass echoes, thick capsule)
3. Case PT-103: Serous Cystadenoma (Large unilocular clear fluid, regular margins)
4. Case PT-104: Mucinous Cystadenoma (Multilocular, internal echogenic septations)
5. Case PT-105: Mature Cystic Teratoma / Dermoid (Echogenic Rokitansky nodule with acoustic shadow)
6. Case PT-106: Normal Ovary (Central stroma with small peripheral antral follicles)
7. Case PT-107: Hemorrhagic Cyst (Internal fibrinous reticular mesh / fishnet pattern)
8. Case PT-108: High-Grade Serous Neoplasm (Thick irregular walls, solid papillary excrescences)
"""

import os
import json
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple


def _create_transvaginal_sector_base(h: int = 256, w: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic transvaginal ultrasound probe sector arc (apex at top center)
    with transducer ring-down reverberations and Rayleigh speckle.
    """
    base = np.zeros((h, w), dtype=np.float32)
    y_idx, x_idx = np.indices((h, w))
    center_x, center_y = w // 2, 8

    # Radial distance & angle
    dist = np.sqrt((x_idx - center_x)**2 + (y_idx - center_y)**2)
    angle = np.arctan2(y_idx - center_y, x_idx - center_x)

    # Sector bounds (approx 55 to 125 degrees)
    sector_mask = (dist >= 28) & (dist <= 245) & (angle >= 0.85) & (angle <= 2.29)

    # Rayleigh acoustic speckle noise
    speckle = np.random.rayleigh(scale=32.0, size=(h, w)).astype(np.float32)

    # Near-field transducer ring-down & gain depth compensation (TGC)
    depth_gain = 0.75 + (dist / 256.0) * 0.45
    base[sector_mask] = (speckle * depth_gain)[sector_mask]

    return base, sector_mask


def generate_all_patient_cases(dataset_root: str) -> List[Dict[str, Any]]:
    """
    Generate the 8 clinical patient ultrasound scans, masks, metadata,
    and simulated cine-loop video clips.
    """
    images_dir = os.path.join(dataset_root, "images")
    masks_dir = os.path.join(dataset_root, "masks")
    meta_path = os.path.join(dataset_root, "patient_metadata.json")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    cases = [
        {
            "patient_id": "PT-101",
            "name": "Case 1: PT-101 (Simple Cyst - 28yo)",
            "age": 28,
            "category": "Simple cyst",
            "clinical_history": "28yo G0 with mild left pelvic fullness during mid-cycle ovulation. No fever or dysmenorrhea.",
            "transducer": "Transvaginal Endocavitary 7.5 MHz",
            "ultrasound_findings": "Well-circumscribed unilocular 4.2 cm anechoic fluid-filled cyst with smooth, thin imperceptible wall (<2 mm) and prominent posterior acoustic enhancement. No internal echoes, septa, or mural nodules.",
            "radiologist_impression": "Benign simple ovarian follicular cyst. Compatible with IOTA Simple Cyst criteria. Routine sonographic follow-up in 6-8 weeks.",
            "file_prefix": "patient_101_simple_cyst"
        },
        {
            "patient_id": "PT-102",
            "name": "Case 2: PT-102 (Chocolate Cyst / Endometrioma - 34yo)",
            "age": 34,
            "category": "Chocolate cyst",
            "clinical_history": "34yo G1P1 presenting with worsening dysmenorrhea, deep dyspareunia, and chronic cyclic pelvic pain.",
            "transducer": "Transvaginal Endocavitary 8.0 MHz",
            "ultrasound_findings": "Round, thick-walled 3.8 cm adnexal mass displaying diffuse, homogeneous low-level internal echoes ('ground-glass' echotexture) characteristic of hemosiderin fluid without internal vascularity.",
            "radiologist_impression": "Classic ovarian endometrioma ('chocolate cyst'). No suspicious solid papillary components.",
            "file_prefix": "patient_102_chocolate_cyst"
        },
        {
            "patient_id": "PT-103",
            "name": "Case 3: PT-103 (Serous Cystadenoma - 46yo)",
            "age": 46,
            "category": "Serous cystadenoma",
            "clinical_history": "46yo perimenopausal female with progressive lower abdominal swelling and urinary frequency.",
            "transducer": "Transabdominal 4.0 MHz & Transvaginal 6.5 MHz",
            "ultrasound_findings": "Large 6.5 cm unilocular cystic lesion in the right ovary containing clear anechoic fluid. Smooth regular outer capsule with thin, delicate avascular single septation. No solid excrescences.",
            "radiologist_impression": "Findings highly consistent with benign serous cystadenoma. Recommend elective gynecologic consultation.",
            "file_prefix": "patient_103_serous_cystadenoma"
        },
        {
            "patient_id": "PT-104",
            "name": "Case 4: PT-104 (Mucinous Cystadenoma - 52yo)",
            "age": 52,
            "category": "Mucinous cystadenoma",
            "clinical_history": "52yo postmenopausal woman presenting with palpable lower abdominal fullness and early satiety.",
            "transducer": "Transabdominal 3.5 MHz Curved Array",
            "ultrasound_findings": "Voluminous, multilocular cystic lesion measuring 7.8 cm with numerous thin internal echogenic septations forming multiple fluid compartments. Variable internal echogenicity across compartments representing thick mucinous fluid.",
            "radiologist_impression": "Multilocular cystic neoplasm, typical appearance of mucinous cystadenoma. Low risk of malignancy; surgical excision indicated.",
            "file_prefix": "patient_104_mucinous_cystadenoma"
        },
        {
            "patient_id": "PT-105",
            "name": "Case 5: PT-105 (Mature Cystic Teratoma - 24yo)",
            "age": 24,
            "category": "Teratoma",
            "clinical_history": "24yo female evaluated during pre-conceptional pelvic sonogram. Asymptomatic.",
            "transducer": "Transvaginal Endocavitary 7.0 MHz",
            "ultrasound_findings": "Complex predominantly cystic mass with a focal, intensely hyperechoic mural nodule (Rokitansky dermoid plug) casting a prominent posterior acoustic shadow ('tip of the iceberg' sign). Fine hyperechoic lines representing hair strands.",
            "radiologist_impression": "Mature cystic teratoma (dermoid cyst) of the left ovary. Classic pathognomonic acoustic features.",
            "file_prefix": "patient_105_teratoma"
        },
        {
            "patient_id": "PT-106",
            "name": "Case 6: PT-106 (Normal Ovary / Antral Follicles - 22yo)",
            "age": 22,
            "category": "Normal ovary",
            "clinical_history": "22yo healthy female screened as prospective oocyte donor. Normal menstrual cycle.",
            "transducer": "Transvaginal Endocavitary 8.5 MHz",
            "ultrasound_findings": "Normal-sized ovary (8.5 cm³ volume) demonstrating homogeneous central echogenic stroma and multiple small, peripheral physiological antral follicles (3 mm to 8 mm). No dominant pathological cyst or mass.",
            "radiologist_impression": "Normal physiological ovary with healthy antral follicle count. No adnexal pathology.",
            "file_prefix": "patient_106_normal_ovary"
        },
        {
            "patient_id": "PT-107",
            "name": "Case 7: PT-107 (Hemorrhagic Corpus Luteum - 29yo)",
            "age": 29,
            "category": "Chocolate cyst",  # MMOTU hemorrhagic/endometrial spectrum
            "clinical_history": "29yo presenting with acute onset of right adnexal tenderness on day 22 of menstrual cycle.",
            "transducer": "Transvaginal Endocavitary 7.5 MHz",
            "ultrasound_findings": "Thick-walled 3.6 cm cystic lesion showing delicate internal reticular lace-like fibrinous strands ('fishnet pattern') and a peripheral retracting clot with concave margins. Peripheral circular flow ('ring of fire') on color Doppler.",
            "radiologist_impression": "Hemorrhagic corpus luteum cyst with internal fibrin mesh. Typical self-limiting appearance; repeat ultrasound in 6 weeks recommended.",
            "file_prefix": "patient_107_hemorrhagic_cyst"
        },
        {
            "patient_id": "PT-108",
            "name": "Case 8: PT-108 (High-Grade Serous / Complex Lesion - 61yo)",
            "age": 61,
            "category": "High-grade serous",
            "clinical_history": "61yo postmenopausal female presenting with unexplained pelvic discomfort, weight loss, and elevated CA-125 (180 U/mL).",
            "transducer": "Transvaginal Endocavitary 7.5 MHz & Transabdominal 3.5 MHz",
            "ultrasound_findings": "Complex, multilocular solid-cystic ovarian mass measuring 5.4 cm with thickened, irregular septa (>3 mm) and multiple vascular solid endocystic papillary projections (>10 mm). Moderate adjacent free pelvic fluid (ascites).",
            "radiologist_impression": "O-RADS Category 5 (High Risk of Malignancy). Complex solid-cystic mass suspicious for high-grade serous ovarian neoplasm. Urgent gynecologic oncology referral.",
            "file_prefix": "patient_108_high_grade_serous"
        }
    ]

    h, w = 256, 256
    y_idx, x_idx = np.indices((h, w))

    for idx, case in enumerate(cases):
        base, sector = _create_transvaginal_sector_base(h, w)
        mask = np.zeros((h, w), dtype=np.uint8)

        cx, cy = w // 2 + (idx % 3 - 1) * 8, h // 2 + 10

        if case["category"] == "Simple cyst":
            # Round, anechoic, smooth thin wall, strong enhancement
            rx, ry = 42, 38
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            # Anechoic interior (very low intensity Rayleigh noise)
            base[mask == 255] = np.random.rayleigh(scale=3.0, size=np.sum(mask == 255))
            # Thin hyperechoic wall
            wall = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(wall, (cx, cy), (rx + 1, ry + 1), 0, 0, 360, 255, 1)
            base[wall == 255] = np.clip(base[wall == 255] * 1.6 + 45, 0, 255)
            # Posterior acoustic enhancement
            enh_zone = (x_idx > cx - rx + 5) & (x_idx < cx + rx - 5) & (y_idx > cy + ry) & sector
            base[enh_zone] = np.clip(base[enh_zone] * 1.45 + 30, 0, 255)

        elif case["category"] == "Chocolate cyst" and "Endometrioma" in case["name"]:
            # Homogeneous ground-glass internal echoes, thickened wall
            rx, ry = 38, 36
            cv2.ellipse(mask, (cx, cy), (rx, ry), 15, 0, 360, 255, -1)
            # Diffuse low-level echoes (scale ~ 16.0)
            base[mask == 255] = np.random.rayleigh(scale=15.0, size=np.sum(mask == 255))
            # Thick fibrotic capsule (2-3px)
            wall = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(wall, (cx, cy), (rx + 2, ry + 2), 15, 0, 360, 255, 2)
            base[wall == 255] = np.clip(base[wall == 255] * 1.5 + 40, 0, 255)
            # Mild posterior enhancement
            enh_zone = (x_idx > cx - rx + 8) & (x_idx < cx + rx - 8) & (y_idx > cy + ry) & sector
            base[enh_zone] = np.clip(base[enh_zone] * 1.25 + 15, 0, 255)

        elif case["category"] == "Serous cystadenoma":
            # Large unilocular, clear fluid, single delicate septum
            rx, ry = 48, 44
            cv2.ellipse(mask, (cx, cy), (rx, ry), -10, 0, 360, 255, -1)
            base[mask == 255] = np.random.rayleigh(scale=4.0, size=np.sum(mask == 255))
            # Delicate septum across center
            septum = np.zeros((h, w), dtype=np.uint8)
            cv2.line(septum, (cx - rx + 10, cy - 10), (cx + rx - 10, cy + 12), 255, 1)
            base[(septum == 255) & (mask == 255)] = 175.0
            # Posterior enhancement
            enh_zone = (x_idx > cx - rx + 5) & (x_idx < cx + rx - 5) & (y_idx > cy + ry) & sector
            base[enh_zone] = np.clip(base[enh_zone] * 1.4 + 25, 0, 255)

        elif case["category"] == "Mucinous cystadenoma":
            # Multilocular, honeycomb septations with variable mucinous echogenicity
            rx, ry = 52, 46
            cv2.ellipse(mask, (cx, cy), (rx, ry), 5, 0, 360, 255, -1)
            base[mask == 255] = np.random.rayleigh(scale=9.0, size=np.sum(mask == 255))
            # Draw multiple septations
            sept = np.zeros((h, w), dtype=np.uint8)
            cv2.line(sept, (cx - 20, cy - ry + 4), (cx - 15, cy + ry - 4), 255, 1)
            cv2.line(sept, (cx + 18, cy - ry + 4), (cx + 10, cy + ry - 4), 255, 1)
            cv2.line(sept, (cx - rx + 5, cy), (cx + rx - 5, cy - 5), 255, 1)
            base[(sept == 255) & (mask == 255)] = 190.0
            # One compartment slightly thicker mucin
            locule = (x_idx > cx - 20) & (x_idx < cx + 10) & (y_idx > cy - ry + 4) & (y_idx < cy) & (mask == 255)
            base[locule] = np.random.rayleigh(scale=20.0, size=np.sum(locule))

        elif case["category"] == "Teratoma":
            # Cyst with hyperechoic Rokitansky nodule casting acoustic shadow
            rx, ry = 42, 38
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            base[mask == 255] = np.random.rayleigh(scale=8.0, size=np.sum(mask == 255))
            # Rokitansky nodule (dermoid plug) in top-left of cyst
            plug_cx, plug_cy = cx - 14, cy - 10
            plug_r = 12
            cv2.circle(base, (plug_cx, plug_cy), plug_r, 235.0, -1)
            # Acoustic shadowing beneath the plug
            shadow_zone = (x_idx >= plug_cx - plug_r) & (x_idx <= plug_cx + plug_r) & (y_idx > plug_cy + plug_r) & sector
            base[shadow_zone] = np.clip(base[shadow_zone] * 0.25, 0, 255)

        elif case["category"] == "Normal ovary":
            # Normal stroma with small peripheral antral follicles (3-8 mm)
            # Oval ovary contour
            ov_rx, ov_ry = 38, 28
            ovary_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(ovary_mask, (cx, cy), (ov_rx, ov_ry), 10, 0, 360, 255, -1)
            # Normal echogenic stroma
            base[ovary_mask == 255] = np.random.rayleigh(scale=38.0, size=np.sum(ovary_mask == 255))
            # Small peripheral follicles (3 to 6 px)
            follicle_offsets = [(-22, -10), (-16, 12), (18, -8), (20, 10), (0, -16), (2, 16)]
            for ox, oy in follicle_offsets:
                fx, fy = cx + ox, cy + oy
                cv2.circle(base, (fx, fy), np.random.randint(3, 6), 5.0, -1)
                cv2.circle(mask, (fx, fy), 4, 255, -1)  # small follicle annotations

        elif "Hemorrhagic" in case["name"]:
            # Corpus luteum with internal reticular lace / fishnet pattern
            rx, ry = 36, 34
            cv2.ellipse(mask, (cx, cy), (rx, ry), -15, 0, 360, 255, -1)
            base[mask == 255] = np.random.rayleigh(scale=7.0, size=np.sum(mask == 255))
            # Reticular meshwork lines
            mesh = np.zeros((h, w), dtype=np.uint8)
            for k in range(-20, 25, 8):
                cv2.line(mesh, (cx - rx + 5, cy + k), (cx + rx - 5, cy + k + 10), 255, 1)
                cv2.line(mesh, (cx + k, cy - ry + 5), (cx + k - 8, cy + ry - 5), 255, 1)
            base[(mesh == 255) & (mask == 255)] = np.random.uniform(90, 140, size=np.sum((mesh == 255) & (mask == 255)))

        elif case["category"] == "High-grade serous":
            # Thick irregular cyst with solid papillary excrescences (>10mm)
            rx, ry = 46, 38
            # Irregular lobulated contour
            pts = []
            for deg in range(0, 360, 20):
                rad = np.radians(deg)
                r_var = rx + np.sin(deg * 3) * 6
                px = int(cx + r_var * np.cos(rad))
                py = int(cy + (ry + np.cos(deg * 2) * 4) * np.sin(rad))
                pts.append([px, py])
            pts = np.array(pts, dtype=np.int32)
            cv2.drawContours(mask, [pts], -1, 255, -1)
            base[mask == 255] = np.random.rayleigh(scale=10.0, size=np.sum(mask == 255))
            # Solid irregular papillary mural nodule
            pap_pts = np.array([[cx - 10, cy + 10], [cx + 8, cy + 14], [cx + 14, cy - 2], [cx, cy - 6]], dtype=np.int32)
            cv2.fillPoly(base, [pap_pts], 180.0)
            # Thickened wall
            cv2.drawContours(base, [pts], -1, 200.0, 3)

        # Boundary clipping and saving
        final_img = np.clip(base, 0, 255).astype(np.uint8)
        img_filename = f"{case['file_prefix']}.png"
        mask_filename = f"{case['file_prefix']}.png"

        img_path = os.path.join(images_dir, img_filename)
        mask_path = os.path.join(masks_dir, mask_filename)

        cv2.imwrite(img_path, final_img)
        cv2.imwrite(mask_path, mask)

        case["image_path"] = img_path
        case["mask_path"] = mask_path
        case["image_filename"] = img_filename

    # Save metadata JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2)

    print(f"[PatientCases] Successfully created 8 clinical patient cases and metadata in {dataset_root}")
    return cases


def load_patient_case_library(dataset_root: str) -> List[Dict[str, Any]]:
    """Load or generate the clinical patient cases metadata."""
    meta_path = os.path.join(dataset_root, "patient_metadata.json")
    if not os.path.exists(meta_path):
        return generate_all_patient_cases(dataset_root)
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
    generate_all_patient_cases(root)
