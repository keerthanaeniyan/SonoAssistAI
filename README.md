# SonoAssist AI: AI-Assisted Ovarian Ultrasound Analysis System

**SonoAssist AI** is an offline, clinical decision-support application designed to enhance low-quality ovarian ultrasound scans, segment cyst boundaries using deep learning, extract morphological descriptors, and generate standardized PDF reports for radiologist review.

---

## 1. System Architecture & Features

- **Offline Operational Capability**: Operates 100% locally with zero internet or cloud dependencies after initial setup.
- **Ultrasound Quality Enhancement**: Multi-stage OpenCV pipeline:
  - Acoustic speckle attenuation (Bilateral, Non-Local Means, or Median filtering)
  - Contrast Limited Adaptive Histogram Equalization (CLAHE)
  - Edge-preserving unsharp masking
  - Non-reference image quality metrics (Contrast Std, Tenengrad Sharpness, Estimated SNR)
- **Deep Learning Cyst Segmentation**:
  - **Attention U-Net** with additive attention gates to focus on hypoechoic cyst lesions against echogenic stroma
  - Input: 256 × 256 grayscale ultrasound images
  - Output: Binary lesion mask + Sigmoid probability map
  - Combined Binary Cross Entropy (BCE) + Dice Loss
- **Quantitative Morphometry**:
  - Area in pixels, Perimeter, Circularity / Isoperimetric Quotient, Aspect Ratio, Solidity, and Boundary Roughness index
- **Uncertainty & Safety Assurance**:
  - 4-fold Test-Time Augmentation (TTA) spatial variance estimation
  - Automatic quality control checks for low dynamic range, severe acoustic blur, and over-segmentation
  - Explicit **Demonstration Mode** indicator when running without pre-trained model weights
- **Lesion Classification**:
  - EfficientNet-B0 backbone supporting the 8 MMOTU ovarian lesion categories:
    1. Simple cyst
    2. Chocolate cyst (Endometrioma)
    3. Serous cystadenoma
    4. Mucinous cystadenoma
    5. Teratoma (Dermoid cyst)
    6. Theca cell tumor
    7. High-grade serous
    8. Normal ovary
- **Ultrasound Cine-Loop Video Processing**:
  - Streaming frame-by-frame memory-safe ingestion (MP4 & AVI)
  - Temporal exponential smoothing between consecutive frames to eliminate boundary flickering
- **Offline Report Engine**:
  - Compiles PDF clinical reports via ReportLab with embedded image exhibits, measurements, QC alerts, and radiologist sign-off sections
- **Local SQLite Repository**:
  - Tracks all examination records, patient metadata, and radiologist review notes offline.

---

## 2. Project Folder Structure

```
SonoAssistAI/
├── app.py                         # Main Streamlit clinical dashboard application
├── requirements.txt               # Python package dependencies
├── README.md                      # Complete system documentation
│
├── models/
│   ├── attention_unet.py          # PyTorch Attention U-Net & loss functions
│   ├── efficientnet.py            # EfficientNet-B0 MMOTU classifier
│   └── trained_weights/           # Model checkpoints (.pth files)
│
├── preprocessing/
│   ├── enhancement.py             # Full enhancement pipeline & quality metrics
│   ├── denoising.py               # Speckle reduction, NLM & median filters
│   └── video_processing.py        # Streaming cine-loop video processor
│
├── detection/
│   ├── segmentation.py            # Segmentation inference engine & demo prior
│   ├── classification.py          # Classification inference engine
│   ├── feature_extraction.py      # Morphological metric extraction
│   └── uncertainty.py             # Test-Time Augmentation & clinical QC
│
├── training/
│   ├── dataset.py                 # MMOTU dataset loader & patient-level split
│   ├── train_segmentation.py      # Segmentation training script
│   ├── train_classifier.py        # Classification training script
│   └── evaluate.py                # Dice, IoU, and precision/recall evaluation
│
├── visualization/
│   ├── annotations.py             # High-contrast clinical overlay generator
│   └── comparison.py              # Side-by-side & 4-panel comparison grids
│
├── reports/
│   └── generate_report.py         # Offline PDF generator (ReportLab)
│
├── storage/
│   └── database.py                # SQLite database interface (sonoassist.db)
│
├── dataset/
│   ├── images/                    # Ultrasound scans (.png / .jpg)
│   └── masks/                     # Binary ground truth masks (.png)
│
└── outputs/
    ├── enhanced/                  # Enhanced scans
    ├── segmented/                 # Predicted masks & overlays
    ├── videos/                    # Processed cine-loop outputs
    └── reports/                   # Generated clinical PDF reports
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10 – 3.14
- PyTorch 2.x

### Step-by-Step Installation
1. Clone or copy the `SonoAssistAI` project directory:
   ```bash
   cd SonoAssistAI
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Clinical Patient Cases Library & MMOTU Dataset

SonoAssist AI includes an interactive **Clinical Patient Benchmark Library** representing 8 real-world ovarian ultrasound cases matching the MMOTU classification scheme and acoustic ultrasound physics:

| Case ID | Patient Profile | MMOTU Pathology | Key Acoustic & Sonographic Features |
| :--- | :--- | :--- | :--- |
| **PT-101** | 28yo G0, ovulatory fullness | **Simple cyst** | Unilocular anechoic fluid, smooth imperceptible wall (<2mm), prominent posterior enhancement |
| **PT-102** | 34yo G1P1, dysmenorrhea | **Chocolate cyst** | Thick-walled, diffuse homogeneous low-level "ground-glass" echoes (endometrioma) |
| **PT-103** | 46yo, abdominal fullness | **Serous cystadenoma** | Large (6.5 cm) unilocular clear fluid, regular margins, delicate avascular single septum |
| **PT-104** | 52yo, early satiety | **Mucinous cystadenoma** | Voluminous multilocular cyst with numerous thin septa, variable internal fluid echogenicity |
| **PT-105** | 24yo, asymptomatic screening | **Teratoma** | Focal hyperechoic Rokitansky nodule (dermoid plug) casting acoustic shadow ("tip of iceberg") |
| **PT-106** | 22yo, healthy donor screen | **Normal ovary** | Central echogenic stroma with multiple small peripheral physiological antral follicles (3-8mm) |
| **PT-107** | 29yo, acute cycle day 22 pain | **Hemorrhagic cyst** | Reticular lace-like fibrinous mesh ("fishnet" pattern) with retracting clot |
| **PT-108** | 61yo, CA-125 180 U/mL | **High-grade serous** | Multilocular solid-cystic mass, thickened irregular septa (>3mm), solid papillary mural nodules |

Each case comes pre-packaged with:
- Clinical transvaginal/transabdominal sector ultrasound scan
- Pixel-level ground-truth segmentation annotation mask
- Anonymized clinical history, age, transducer specifications, and radiologist impression
- Pre-seeded local SQLite records with radiologist sign-offs

### Full MMOTU Dataset Setup
The system is also compatible with the full **MMOTU (Multi-Modality Ovarian Tumor Ultrasound) Dataset**:
- **GitHub Repository**: [https://github.com/cv516Buaa/MMOTU_DS2Net](https://github.com/cv516Buaa/MMOTU_DS2Net)
- **Zenodo Archive**: [https://zenodo.org/records/21128657](https://zenodo.org/records/21128657)

### Directory Arrangement
Extract the downloaded dataset images and segmentation masks into the `dataset/` directory:
```
SonoAssistAI/dataset/
├── images/
│   ├── patient001_scan01.png
│   ├── patient002_scan01.png
│   └── ...
└── masks/
    ├── patient001_scan01.png
    ├── patient002_scan01.png
    └── ...
```

> **Offline Demonstration Note**: If the MMOTU dataset is not yet downloaded, SonoAssist AI automatically generates synthetic demonstration ultrasound scans on first startup so that all 8 pages of the application can be tested immediately.

---

## 5. Model Training Instructions

### Train the Attention U-Net Segmentation Model:
```bash
python training/train_segmentation.py --data_dir dataset --epochs 15 --batch_size 4 --lr 0.0001
```
- Trains with combined BCE + Dice loss
- Performs patient-level cross-validation
- Automatically saves the best model checkpoint to `models/trained_weights/attention_unet_best.pth`
- Saves convergence curves to `models/trained_weights/segmentation_training_curve.png`

### Train the EfficientNet-B0 Classification Model:
```bash
python training/train_classifier.py --data_dir dataset --epochs 10 --batch_size 4
```
- Automatically saves the best classification checkpoint to `models/trained_weights/efficientnet_best.pth`

---

## 6. Running the Application

Launch the Streamlit clinical dashboard:
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

### Navigation Workflow:
1. **1. Dashboard**: View system stats, active AI engines, and offline status.
2. **2. Upload Ultrasound**: Upload an ovarian ultrasound image or choose a built-in MMOTU demo scan.
3. **3. Image Enhancement**: Tune speckle denoising (Bilateral, NLM, Median) and CLAHE contrast.
4. **4. AI Segmentation**: Run Attention U-Net inference, view diagnostic overlays, and inspect morphological metrics.
5. **5. Classification**: Inspect predicted MMOTU lesion category and class probability distribution.
6. **6. Video Analysis**: Upload cine-loop ultrasound clips (MP4/AVI) for temporal smoothed analysis.
7. **7. Analysis Results**: Review historical scans in the offline SQLite database and enter radiologist sign-off notes.
8. **8. Generate Report**: Export a publication-quality PDF report with radiologist sign-off sections.

---

## 7. Clinical Decision Support Disclaimer

> **IMPORTANT**:
> "AI-generated findings are for research and decision-support purposes only. They are not a medical diagnosis. All findings require review by a qualified radiologist."
