#  Brain Tumor MRI Classification with Grad-CAM

>  **Research & Educational Use Only** — Not for clinical diagnosis.

## Project Structure

```
├── brain_tumor_classification.ipynb  # 19-section notebook (full pipeline)
├── app.py                            # Streamlit web application
├── requirements.txt                  # All dependencies
├── README.md                         # This file
└── models/                           # Saved .h5 models (after training)
    ├── best_model_EfficientNetB0.h5
    └── metadata.json
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset (Kaggle)
```bash
# Place ~/.kaggle/kaggle.json first, then:
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip -p ./brain_tumor_dataset
```

### 3. Run the Jupyter Notebook (full training)
```bash
jupyter notebook brain_tumor_classification.ipynb
```

### 4. Launch Streamlit App
```bash
streamlit run app.py
# → http://localhost:8501
```

## Cloud Deployment

### Streamlit Community Cloud (free)
1. Push to GitHub
2. Go to share.streamlit.io → New App
3. Set main file: `app.py`
4. Deploy

### Hugging Face Spaces
1. Create Space with Streamlit SDK
2. Upload `app.py` + `requirements.txt`

## Architecture Overview

| Model | Params | Strategy | Best Use |
|-------|--------|----------|----------|
| Baseline CNN | 0.5M | From scratch | Baseline reference |
| ResNet50 | 25.6M | Transfer (ImageNet) | Max accuracy |
| EfficientNetB0 | 5.3M | Transfer (ImageNet) | Best acc/param ratio |
| MobileNetV2 | 3.4M | Transfer (ImageNet) | Edge deployment |

## Notebook Sections (19 total)

1. Installation & Setup
2. Imports & Configuration
3. Dataset Download (Kaggle + synthetic fallback)
4. Medical Preprocessing (CLAHE, skull strip)
5. Data Loading & Augmentation
6. Model Architectures
7. Two-Phase Transfer Learning Training
8. Training Diagnostics & Learning Curves
9. Comprehensive Evaluation (AUC, F1, ROC)
10. Grad-CAM Explainability
11. Monte Carlo Dropout Uncertainty
12. t-SNE Feature Visualization
13. Ensemble Model Fusion
14. Model Export (H5, TFLite)
15. Per-Class Error Analysis
16. Clinical Threshold Calibration
17. Grad-CAM++ vs Standard Comparison
18. Final Benchmark Report
19. Launch Instructions

## Streamlit App Features

-  Upload any brain MRI (JPG/PNG)
-  Real-time 4-class prediction
-  Interactive Grad-CAM heatmap with opacity slider
-  Probability charts with MC Dropout error bars
-  Uncertainty estimation with clinical review flag
-  Model selector (switch architectures live)
-  Full benchmark comparison dashboard
-  Methods & clinical context documentation

## References

- Selvaraju et al. (2017). Grad-CAM. ICCV.
- Chattopadhyay et al. (2018). Grad-CAM++. WACV.
- Gal & Ghahramani (2016). MC Dropout. ICML.
- Nickparvar (2021). Brain Tumor MRI Dataset. Kaggle.
