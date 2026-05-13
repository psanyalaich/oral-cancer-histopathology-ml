# Histopathology Image Classification using Classical Machine Learning
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

A classical machine learning pipeline for Oral Squamous Cell Carcinoma (OSCC) histopathological image classification using handcrafted texture and color features.

This project evaluates:

- Random Forest vs Support Vector Machine (SVM)
- Basic handcrafted features vs Haralick texture features
- Balanced vs imbalanced datasets
- 100x vs 400x histopathology magnifications

The pipeline was designed as a reproducible experimental framework for histopathology-based cancer classification research.

---

## Highlights

- Automated experiment pipeline
- 24 reproducible experiments
- Cross-validation evaluation
- ROC curve generation
- Feature importance visualization
- Support for Haralick texture analysis
- Config-driven experiment system

---

# Project Structure

```text
oral-cancer-histopathology-ml/
│
├── dataset.py                 # Dataset construction
├── train_model.py             # Main experiment runner
├── experiments.py             # Experiment configurations
├── visualize_results.py       # ROC + feature importance plots
├── requirements.txt
│
├── src/
│   ├── preprocessing.py       # Tissue segmentation + image loading
│   ├── features.py            # Feature extraction functions
│
├── results/
│   ├── experiment_name/
│   │   ├── metrics.txt
│   │   ├── roc_curve.png
│   │   └── feature_importance.png
│
└── results_summary.csv
```

---

# Installation
- Clone Repository: ```git clone https://github.com/yourusername/oral-cancer-histopathology-ml.git```
- ```cd oral-cancer-histopathology-ml```
- Create virtual env: ```python -m venv venv```
- Activate venv:
    - Windows: ```venv\Scripts\activate```
    - Linux/Mac: ```source venv/bin/activate```
- Install dependencies: ```pip install -r requirements.txt```
- All experiments are configured inside: ```experiments.py```
- To run the full experiment pipeline: ```python train_model.py```

# Dataset

Source: 
- Rahman, Tabassum Yesmin; Mahanta, Lipi B.; Das, Anup  K.; Sarma, Jagannath D. (2023), “Histopathological imaging database for Oral Cancer analysis ”, Mendeley Data, V2, doi: 10.17632/ftmp4cvtmb.2
- [Histopathological imaging database for Oral Cancer analysis](https://data.mendeley.com/datasets/ftmp4cvtmb/2)

Dataset description:
* H&E stained tissue slides
* Captured using Leica ICC50 HD microscope
* Images collected from 230 patients

## Magnifications Used

### 100x

* 89 normal images
* 439 tumour images

### 400x

* 201 normal images
* 495 tumour images

---

# Reproducibility

Fixed random seeds were used throughout the project: ```random_state = 42```

This was applied to:

- Random Forest
- SVM
- Stratified K-Fold splitting

to ensure reproducible experimental results.

---

# Preprocessing Pipeline

## 1. Image Loading

Images are:

* loaded using OpenCV
* converted from BGR to RGB
* resized to 512 × 512

## 2. Tissue Segmentation

A tissue mask is generated using:

* grayscale conversion
* Gaussian blur
* Otsu thresholding
* morphological closing

This helps remove background regions and focuses feature extraction on tissue regions.

---

# Feature Extraction

## 1. Color Features

For each RGB channel:

* mean intensity
* standard deviation

Total color features: 6

---

## 2. Local Binary Pattern (LBP)

LBP texture descriptors were extracted from grayscale tissue regions.

Histogram bins:

* 10 uniform LBP bins

Total LBP features: 10

---

## 3. Haralick Texture Features

Gray-Level Co-occurrence Matrix (GLCM) features were extracted.

Features used:

* contrast
* correlation
* energy
* homogeneity

Total Haralick features: 4

---

# Machine Learning Models

## Random Forest

Parameters:

```python
RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)
```

---

## Support Vector Machine (SVM)

Parameters:

```python
SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)
```

---

# Evaluation Strategy

## Cross Validation

5-fold stratified cross validation was used.

## Metrics

The following metrics were computed:

* Accuracy
* AUC
* Sensitivity
* Specificity

ROC curves were also generated for each experiment.

---

# Output Files

For each experiment, the pipeline automatically generates:

- `metrics.txt`
- `roc_curve.png`
- `feature_importance.png` (Random Forest only)

All experiment outputs are stored inside: ```results/<experiment_name>/```

A summary of all experiments is also stored in: ```results_summary.csv```

---

# Experiments Performed

## Best Performing Configuration

| Model | Features | Magnification | Accuracy | AUC |
|-------|-----------|---------------|----------|-----|
| SVM | Haralick | 100x | 0.933 | 0.983 |

## Sample Outputs
### ROC Curve - ```svm_haralick_full_100x```
![ROC](results/sample_outputs/roc_curve.png)

### Feature Importance - ```rf_haralick_20img_100x```
![Feature Importance](results/sample_outputs/feature_importance.png)

---

## 100x Magnification

### Random Forest + Basic Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.925    | 0.975 |
| 89 vs 89   | 0.865    | 0.941 |
| 89 vs 439  | 0.867    | 0.901 |

---

### SVM + Basic Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.800    | 0.962 |
| 89 vs 89   | 0.916    | 0.972 |
| 89 vs 439  | 0.871    | 0.944 |

---

### Random Forest + Haralick Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.950    | 0.950 |
| 89 vs 89   | 0.882    | 0.955 |
| 89 vs 439  | 0.888    | 0.914 |

---

### SVM + Haralick Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.925    | 0.988 |
| 89 vs 89   | 0.933    | 0.983 |
| 89 vs 439  | 0.898    | 0.948 |

---

# 400x Magnification

### Random Forest + Basic Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.900    | 0.894 |
| 201 vs 201 | 0.804    | 0.892 |
| 201 vs 495 | 0.800    | 0.814 |

---

### SVM + Basic Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.850    | 0.950 |
| 201 vs 201 | 0.833    | 0.906 |
| 201 vs 495 | 0.777    | 0.857 |

---

### Random Forest + Haralick Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.875    | 0.912 |
| 201 vs 201 | 0.806    | 0.884 |
| 201 vs 495 | 0.805    | 0.847 |

---

### SVM + Haralick Features

| Experiment | Accuracy | AUC   |
| ---------- | -------- | ----- |
| 20 vs 20   | 0.900    | 0.962 |
| 201 vs 201 | 0.841    | 0.905 |
| 201 vs 495 | 0.795    | 0.872 |

---

# Observations

## Key Findings

* SVM generally performed better than Random Forest.
* Haralick texture features improved overall performance.
* Balanced datasets produced more reliable specificity.
* Small datasets produced inflated accuracies due to overfitting.
* Dataset imbalance increased sensitivity while reducing specificity.
* 100x magnification generally achieved better performance than 400x.

# Limitations

- Dataset imbalance affects specificity in some experiments
- No patient-wise splitting was available in the dataset
- Classical handcrafted features may not capture all tissue morphology patterns
- Small datasets may produce optimistic performance estimates due to overfitting.

Date Updated: 13-05-2026