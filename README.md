# Histopathology Image Classification using Classical Machine Learning

## Overview

This project focuses on the classification of oral histopathological images into:

* Normal tissue
* Oral Squamous Cell Carcinoma (OSCC)

The project uses classical machine learning techniques along with handcrafted image features extracted from histopathology slides.

The goal was to study how different parameters affect classification performance.

* magnifications
* dataset sizes
* feature extraction methods
* classifiers

---

# Dataset

## Source

[https://data.mendeley.com/datasets/ftmp4cvtmb/2](Open-source histopathological image dataset of Oral Squamous Cell Carcinoma (OSCC).)

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

# Experiments Performed

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

Date Updated: 08-05-2026