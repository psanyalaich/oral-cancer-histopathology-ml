# Histopathology Image Classification using Classical Machine Learning
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

A classical machine learning pipeline for Oral Squamous Cell Carcinoma (OSCC) histopathological image classification using handcrafted texture and color features.

The pipeline was designed as a reproducible experimental framework for histopathology-based cancer classification research, with emphasis on statistically rigorous evaluation, reproducible experimentation, and interpretable classical machine learning methods.

This project evaluates:
- Random Forest vs Support Vector Machine (SVM)
- Systematic handcrafted feature ablation experiments
- Color, LBP, and Haralick texture descriptors
- Balanced vs imbalanced datasets
- 100x vs 400x histopathology magnifications
- Feature scaling effects for SVM models
- Statistical significance of model performance differences
- Reproducibility and experimental robustness in histopathology ML

## Highlights
- Automated experiment pipeline
- 126 configurable experiments
- Repeated stratified cross-validation
- Nested hyperparameter tuning using GridSearchCV
- ROC and Precision-Recall curve generation
- Feature importance and SHAP visualization
- Support for Haralick texture analysis
- Statistical significance testing
- Confidence interval computation
- Reproducible dataset splits and random seeds
- Config-driven experiment system
- Configurable feature ablation framework
- Feature caching system for faster experimentation
- Optional SVM feature scaling analysis
- Segmentation robustness safeguards
- SHAP and permutation importance visualization

# Installation
- Clone Repository: 
    - ```git clone https://github.com/yourusername/oral-cancer-histopathology-ml.git```
    - ```cd oral-cancer-histopathology-ml```
- Create virtual env: ```python -m venv venv```
- Activate venv:
    - Windows: ```venv\Scripts\activate```
    - Linux/Mac: ```source venv/bin/activate```
- Install dependencies: ```pip install -r requirements.txt```
- All experiments are configured inside: ```experiments.py```

- To run the full experiment pipeline: ```python train_model.py```
    - loads images
    - preprocesses tissue regions
    - extracts features
    - trains models
    - performs cross-validation
    - computes evaluation metrics
    - generates predictions
    - saves experiment outputs
- To run statistical analysis: ```python stats_analysis.py```
    - paired t-tests
    - Wilcoxon signed-rank tests
    - effect size calculations
    - saves dataset into ```statistical_tests.csv```

# Dataset
Source: 
- Rahman, Tabassum Yesmin; Mahanta, Lipi B.; Das, Anup  K.; Sarma, Jagannath D. (2023), “Histopathological imaging database for Oral Cancer analysis ”
- DOI:  10.17632/ftmp4cvtmb.2
- [Histopathological imaging database for Oral Cancer analysis](https://data.mendeley.com/datasets/ftmp4cvtmb/2)

Dataset description:
* H&E stained tissue slides
* Captured using Leica ICC50 HD microscope
* Images collected from 230 patients
* Binary classification:
    * Normal tissue
    * OSCC tumour tissue

## Magnifications Used
### 100x
* 89 normal images
* 439 tumour images
### 400x
* 201 normal images
* 495 tumour images

# Reproducibility

Fixed random seeds were used throughout the project: ```random_state = 42```

This was applied to the following to ensure reproducible experimental results:
- Random Forest
- SVM
- Stratified K-Fold splitting
- NumPy randomization

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

Additional preprocessing safeguards were implemented to improve robustness:

* dark-image rejection
* connected-component noise removal
* minimum tissue-area validation
* excessive tissue-mask rejection

These checks help reduce segmentation failures and unstable feature extraction caused by noisy or low-information images.

# Feature Extraction
The machine learning models do not directly analyze raw images.

Each image is converted into a numerical feature vector representing:
* color distribution
* texture patterns
* tissue heterogeneity
* structural organization

The framework also supports configurable feature subset experiments for systematic feature ablation analysis.

## 1. Color Features
For each RGB channel:
* mean intensity
* standard deviation

Total color features: 6

These help capture stain distribution and color variation.

## 2. Local Binary Pattern (LBP)
LBP texture descriptors were extracted from grayscale tissue regions.

For each pixel:
* neighboring pixels are compared
* binary texture patterns are generated
* histogram representation is computed

Histogram bins: 10 uniform LBP bins
Total LBP features: 10 features

LBP captures:
* local cellular texture
* nucleus density variation
* structural irregularities


## 3. Haralick Texture Features
Gray-Level Co-occurrence Matrix (GLCM) features were extracted from segmented tissue regions.

Features used:
* contrast
* correlation
* energy
* homogeneity

Total Haralick features: 4

Additional safeguards were implemented for stable Haralick extraction:
* tissue-region cropping
* low-tissue rejection
* sparse-mask filtering
* gray-level quantization

These steps help reduce unstable GLCM estimates caused by noisy background regions.

## Feature Ablation Framework

The pipeline supports multiple configurable feature subsets:

| Feature Set | Included Features |
|---|---|
| `color` | RGB statistical features |
| `lbp` | Local Binary Pattern features |
| `haralick` | GLCM Haralick texture features |
| `color_lbp` | Color + LBP |
| `color_haralick` | Color + Haralick |
| `lbp_haralick` | LBP + Haralick |
| `all` | Color + LBP + Haralick |

This allows controlled experiments to evaluate how different handcrafted feature families contribute to histopathology classification performance.

# Machine Learning Models

## Random Forest
GridSearchCV optimized:
* number of estimators
* maximum depth

```python
RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)
```

## Support Vector Machine (SVM)
GridSearchCV optimized:
* C
* gamma

```python
SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)
```

SVM experiments were performed both with and without feature standardization to evaluate scaling sensitivity of handcrafted descriptors.

# Evaluation Strategy

## Repeated Cross-Validation
The experiments used 25 repeated stratified train/test splits.

Pre-generated dataset splits were reused across experiments to ensure fair paired statistical comparison between models.

## Nested Hyperparameter Tuning
Hyperparameters were optimized inside each training fold using GridSearchCV with stratified inner cross-validation.

This prevents information leakage between training and evaluation data.

A small inner cross-validation split count was used because some experiments contained limited training samples.

## Metrics
The following metrics were computed:
* Accuracy
* AUC
* PR-AUC
* Precision
* Sensitivity
* Specificity
* F1-score
* MCC

Sensitivity was used instead of recall because both metrics represent the same quantity:

TP / (TP + FN)

Sensitivity terminology was retained because the project is focused on medical image classification.

ROC curves and confusion matrices were also generated for each experiment.

Mean ± standard deviation values in the summary tables were computed from fold-level metrics.

ROC curve AUC values were computed using pooled predictions aggregated across all folds.

## Statistical Analysis

Additional statistical testing was performed using:
* Nadeau-Bengio corrected paired t-test
* Wilcoxon signed-rank test
* Welch’s t-test
* Mann-Whitney U test
* Cohen's d effect size

Paired statistical tests were only applied when experiments reused identical cross-validation splits.

Independent statistical tests were used for experiments involving different datasets or magnification settings.

Bonferroni-corrected p-values were computed to reduce false positives from multiple statistical comparisons.

# Output Files

For each experiment, the pipeline automatically generates:
- `metrics.txt`
- `fold_metrics.csv`
- `fold_predictions.csv`
- `roc_curve.png`
- `confusion_matrix.png`
- `feature_importance.png` (Random Forest only)
- `permutation_importance.png`
- `permutation_importance.csv`
- `shap_summary.png`
- `dataset_manifest.json`
- `config.json`

Stored results:
* All experiment outputs are stored inside: ```results/<experiment_name>/```
* A summary of all experiments is also stored in: ```results_summary.csv```
* Statistical test outputs are stored in: ```statistical_tests.csv```

## Dataset Feature Caching

Extracted feature vectors are cached to disk to avoid repeated preprocessing and feature extraction across experiments.

Cache keys are automatically generated using:
* magnification
* dataset size
* feature configuration

This substantially reduces runtime for large experimental grids.

# Experiments Performed

## Best Performing Configuration

| Model | Features | Magnification | Accuracy | AUC |
|-------|-----------|---------------|----------|-----|
| SVM | Haralick | 100x | 0.947 | 0.986 |

## Sample Outputs
### ROC Curve - ```svm_haralick_full_100x```
![ROC](results/sample_outputs/roc_curve.png)

### Feature Importance - ```rf_haralick_20img_100x```
![Feature Importance](results/sample_outputs/feature_importance.png)

## Statistical Testing

### SVM vs RF (Full 100x Dataset)
| Metric   | p-value    |
| -------- | ---------- |
| Accuracy | 0.00015    |
| AUC      | < 0.000001 |
| PR-AUC   | < 0.000001 |
| F1-score | 0.0115     |
| MCC      | < 0.000001 |

This suggests SVM significantly outperformed RF on the full 100x dataset.

### Haralick vs Non-Haralick (SVM Full 100x)
No statistically significant improvement was observed despite slight metric increases.

This suggests:
* LBP already captured substantial texture information
* Haralick features added limited complementary information

# Key Findings
* SVM generally performed better than Random Forest.
* Feature subset performance varied substantially across experiments.
* Haralick texture features improved some configurations but not all.
* Feature scaling improved SVM stability in several configurations.
* Balanced datasets produced more reliable specificity.
* Small datasets sometimes produced unstable or optimistic performance estimates.
* Dataset imbalance increased sensitivity while reducing specificity.
* 100x magnification generally achieved better performance than 400x.
* SVM significantly outperformed RF statistically on the full 100x dataset.

# Reproducibility and Experimental Integrity

To improve reproducibility and fair model comparison:
- identical dataset splits were reused across paired experiments
- dataset manifests were saved for each experiment
- experiment configurations were stored as JSON files
- fold-level predictions and metrics were saved
- statistical comparison was only performed after verifying matching splits
- feature extraction settings were stored with experiment outputs
- cached datasets preserved preprocessing consistency
- split files were pre-generated and reused across experiments

# Limitations
- Dataset imbalance affects specificity in some experiments
- No patient-wise splitting was available in the dataset
- Classical handcrafted features may not capture all tissue morphology patterns
- Small datasets may produce optimistic performance estimates due to overfitting.
- Hyperparameter tuning on very small datasets may be unstable because inner validation folds contain few samples.
- External validation dataset was not available
- No stain normalization was applied
- Image-level splitting may still introduce hidden patient-level leakage
- Patient-wise cross-validation could not be performed because patient IDs were unavailable
- The dataset originates from a single acquisition source, which may limit generalization across scanners, staining protocols, and institutions.

---
### *If you use this project, please cite the original dataset creators.*
---

Date Updated: <!--LAST_UPDATED--> 21-05-2026
