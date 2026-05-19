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
- Statistical significance of model performance differences

The pipeline was designed as a reproducible experimental framework for histopathology-based cancer classification research.

## Highlights
- Automated experiment pipeline
- 24 configurable experiments
- Cross-validation evaluation
- ROC curve generation
- Feature importance visualization
- Support for Haralick texture analysis
- Statistical significance testing
- Confidence interval computation
- Config-driven experiment system
- Reproducible random seed setup

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
- To generate visualization: ```python visualize_results.py```
    - ROC curves
    - confusion matrices
    - feature importance plots
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

This was applied to the folloring to ensure reproducible experimental results:
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

# Feature Extraction
The machine learning models do not directly analyze raw images.

Each image is converted into a numerical feature vector representing:
* color distribution
* texture patterns
* tissue heterogeneity
* structural organization

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
Gray-Level Co-occurrence Matrix (GLCM) features were extracted.

Features used:
* contrast
* correlation
* energy
* homogeneity

Total Haralick features: 4

# Machine Learning Models

## Random Forest
GridSearchCV optimized:
* number of estimators
* maximum depth
* minimum samples split

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
* kernel

```python
SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=42
)
```

# Evaluation Strategy
- **Cross Validation**: 25-fold stratified cross validation was used.

## Metrics
The following metrics were computed:
* Accuracy
* AUC
* PR-AUC
* Precision
* Recall/Sensitivity
* Specificity
* F1-score
* MCC

ROC curves and confusion matrices were also generated for each experiment.

## Statistical Analysis
Additional statistical testing was performed using:
* Paired t-test
* Wilcoxon signed-rank test
* Cohen's d effect size

Purpose:
* verify whether performance differences are statistically significant
* avoid relying only on average accuracy values

# Output Files

For each experiment, the pipeline automatically generates:
- `metrics.txt`
- `fold_metrics.csv`
- `predictions.csv`
- `roc_curve.png`
- `confusion_matrix.png`
- `feature_importance.png` (Random Forest only)'

Stored results:
* All experiment outputs are stored inside: ```results/<experiment_name>/```
* A summary of all experiments is also stored in: ```results_summary.csv```
* Statistical test outputs are stored in: ```statistical_tests.csv```

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

#  Key Findings
* SVM generally performed better than Random Forest.
* Haralick texture features improved overall performance.
* Balanced datasets produced more reliable specificity.
* Small datasets produced inflated accuracies due to overfitting.
* Dataset imbalance increased sensitivity while reducing specificity.
* 100x magnification generally achieved better performance than 400x.
* SVM significantly outperformed RF statistically on the full 100x dataset.

# Limitations
- Dataset imbalance affects specificity in some experiments
- No patient-wise splitting was available in the dataset
- Classical handcrafted features may not capture all tissue morphology patterns
- Small datasets may produce optimistic performance estimates due to overfitting.
- External validation dataset was not available

### *If you use this project, please cite the original dataset creators.*

---

Date Updated: 19-05-2026
