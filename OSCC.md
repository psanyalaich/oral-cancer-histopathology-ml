# Systematic Notes

## 1. Project Overview & Scope

**Task:** binary classification of H&E oral histopathology images (Normal vs OSCC) with classical ML on handcrafted features.

**Objectives:** build a complete image-classification pipeline; extract handcrafted histopathology features; compare classical models; evaluate across dataset sizes and magnifications; automate large-scale experimentation; validate the workflow before using real lab data; study feature/imbalance effects; produce reproducible, publication-grade results.

**Scope philosophy (recurring throughout):** the goal is *"a rigorous, interpretable, publication-quality undergraduate computational pathology pipeline,"* **not** a state-of-the-art methods paper. The project deliberately avoids foundation pathology models, deep segmentation, multi-center domain adaptation, AutoML/Bayesian optimization, and excessive OOP/orchestration. Coherent rigor is valued over maximal complexity, and overengineering is treated as a failure mode.

---

## 2. Dataset

**Source:** Rahman, T. Y.; Mahanta, L. B.; Das, A. K.; Sarma, J. D. (2023). *"Histopathological imaging database for Oral Cancer analysis."* Mendeley Data, V2.
DOI: `https://doi.org/10.17632/ftmp4cvtmb.2` · Link: `https://data.mendeley.com/datasets/ftmp4cvtmb/2`

**Composition:** H&E-stained oral histopathology images from **230 patients**, captured with a **Leica ICC50 HD** microscope. Two classes: Normal oral tissue and OSCC. Two magnification groups (never mixed during evaluation):

| Magnification | Normal | Tumour |
|---|---|---|
| 100× | 89 | 439 |
| 400× | 201 | 495 |

---

## 3. Repository Architecture

### 3.1 Final structure

```text
src/
├── experiment_runner.py     # per-fold training/evaluation (run_fold)
├── model_utils.py           # model construction + hyperparameter grids
├── results_utils.py         # persistence + reporting (summary CSV, metrics.txt)
├── evaluation_utils.py      # ROC/PR, overall confusion matrix, pooled AUC
├── rf_analysis.py           # RF interpretability (feature importance, SHAP)
├── metrics_utils.py         # metric computation
├── statistics_utils.py      # CIs, corrected tests, split verification
├── visualize_results.py     # all plotting
├── explainability.py        # permutation importance, SHAP
├── cache_utils.py           # dataset/feature caching
├── analysis_plots.py        # class distribution, correlation
├── features.py              # handcrafted feature extraction
├── preprocessing.py         # loading, segmentation, stain normalization
├── dataset.py               # build_dataset (moved here from repo root — §3.3)
├── data_quality.py          # blur/dark/bright/contrast/duplicate detection
└── learning_curve_analysis.py
train_model.py               # orchestration entry point
generate_splits.py           # precompute CV splits
experiments.py               # experiment config generation
configs/
├── experiment_config.py     # folds, seeds, models, magnifications, feature sets, norm options
├── model_config.py          # RF + SVM grids
└── preprocessing_config.py  # image size, thresholds, morphology, tissue limits, norm options
tests/                       # test_preprocessing/features/dataset/splitting + conftest.py
```

Module responsibilities (condensed):

- **`train_model.py`** — iterate experiments; create result folders; load cached dataset or build; call fold runner; save metrics/summary rows/plots; run RF analysis when enabled; plot learning curves at the end.
- **`experiment_runner.py`** — load train/test indices; fit model; run `GridSearchCV`; compute fold metrics; collect predictions; save fold plots; optional permutation importance.
- **`cache_utils.py`** — build a cache key (magnification, sample counts, feature set, stain normalization, seed); load existing cache or build+save. One of the biggest speedups.
- **`metrics_utils.py`** — accuracy, ROC-AUC, PR-AUC, precision, sensitivity/recall, specificity, F1, MCC, Brier, confusion-matrix counts.
- **`results_utils.py`** — summary CSV init, fold metrics, predictions, hyperparameters, final `metrics.txt`, aggregate summary rows.
- **`logging_utils.py`** — experiment failure logging (name, fold, exception type/message, traceback).

### 3.2 The `train_model.py` refactor (monolith → structured functions)

**Severity:** High. The original was a single ~640-line module-level script: importing it (`import train_model`) immediately ran all 24 experiments; fold execution, GridSearchCV, plotting, saving, aggregation, and metrics were all interleaved at module scope; error handling only `print`-ed the exception (no traceback, no fold context); and **partial fold failures silently corrupted summaries** — if fold 3 of 5 crashed, `fold_rows` held only 2 folds yet `fold_df.agg(["mean","std"])` still produced "valid-looking" statistics with no warning.

Reorganised into four units (no trainer classes, no DI, no frameworks — kept intentionally minimal):

| Function | Purpose |
|---|---|
| `get_model()` | construct model pipelines (retained as-is) |
| `run_fold()` | execute one CV fold |
| `run_experiment()` | execute one full experiment |
| `main()` | loop through all experiments safely |

`run_fold()` loads precomputed splits, builds train/test sets, creates the model pipeline, defines the param grid, runs `GridSearchCV`, predicts, computes metrics, saves the fold confusion matrix, and **returns structured output instead of mutating globals**:

```python
return {
    "fold_row": fold_row,
    "prediction_rows": prediction_rows,
    "y_test": y_test, "y_pred": y_pred, "y_prob": y_prob,
    "best_params": {...},
}
```

`run_experiment()` reads config, makes the results dir, builds the dataset, generates feature names, runs the fold loop (calling `run_fold`), aggregates fold outputs, and computes summary metrics. `main()` loops experiments with robust failure handling:

```python
logging.exception(f"Experiment failed: {config['name']}")
failed_experiments.append(config["name"])
```

The module-scope experiment loop was removed; execution happens only via `main()` behind `if __name__ == "__main__":`. Logging configured as:

```python
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
```

Later, `train_model.py` became primarily an orchestration script delegating to the `src/` utilities:

```python
run_fold(...); save_fold_metrics(...); save_predictions(...)
write_metrics_txt(...); save_overall_evaluation(...); run_rf_analysis(...)
```

### 3.3 Structural cleanups

- **`dataset.py` moved into `src/`.** Old `from dataset import build_dataset` relied on an implicit `sys.path` dependency; changed to `from src.dataset import build_dataset` (and `from .preprocessing import ...` etc.). Recommendations recorded: add `__init__.py`, use absolute imports consistently, avoid `sys.path.append()`.
- **`generate_splits.py` execution guard.** It ran on import (side effects, poor reusability, hard to test). Wrapped its logic in a `main()` behind `if __name__ == "__main__":`.
- **Duplicate `get_model` header bug** — two stacked `def get_model(...)` signatures during refactor; removed the outdated one.
- **Import-time file deletion removed** — summary-CSV deletion happened at import; replaced with an explicit `initialize_summary_csv()` called at runtime (see §3.2 corruption note and §12).

---

## 4. Preprocessing (`src/preprocessing.py`)

### 4.1 Image loading

Load with OpenCV, convert BGR→RGB, resize to 512×512:

```python
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (512, 512))
```

**Known limitation (fixed 512×512 resize):** LBP and GLCM are scale-sensitive, so resizing alters their physical interpretation, and 100×/400× nuclei occupy different pixel sizes even at equal dimensions. Kept as-is because the dataset is single-scanner, magnifications are evaluated separately, and fixed resizing is standard. Documented as a limitation; optional future fix = multi-scale LBP (`P=8,R=1` and `P=16,R=2`).

### 4.2 Tissue segmentation + quality control

Pipeline: grayscale → Gaussian blur → **Otsu with `THRESH_BINARY_INV`** → morphological closing → connected-component cleanup → tissue-fraction validation.

```python
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, mask = cv2.threshold(blurred, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
```

**`THRESH_BINARY_INV` is correct** (an early note wrongly documented `THRESH_BINARY`): H&E slides have a bright background and darker tissue, so inversion maps tissue→foreground (255), background→0.

QC safeguards added (segmentation was originally validated only for completely empty masks):

```python
if gray.mean() < 15:                      # dark-image guard
    raise ValueError("Image too dark for reliable segmentation.")

# connected-component filtering removes tiny artifacts / noise blobs
cv2.connectedComponentsWithStats(...)

MIN_TISSUE_FRACTION = 0.05                 # detect empty masks / segmentation failure
MAX_TISSUE_FRACTION = 0.95                 # detect oversegmentation (mask covers whole slide)
if tissue_fraction < MIN_TISSUE_FRACTION:  # ...
if tissue_fraction > MAX_TISSUE_FRACTION:  # ...
```

Otsu assumes bimodal, separable intensities, which histopathology does not always satisfy — hence the coverage checks. `MIN_TISSUE_FRACTION` was later **centralised**: it had been defined in both `preprocessing_config.py` (canonical) and `features.py` (duplicate), risking silent divergence; the local copy in `features.py` was deleted and imported from config instead.

`src/data_quality.py` adds blur, too-dark/too-bright, low-contrast, and **duplicate detection via perceptual hash**.

### 4.3 Stain normalization — three implementations, two silent no-op bugs

Histopathology features can encode staining/scanner artifacts rather than morphology. A configurable `normalize_staining()` abstraction was added and evolved through a stub → Reinhard → Macenko progression. Normalization is applied **inside the dataset loops, immediately after loading, before segmentation**:

```python
img = load_image(path)
if stain_normalization is not None:
    img = normalize_staining(img, method=stain_normalization, normalizer=normalizer)
mask = segment_tissue(img)
features = extract_features(img, mask, feature_set=feature_set)
```

**(a) Stub phase.** Initial placeholder returned the image unchanged for `"reinhard"` (architecture-first design). Experiment names embed normalization state, e.g. `rf_unscaled_all_reinhard_full_400x` vs `..._no_norm_...`, and the cache key was extended with `f"stain_{stain_normalization}"`.

**(b) Reinhard (via `staintools`) — and the first scientific-integrity bug.** The `"reinhard"` branch had been left as `raise NotImplementedError(...)` while configs still generated `"reinhard"` experiments, caches, models, metrics, and **statistical tests**. Because no transform occurred, `no_norm == reinhard` for every experiment; every stain comparison produced `t = 0.0000, p = 1.000000`, and those null comparisons still inflated the Bonferroni correction factor, penalising *unrelated* experiments. Forensic review (suspiciously perfect equivalence) caught it. Real Reinhard was then implemented:

```python
# pip install staintools   (Windows: pip install spams-bin)
def load_target_image():
    return load_image(STAIN_NORMALIZATION_TARGET_IMAGE)   # "data/target_stain_image.jpg"

def create_reinhard_normalizer(target_image):
    normalizer = staintools.ReinhardColorNormalizer()
    normalizer.fit(target_image)
    return normalizer

def normalize_staining(img, method=None, normalizer=None):
    if method is None:
        return img
    if method == "reinhard":
        if normalizer is None:
            raise ValueError("Reinhard normalizer required.")
        normalized = normalizer.transform(img)
        return np.clip(normalized, 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown normalization method: {method}")
```

The normalizer is fit **once** to a single representative target image (balanced H&E, no blur/artifacts/overexposure, not extreme eosin/hematoxylin) and reused across all images. Using `target=image` (self-normalization) was deliberately avoided — it degenerates to near-identity. A regression test now guards against silent no-ops:

```python
def test_reinhard_changes_image(sample_image):
    normalizer = create_reinhard_normalizer(sample_image)
    normalized = normalize_staining(sample_image, method="reinhard", normalizer=normalizer)
    assert normalized.shape == sample_image.shape
    assert not np.array_equal(normalized, sample_image)
```

All prior `"reinhard"` caches were invalidated and `CACHE_VERSION = "v2"` was set. A `.jpg`/`.png` extension mismatch on the target path was also fixed.

**(c) Macenko (via `torchstain`).** Estimates separate hematoxylin/eosin stain vectors from the target image.

```python
import torch, torchstain
def create_macenko_normalizer(target_image):
    normalizer = torchstain.normalizers.MacenkoNormalizer()
    target_tensor = torch.from_numpy(target_image).permute(2, 0, 1).float()
    normalizer.fit(target_tensor)
    return normalizer
```

**torchstain tensor-convention bugs (recurred several times):** `torchstain` already returns `(H, W, C)`, but the pipeline added a permutation, producing `(512, 3, 512)`. This caused `TypeError: Invalid shape (512, 3, 512)` in matplotlib and `cv2.error: Invalid number of channels in input image, 'scn' is 512` in OpenCV. A related `.permute(0, 2, 1)` produced `(512, 3, 512)` too. Diagnosed by printing `tensor.shape`/`dtype` in an isolated `test.py`. Also: calling `.numpy()` on an already-NumPy array → `AttributeError: 'numpy.ndarray' object has no attribute 'numpy'`; calling `.astype` on a tensor → `AttributeError: 'Tensor' object has no attribute 'astype'`. torchstain expects **0–255 float** tensors, not 0–1 (a 0–1 scaling attempt collapsed the image to near-blank). Final working block:

```python
if method == "macenko":
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    normalized, _, _ = normalizer.normalize(tensor)
    normalized = normalized.numpy()                       # already (H, W, C); no extra permute
    return np.clip(normalized, 0, 255).astype(np.uint8)
```

Even when functionally correct, output showed neon-purple/magenta tissue — caused by **poor stain-basis estimation from a non-representative reference image** (hematoxylin-dominant, dark, low eosin variation). Macenko is highly reference-dependent; remaining work is reference-image selection/benchmarking (3–5 candidates), a `results/debug_normalization/` Original|Reinhard|Macenko grid, and post-normalization segmentation-stability checks.

**(d) The second silent no-op bug (cache propagation).** `train_model.py` called `get_or_cache_dataset(...)` **without passing `stain_normalization`**, so it defaulted to `None` while the cache key still produced names like `stain_reinhard`/`stain_macenko`. *All* Reinhard and Macenko experiments silently ran on unnormalized images. Fix:

```python
STAIN_NORMALIZATION = config.get("stain_normalization", None)  # .get avoids KeyError
X, y, used_files, image_paths = get_or_cache_dataset(
    magnification=MAGNIFICATION, num_normal=NUM_NORMAL, num_tumour=NUM_TUMOUR,
    feature_set=FEATURE_SET, seed=SEED, stain_normalization=STAIN_NORMALIZATION)
```

All stain caches were invalidated. Validation recommendations recorded: log the active method, save debug example triplets, and confirm feature distributions actually shift across `none`/`reinhard`/`macenko` (identical features ⇒ still broken). Future-proofing: per-cache JSON metadata (incl. `preprocessing_version`) and a `PREPROCESSING_VERSION` in the cache key.

**Preprocessing hygiene cleanup:** `normalize_staining()` had leftover debug `print()`s and an unreachable second `return normalized` after `return normalized.astype(np.uint8)`; both removed. `astype(np.uint8)` is kept (OpenCV, visualization, saving, feature extraction all expect uint8); debug output should use `logger.debug(...)`.

---

## 5. Feature Extraction (`src/features.py`)

Three handcrafted families. Basic set (color + LBP) ≈ **16 features**; adding Haralick brings it to ~20.

### 5.1 Color (6 features)
Mean and standard deviation of each RGB channel. Captures global staining/color-distribution differences.

**Caveat (stain sensitivity):** feature-importance analysis showed `mean_r`, `mean_g`, `mean_b` as the top features (hematoxylin lowers red; eosin raises red/pink), which is biologically plausible but reflects scanner/staining/illumination/protocol variability and limits cross-institution generalization. No normalization was applied for the single-center dataset; the limitation is acknowledged explicitly (this motivated the stain-normalization experiments above).

### 5.2 Local Binary Pattern (LBP)
Uniform LBP on grayscale tissue, `P = 8`, `R = 1`, histogram representation with **10 bins**. Captures micro-texture / cellular texture differences.

### 5.3 Haralick (GLCM-based, 4 features) — the background-contamination saga
Features: **contrast, correlation, energy, homogeneity**. This was the **most important scientific correction** in the project and was iterated on several times.

**Original (flawed):** GLCM computed over the *entire 512×512 image* after zeroing background, so the co-occurrence matrix counted background↔background (`0→0`), background↔tissue, and tissue↔background transitions instead of pure tissue texture — distorting contrast and inflating energy/homogeneity. Critical because tissue may occupy only 20–30% of an image; the features partly encoded tissue coverage / background proportion / artificial black boundaries. Also weak methodologically: single distance (`[1]`), single angle (`[0]`, directionally biased — tissue has no preferred orientation), full `levels=256` (large, sparse, noisy GLCM).

```python
# ORIGINAL — contaminated
masked_gray = gray.copy(); masked_gray[mask == 0] = 0
glcm = graycomatrix(masked_gray, distances=[1], angles=[0],
                    levels=256, symmetric=True, normed=True)
```

**Fixes applied:** restrict to tissue, quantize 256→16 gray levels (`// 16`), and average over four angles (0, 45, 90, 135°). The region-restriction method itself evolved: first a **bounding-box crop**, later switched to **masking inside the region** (`tissue_gray[mask==0]=0`) plus a tissue-fraction guard. `distances=[1,2]` was considered but simplified to `[1]` for interpretability (multi-angle already gives the main robustness gain). The bounding box leaves some internal non-tissue pixels — a fully masked pairwise GLCM is the rigorous version, deferred as out of scope.

```python
# FINAL — masked, quantized, multi-angle
def extract_haralick_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    tissue_fraction = np.mean(mask > 0)
    if tissue_fraction < MIN_TISSUE_FRACTION:
        return [0.0, 0.0, 0.0, 0.0]
    tissue_gray = gray.copy()
    tissue_gray[mask == 0] = 0
    tissue_gray = (tissue_gray // 16).astype(np.uint8)
    glcm = graycomatrix(tissue_gray, distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=16, symmetric=True, normed=True)
    features = []
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        features.append(float(graycoprops(glcm, prop).mean()))
    return features
```

> **Documentation statement (for report/paper):** *"Haralick texture features were computed on grayscale tissue regions after excluding background using segmentation masks. Grayscale intensities were quantized from 256 to 16 levels before GLCM computation to reduce sparsity and noise. Texture statistics were averaged across four orientations (0°, 45°, 90°, 135°)."*

The same GLCM bug also lived in a standalone **`demo.py`** (`compute_haralick()` running GLCM on full background-containing images); it was fixed identically (bounding-box crop, tight tissue crop, density validation) so `demo.py` matches the pipeline. `demo.py` was kept for visualization/illustration.

### 5.4 `FEATURE_GROUPS` architecture
The original `use_haralick=True/False` toggle only allowed base / base+Haralick and prevented isolating families. Replaced by `feature_set="..."` with a modular group map enabling full ablations:

```python
FEATURE_GROUPS = {
    "color": ["color"], "lbp": ["lbp"], "haralick": ["haralick"],
    "color_lbp": ["color", "lbp"],
    "color_haralick": ["color", "haralick"],
    "lbp_haralick": ["lbp", "haralick"],
    "all": ["color", "lbp", "haralick"],
}
```

Extraction became modular (`if "color" in feature_groups: features += extract_color_features(...)`), and `get_feature_names(feature_set)` keeps names aligned with permutation importance, SHAP, and correlation plots.

**Critical runtime bug fixed:** `extract_features(img, mask, feature_set="feature_set")` passed the literal string instead of the variable — always failing feature lookup. Corrected to `feature_set=feature_set`.

---

## 6. Dataset Construction & Caching

`build_dataset(magnification, num_normal, num_tumour, feature_set, stain_normalization, seed)` loads images, preprocesses tissue, extracts features, assigns labels, and builds `X` (feature matrix) and `y` (label vector). It enabled experiments across magnifications, balanced/imbalanced sizes, and feature sets.

**Sampling fix (deterministic → reproducible random).** `_limit_files()` originally returned `files[:limit]` after lexicographic sort, so reduced-size experiments were *nested deterministic subsets* (`20 ⊂ 50 ⊂ 89`) correlated with acquisition/storage order rather than independent samples — weakening "performance under small data" claims. Replaced with seeded random sampling (optionally different seeds per class, e.g. 42 normal / 43 tumour):

```python
def _limit_files(files, limit, seed=42):
    if limit is None or limit >= len(files):
        return files
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(files), size=limit, replace=False))
    return [files[i] for i in idx]
```

**Caching (`cache_utils.py`).** Multiple experiments reuse the same dataset (e.g. `rf_full_100x`, `svm_full_100x`, `rf_haralick_full_100x`, …), so feature extraction was repeated ~4× (and Haralick ~2×) per condition. Caching stores already-computed `(X, y, used_files, image_paths)`; it does **not** change results (identical features/CV/seeds/training), only runtime.

```python
def get_or_cache_dataset(magnification, num_normal, num_tumour,
        use_haralick, cache_dir="feature_cache"):
    key = f"{magnification}_{num_normal}_{num_tumour}_haralick_{use_haralick}"
    cache_path = Path(cache_dir) / f"{key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    X, y, used_files, image_paths = build_dataset(
        magnification=magnification, num_normal=num_normal,
        num_tumour=num_tumour, use_haralick=use_haralick)
    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((X, y, used_files, image_paths), f)
    return X, y, used_files, image_paths
```

The cache key later evolved from `use_haralick` to `feature_set`, and gained `stain_normalization` and `seed` components (see §4.3d for the propagation bug). A Pylance/Ruff "undefined name" set of errors during this work was an indentation/scope issue (key creation must stay inside the function body).

**Traceability:** `dataset_manifest.json` (magnification, image filenames, composition) and `dataset_index.csv` (image path, label, index).

**QC changes dataset cardinality (important downstream effect):** stain normalization and QC filtering (blur/duplicate/tissue-coverage) drop images dynamically, so the post-QC dataset size can differ from the size used when CV splits were generated — the root of the split-mismatch failures in §8.

---

## 7. Models & Hyperparameters

Two classifiers in an sklearn `Pipeline`. Both use `class_weight="balanced"` and a per-experiment `random_state=seed`.

```python
RandomForestClassifier(class_weight="balanced", random_state=42)   # n_estimators via grid
SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
```

`n_estimators` was removed from the RF constructor because it is controlled by the grid (keeping both was redundant/confusing).

**Scaling pipeline (unified, toggleable).** Originally `StandardScaler` was always applied; now controlled via `get_model(model_type, use_scaling=True)` so scaled vs unscaled SVM can be compared (SVMs are scale-sensitive):

```python
steps = []
if use_scaling:
    steps.append(("scaler", StandardScaler()))
steps.append(("classifier", model))
```

**Hyperparameter grids.** Inner search initially tiny; broadened (but kept compact, no AutoML). Final grids:

```python
# Random Forest
{"classifier__n_estimators": [100, 200],
 "classifier__max_depth": [None, 10, 20],
 "classifier__min_samples_split": [2, 5],
 "classifier__min_samples_leaf": [1, 2]}
# SVM
{"classifier__C": [0.1, 1, 10],
 "classifier__gamma": ["scale", 0.01, 0.001]}
```

Rationale: `n_estimators` (ensemble size/stability), `max_depth` (complexity vs overfitting), `min_samples_split`/`min_samples_leaf` (regularization, leaf noise); `C` (regularization strength), `gamma` (RBF locality). `GridSearchCV` uses explicit `refit=True`; per-fold tracking records `cv_std` (`std_test_score` at `best_index_`) and `n_hyperparameter_configs` (`len(cv_results_["params"])`). A missing-comma syntax bug in that dict (before `**grid.best_params_`) was fixed.

> Note: on very small datasets the **inner** hyperparameter CV uses small validation folds (`StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)`), making selection noisier. This does **not** invalidate outer-CV estimates (the whole model-selection pipeline is evaluated within each outer fold); grids were kept small and the limitation stated explicitly.

**Calibration analysis** (probability reliability — important clinically; evaluation only, no isotonic/Platt correction):

- **Brier score** added to `metrics_utils.py` (`from sklearn.metrics import ... brier_score_loss`; `"brier_score": brier_score_loss(y_true, y_prob)`; lower = better).
- **Calibration curve** in `visualize_results.py` via `from sklearn.calibration import calibration_curve` — `plot_calibration_curve(y_true, y_prob, save_path, n_bins=10, csv_path=None)`, `strategy="uniform"`, plotting predicted vs observed positive fraction against the diagonal, optionally exporting `mean_predicted_probability`/`fraction_of_positives` to CSV. For small datasets, `n_bins=5` gives more stable estimates than `n_bins=10`. Integrated per fold (`calibration_curve_fold_{fold}.png`).

---

## 8. Cross-Validation & Splits

**Strategy.** Day-1 used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Upgraded to **`RepeatedStratifiedKFold(n_splits=5, n_repeats=5)` = 25 evaluations** for better variance estimation.

**Precomputed splits (`generate_splits.py`).** Splits are generated once and saved to `cv_splits/` as `.npy` index files, so every experiment shares identical folds — enabling valid paired tests and full reproducibility. Split directories include a `seed_{seed}` identifier (omitting it caused collisions/overwrites/incorrect reuse).

**Repeated-CV loop bug (Severity: High).** Splits were generated for all 25 iterations, but the training loop iterated only `for fold in range(1, N_SPLITS + 1)` — i.e. folds 1–5 (one repeat), silently reducing 5×5 repeated CV to a single-repeat 5-fold CV. All prior metrics, SDs, CIs, and significance tests were based on 5 evaluations, not 25 ("not invalid, but statistically incomplete"). Fix: nested loop + repeat-aware filenames so artifacts don't overwrite.

```python
for repeat in range(1, N_REPEATS + 1):
    for fold in range(1, N_SPLITS + 1):
        try:
            result = run_fold(repeat=repeat, fold=fold, ...)   # was: duplicate fold=fold arg bug
        except Exception:
            logger.exception(f"Repeat {repeat}, Fold {fold} failed.")
```

Naming: `train_idx_fold_1.npy` → `train_idx_repeat_1_fold_1.npy` (same for test indices, confusion matrices, calibration curves, permutation/SHAP). Fold/prediction rows gained a `"repeat"` field; `verify_same_splits(exp1_dir, exp2_dir, n_splits=5, n_repeats=5)` was fixed to iterate all repeats/folds with the new filenames. After the naming change, `rm -rf cv_splits/ && python generate_splits.py`; key/best experiments rerun.

**Split ↔ dataset cardinality mismatch (Severity: High).** With Macenko/QC active, experiments failed with `IndexError: index 391 is out of bounds for axis 0 with size 391` (and 672/672). Logs showed QC dropping images (`Blurry image detected`, `Segmentation failed`, `Duplicate image detected`). Root cause: **`generate_splits.py` built splits without stain normalization**, but training built the dataset *with* Macenko + QC, so old indices referenced removed samples:

```python
# generate_splits.py — BEFORE (no normalization → wrong cardinality)
X, y, used_files, image_paths = build_dataset(
    magnification=magnification, num_normal=num_normal,
    num_tumour=num_tumour, feature_set="all")
# AFTER — match training exactly
X, y, used_files, image_paths = build_dataset(
    magnification=magnification, num_normal=num_normal, num_tumour=num_tumour,
    feature_set=config["feature_set"], stain_normalization=stain_normalization, seed=seed)
```

Split key corrected so **stain normalization (not feature set) participates** (feature set doesn't change which images survive QC; normalization does):

```python
# BEFORE: (magnification, num_normal, num_tumour, seed, config["feature_set"])
split_key = (magnification, num_normal, num_tumour, stain_normalization, seed)
```

Cleanup + regeneration order: delete broken `cv_splits/400x_201_201_seed_42`, `cv_splits/400x_201_495_seed_42` and broken `feature_cache/v3_400x_..._stain_macenko_seed_42.pkl`, then `python generate_splits.py` → `python train_model.py`.

**Patient-level leakage (future-proofing).** Image-level stratified splitting risks train/test contamination if multiple images share a patient, inflating metrics. Documented future fix: `StratifiedGroupKFold(groups=groups)` to keep a patient's images together (requires patient metadata not yet in the dataset).

---

## 9. Metrics & Evaluation

**Metric set** (`compute_all_metrics(y_true, y_pred, y_prob)`): accuracy, ROC-AUC, PR-AUC, precision, sensitivity (= recall), specificity, F1, MCC, Brier, plus TN/FP/FN/TP. PR-AUC and MCC are emphasised for the imbalanced setting.

```python
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef)

def compute_all_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy   = accuracy_score(y_true, y_pred)
    auc        = roc_auc_score(y_true, y_prob)
    pr_auc     = average_precision_score(y_true, y_prob)
    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc        = matthews_corrcoef(y_true, y_pred)
    return {"accuracy": accuracy, "auc": auc, "pr_auc": pr_auc, "precision": precision,
            "recall": recall, "f1_score": f1, "sensitivity": sensitivity,
            "specificity": specificity, "tn": tn, "fp": fp, "fn": fn, "tp": tp, "mcc": mcc}
```

**Recall vs sensitivity duplication removed** — they are identical (`TP/(TP+FN)`); the biomedical term **sensitivity** was kept everywhere and `recall` dropped from fold metrics, summaries, `metrics.txt`, CSVs, and reporting loops. A duplicate `"sensitivity"` entry in a metric list was also removed; canonical list: `["accuracy","auc","pr_auc","precision","sensitivity","f1_score","specificity","mcc"]`. (One residual exception: the PR-curve x-axis label was set to `"Recall"` to match standard precision–recall terminology.)

**Fold-wise saving** (needed for variance/paired tests): each fold stores its metrics, predictions, probabilities, image paths, confusion matrix, and train/test indices (`fold_metrics.csv`, `fold_predictions.csv`, `train_idx_*`/`test_idx_*`).

**Mean fold AUC vs pooled ROC AUC** are not equal and are reported separately to avoid ambiguity. Mean fold AUC (averages per-fold AUCs; equal fold weight; good for variability) goes in summary tables as `Mean Fold AUC ± SD`; pooled AUC (`roc_auc_score(all_y_test, all_y_prob)` over concatenated fold predictions; one global curve, smoother) is labelled `Pooled ROC AUC` on ROC figures. Standard deviations use `std(ddof=1)` throughout (sample SD).

**Outputs per experiment:** `roc_curve.png`/`roc_curve_data.csv`, `pr_curve.png`/`pr_curve_data.csv`, per-fold + overall confusion matrices (PNG + CSV, with `normalize=True`), `best_hyperparameters.csv` (fold, best CV AUC, best params), calibration curves.

---

## 10. Statistical Testing (`statistics_utils.py`, `stats_analysis.py`)

Lets the project report e.g. *"SVM significantly outperformed RF (p < 0.05)"* instead of "looks better." The recurring theme: **repeated CV violates independence**, so naive tests are anti-conservative.

**Paired vs independent comparisons** (originally everything was treated as paired, which is invalid across magnifications/independent datasets):
- **Paired** (same splits/samples, e.g. model A vs B on identical folds): corrected paired t-test + Wilcoxon signed-rank. Guarded by `verify_same_splits()` so paired stats are only used when folds match exactly.
- **Independent** (different datasets/magnifications/non-matching splits): Welch's t-test + Mann-Whitney U.

**Nadeau–Bengio corrected variance** (overlapping repeated-CV test sets make fold estimates positively correlated and naive SEs too small / CIs too narrow / t-statistics inflated):

$$SE_{corrected} = s\sqrt{\tfrac{1}{n} + \tfrac{\rho}{1-\rho}},\qquad \rho = \tfrac{1}{k}$$

where `k` = number of CV splits, so for 5-fold CV `ρ = 0.2` (the CV test fraction). A misleading parameter `n_test` (read as "5 samples," which would massively overcorrect for hundreds of images) was renamed to `test_fraction`:

```python
# CV_N_SPLITS = 5 ; TEST_FRACTION = 1.0 / CV_N_SPLITS  → 0.2
def paired_ttest_corrected(a, b, test_fraction):
    diff = np.asarray(a) - np.asarray(b)
    n = len(diff)
    mean_diff, std_diff = diff.mean(), diff.std(ddof=1)
    if np.isclose(std_diff, 0):
        return 0.0, 1.0
    rho = test_fraction
    corrected_se = std_diff * np.sqrt((1.0 / n) + (rho / (1.0 - rho)))
    t_stat = mean_diff / corrected_se
    p_val = 2 * (1 - t.cdf(abs(t_stat), df=n - 1))
    return t_stat, p_val
```

The corrected **confidence interval** uses the same correction:

```python
def confidence_interval_corrected(values, test_fraction, confidence=0.95):
    values = np.asarray(values); n = len(values)
    mean, std = np.mean(values), values.std(ddof=1)
    rho = test_fraction
    corrected_se = std * np.sqrt((1.0 / n) + (rho / (1.0 - rho)))
    margin = corrected_se * t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - margin, mean + margin
```

A separate **CI consistency fix**: a "naive" `confidence_interval()` (`sem = std/√n`; assumes IID) coexisted with the corrected version and was being used inconsistently. The naive function was renamed/demoted and the corrected one promoted to the primary, with parameter naming clarified. Earlier dynamic test-size extraction (`n_test = len(np.load(".../test_idx_fold_1.npy"))`) made the correction dataset-independent before the `test_fraction` reframing.

**Wilcoxon edge case** (identical fold metrics → numerical failure):

```python
if np.allclose(diff, 0):
    w_stat, w_p = 0.0, 1.0
else:
    w_stat, w_p = wilcoxon(a, b)
```

**Effect size** is reported for both paired and independent comparisons. **Bonferroni** correction for multiple tests (`p_corr = min(p × m, 1.0)`):

```python
n_tests = len(results_df)
results_df["paired_t_p_bonferroni"]  = np.minimum(results_df["paired_t_p"]  * n_tests, 1.0)
results_df["wilcoxon_p_bonferroni"]  = np.minimum(results_df["wilcoxon_p"]  * n_tests, 1.0)
```

(Conservative even though metrics are correlated — acceptable for biomedical ML.) Comparisons cover: model (RF vs SVM), Haralick effect, magnification effect, dataset-balance effect. A **dual `verify_same_splits()`** implementation existed in both `stats_analysis.py` and `verify_splits.py`; consolidated. Output: `statistical_tests.csv` (p-values, t/Wilcoxon statistics, effect sizes, Bonferroni-adjusted p-values).

> Methodological limitations stated: results still depend on repeated-CV correlation structure; small fold counts; no external validation yet.

---

## 11. Explainability (`explainability.py`, `rf_analysis.py`)

Three methods: **RF Mean Decrease Impurity (MDI)**, **permutation importance**, **SHAP** (`shap.TreeExplainer(rf)`). Outputs: `feature_importance.png`, `permutation_importance.png`, `shap_summary.png`, plus CSV exports for downstream tables.

**Known issue — permutation importance on a full-data model (Severity: High, acknowledged not fixed).** The final RF is fit on all data (`final_rf.fit(X, y)`) and permutation importance is computed on `last_X_test` (fold-25 test set), which the model has already seen — reflecting memorization, not generalization. This does **not** affect the main CV metrics (folds are properly separated; tuning is inside training folds). SHAP-on-full-data is more defensible for global interpretability; permutation importance ideally needs strictly unseen data. Recommended fixes: a dedicated 10–15% explainability holdout, or fold-aggregated permutation importance (train on fold train, score on fold test, aggregate).

**Interpretation honesty (Severity: High).** Early notes claimed *"texture entropy and contrast dominated,"* but results showed color dominates: `mean_r` > `mean_g` > `mean_b`, Haralick contrast next, LBP low. The accurate framing: *"first-order color statistics, particularly mean RGB intensity, contributed more strongly than handcrafted texture; among texture features Haralick contrast was highest while LBP was low — color dominance may partly reflect staining variability."* **Key lesson: narrate the biology the analysis supports, not the biology expected.**

**Method reconciliation.** MDI/permutation/SHAP disagree because they measure different things (MDI is biased toward high-cardinality features; permutation is unstable on small test sets; SHAP uses a sampled subset). Recommended hierarchy: MDI = quick global overview, **SHAP = primary**, permutation = supplementary robustness check — stated explicitly so reviewers expect non-identical rankings.

> During the large sweep, all explainability (`plot_permutation_importance`, `plot_shap_summary_rf`, `run_rf_analysis`) was **disabled for speed** and re-enabled only for shortlisted models (§16).

---

## 12. Experiment Automation (`experiments.py`)

Manual per-experiment dict editing didn't scale (duplication, copy-paste errors, inconsistent naming). Replaced with **config-driven factorial generation**:

```python
MODELS = ["rf", "svm"]
MAGNIFICATIONS = ["100x", "400x"]
HARALICK_OPTIONS = [False, True]
DATASET_CONFIGS = {
    "100x": [("20img", 20, 20), ("89img", 89, 89), ("full", 89, 439)],
    "400x": [("20img", 20, 20), ("201img", 201, 201), ("full", 201, 495)],
}
from itertools import product
for model, magnification, haralick in product(MODELS, MAGNIFICATIONS, HARALICK_OPTIONS):
    ...   # → e.g. {"name": "rf_haralick_full_100x", "model": "rf", "haralick": True,
          #          "magnification": "100x", "num_normal": 89, "num_tumour": 439}
```

Day-1 grid = 2 models × 2 magnifications × 2 feature toggles × 3 sizes = **24 experiments**. The space was later expanded to a full factorial over model × magnification × **feature_set** × **scaling** × size × **stain normalization** × **seed**, which is *combinatorial* — the cause of the long runtimes in §16. Stain options `[None, "reinhard"]` (later `"macenko"`), seeds `[42, 123, 456, 789, 2024]`.

**Self-documenting names** encode all metadata, e.g. `rf_unscaled_color_no_norm_20img_100x_seed_42`. Outputs are later organised under seed folders: `results/seed_42/<experiment_name>/`.

**Validation (`validate_experiments`) + sanity tests:** count check (`assert len(EXPERIMENTS) == 24`), duplicate-name detection (`assert len(names) == len(set(names))`), required keys, positive dataset sizes (`assert exp["num_normal"] > 0`), equivalence with old manual configs.

**Early-return generator bug (severe):** `return experiments` was indented *inside* the loop, silently truncating to a subset of experiments (missing runs, incomplete results, misleading summaries). Fix: the return must be outside all loops.

**Skip-completed experiments (resumability).** `metrics.txt` is the completion marker (written only on full success):

```python
RESULTS_DIR = os.path.join("results", EXPERIMENT_NAME)
os.makedirs(RESULTS_DIR, exist_ok=True)
metrics_file = os.path.join(RESULTS_DIR, "metrics.txt")
if os.path.exists(metrics_file):
    print(f"Skipping completed experiment: {EXPERIMENT_NAME}")
    return
```

**Structured failure logging:** `logging.exception(f"Fold {fold} failed in {EXPERIMENT_NAME}")`, a `failed_experiments = []` list, and a final failure summary — replacing bare `print`.

---

## 13. Visualization (`visualize_results.py`, `analysis_plots.py`)

**Headless backend.** A `RuntimeError: main thread is not in main loop` / `Tcl_AsyncDelete: async handler deleted by the wrong thread` crash occurred because matplotlib used the Tkinter GUI backend while `GridSearchCV(n_jobs=-1)` multiprocessed and plotting ran repeatedly (unstable on Windows). Fixed by forcing the non-GUI Agg backend at the top of `train_model.py`, `analysis_plots.py`, `visualize_results.py`, `explainability.py`:

```python
import matplotlib
matplotlib.use("Agg")
```

Also, blocking `plt.show()` was replaced with `plt.savefig(path); plt.close()`.

**Plots:** ROC, PR, calibration, confusion matrices (per-fold + overall, normalized, with CSV export), feature importance, SHAP, permutation importance, class distribution (`class_distribution.png`), Pearson feature-correlation heatmap (`corr = np.corrcoef(X, rowvar=False)`), learning curves.

**Visualization refactor.** Save-directory creation, confusion-matrix CSV export, and plot titles were centralised (previously duplicated across functions), and `evaluation_utils.py` was slimmed. Two `NameError`-class bugs from an `ensure_save_dir` helper referenced an undefined `csv_path` in `ensure_save_dir` and in `plot_feature_importance`; fixed by correcting the parameter flow. An unused `pooled_auc` parameter was removed from one plotting signature.

---

## 14. Results & Scientific Conclusions

> **Important context:** the concrete numbers below are the **Day-1 baseline benchmark** (`StratifiedKFold` 5-fold, basic vs Haralick, before the GLCM-contamination fix, the repeated-CV fix, and the stain-normalization work). They were later understood to be *statistically incomplete* (5 evaluations, not the intended 25 — §8) and affected by the GLCM/stain issues. Treat them as initial baselines; the later large sweep produced richer but largely qualitative conclusions (below the tables).

### 14.1 Day-1 baseline (Accuracy / AUC)

**100× magnification**

| Config | 20 vs 20 (Acc/AUC) | 89 vs 89 | 89 vs 439 |
|---|---|---|---|
| RF + basic | 0.925 / 0.975 | 0.865 / 0.941 | 0.867 / 0.901 |
| SVM + basic | 0.800 / 0.962 | 0.916 / 0.972 | 0.871 / 0.944 |
| RF + Haralick | 0.950 / 0.950 | 0.882 / 0.955 | 0.888 / 0.914 |
| SVM + Haralick | 0.925 / 0.988 | 0.933 / 0.983 | 0.898 / 0.948 |

**400× magnification**

| Config | 20 vs 20 (Acc/AUC) | 201 vs 201 | 201 vs 495 |
|---|---|---|---|
| RF + basic | 0.900 / 0.894 | 0.804 / 0.892 | 0.800 / 0.814 |
| SVM + basic | 0.850 / 0.950 | 0.833 / 0.906 | 0.777 / 0.857 |
| RF + Haralick | 0.875 / 0.912 | 0.806 / 0.884 | 0.805 / 0.847 |
| SVM + Haralick | 0.900 / 0.962 | 0.841 / 0.905 | 0.795 / 0.872 |

### 14.2 Day-1 observations

1. **SVM generally outperformed RF** (higher AUC, more balanced sensitivity/specificity).
2. **Haralick features helped**, especially for SVM.
3. **Tiny (20 vs 20) datasets gave inflated scores** (overfitting, low variability, optimistic evaluation).
4. **Imbalance raised sensitivity but cut specificity** — the model over-predicts tumour. Example (`rf_full_100x`): sensitivity = 0.982, specificity = 0.302.
5. **100× was more stable than 400×** (higher AUC, stronger specificity).

### 14.3 Large-sweep scientific conclusions

A. More data usually helps (the 20-image subset was far less stable). B. PR-AUC is often more honest than accuracy under imbalance. C. **Specificity exposed the real weakness** (high sensitivity, poor specificity in some runs). D. Neither RF nor SVM universally dominated. E. Magnification matters (100× ≠ 400×). F. **Stain normalization is not universally beneficial** (Reinhard helped some settings, not all). G. Feature combinations matter (some families clearly better). H. **MCC is important** — it exposed when imbalance made performance look better than it was.

Reporting guidance: foreground MCC, specificity, and PR-AUC; don't rely on accuracy; compare magnifications carefully; don't assume normalization always helps; use one seed for screening and multiple for validation; reserve heavy explainability for shortlisted models. The project's narrative is a **study of how handcrafted histopathology features behave under different data conditions**, not merely a classifier benchmark.

---

## 15. Reproducibility & Repository Engineering

- **Seeds.** Multi-seed list `[42, 123, 456, 789, 2024]`. A bug where the seed was set in configs but not propagated was fixed by threading `seed`/`random_state=seed` into `run_fold`, the inner `StratifiedKFold`, `RandomForestClassifier`/`SVC`, permutation importance, and split-directory names (`seed_{seed}`).
- **Configuration transparency.** Scattered thresholds/seeds/folds/sizes/hyperparameters were centralised into `configs/preprocessing_config.py`, `experiment_config.py`, `model_config.py`, replacing magic numbers with imports.
- **Unit tests.** `tests/` covers preprocessing (mask validity, tissue thresholds, dark-image guard), dataset logic (deterministic sampling, subset consistency), feature extraction (output shape, NaN prevention), and splitting (reproducibility, leakage prevention). A `ModuleNotFoundError: No module named 'src'`/`'dataset'` under pytest was fixed by adding repo root to the path in `conftest.py`:
  ```python
  import sys; from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  ```
  Result: `11 passed in 3.06s`.
- **Recommended execution order:** `python generate_splits.py` → `python train_model.py` → `python stats_analysis.py`.
- **Repo hygiene.** `.gitignore` redesigned with selective artifact tracking; git staging issues debugged; **MIT license** added; repository frozen/versioned as a clean milestone.
- **Git history cleanup:** `git reset --soft HEAD~1` (drop last commit, keep changes staged) then `git push --force` (rewrite remote branch) so the repo could receive one clean commit for the stabilized state.
- **README** expanded with research objective, experiment scale, reproducibility, repository structure, limitations, future directions, and a Mermaid pipeline diagram.
- **Summary-CSV deletion bug + schema fixes** — see §17 (these are reproducibility-critical).

---

## 16. Operational Notes

- **Environment.** Python venv on Windows. `python -m venv venv` failed with `Permission denied: venv\Scripts\python.exe` due to **OneDrive file-locking** — fixed by moving the project outside OneDrive, then `venv\Scripts\activate` + `pip install -r requirements.txt`. A later `ModuleNotFoundError: No module named 'cv2'` was just dependencies not installed inside the venv.
- **Reproducibility seed (day-1):** `random_state=42` everywhere.
- **Runtime explosion.** Felt "impossibly slow" not from a single bug but the multiplicative experiment grid (model × scaling × feature set × stain × magnification × size × seed → hundreds–thousands of runs). The project had become a benchmark suite, not a script.
- **Thermals.** Loud fans but stable CPU/memory/disk/temps = busy, not failing. The laptop had been propped **upside-down** for airflow — corrected (risk of screen/hinge damage); proper setup = flat hard surface, clear vents, small rear lift.
- **Sleep/hibernate killed overnight runs** (Windows logged out, closed VS Code). This motivated resumable experiments (skip-completed via `metrics.txt`, §12) and inferring progress from result folders/CSVs/timestamps.
- **Speed levers identified:** pre-generated CV splits; dataset loaded once per experiment; small feature dimensionality (e.g. 178 samples × 16 features); Haralick disabled in some experiments; SHAP run once (not per fold). Disabling explainability during the sweep was the biggest win; timestamps showed each experiment then finished in minutes.

---

## 17. Consolidated Debugging Catalogue

Every distinct error encountered, with root cause and fix.

**Environment / tooling**
- `Permission denied: venv\Scripts\python.exe` — OneDrive file lock → move project out of OneDrive.
- `ModuleNotFoundError: No module named 'cv2'` — deps not in venv → `pip install -r requirements.txt`.
- `ModuleNotFoundError: No module named 'src'` / `'dataset'` (pytest) — repo root not on path → add it in `tests/conftest.py`.
- Ruff/Pylance "Undefined name 'magnification'…" — indentation/scope; keep cache-key creation inside the function.
- Pylance greyed-out `rmin/rmax/cmin/cmax` — expected: unused after switching from bounding-box crop to masking.

**Crashes / runtime**
- `RuntimeError: main thread is not in main loop` + `Tcl_AsyncDelete…` — Tkinter backend + multiprocessing on Windows → `matplotlib.use("Agg")`.
- Blocking `plt.show()` pausing runs → `plt.savefig(); plt.close()`.
- `KeyboardInterrupt` during plotting — manual Ctrl+C, not a bug.

**Feature / preprocessing correctness**
- Haralick GLCM over full background-containing image — contamination → mask + 16-level quantize + 4-angle averaging (`features.py` and `demo.py`).
- `extract_features(..., feature_set="feature_set")` — literal string passed → use the variable.
- Unreachable `return normalized` after `return normalized.astype(np.uint8)` + leftover debug prints → removed.
- Duplicate `MIN_TISSUE_FRACTION` in `features.py` and config → import from config.

**Stain normalization**
- `"reinhard"` branch = `raise NotImplementedError` while running experiments → `no_norm == reinhard`, all stain tests `t=0, p=1`, inflated Bonferroni → implement real Reinhard (staintools) + regression test.
- `stain_normalization` not passed to `get_or_cache_dataset` → silently `None`; Reinhard/Macenko ran unnormalized despite cache names → pass `config.get("stain_normalization", None)`; invalidate caches.
- torchstain: `TypeError: Invalid shape (512, 3, 512)` / `cv2.error … 'scn' is 512` — extra permute on already-`(H,W,C)` output → drop the permute.
- `AttributeError: 'numpy.ndarray' object has no attribute 'numpy'` — `.numpy()` on a NumPy array → only call on tensors.
- `AttributeError: 'Tensor' object has no attribute 'astype'` — `.astype` on a tensor → `.numpy()` first.
- Neon-purple Macenko output — poor stain-basis estimation from a non-representative reference image → choose/benchmark a balanced reference. (0–1 scaling attempt collapsed the image; torchstain expects 0–255 floats.)
- `ValueError: Could not load image: data/target_stain_image.png` — extension mismatch → `.jpg`.
- `ImportError: cannot import name 'normalize_staining'` — function removed in refactor → restored.

**Experiment / results pipeline**
- Generator `return experiments` indented inside loop → silent experiment truncation → move outside loops.
- `initialize_summary_csv` deleting the summary every start (`os.remove`) → empty master report after skips; changed to "return if exists, else write header."
- `KeyError: 'stain_normalization'` — learning-curve filter wanted a column the summary schema lacked → add column to header + `append_summary_row`.
- `KeyError: 'Column not found: auc'` — summary stores `auc_mean`/`auc_std`, not `auc` → learning curve uses the aggregate columns.
- `ValueError: No experiments matched learning curve filter.` — filter too strict vs stored values → broaden/match.
- Empty summary after rebuild — rebuild scanned `results/` directly; experiments are nested under `results/seed_42/<name>/` → recurse into seed folders.
- `ImportError: cannot import name 'plot_permutation_importance' from 'src.explainability'` — call removed but import left → remove the import too.

**Cross-validation**
- Only folds 1–5 evaluated (one repeat) → nested `repeat × fold` loop + repeat-aware filenames.
- Duplicate `fold=fold` kwarg in `run_fold(...)` → remove duplicate.
- `IndexError: index N is out of bounds for axis 0 with size N` (391, 672) — splits built without normalization/QC, training built with them → regenerate splits with matching `stain_normalization`/`feature_set`/`seed`; put `stain_normalization` (not `feature_set`) in the split key; delete stale `cv_splits/` and Macenko caches.

**Statistics**
- IID-assuming CI / paired t-test under repeated CV → Nadeau–Bengio correction (`ρ = test_fraction = 1/k`).
- Misleading `n_test` param → renamed `test_fraction`.
- Wilcoxon failure on identical fold metrics → `if np.allclose(diff, 0): (0.0, 1.0)`.
- Dual `verify_same_splits()` in two files → consolidate; fix undefined `n_folds`, missing repeat iteration, outdated filenames.
- Everything treated as paired → split into paired (corrected t + Wilcoxon) vs independent (Welch + Mann-Whitney).

---

## 18. Lessons Learned

- **Tensor conventions from external libraries are dangerous** — never assume ordering; `print(tensor.shape)` early; build isolated tests (`test.py`) before rerunning multi-hour experiments.
- **Cached artifacts propagate silent bugs** — feature caches and CV splits go stale when preprocessing/QC/normalization/filtering change; version caches and validate them; cache *names* are not validation.
- **Preprocessing parameters must propagate explicitly** — silent defaults (`stain_normalization=None`) and unimplemented branches (`NotImplementedError`) created the two most serious scientific-integrity bugs; add verification artifacts and regression tests that assert outputs actually changed.
- **CV splits must match the exact training dataset** — including normalization, filtering, QC removal, and duplicate handling; QC changes cardinality even in deterministic pipelines.
- **Statistical correctness matters** — repeated CV needs corrected inference; report the right uncertainty, not the prettiest.
- **Interpretability must be built in and honest** — narrate the biology the analysis supports; reconcile MDI/SHAP/permutation explicitly.
- **Experiment automation prevents human error**, but factorial grids are combinatorial — stop expanding the brute-force sweep once the matrix is large enough.
- **Presentation matters** — clean repo, README, license, versioning, and self-documenting artifact names make the work reviewable and resumable.

---

## 19. Limitations & Future Work

**Current limitations:** single-center/single-scanner data (uncertain cross-lab generalization); classical handcrafted features only; no external validation; no patient metadata yet (potential patient-level leakage); color features remain stain-sensitive; inner hyperparameter CV uses small validation folds; results depend on repeated-CV correlation structure and small fold counts; permutation importance currently computed on a full-data model.

**Future work:** dedicated explainability holdout or fold-aggregated permutation importance; finish/benchmark Macenko (reference-image selection, magnification-specific targets, debug triplets, downstream-task evaluation rather than pixel similarity); `StratifiedGroupKFold` for patient-level validation; external-dataset validation; statistical comparison between feature groups; cache metadata + `PREPROCESSING_VERSION`; optional multi-scale LBP; and engineering polish (dataclass result objects, YAML/Hydra/pydantic configs, package subdivision, environment/version logging).

**Key Macenko hypothesis to test:** if RGB feature importance drops after Macenko while texture features stay predictive, morphology contributes independently of staining artifacts.

---

## Appendix A — `results/` directory layout (per experiment)

```text
results/seed_<seed>/<experiment_name>/
├── config.json
├── dataset_manifest.json
├── cross_validation_config.json   # documents that splits are PRECOMPUTED (loaded from cv_splits/)
├── fold_metrics.csv
├── fold_predictions.csv
├── summary_metrics.csv
├── metrics.txt                    # completion marker
├── best_hyperparameters.csv
├── roc_curve.png / roc_curve_data.csv
├── pr_curve.png  / pr_curve_data.csv
├── confusion_matrix_repeat_<r>_fold_<f>.png
├── confusion_matrix_overall.png / .csv
├── calibration_curve_repeat_<r>_fold_<f>.png
├── feature_importance.png / shap_summary.png / permutation_importance.png
├── class_distribution.png / feature_correlation.png
└── train_idx_repeat_<r>_fold_<f>.npy / test_idx_repeat_<r>_fold_<f>.npy
```

> `cross_validation_config.json` (`n_splits`, `n_repeats`, `random_state`) once implied experiments generated their own splits; actual provenance is the precomputed `cv_splits/` directory — documented to avoid the false impression of self-contained reproducibility.

## Appendix B — `requirements.txt`

```text
contourpy==1.3.3
cycler==0.12.1
fonttools==4.62.1
ImageIO==2.37.3
joblib==1.5.3
kiwisolver==1.5.0
lazy-loader==0.5
matplotlib==3.10.9
networkx==3.6.1
numpy==2.4.4
opencv-python==4.13.0.92
packaging==26.2
pandas==3.0.2
pillow==12.2.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
scikit-image==0.26.0
scikit-learn==1.8.0
scipy==1.17.1
six==1.17.0
threadpoolctl==3.6.0
tifffile==2026.5.2
tzdata==2026.2
```
Additional stain-normalization dependencies installed later: `staintools` (Reinhard; Windows: `spams-bin`), `torch` + `torchstain` (Macenko).
