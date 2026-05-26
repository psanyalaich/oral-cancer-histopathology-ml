from configs.preprocessing_config import STAIN_NORMALIZATION_METHODS

N_SPLITS = 5
N_REPEATS = 5
TOTAL_CV_ITERATIONS = N_SPLITS * N_REPEATS
SEEDS = [42] # 42 103 368
MODELS = ["rf", "svm"]
MAGNIFICATIONS = ["100x", "400x"]
STAIN_NORMALIZATION_OPTIONS = (STAIN_NORMALIZATION_METHODS)

FEATURE_SETS = [
    "color",
    "lbp",
    "haralick",
    "color_lbp",
    "color_haralick",
    "lbp_haralick",
    "all"
]
