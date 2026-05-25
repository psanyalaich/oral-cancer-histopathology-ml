N_FOLDS = 25
SEEDS = [42] # 42 103 368
MODELS = ["rf", "svm"]
MAGNIFICATIONS = ["100x", "400x"]
STAIN_NORMALIZATION_OPTIONS = [None, "reinhard"]

FEATURE_SETS = [
    "color",
    "lbp",
    "haralick",
    "color_lbp",
    "color_haralick",
    "lbp_haralick",
    "all"
]
