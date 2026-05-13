# experiment configurations

EXPERIMENTS = [

    # 100x RF
    {
        "name": "rf_20img_100x",
        "model": "rf",
        "haralick": False,
        "magnification": "100x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "rf_89img_100x",
        "model": "rf",
        "haralick": False,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 89,
    },

    {
        "name": "rf_full_100x",
        "model": "rf",
        "haralick": False,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 439,
    },

    # 100x SVM
    {
        "name": "svm_20img_100x",
        "model": "svm",
        "haralick": False,
        "magnification": "100x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "svm_89img_100x",
        "model": "svm",
        "haralick": False,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 89,
    },

    {
        "name": "svm_full_100x",
        "model": "svm",
        "haralick": False,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 439,
    },

    # 100x RF HARALICK
    {
        "name": "rf_haralick_20img_100x",
        "model": "rf",
        "haralick": True,
        "magnification": "100x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "rf_haralick_89img_100x",
        "model": "rf",
        "haralick": True,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 89,
    },

    {
        "name": "rf_haralick_full_100x",
        "model": "rf",
        "haralick": True,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 439,
    },

    # 100x SVM HARALICK
    {
        "name": "svm_haralick_20img_100x",
        "model": "svm",
        "haralick": True,
        "magnification": "100x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "svm_haralick_89img_100x",
        "model": "svm",
        "haralick": True,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 89,
    },

    {
        "name": "svm_haralick_full_100x",
        "model": "svm",
        "haralick": True,
        "magnification": "100x",
        "num_normal": 89,
        "num_tumour": 439,
    },

    # 400x RF
    {
        "name": "rf_20img_400x",
        "model": "rf",
        "haralick": False,
        "magnification": "400x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "rf_201img_400x",
        "model": "rf",
        "haralick": False,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 201,
    },

    {
        "name": "rf_full_400x",
        "model": "rf",
        "haralick": False,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 495,
    },

    # 400x SVM
    {
        "name": "svm_20img_400x",
        "model": "svm",
        "haralick": False,
        "magnification": "400x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "svm_201img_400x",
        "model": "svm",
        "haralick": False,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 201,
    },

    {
        "name": "svm_full_400x",
        "model": "svm",
        "haralick": False,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 495,
    },

    # 400x RF HARALICK
    {
        "name": "rf_haralick_20img_400x",
        "model": "rf",
        "haralick": True,
        "magnification": "400x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "rf_haralick_201img_400x",
        "model": "rf",
        "haralick": True,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 201,
    },

    {
        "name": "rf_haralick_full_400x",
        "model": "rf",
        "haralick": True,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 495,
    },

    # 400x SVM HARALICK
    {
        "name": "svm_haralick_20img_400x",
        "model": "svm",
        "haralick": True,
        "magnification": "400x",
        "num_normal": 20,
        "num_tumour": 20,
    },

    {
        "name": "svm_haralick_201img_400x",
        "model": "svm",
        "haralick": True,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 201,
    },

    {
        "name": "svm_haralick_full_400x",
        "model": "svm",
        "haralick": True,
        "magnification": "400x",
        "num_normal": 201,
        "num_tumour": 495,
    },

]
