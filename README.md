# COMP90051 Statistical Machine Learning Project

## Installation

From the project root:

```bash
pip install -r requirements.txt
```

Key dependencies: NumPy, pandas, Pillow, SciPy, scikit-image, scikit-learn, matplotlib, threadpoolctl, LightGBM.

---

## Overview

This project studies scene image classification on the **Intel Image Classification** dataset, with a focus on the **robustness** of different models and feature representations under image degradation.

Beyond comparing classification accuracy on clean images, the project investigates:

1. Whether different feature representations differ meaningfully in classification performance on clean images.
2. Whether different features and models degrade differently under **Gaussian blur** or **brightness reduction**.

To ensure fair comparison, all models share the same data, feature files, corruption conditions, and nested cross-validation splits.

---

## Research Question

> **How robust are different image classification models and feature representations to increasing levels of image degradation?**

Four hand-crafted visual features are compared:

- **Colour Histogram**
- **HOG**
- **LBP**
- **Spatial Features**

Across three models of increasing complexity:

| Model | Role |
|---|---|
| Multiclass Logistic Regression | Simple model |
| Multilayer Perceptron (MLP) | Medium-complexity model |
| LightGBM | Complex model |

---

## Dataset

**Intel Image Classification** dataset from Kaggle:

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Six scene categories are used:

- buildings, forest, glacier, mountain, sea, street

Images are loaded from both `seg_train` and `seg_test`. Images whose original width or height is below 100 pixels are excluded. All remaining images are converted to RGB and resized to **150 × 150**.

```text
img_data/
├── seg_train/
│   ├── buildings/  forest/  glacier/  mountain/  sea/  street/
└── seg_test/
    ├── buildings/  forest/  glacier/  mountain/  sea/  street/
```

---

## Feature Representations

`Project.ipynb` extracts four feature representations per image.

| Feature | Description |
|---|---|
| **Colour Histogram** | Captures overall colour distribution across RGB channels |
| **HOG** | Captures edge orientations and local shape structure |
| **LBP** | Captures local texture patterns |
| **Spatial Features** | Divides the image into a spatial grid and combines local Colour, HOG, and LBP features to preserve spatial layout |

Features are shared across all models, so comparisons are not affected by differences in preprocessing.

---

## Robustness Conditions

Models are evaluated not only on clean images but also under two types of controlled corruption.

### Gaussian Blur

Simulates loss of image sharpness:

```text
sigma = 0, 1, 2, 3  →  clean, mild, medium, severe
```

### Brightness Reduction

Simulates decreasing illumination:

```text
factor = 1.0, 0.8, 0.6, 0.4  →  clean, mild, medium, severe
```

All corruptions are applied to images **before** feature extraction, so models receive features derived from the degraded images.

Generated feature files:

```text
features/
├── clean.npz
├── blur_mild.npz
├── blur_medium.npz
├── blur_severe.npz
├── brightness_mild.npz
├── brightness_medium.npz
├── brightness_severe.npz
└── corruption_visualization.png
```

`corruption_visualization.png` shows the same image across all degradation levels.

---

## Experimental Protocol

All three models use the same **nested cross-validation** scheme:

```text
Outer CV: 10 folds
Inner CV: 3 folds
```

Per outer fold:

```text
Outer training set
      ↓
Inner 3-fold CV  →  hyperparameter selection
      ↓
Retrain on full outer training set
      ↓
Evaluate on outer test set
```

Hyperparameter tuning uses **clean training data only**. The trained model is then evaluated on the same test indices across all seven conditions:

```text
clean, blur_mild, blur_medium, blur_severe,
brightness_mild, brightness_medium, brightness_severe
```

This ensures that robustness comparisons reflect the effect of image degradation, not differences in data splits.

The shared fold indices are saved to `results/outer_and_inner_folds.pkl`. The Logistic Regression notebook creates this file on first run; the remaining model notebooks reuse the same splits.

---

## Evaluation Metrics

All models are evaluated using:

- Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1

Results are aggregated across 10 outer folds and reported as mean ± standard deviation.

---

## Notebook Execution Order

### 1. `Project.ipynb`

Handles the full preprocessing and feature extraction pipeline:

- Load images from `seg_train` and `seg_test` as one combined dataset
- Filter undersized images; convert to RGB; resize to 150 × 150
- Apply Gaussian blur and brightness corruption at each severity level
- Extract the four feature representations for each condition
- Save all feature files to `features/`

### 2. `Logistic_Regression_Robustness.ipynb`

Runs the Logistic Regression experiment:

- Nested cross-validation with hyperparameter `C` tuning
- Evaluation on clean and all six corrupted conditions
- Saves fold-level results and inner tuning results

### 3. `MLP_Robustness.ipynb`

Runs the MLP experiment, reusing the CV splits created by the LR notebook:

- Single hidden layer MLP with nested CV for `alpha` tuning
- Per-feature checkpointing to support resuming after interruption
- Same robustness evaluation across seven conditions
- Saves fold-level results and inner tuning results

### 4. `LightGBM_Robustness.ipynb`

Runs the LightGBM experiment, reusing the same CV splits:

- Nested CV with `num_leaves` tuning
- Per-fold checkpointing to support resuming after interruption
- Same robustness evaluation across seven conditions
- Saves fold-level results and inner tuning results

### 5. Analysis

Each model notebook includes an analysis section at the end showing clean performance and robustness degradation curves per feature representation. Cross-model comparison is done by combining the three `*_results.csv` files.

---

## Results

Each model's results are saved independently to avoid overwriting across experiments:

```text
results/
├── outer_and_inner_folds.pkl
├── logistic_regression/
│   ├── logistic_regression_results.csv
│   └── logistic_regression_inner_tuning.csv
├── mlp/
│   ├── mlp_results.csv
│   └── mlp_inner_tuning.csv
└── lightgbm/
    ├── lightgbm_results.csv
    ├── lightgbm_inner_tuning.csv
    └── checkpoints/
```

`*_results.csv` records evaluation metrics per outer fold, feature representation, and corruption condition.

`*_inner_tuning.csv` records the inner CV performance of each hyperparameter candidate.

---

## Project Structure

```text
SML-Project/
├── Project.ipynb                        # preprocessing & feature extraction
├── Logistic_Regression_Robustness.ipynb # simple model
├── MLP_Robustness.ipynb                 # medium model
├── LightGBM_Robustness.ipynb            # complex model
├── README.md
├── requirements.txt
│
├── img_data/
├── features/
└── results/
    ├── outer_and_inner_folds.pkl
    ├── logistic_regression/
    ├── mlp/
    └── lightgbm/
```

> **Preprocessing shared once → models trained separately → results compared together**

This structure keeps each model's code and results self-contained, while ensuring all models operate on identical data, features, and splits.

---
