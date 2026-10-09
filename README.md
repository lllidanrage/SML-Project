# COMP90051 Statistical Machine Learning Project

## 安装依赖

在项目根目录运行：

```bash
pip install -r requirements.txt
```

项目主要使用：

- NumPy
- pandas
- Pillow
- SciPy
- scikit-image
- scikit-learn
- matplotlib
- threadpoolctl
- LightGBM


## 项目概述

本项目基于 **Intel Image Classification** 数据集研究场景图像分类，并重点考察不同模型和不同图像特征在图像质量下降时的 **robustness（鲁棒性）**。

与单纯比较分类准确率不同，本项目同时关注两个问题：

1. 不同 feature representation 在正常图像上的分类能力是否存在明显差异；
2. 当图像出现 **Gaussian blur** 或 **brightness reduction** 时，不同特征与模型的性能下降模式是否不同。
3. 不同 feature representations 是分别使用更好，还是将多种 feature 信息结合起来能够获得更稳定的分类表现和更强的 robustness。

为了保证模型之间的比较公平，所有模型使用相同的数据、feature files、corruption conditions，以及完全相同的 nested cross-validation splits。

---

## Research Question

> **How robust are different image classification models and feature representations to increasing levels of image degradation?**

项目主要比较四种人工构造的视觉特征：

- **Colour Histogram**
- **HOG**
- **LBP**
- **Spatial Features**

并使用三个不同复杂度的分类模型：

| Model | Role |
|---|---|
| Multiclass Logistic Regression | Simple model |
| Multilayer Perceptron (MLP) | Medium-complexity model |
| LightGBM | Complex model |

LightGBM 作为较新的复杂模型，用于与前两个模型在相同实验条件下进行最终比较。

---

## 数据集

本项目使用 Kaggle 上的 **Intel Image Classification** 数据集：

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

使用的六个场景类别为：

- buildings
- forest
- glacier
- mountain
- sea
- street

项目读取 `seg_train` 和 `seg_test` 中带标签的图像。原始宽度或高度小于 100 pixels 的图像在 preprocessing 前被排除，其余图像统一转换为 RGB，并 resize 为 **150 × 150**。

数据目录：

```text
img_data/
├── seg_train/
│   ├── buildings/
│   ├── forest/
│   ├── glacier/
│   ├── mountain/
│   ├── sea/
│   └── street/
└── seg_test/
    ├── buildings/
    ├── forest/
    ├── glacier/
    ├── mountain/
    ├── sea/
    └── street/
```

---

## Feature Representations

`Project.ipynb` 为每张图像提取四种 feature representations。

| Feature | Description |
|---|---|
| **Colour Histogram** | 描述整体颜色分布 |
| **HOG** | 描述边缘方向和局部形状结构 |
| **LBP** | 描述局部纹理模式 |
| **Spatial Features** | 将图像划分为空间区域，并结合局部 Colour、HOG 和 LBP 信息，以保留一定的空间布局 |

这些 features 在所有模型之间共享，因此模型比较不会受到不同 preprocessing pipeline 的影响。

---

## Robustness Conditions

模型不仅在 clean images 上测试，还会在两种 controlled corruption 下测试。

### Gaussian Blur

用于模拟图像清晰度下降：

```text
sigma = 0, 1, 2, 3
```

对应：

```text
clean → mild → medium → severe
```

### Brightness Reduction

用于模拟光照逐渐变暗：

```text
factor = 1.0, 0.8, 0.6, 0.4
```

对应：

```text
clean → mild → medium → severe
```

所有 corruption 都在 **feature extraction 之前**应用于图像，因此模型实际接收到的是对应 degraded image 提取出的 features。

生成的 feature files：

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

`corruption_visualization.png` 用于直观展示同一张图像在不同 degradation level 下的变化。

---

## Experimental Protocol

三个模型使用同一套 **Nested Cross-Validation**：

```text
Outer CV: 10 folds
Inner CV: 3 folds
```

每个 outer fold 的流程为：

```text
Outer training fold
        ↓
Inner 3-fold CV
        ↓
Hyperparameter selection
        ↓
Retrain on the full outer training fold
        ↓
Evaluate on the corresponding outer test fold
```

Hyperparameter tuning 只使用 **clean training data**。

模型训练完成后，同一个 outer-fold model 会在完全相同的 test indices 上分别测试：

```text
clean
blur_mild
blur_medium
blur_severe
brightness_mild
brightness_medium
brightness_severe
```

因此 robustness comparison 反映的是图像 degradation 对模型预测的影响，而不是不同数据划分造成的差异。

共享的 fold indices 保存在：

```text
results/outer_and_inner_folds.pkl
```

Logistic Regression 在该文件不存在时创建 folds；其余模型直接复用同一份划分。

---

## Evaluation Metrics

所有模型使用相同的评价指标：

- Accuracy
- Macro-Precision
- Macro-Recall
- Macro-F1

最终结果基于 10 个 outer folds 汇总，并报告 mean 和 standard deviation，用于比较不同模型、不同 feature representations 和不同 corruption severity 下的表现。

---

## Notebook 运行顺序

### 1. `Project.ipynb`

负责整个 preprocessing 和 feature extraction pipeline：

- 读取并整理图像；
- 过滤不符合尺寸要求的原始图像；
- RGB conversion 与 resizing；
- 构造 Gaussian blur 和 brightness conditions；
- 提取四种 feature representations；
- 保存所有 feature files。

### 2. `Logistic_Regression_Robustness.ipynb`

完成 Logistic Regression 实验，包括：

- nested cross-validation；
- hyperparameter `C` tuning；
- clean performance evaluation；
- blur / brightness robustness evaluation；
- 保存 fold-level results 和 tuning results。

### 3. `MLP_Robustness.ipynb`

完成 MLP 实验，并复用 Logistic Regression 创建的相同 CV splits。

MLP 使用单隐藏层网络，并通过 nested CV 对正式 hyperparameter 进行选择，然后在七种 image conditions 下进行相同的 robustness evaluation。

### 4. LightGBM

LightGBM 使用相同的 features、outer/inner folds、evaluation metrics 和 robustness conditions，从而与 Logistic Regression 和 MLP 保持一致的实验设计。

### 5. Final Model Comparison

三个模型完成后，将各自保存的 results 合并，用于最终比较：

- clean classification performance；
- robustness under Gaussian blur；
- robustness under brightness reduction；
- 不同 feature representation 的稳定性；
- 三个模型在相同实验条件下的性能差异。

---

## Results

每个模型的结果单独保存，避免不同实验之间相互覆盖：

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
    └── lightgbm_inner_tuning.csv
```

`*_results.csv` 保存每个 outer fold、feature representation 和 corruption condition 下的 evaluation metrics。

`*_inner_tuning.csv` 保存 inner CV 中不同 hyperparameter candidates 的表现。

这种结构使模型训练和最终模型比较彼此独立，同时方便后续统一生成表格和 robustness figures。

---

## 项目目录结构

```text
SML-Project/
├── Project.ipynb
├── Logistic_Regression_Robustness.ipynb
├── MLP_Robustness.ipynb
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

整体结构遵循：

> **Preprocessing shared once → models trained separately → results compared together**

这样既保持了实验流程的一致性，也使每个模型的代码和结果保持独立、清晰且容易复现。

---

