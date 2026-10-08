# SML-Project

COMP90051 Statistical Machine Learning — Group Project

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖库：
- `numpy` — 数组计算
- `Pillow` — 图片加载
- `scipy` — Gaussian blur（图像干扰生成）
- `scikit-image` — HOG、LBP 特征提取
- `matplotlib` — 结果可视化

## 数据集

Intel Image Classification（Kaggle）：https://www.kaggle.com/datasets/puneet6060/intel-image-classification

下载后将数据解压至项目根目录，目录结构如下：

```
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
    └── ...
```

## 运行方式

按顺序执行 `Project.ipynb` 中的所有 cell：

1. 加载图片
2. 提取视觉特征（Colour Histogram、HOG、LBP、Spatial Combined）
3. 生成图像干扰（Gaussian Blur / Brightness 各 4 个等级）
4. 提取干扰图片的特征
5. 将所有特征保存至 `features/` 目录

特征文件生成后，后续模型训练直接读取 `features/train_feats.pkl` 和 `features/test_feats.pkl`。
