# ⌚ med-har-transformer-analysis
**Real-Time Human Activity Recognition (HAR) via Time-Series Transformers**

[![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Analysis-Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-HAR--70-20BEB6?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/)

## 📘 Project Overview
This repository implements a high-performance **Time-Series Transformer** architecture to classify human activities in real-time using multivariate sensor data from wearable devices.

Unlike traditional RNNs or LSTMs, this project leverages the **Attention Mechanism** to capture long-range temporal dependencies and inter-sensor correlations, significantly improving accuracy in complex movement patterns.

### 🏗️ Technical Approach
- **Data Pipeline**: Processing of the HAR-70 dataset, involving sliding-window segmentation and Z-score normalization.
- **Architecture**: A Transformer Encoder with Multi-Head Attention and Positional Encoding to preserve the temporal order of accelerometer and gyroscope readings.
- **Analysis**: End-to-end workflow from Signal Processing $\rightarrow$ Feature Engineering $\rightarrow$ Transformer Training $\rightarrow$ Validation.

---

## 📂 Repository Structure
```text
├── project_work.ipynb           # Master analysis & training notebook
├── HAR_Transformer_Documentation.pdf # Full technical report
└── README.md                   # Project documentation
```
*(Note: For production deployment, the logic in `project_work.ipynb` is designed to be modularized into `src/` as described in the technical report).*

---

## ⚙️ Workflow & Implementation

### 1. Signal Processing
The pipeline handles multivariate streams (X, Y, Z axes) from multiple sensors, converting raw signals into fixed-length overlapping windows to ensure no temporal information is lost at the boundaries.

### 2. Transformer Architecture
- **Multi-Head Attention**: Allows the model to attend to different parts of the activity window simultaneously.
- **Positional Encoding**: Adds spatial/temporal context to the sensor readings.
- **MLP Head**: A final classification layer to map attention outputs to activity labels (Walking, Running, etc.).

### 3. Evaluation Metrics
The model is evaluated using:
- **Confusion Matrices**: To identify specific activity misclassifications.
- **F1-Score & Accuracy**: Balancing precision and recall across all activity classes.
- **ROC Curves**: Analyzing the trade-off between true positive and false positive rates.

---

## 🚀 Getting Started
To reproduce the results:
1. Clone the repository.
2. Install dependencies: `pip install torch pandas scikit-learn matplotlib seaborn`.
3. Run the `project_work.ipynb` notebook.

---
*Developed by Adelugba Adejare*
