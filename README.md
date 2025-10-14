
# 🧠 Real-Time Human Activity Recognition Using Time Series Transformers & Wearable Sensors (HAR 70+ Dataset)

## 📘 Project Overview

This project applies Time Series Transformers to the task of real-time Human Activity Recognition (HAR) using data collected from wearable sensors.
The goal is to classify human movements (like walking, sitting, standing, running, etc.) from continuous multivariate sensor readings.

The notebook demonstrates a full end-to-end workflow — from exploring the HAR dataset to training and evaluating a transformer-based deep learning model capable of understanding temporal dependencies in sensor data.



## 📊 Dataset Description

Dataset: HAR 70+ (Human Activity Recognition Dataset with Wearable Sensors)

The dataset contains sensor readings from devices such as smartphones, accelerometers, and gyroscopes worn on different parts of the body. Each data point corresponds to a small time window of motion readings.

Key features include:

* Sensor channels: Acceleration, gyroscope, and orientation values (x, y, z).
* Sampling rate: Typically recorded at fixed intervals (e.g., 50–100 Hz).
* Target labels: Activity types like walking, running, sitting, standing, lying, climbing stairs, etc.

Data columns (general structure):

* Timestamp
* Sensor ID / Subject ID
* Accelerometer (X, Y, Z)
* Gyroscope (X, Y, Z)
* Activity Label



## ⚙️ Code Structure

### 1. Data Loading and Preprocessing

* Imports necessary libraries (Pandas, NumPy, Matplotlib, PyTorch, etc.).
* Loads the HAR 70+ dataset and merges sensor data files if needed.
* Normalizes sensor readings and encodes activity labels.
* Splits the data into training, validation, and test sets.

### 2. Exploratory Data Analysis (EDA)

* Visualizes activity distributions and sensor signal patterns.
* Examines correlations between accelerometer and gyroscope axes.
* Plots sample activity waveforms to illustrate differences in movement.

### 3. Feature Scaling and Windowing

* Segments continuous sensor streams into fixed-length time windows.
* Extracts overlapping frames to preserve temporal continuity.
* Scales numeric features for consistent model input.

### 4. Model Design — Time Series Transformer

* Implements a **Transformer Encoder** architecture adapted for time series.
* Uses positional encoding to retain order information.
* Employs multi-head attention to capture relationships between sensor readings over time.
* Outputs activity class probabilities.

### 5. Training and Validation

* Trains the model using an optimizer such as **Adam** and a loss function like CrossEntropyLoss.
* Monitors training and validation accuracy/loss across epochs.
* Saves the best-performing model based on validation performance.

### 6. Evaluation and Visualization

* Evaluates model performance using:

  * Accuracy
  * Confusion Matrix
  * Classification Report (Precision, Recall, F1-Score)
  * ROC Curves for multi-class activity prediction
* Generates visual graphs such as:

  * Accuracy vs. Epochs
  * Loss vs. Epochs
  * ROC Curve
  * Feature Importance Visualization (if applicable)



## 📈 Results / Output

* A trained ransformer-based HAR model capable of classifying activities in real time.
* Detailed performance metrics highlighting model effectiveness.
* Visual insights into model behavior, training dynamics, and feature contributions.
* Example plots:

  * Training/validation accuracy and loss curves
  * Confusion matrix of predicted vs. true activity labels
  * ROC curves for each class



## 🧠 Key Takeaways

* Transformers can effectively handle **multivariate time series data** without relying on recurrent architectures like LSTMs.
* The **attention mechanism** helps capture both short-term and long-term dependencies in motion sequences.
* Proper **data segmentation and normalization** are crucial for accurate HAR model performance.
* Visualization is essential for diagnosing model performance and feature relevance.


## 🙌 Acknowledgments

This work leverages contributions from the open-source ML community, especially:

* PyTorch — for deep learning implementation
* scikit-learn — for evaluation and preprocessing
* Matplotlib / Seaborn — for visualization
* HAR 70+ Dataset From Kaggle — for providing real-world wearable sensor data

Author: Adelugba Adejare
Email: adelugbaadejare03@gmail.com

