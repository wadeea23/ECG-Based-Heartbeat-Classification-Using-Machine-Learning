# ECG-Based Heartbeat Classification Using Machine Learning

## Project Overview

This project uses Machine Learning and Deep Learning to classify ECG heartbeats into normal and abnormal categories.

The main goal is to develop an ECG monitoring system that can analyze heartbeat signals, detect abnormal heartbeat patterns, and display the classification results.

The project combines a trained CNN model with an ECG sensor and an Arduino Uno for real-time signal collection.

**Note:** This project is for educational and experimental purposes only. It is not a medical device and cannot diagnose heart diseases.

## Live Application

[Open the Streamlit App](https://ecg-based-heartbeat-classification-wadeea21.streamlit.app/)

## GitHub Repository

[View the Project on GitHub](https://github.com/wadeea23/ECG-Based-Heartbeat-Classification-Using-Machine-Learning)

---

## Dataset

The project uses the **MIT-BIH Arrhythmia Database** from PhysioNet.

Dataset link: https://physionet.org/content/mitdb/1.0.0/

The database contains:

- 48 half-hour ECG recordings.
- ECG signals recorded at 360 Hz.
- Two ECG channels.
- Expert-reviewed heartbeat annotations.

### Heartbeat Classes

The dataset includes the following heartbeat categories:

| Symbol | Description |
|---|---|
| N | Normal Beat |
| S | Supraventricular Premature Beat |
| V | Premature Ventricular Contraction |
| F | Fusion of Ventricular and Normal Beat |


The classes are imbalanced, meaning that some heartbeat categories have significantly more samples than others.

---

## Data Preprocessing

The following steps were used to prepare the ECG signals:

1. Load the ECG recordings and their annotations.
2. Detect heartbeat locations.
3. Extract heartbeat segments.
4. Prepare each heartbeat using 360 samples.
5. Encode the heartbeat labels.
6. Split the data into training , validations and testing sets.
7. Apply class sample weights to help handle class imbalance.

## Machine Learning Model

A **Convolutional Neural Network (CNN)** was selected for heartbeat classification.

The model has two outputs:

### 1. Binary Classification

Classifies each heartbeat into:

- Normal
- Abnormal

### 2. Heartbeat Type Classification

Identifies the heartbeat category:

- N: Normal Beat
- S: Supraventricular Premature Beat
- V: Premature Ventricular Contraction
- F: Fusion Beat


The model uses a 360-sample heartbeat segment as its input.

### Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Recall is particularly important because the dataset contains imbalanced heartbeat classes.

---

## System Features

The Streamlit application provides two main functions:

### 1. Upload ECG File

- Upload an ECG CSV file.
- Analyze the ECG data.
- Extract heartbeat segments.
- Classify the detected heartbeats.
- Display the classification results.

### 2. Live ECG Monitoring

- Receive ECG readings from the Arduino Uno through a serial connection.
- Display the collected ECG signal.
- Detect disconnected electrodes.
- Extract heartbeat segments.
- Classify the detected heartbeats.
- Display the overall classification result.

### Overall Result

The application displays one of the following results:

- **NORMAL:** The percentage of abnormal beats is within the project's defined 5% threshold.
- **ABNORMAL:** The percentage of abnormal beats is above the project's defined 5% threshold.
- **NO RESULTS:** No valid heartbeat segments were detected.

The 5% threshold is a project-specific rule and is not a clinical medical standard.

---

## Hardware Components

The hardware setup includes:

| Component | Purpose |
|---|---|
| Arduino Uno | Collects sensor readings |
| AD8232 ECG Sensor | Measures ECG signals |
| ECG Electrodes | Collect ECG electrical activity |
| 16x2 I2C LCD | Intended to display results |

The Arduino Uno sends ECG readings to the computer, where the Streamlit application processes and classifies the signal.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- SciPy
- Arduino IDE
- Serial Communication

---

## Project Files

```text
ECG-Based-Heartbeat-Classification-Using-Machine-Learning/
│
├── app.py
├── ECG.ipynb
├── best_model2.keras
├── encoder.pkl
├── requirements.txt
└── README.md
```

| File | Description |
|---|---|
| `app.py` | Streamlit application |
| `ECG.ipynb` | Data analysis, preprocessing, and model training |
| `best_model2.keras` | Trained CNN model |
| `encoder.pkl` | Label encoder |
| `requirements.txt` | Required Python libraries |
| `README.md` | Project documentation |

---

## Author

Wadeea Almutawa
