# ECG-Based Heartbeat Classification Using Machine Learning

## Live Demo
Try the application here:

[ECG Heartbeat Classification – Streamlit App](https://ecg-based-heartbeat-classification-wadeea.streamlit.app/)

## GitHub Repository
[View the Project on GitHub](https://github.com/wadeea23/ECG-Based-Heartbeat-Classification-Using-Machine-Learning)

## About the Project
This project uses Machine Learning and Deep Learning to classify ECG heartbeats. It aims to detect normal and abnormal heartbeats and identify the type of abnormal heartbeat.

The project includes a Streamlit application that allows users to upload ECG data or test ECG signals in real time using a connected hardware device.

The hardware setup includes an Arduino Uno, an AD8232 ECG sensor, and an LCD screen to display the classification results and electrode connection status.

## Problem Statement & Motivation
Abnormal heartbeats can indicate potential heart rhythm problems. Detecting these patterns from ECG signals can help demonstrate how Machine Learning can support automated heartbeat analysis.

The main motivation behind this project is to apply my Electronic Engineering background and Data Science skills to a practical healthcare-related application.

## Dataset
The project uses the **MIT-BIH Arrhythmia Database** from PhysioNet.

Dataset source:  
https://physionet.org/content/mitdb/1.0.0/

Dataset information:
- 48 approximately half-hour ECG recordings.
- Two ECG channels per recording.
- Sampling frequency of 360 Hz.
- Expert-reviewed heartbeat annotations.

The project focuses on four heartbeat classes:

| Class | Description |
|---|---|
| N | Normal Beat |
| S | Supraventricular Ectopic Beat |
| V | Ventricular Ectopic Beat |
| F | Fusion Beat |

The ECG data is prepared into individual heartbeat segments containing 360 samples each.

## Models & Methodology

Different models were trained and compared, including:

- Decision Tree
- K-Nearest Neighbors (KNN)
- Convolutional Neural Network (CNN)

The final classification process uses two stages:

1. **Binary Classification:** A CNN model classifies heartbeats as Normal or Abnormal.
2. **Multi-Class Classification:** A second CNN model identifies the type of abnormal heartbeat (S, V, or F).

## Evaluation Metrics

Because the dataset is imbalanced, **Recall was selected as the primary evaluation metric instead of Accuracy**.

Accuracy can be misleading when one class has significantly more samples than other classes. A model may achieve high accuracy by predicting the majority class while missing many abnormal heartbeats.

Recall helps measure how many actual positive cases the model successfully detects.

The evaluation also considers the confusion matrix and other classification metrics to better understand model performance.

## Model Results

| Model | Evaluation Metric |
|---|---|
| Decision Tree | Recall |
| KNN | Recall |
| CNN | Recall |

## Streamlit Application

The application provides two main features:

**1. Upload ECG CSV**
- Upload an ECG dataset in CSV format.
- Display the ECG signal.
- Classify the heartbeat signal.
- Display the detected heartbeat type.

**2. Live ECG**
- Read ECG data from a connected Arduino device.
- Display the live ECG signal.
- Detect whether the electrodes are disconnected or the signal is unavailable.
- Classify the detected heartbeat.
- Send the classification result to the LCD screen.

## Hardware Components

The hardware setup includes:

- Arduino Uno
- AD8232 ECG Sensor
- 16x2 I2C LCD Display
- ECG Electrodes

**Hardware Functionality:**
- The AD8232 sensor captures ECG signals.
- Arduino Uno reads and sends the ECG data to the Streamlit application.
- The LCD displays the system status and classification result.
- The LCD also indicates whether the ECG electrodes are connected or disconnected.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- SciPy
- Arduino


## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/wadeea23/ECG-Based-Heartbeat-Classification-Using-Machine-Learning.git
cd ECG-Based-Heartbeat-Classification-Using-Machine-Learning
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

## Repository Structure

```text
ECG-Based-Heartbeat-Classification/
│
├── app.py
├── ECG.ipynb
├── best_model2.keras
├── encoder.pkl
├── requirements.txt
└── README.md
```

## Limitations & Future Work
- Test the system with more real-world ECG recordings.
- Improve the classification of minority heartbeat classes.
- Test the system with different ECG hardware devices.
- Improve real-time signal processing and classification.
- Enhance the recovery monitoring capabilities of the system.

## License & Acknowledgments

The MIT-BIH Arrhythmia Database is provided by PhysioNet under the Open Data Commons Attribution License (ODC-By) v1.0.

Dataset reference:

MIT-BIH Arrhythmia Database (version 1.0.0). PhysioNet. https://doi.org/10.13026/C2F305


##Author 

WADEEA ALMUTAWA
## Author

**Wadeea Almutawa**
