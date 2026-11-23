# ECG-Based Heartbeat Classification Using Machine Learning

## Project Overview

This project uses machine learning to look at ECG heartbeats and decide if each one is **normal** or **abnormal**. If a heartbeat is abnormal, the model also tells which type it is.

The project has three parts:
- A trained CNN model that classifies heartbeats.
- A Streamlit app that shows the results.
- A small hardware setup (ECG sensor and Arduino Uno) that collects a live ECG signal.

> **Note:** This project is only for learning and experiments. It is not a medical device, and it cannot diagnose heart diseases.

## Links

- **Live app:** [Streamlit app](https://ecg-based-heartbeat-classification-wadeea21.streamlit.app/)
- **Demo video:** [video demo ](https://drive.google.com/file/d/1xaYw5IbxyfQt4Sz22Rv8mfIQ5S4i_eR9/view?usp=sharing)
- **GitHub:** [GitHub](https://github.com/wadeea23/ECG-Based-Heartbeat-Classification-Using-Machine-Learning)

## Why This Project?

A half-hour ECG recording has thousands of heartbeats, and the abnormal ones are rare. Checking all of them by hand takes a long time. This project is an automatic first check. It marks the beats that look different, so an expert can look at them first.

## Dataset

I used the **MIT-BIH Arrhythmia Database** from PhysioNet.

Dataset link: https://physionet.org/content/mitdb/1.0.0/

The database has:
- 48 half-hour ECG recordings.
- ECG signals recorded at 360 Hz.
- Two ECG channels.
- Heartbeat labels checked by experts.

### Heartbeat Classes

| Symbol | Description |
|---|---|
| N | Normal Beat |
| S | Supraventricular Premature Beat |
| V | Premature Ventricular Contraction |
| F | Fusion of Ventricular and Normal Beat |

The classes are not balanced. Some classes have many more samples than others.

## Data Preprocessing

These are the steps I used to prepare the ECG signals:

1. Load the ECG recordings and their labels.
2. Cut out each heartbeat segment.
3. Use 360 samples for each heartbeat.
4. Encode the heartbeat labels.
5. Split the data into training, validation, and testing sets.
6. Use class weights to help with the imbalanced classes.

## Model

I used a **Convolutional Neural Network (CNN)**. A CNN is good at finding patterns in shapes, and a heartbeat has a shape.

The model takes one heartbeat (360 samples) and gives two outputs:

1. **Normal or Abnormal**
2. **Heartbeat type:** N, S, V, or F

### Evaluation

I checked the model with:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Recall is very important here, because the classes are imbalanced and we do not want to miss abnormal beats.

## Results

On the test data:

| Result | Score |
|---|---|
| Recall for normal vs abnormal (macro avg) | **97%** |
| Recall for abnormal type (S, V, F) | **93%** |

## Streamlit App

The app has two tabs.

### 1. Upload CSV

1. Upload an ECG CSV file.
2. Click **Classify ECG**.
3. The app finds the heartbeats and classifies each one.
4. It shows the result, the number of normal and abnormal beats, and the types of abnormal beats found.
5. It explains each type in simple words.
6. You can download a one-page **PDF report**.

### 2. Live ECG

The Live ECG tab works **locally with the Arduino**. Run the app on your own computer, connect the Arduino Uno with a USB cable, and open the app in **Google Chrome or Microsoft Edge (desktop)**. The app reads the Arduino directly from the browser using the Web Serial API.

1. Connect the sensor and the Arduino to the computer.
2. Click **Connect to Arduino & Read 10s** and choose the Arduino's serial port.
3. The app reads the ECG signal for 10 seconds and shows it live on the screen.
4. If an electrode comes off, the app shows a warning and stops.
5. The app finds the heartbeats, classifies them, and shows the overall result.
6. It also creates a **PDF report**, the same as the Upload CSV tab.

> **Important:** Live ECG needs the Arduino to be connected to the computer that opens the app, and it only works in Chrome or Edge on desktop. The online Streamlit link is used for Upload CSV.

### Overall Result

- **NORMAL:** the abnormal beats are 5% or less.
- **ABNORMAL:** the abnormal beats are more than 5%.
- **NO RESULTS:** no valid heartbeats were found.

The 5% limit is a rule I chose for this project. It is not a medical standard.

## Hardware

| Component | What it does |
|---|---|
| Arduino Uno | Reads the sensor and sends the data to the computer |
| AD8232 ECG Sensor | Measures the ECG signal |
| ECG Electrodes | Pick up the electrical activity of the heart |
| 16x2 I2C LCD | I planned to show the result on it, but it is not working yet |

The Arduino sends the readings to the computer. The Streamlit app then processes and classifies the signal.

## Limitations

- The result depends on the signal quality and where the electrodes are placed.
- The model only knows the four trained classes (N, S, V, F).
- I tested the model on a random split of heartbeats, so I did not test it on completely new patients yet.
- Live ECG works locally with the Arduino connected, in Chrome or Edge on desktop.

## Future Work

- Test the model on new patients.
- Use more real-world ECG data.
- Add recovery-time monitoring.
  

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, TensorFlow / Keras, Streamlit, SciPy, Arduino IDE, Serial Communication

## Project Files

```text
ECG-Based-Heartbeat-Classification-Using-Machine-Learning/
│
├── app.py
├── ECG.ipynb
├── best_model2.keras
├── encoder.pkl
├── requirements.txt
├── README.md
└── arduino_ad8232/
```

| File | Description |
|---|---|
| `app.py` | Streamlit app |
| `ECG.ipynb` | Data analysis, preprocessing, and model training |
| `best_model2.keras` | Trained CNN model |
| `encoder.pkl` | Label encoder |
| `requirements.txt` | Python libraries needed |
| `arduino_ad8232/` | Arduino code for the sensor |

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

To use Live ECG, connect the Arduino to your computer before opening the app, then open it in Chrome or Edge.

## Author

Wadeea Almutawa
