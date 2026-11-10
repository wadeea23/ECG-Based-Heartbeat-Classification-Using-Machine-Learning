# ECG Heartbeat Classification Using Machine Learning

This project is about classifying ECG heartbeats using Machine Learning. The app allows users to upload a CSV file and see the heartbeat types and results.

## What the App Does

1. Upload an ECG CSV file.
2. Analyze all heartbeats in the file.
3. Show the total number of heartbeats.
4. Show normal and abnormal heartbeats.
5. Show the number of each heartbeat type in a table and chart.

## Heartbeat Types

| Class | Type |
|---|---|
| N | Normal |
| S | Supraventricular |
| V | Ventricular |
| F | Fusion |
| Q | Unknown |

## Dataset

The project uses the MIT-BIH Arrhythmia Database from PhysioNet.

Each heartbeat has 187 values, and there are 5 heartbeat classes.

## Models

I trained and compared four models:

- Decision Tree
- KNN
- Neural Network
- CNN

The best model is saved in a file called `best_model.keras` and used in the Streamlit app.

## Project Files

- `app.py`: The Streamlit app.
- `best_model.keras`: The trained model.
- `requirements.txt`: The required libraries.
- `README.md`: Project information.

## How to Run

1. Install the required libraries.
2. Run the Streamlit app."https://ecg-based-heartbeat-classification-wadeea.streamlit.app/"
3. Open the link in your browser.

## How to Use

1. Upload a CSV file.
2. Click **Analyze ECG**.
3. View the results.

The CSV file should have no header, and each row should contain 187 ECG values.

## Future Work

- Connect the project to ESP32 with an LCD and LEDs.
- Try to connect an Apple Watch to get live heartbeat data.

## Dataset & Sources 

"https://www.physionet.org/content/ptbdb/1.0.0/"

"https://www.physionet.org/physiobank/database/mitdb/"

"https://www.kaggle.com/datasets/shayanfazeli/heartbeat"

## Author

Wadeea Almutawa
