import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import serial
import time
from scipy.signal import find_peaks

st.title("ECG Heartbeat Classification")

model = tf.keras.models.load_model("best_model2.keras")
encoder = joblib.load("encoder2.pkl")

file = st.file_uploader("Upload ECG CSV", type="csv")

if file:
    data = pd.read_csv(file, header=None)
    beats = data.to_numpy(dtype=float)

    if beats.shape[1] == 360:
        signal = beats.flatten()
    else:
        signal = beats[:, 0]

    st.line_chart(signal)

    if beats.shape[1] != 360:
        peaks, _ = find_peaks(signal, distance=100)
        beats = np.array([signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)])

    if len(beats) > 0:
        beats = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-8)
        beats = beats.reshape(-1, 360, 1)

        result = model.predict(beats, verbose=0)
        abnormal = result[0].flatten() > 0.5

        if abnormal.any():
            types = encoder.inverse_transform(np.argmax(result[1][abnormal], axis=1))
            names = {"F": "Fusion Beat", "S": "Supraventricular Ectopic Beat", "V": "Ventricular Ectopic Beat"}
            st.error("Abnormal Signal")
            st.write("Detected Type:", names.get(max(set(types), key=list(types).count), "Unknown"))
        else:
            st.success("Normal Signal")
    else:
        st.error("No heartbeats found.")

st.subheader("Live ECG")

port = st.text_input("Arduino Port", "COM7")

if st.button("Start Reading"):
    signal = []
    try:
        arduino = serial.Serial(port, 115200, timeout=0.1)
        time.sleep(2)

        start = time.time()
        while time.time() - start < 10:
            value = arduino.readline().decode(errors="ignore").strip()
            try:
                ecg, lo_plus, lo_minus = map(float, value.split(","))
                if lo_plus == 1 or lo_minus == 1:
                    st.error("Electrode Disconnected!")
                    signal = []
                    break
                signal.append(ecg)
            except ValueError:
                pass

        arduino.close()

        if len(signal) > 0:
            st.line_chart(signal)
    except Exception as e:
        st.error(f"Arduino Error: {e}")
