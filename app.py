
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import serial
import time
import altair as alt
from scipy.signal import find_peaks

st.title("ECG Heartbeat Classification")
model = tf.keras.models.load_model("best_model2.keras")
encoder = joblib.load("encoder.pkl")
port = "COM6"
def classify(beats):
    if len(beats) == 0:
        st.error("No heartbeats found.")
        return None
    beats = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-8)
    result = model.predict(beats.reshape(-1, 360, 1), verbose=0)
    abnormal = result[0].flatten() > 0.5
    if abnormal.any():
        types = encoder.inverse_transform(np.argmax(result[1][abnormal], axis=1))
        st.error("ABNORMAL")
        st.write("Detected Type:", max(set(types), key=list(types).count))
        return "ABNORMAL"
    st.success("NORMAL")
    return "NORMAL"
def send_result(arduino, result):
    if result:
        arduino.write((result + "\n").encode())

tab1, tab2 = st.tabs(["Upload CSV", "Live ECG"])

with tab1:
    file = st.file_uploader("Upload ECG CSV", type="csv")
    if file:
        data = pd.read_csv(file, header=None)
        if data.shape[1] == 360:
            beats = data.to_numpy(dtype=float)
        else:
            signal = pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna().to_numpy()
            st.line_chart(signal)
            peaks, _ = find_peaks(signal, distance=100)
            beats = np.array([signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)])
        if len(beats) > 0:
            st.line_chart(beats[0])
            try:
                arduino = serial.Serial(port, 115200, timeout=1)
                time.sleep(2)
                send_result(arduino, classify(beats))
                arduino.close()
            except Exception as e:
                st.error(f"LCD Error: {e}")

with tab2:
    if st.button("Start ECG Reading"):
        signal = []
        chart = st.empty()
        status = st.empty()
        try:
            arduino = serial.Serial(port, 115200, timeout=0.1)
            time.sleep(2)
            start = time.time()
            disconnected = False

            while time.time() - start < 10:
                value = arduino.readline().decode(errors="ignore").strip()
                try:
                    ecg, lo_plus, lo_minus = map(float, value.split(","))
                    if lo_plus == 1 or lo_minus == 1:
                        disconnected = True
                        status.error("Electrode Disconnected!")
                        arduino.write(b"DISCONNECTED\n")
                        break
                    signal.append(ecg)
                    data = pd.DataFrame({"ECG": signal})
                    chart.altair_chart(alt.Chart(data).mark_line().encode(x=alt.X("index:Q"), y=alt.Y("ECG:Q", scale=alt.Scale(zero=False))).transform_window(index="row_number()").properties(height=300), use_container_width=True)
                    status.success("Electrodes Connected")
                except ValueError:
                    pass

            if disconnected:
                st.warning("Classification stopped because electrodes are disconnected.")
            elif len(signal) > 0:
                peaks, _ = find_peaks(signal, distance=100)
                beats = np.array([signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)])
                result = classify(beats)
                send_result(arduino, result)
            else:
                status.error("No ECG Signal!")

            arduino.close()
        except Exception as e:
            st.error(f"Arduino Error: {e}")
