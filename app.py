
import streamlit as st
import pandas as pd
import numpy as np
import serial
import time
import joblib
import tensorflow as tf
import altair as alt
from scipy.signal import find_peaks
from collections import Counter

st.title("ECG Heartbeat Classification")

model1 = tf.keras.models.load_model("best_bainary_model.keras")
model2 = tf.keras.models.load_model("cnn2_model.keras")
encoder = joblib.load("encoder2.pkl")

tab1, tab2 = st.tabs(["Upload CSV", "Live ECG"])
signal, beats = None, None
live, disconnected = False, False

def show_signal(values, area):
    data = pd.DataFrame({"Sample": range(len(values)), "ECG": values})
    plot = alt.Chart(data).mark_line(strokeWidth=1.5).encode(
        x=alt.X("Sample:Q", title=None),
        y=alt.Y("ECG:Q", title=None, scale=alt.Scale(zero=False))
    ).properties(height=300).configure_view(stroke=None)
    area.altair_chart(plot, use_container_width=True)

with tab1:
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        data = pd.read_csv(file, header=None)
        beats = data.to_numpy(dtype=float) if data.shape[1] == 360 else None
        signal = beats.flatten() if beats is not None else pd.to_numeric(data[0], errors="coerce").dropna().to_numpy()
        st.subheader("ECG Signal")
        show_signal(signal, st)

with tab2:
    port = st.text_input("Arduino Port", "COM8")
    if st.button("Start Reading"):
        live, signal = True, []
        chart = st.empty()
        try:
            arduino = serial.Serial(port, 115200, timeout=0.1)
            time.sleep(2)
            start = time.time()
            while time.time() - start < 10:
                value = arduino.readline().decode(errors="ignore").strip()
                try:
                    ecg, lo_plus, lo_minus = map(float, value.split(","))
                    if lo_plus == 1 or lo_minus == 1:
                        disconnected = True
                    signal.append(ecg)
                    if len(signal) % 20 == 0:
                        show_signal(signal[-500:], chart)
                except ValueError:
                    pass
            arduino.close()
        except Exception as e:
            st.error(f"Arduino Error: {e}")
            signal = None

if signal is not None and len(signal) > 0:
    if live and (disconnected or np.std(signal) < 2 or np.ptp(signal) < 5):
        st.error("Electrode Disconnected! No classification.")
    else:
        if beats is None:
            peaks, _ = find_peaks(signal, distance=100, prominence=max(np.std(signal)*0.2, 1))
            beats = np.array([signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)])
        if len(beats) > 0:
            beats = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-8)
            beats = beats.reshape(-1, 360, 1)
            result = model1.predict(beats, verbose=0)
            abnormal = result[:, 0] > 0.5
            if abnormal.any():
                result = model2.predict(beats[abnormal], verbose=0)
                labels = encoder.inverse_transform(np.argmax(result, axis=1))
                names = {"F": "Fusion Beat", "S": "Supraventricular Ectopic Beat", "V": "Ventricular Ectopic Beat"}
                detected = [names.get(x, str(x)) for x in labels]
                st.error("Abnormal Signal")
                st.write("Detected Type:", Counter(detected).most_common(1)[0][0])
            else:
                st.success("Normal Signal")
        else:
            st.error("No heartbeats found.")
elif live:
    st.error("No ECG Signal!")
