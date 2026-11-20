import streamlit as st
import pandas as pd
import numpy as np
import serial, time, joblib, tensorflow as tf, requests, io
from collections import Counter
from scipy.signal import find_peaks
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

st.set_page_config(page_title="ECG Clasiffy", layout="wide")
st.title("ECG clasification")

model = tf.keras.models.load_model("best_model2.keras")
encoder = joblib.load("encoder.pkl")
st.session_state.setdefault("report", None)
st.session_state.setdefault("pdf", None)

names = {"N": "Normal Beat", "S": "Supraventricular Premature Beat",
         "V": "Premature Ventricular Contraction", "F": "Fusion of Ventricular and Normal Beat",
         "Q": "Unknown Beat"}

def get_beats(signal):
    peaks, _ = find_peaks(signal, distance=100)
    return [signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)]

def classify(beats):
    beats = np.array(beats, dtype=float)
    beats = (beats-beats.mean(axis=1, keepdims=True))/(beats.std(axis=1, keepdims=True)+1e-8)
    binary, beat_type = model.predict(beats.reshape(-1, 360, 1), verbose=0)
    binary = (binary.flatten() > 0.5).astype(int)
    normal, abnormal = int(sum(binary == 0)), int(sum(binary == 1))
    total = normal + abnormal
    labels = encoder.inverse_transform(np.argmax(beat_type, axis=1))
    result = "NO RESULTS" if total == 0 else "NORMAL" if abnormal/total <= 0.05 else "ABNORMAL"
    types = Counter(names.get(str(x), str(x)) for x in labels[binary == 1])

    st.write("**Detected Result:**", result)
    st.write("**Normal Beats:**", normal, "| **Abnormal Beats:**", abnormal)
    if result == "ABNORMAL":
        st.write("**Detected Beat Types:**")
        for name, count in types.items():
            st.write(f"- {name}: {count}")
    return result, normal, abnormal, types

def make_report(result, normal, abnormal, types):
    beat_types = dict(types) if result == "ABNORMAL" else "No abnormal types to display."
    prompt = f"""
    Write a simple English ECG report with:
    1. Overall Result: {result}. Use this result exactly.
    2. Detected Beat Types: {beat_types}
    Explain PVC as an early heartbeat starting in the lower chambers.
    Explain Supraventricular Premature Beat as an early heartbeat
    starting above the ventricles.
    Explain Fusion Beat as a combination of normal and ventricular signals.
    Only list abnormal types if the result is ABNORMAL.
    3. Explanation: Explain the result simply. Recommend consulting a doctor
    if ABNORMAL.
    4. Important Note: This is an experimental machine learning project,
    not a medical device. It cannot diagnose diseases.
    Do not show beat counts, classification letters, BPM, or Recovery Time.
    Do not invent medical information.
    """
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": "openrouter/auto", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
        timeout=60)
    r.raise_for_status()
    report = r.json()["choices"][0]["message"]["content"]
    pdf = io.BytesIO()
    content = [Paragraph("ECG Report", getSampleStyleSheet()["Title"]), Spacer(1, 15)]
    for line in report.splitlines():
        if line.strip():
            content += [Paragraph(escape(line), getSampleStyleSheet()["Normal"]), Spacer(1, 7)]
    SimpleDocTemplate(pdf).build(content)
    st.session_state.report, st.session_state.pdf = report, pdf.getvalue()

def show_pdf(key):
    if st.session_state.report and st.session_state.pdf:
        st.markdown(st.session_state.report)
        st.download_button("Download PDF", st.session_state.pdf, "ECG_Report.pdf",
                           mime="application/pdf", key=key)

tab1, tab2 = st.tabs(["Upload CSV", "Live ECG"])

with tab1:
    upload = st.file_uploader("Upload ECG File", type=["csv"], key="upload")
    if upload and st.button("Classify ECG", key="classify"):
        data = pd.read_csv(upload)
        beats = data.values if data.shape[1] == 360 else get_beats(
            pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna().values)
        if len(beats):
            try:
                st.session_state.report = st.session_state.pdf = None
                make_report(*classify(beats))
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.error("No valid ECG beats found.")
    show_pdf("upload_pdf")

with tab2:
    if st.button("Start ECG Reading"):
        try:
            arduino = serial.Serial("COM6", 115200, timeout=1)
            time.sleep(2)
            signal, disconnected = [], False
            start = time.time()
            while time.time()-start < 10:
                values = arduino.readline().decode("utf-8", errors="ignore").strip().split(",")
                if len(values) == 3:
                    ecg, plus, minus = float(values[0]), int(values[1]), int(values[2])
                    if plus == 1 or minus == 1:
                        disconnected = True
                        break
                    signal.append(ecg)
            arduino.close()
            if disconnected:
                st.error("Electrodes are not connected.")
            else:
                st.line_chart(signal)
                beats = get_beats(np.array(signal))
                if len(beats):
                    st.session_state.report = st.session_state.pdf = None
                    make_report(*classify(beats))
                else:
                    st.error("Not enough ECG data. Try again.")
        except Exception as e:
            st.error(f"Error: {e}")
    show_pdf("live_pdf")
