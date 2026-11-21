import streamlit as st
import pandas as pd
import numpy as np
import serial, time, joblib, tensorflow as tf, requests, io
from collections import Counter
from scipy.signal import find_peaks
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

st.set_page_config(page_title="ECG Classify", layout="wide")
st.title("ECG Classification")

model = tf.keras.models.load_model("best_model2.keras")
encoder = joblib.load("encoder.pkl")
st.session_state.setdefault("report", None)
st.session_state.setdefault("pdf", None)

names = {"N":"Normal", "S":"Supraventricular", "V":"PVC", "F":"Fusion"}

def get_beats(signal):
    peaks, _ = find_peaks(signal, distance=100)
    return [signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)]

def classify(beats):
    beats = np.array(beats, dtype=float)
    beats = (beats-beats.mean(axis=1, keepdims=True))/(beats.std(axis=1, keepdims=True)+1e-8)
    binary, types = model.predict(beats.reshape(-1,360,1), verbose=0)
    binary = (binary.flatten() > 0.5).astype(int)
    normal, abnormal = sum(binary == 0), sum(binary == 1)
    labels = encoder.inverse_transform(np.argmax(types, axis=1))
    result = "NO RESULTS" if len(binary) == 0 else "NORMAL" if abnormal/len(binary) <= 0.05 else "ABNORMAL"
    counts = Counter(names.get(str(x), str(x)) for x in labels[binary == 1])
    st.subheader("Classification Result")
    st.write("**Detected Result:**", result)
    st.write("**Normal Beats:**", normal)
    st.write("**Abnormal Beats:**", abnormal)
    if result == "ABNORMAL":
        st.write("**Detected Beat Types:**")
        for name, count in counts.items():
            st.write(f"- {name}: {count}")
    return result, normal, abnormal, counts

def send_to_lcd(result):
    try:
        lcd = serial.Serial("COM7", 115200, timeout=1)
        time.sleep(2)
        lcd.write((result + "\n").encode())
        lcd.close()
    except Exception as e:
        st.warning(f"LCD connection error: {e}")

def make_report(result, types):
    type_text = "\n".join([f"{name}: {count}" for name, count in types.items()])
    if result != "ABNORMAL":
        type_text = "No abnormal types detected."
    prompt = f"""Write a simple English ECG report.
    Overall Result: {result}. Keep this result exactly.
    Detected Beat Types:
    {type_text}
    Explain PVC, Supraventricular and Fusion beats simply.
    Explain the result and recommend consulting a doctor if ABNORMAL.
    This is an experimental project, not a medical device.
    Do not show beat counts, class letters, BPM or Recovery Time."""
    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}"},
        json={"model":"openrouter/auto","messages":[{"role":"user","content":prompt}],"temperature":0.3},
        timeout=60)
    response.raise_for_status()
    report = response.json()["choices"][0]["message"]["content"]
    pdf = io.BytesIO()
    styles = getSampleStyleSheet()
    content = [Paragraph("ECG Report", styles["Title"]), Spacer(1,15)]
    for line in report.splitlines():
        if line.strip():
            content += [Paragraph(escape(line), styles["Normal"]), Spacer(1,7)]
    SimpleDocTemplate(pdf).build(content)
    st.session_state.report, st.session_state.pdf = report, pdf.getvalue()

def show_report(key):
    if st.session_state.report:
        st.markdown(st.session_state.report)
        st.download_button("Download PDF", st.session_state.pdf, "ECG_Report.pdf", key=key)

def finish(beats):
    if len(beats) == 0:
        st.error("No valid ECG beats found.")
        return
    st.session_state.report = st.session_state.pdf = None
    result, normal, abnormal, types = classify(beats)
    send_to_lcd(result)
    make_report(result, types)

tab1, tab2 = st.tabs(["Upload CSV", "Live ECG"])

with tab1:
    file = st.file_uploader("Upload ECG CSV", type=["csv"])
    if file and st.button("Classify ECG"):
        data = pd.read_csv(file)
        beats = data.values if data.shape[1] == 360 else get_beats(pd.to_numeric(data.iloc[:,0], errors="coerce").dropna().values)
        finish(beats)
    show_report("upload_pdf")

with tab2:
    if st.button("Start ECG Reading"):
        try:
            arduino = serial.Serial("COM7", 115200, timeout=1)
            time.sleep(2)
            signal = []
            disconnected = False
            start = time.time()
            while time.time()-start < 10:
                values = arduino.readline().decode(errors="ignore").strip().split(",")
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
                finish(get_beats(np.array(signal)))
        except Exception as e:
            st.error(f"Error: {e}")
    show_report("live_pdf")
