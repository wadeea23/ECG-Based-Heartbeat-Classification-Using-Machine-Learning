import streamlit as st
import pandas as pd
import numpy as np
import serial, time, joblib, tensorflow as tf, requests, io
from collections import Counter
from scipy.signal import find_peaks
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape
# MODIFIED: Added streamlit components to allow embedding custom HTML/JavaScript
import streamlit.components.v1 as components

st.set_page_config(page_title="ECG Monitor", layout="wide")
st.title("ECG Monitor")

model = tf.keras.models.load_model("best_model2.keras")
encoder = joblib.load("encoder.pkl")
st.session_state.setdefault("report", None)
st.session_state.setdefault("pdf", None)

names = {"N":"Normal Beat", "S":"Supraventricular Premature Beat",
         "V":"Premature Ventricular Contraction", "F":"Fusion of Ventricular and Normal Beat", "Q":"Unknown Beat"}

def get_beats(signal):
    peaks, _ = find_peaks(signal, distance=100)
    return [signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)]

def classify(beats):
    beats = np.array(beats, dtype=float)
    beats = (beats-beats.mean(axis=1, keepdims=True))/(beats.std(axis=1, keepdims=True)+1e-8)
    binary, types = model.predict(beats.reshape(-1,360,1), verbose=0)
    binary = (binary.flatten() > 0.5).astype(int)
    normal, abnormal = int(sum(binary==0)), int(sum(binary==1))
    total = normal + abnormal
    labels = encoder.inverse_transform(np.argmax(types, axis=1))
    result = "NO RESULTS" if total == 0 else "NORMAL" if abnormal/total <= 0.05 else "ABNORMAL"
    counts = Counter(names.get(str(x), str(x)) for x in labels[binary==1])
    st.write("**Detected Result:**", result)
    st.write("**Normal Beats:**", normal, "| **Abnormal Beats:**", abnormal)
    if result == "ABNORMAL":
        st.write("**Detected Beat Types:**", dict(counts))
    return result, normal, abnormal, counts

def send_lcd(result):
    lcd = serial.Serial("COM6", 115200, timeout=1)
    time.sleep(2)
    lcd.write((result+"\n").encode())
    lcd.close()

def make_report(result, normal, abnormal, types):
    beat_types = dict(types) if result == "ABNORMAL" else "No abnormal types to display."
    prompt = f"""Write a simple English ECG report.
    Overall Result: {result}. Use this result exactly.
    Detected Beat Types: {beat_types}
    Explain PVC, Supraventricular Premature Beat and Fusion Beat simply.
    Explain the result and recommend consulting a doctor if ABNORMAL.
    Note: This is an experimental machine learning project, not a medical device.
    Do not show beat counts, classification letters, BPM or Recovery Time.
    Do not invent medical information."""
    r = requests.post("https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization":f"Bearer {st.secrets['OPENROUTER_API_KEY']}", "Content-Type":"application/json"},
        json={"model":"openrouter/auto","messages":[{"role":"user","content":prompt}],"temperature":0.3}, timeout=60)
    r.raise_for_status()
    report = r.json()["choices"][0]["message"]["content"]
    pdf = io.BytesIO()
    styles = getSampleStyleSheet()
    content = [Paragraph("ECG Report", styles["Title"]), Spacer(1,15)]
    for line in report.splitlines():
        if line.strip():
            content += [Paragraph(escape(line), styles["Normal"]), Spacer(1,7)]
    SimpleDocTemplate(pdf).build(content)
    st.session_state.report, st.session_state.pdf = report, pdf.getvalue()

def show_pdf(key):
    if st.session_state.report and st.session_state.pdf:
        st.markdown(st.session_state.report)
        st.download_button("Download PDF", st.session_state.pdf, "ECG_Report.pdf", mime="application/pdf", key=key)


# MODIFIED: Added the HTML code block to handle Web Serial API from the browser.
# This replaces the Python serial loop.
html_code = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: sans-serif; text-align: center; margin-top: 10px; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #FF4B4B; color: white; border: none; border-radius: 5px; }
        #status { margin-top: 10px; font-weight: bold; color: #555; }
    </style>
</head>
<body>
    <button id="connectBtn">Connect to Arduino & Read 10s</button>
    <div id="status">Waiting for connection...</div>

    <script>
        function sendDataToPython(dataArray) {
            window.parent.postMessage({
                isStreamlitMessage: true,
                type: "setComponentValue",
                value: dataArray
            }, "*");
        }

        document.getElementById('connectBtn').addEventListener('click', async () => {
            const statusLabel = document.getElementById('status');
            let port;
            
            try {
                port = await navigator.serial.requestPort();
                await port.open({ baudRate: 115200 });
                statusLabel.innerText = "Connected! Reading data for 10 seconds...";
                
                const textDecoder = new TextDecoderStream();
                const readableStreamClosed = port.readable.pipeTo(textDecoder.writable);
                const reader = textDecoder.readable.getReader();
                
                let buffer = "";
                let ecgData = [];
                let startTime = Date.now();
                
                while (Date.now() - startTime < 10000) {
                    const { value, done } = await reader.read();
                    if (done) break;
                    
                    buffer += value;
                    let lines = buffer.split('\\n');
                    buffer = lines.pop(); // Keep the incomplete line for the next loop
                    
                    for (let line of lines) {
                        let parts = line.split(',');
                        if (parts.length === 3) {
                            let ecgVal = parseFloat(parts[0]);
                            let plus = parseInt(parts[1]);
                            let minus = parseInt(parts[2]);
                            
                            if (plus === 1 || minus === 1) {
                                statusLabel.innerText = "Error: Electrodes disconnected!";
                                reader.cancel();
                                port.close();
                                sendDataToPython([]); // Send empty array on error
                                return;
                            }
                            ecgData.push(ecgVal);
                        }
                    }
                }
                
                reader.cancel();
                await readableStreamClosed.catch(() => {});
                await port.close();
                
                statusLabel.innerText = "Finished reading. Sending to AI Model...";
                sendDataToPython(ecgData);
                
            } catch (error) {
                statusLabel.innerText = "Connection failed or cancelled.";
                console.error(error);
            }
        });
    </script>
</body>
</html>
"""

tab1, tab2 = st.tabs(["Upload CSV", "Live ECG"])

with tab1:
    upload = st.file_uploader("Upload ECG File", type=["csv"], key="upload")
    if upload and st.button("Classify ECG", key="classify"):
        try:
            data = pd.read_csv(upload)
            beats = data.values if data.shape[1] == 360 else get_beats(pd.to_numeric(data.iloc[:,0], errors="coerce").dropna().values)
            if len(beats):
                st.session_state.report = st.session_state.pdf = None
                result, normal, abnormal, types = classify(beats)
                send_lcd(result)
                make_report(result, normal, abnormal, types)
            else:
                st.error("No valid ECG beats found.")
        except Exception as e:
            st.error(f"Error: {e}")
    show_pdf("upload_pdf")

with tab2:
    # MODIFIED: Completely changed tab 2 logic.
    # Removed the st.button() and Python serial loop.
    st.warning("Note: Web Serial only works on Google Chrome or Microsoft Edge desktop browsers.")
    
    # Render the custom HTML button
    ecg_data_from_js = components.html(html_code, height=120)
    
    # This block triggers automatically when the JavaScript finishes 10s and sends data back
    if isinstance(ecg_data_from_js, list):
        if len(ecg_data_from_js) > 0:
            st.success(f"Received {len(ecg_data_from_js)} readings from Arduino.")
            
            # Plot the whole graph at once (live animation is not possible with this method)
            st.line_chart(ecg_data_from_js)
            
            beats = get_beats(np.array(ecg_data_from_js))
            if len(beats):
                st.session_state.report = st.session_state.pdf = None
                result, normal, abnormal, types = classify(beats)
                
                # MODIFIED: Auto-triggering LCD Write workaround!
                # This injects a small HTML block that automatically finds the 
                # pre-approved USB port and shoots the result text to the LCD
                # without requiring the user to click a second button.
                lcd_writer_html = f"""
                <!DOCTYPE html>
                <html>
                <body>
                    <div id="lcd-status" style="font-family: sans-serif; font-weight: bold; color: #555;">
                        Sending '{result}' to LCD...
                    </div>
                    <script>
                        (async function() {{
                            const status = document.getElementById('lcd-status');
                            try {{
                                // Find ports the user has already approved
                                const ports = await navigator.serial.getPorts();
                                if (ports.length > 0) {{
                                    const port = ports[0];
                                    await port.open({{ baudRate: 115200 }});
                                    const encoder = new TextEncoder();
                                    const writer = port.writable.getWriter();
                                    
                                    // Send the result to the Arduino
                                    await writer.write(encoder.encode("{result}\\n"));
                                    writer.releaseLock();
                                    await port.close();
                                    
                                    status.innerText = "✓ Sent '{result}' to LCD successfully!";
                                    status.style.color = "green";
                                }} else {{
                                    status.innerText = "Error: No pre-approved USB ports found.";
                                    status.style.color = "red";
                                }}
                            }} catch (err) {{
                                status.innerText = "Error sending to LCD: " + err.message;
                                status.style.color = "red";
                                console.error(err);
                            }}
                        }})();
                    </script>
                </body>
                </html>
                """
                components.html(lcd_writer_html, height=50)
                
                make_report(result, normal, abnormal, types)
            else:
                st.error("Not enough ECG data. Try again.")
        else:
            st.error("Electrodes are not connected or reading was cancelled.")
            
    show_pdf("live_pdf")
Show
