import streamlit as st
import pandas as pd
import numpy as np
import joblib, tensorflow as tf, requests, io, tempfile, re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import Counter
from scipy.signal import find_peaks
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from xml.sax.saxutils import escape
import streamlit.components.v1 as components

st.set_page_config(page_title="ECG Monitor", layout="wide")
st.title("ECG Monitor")

model = tf.keras.models.load_model("best_model2.keras")
encoder = joblib.load("encoder.pkl")

st.session_state.setdefault("report", None)
st.session_state.setdefault("pdf", None)

names = {
    "N": "Normal Beat",
    "S": "Supraventricular Premature Beat",
    "V": "Premature Ventricular Contraction",
    "F": "Fusion of Ventricular and Normal Beat",
    "Q": "Unknown Beat",
}


def get_beats(signal):
    peaks, _ = find_peaks(signal, distance=100)
    return [signal[p-120:p+240] for p in peaks if p >= 120 and p+240 <= len(signal)]


def classify(beats):
    beats = np.array(beats, dtype=float)
    beats = (beats - beats.mean(axis=1, keepdims=True)) / (beats.std(axis=1, keepdims=True) + 1e-8)

    binary, types = model.predict(beats.reshape(-1, 360, 1), verbose=0)
    binary = (binary.flatten() > 0.5).astype(int)

    normal, abnormal = int(sum(binary == 0)), int(sum(binary == 1))
    total = normal + abnormal
    labels = encoder.inverse_transform(np.argmax(types, axis=1))

    result = ("NO RESULTS" if total == 0
              else "NORMAL" if abnormal / total <= 0.05
              else "ABNORMAL")

    counts = Counter(names.get(str(x), str(x)) for x in labels[binary == 1])

    st.write("**Detected Result:**", result)
    st.write("**Normal Beats:**", normal, "| **Abnormal Beats:**", abnormal)
    if result == "ABNORMAL":
        st.write("**Detected Beat Types:**", ", ".join(f"{k}: {v}" for k, v in counts.items()))

    return result, normal, abnormal, counts


def make_report(result, normal, abnormal, types):
    beat_types = ", ".join(types.keys()) if result == "ABNORMAL" else "No abnormal types to display."

    prompt = f"""
Write a simple English ECG report in the style of a medical report.

Use these sections:
Overall Result, Findings, Interpretation, Recommendation, Disclaimer.

Overall Result: {result}. Use this result exactly.

Detected Beat Types: {beat_types}

Explain PVC, Supraventricular Premature Beat and Fusion Beat simply.

Explain the result and recommend consulting a doctor if ABNORMAL.

In the Disclaimer section, state clearly that this is an experimental machine learning project, not a medical device, and that it must not be relied on for any medical decision.

Do not show beat counts, classification letters, BPM or Recovery Time.

Do not invent medical information.
"""

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {st.secrets['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": "openrouter/auto",
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.3},
        timeout=60,
    )
    r.raise_for_status()
    report = r.json()["choices"][0]["message"]["content"]

    # ---------- PDF ----------
    navy = colors.HexColor("#1F3A5F")
    line_c = colors.HexColor("#B0BEC5")
    res_color = {"NORMAL": "#1E8449", "ABNORMAL": "#C0392B"}.get(result, "#7F8C8D")

    base = getSampleStyleSheet()
    title_st = ParagraphStyle("t", parent=base["Title"], fontSize=22, textColor=navy, spaceAfter=2)
    sub_st = ParagraphStyle("s", parent=base["Normal"], alignment=1, fontSize=10, textColor=colors.grey)
    head_st = ParagraphStyle("h", parent=base["Heading2"], fontSize=13, textColor=navy, spaceBefore=12, spaceAfter=4)
    body_st = ParagraphStyle("b", parent=base["Normal"], fontSize=11, leading=16, spaceAfter=6)
    cell_st = ParagraphStyle("c", parent=base["Normal"], fontSize=11)

    P = lambda t: Paragraph(t, cell_st)
    info = Table([
        [P("<b>Report Date</b>"), P(datetime.now(timezone(timedelta(hours=3))).strftime("%d %B %Y, %H:%M"))],
        [P("<b>Report Type</b>"), P("ECG Heartbeat Classification")],
        [P("<b>Overall Result</b>"), P(f'<font color="{res_color}"><b>{result}</b></font>')],
    ], colWidths=[4.5*cm, 12.5*cm])

    info.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF0F7")),
        ("GRID", (0, 0), (-1, -1), 0.5, line_c),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    content = [
        Paragraph("ECG Analysis Report", title_st),
        Paragraph("Machine Learning Based Heartbeat Classification", sub_st),
        Spacer(1, 8),
        HRFlowable(width="100%", thickness=1.5, color=navy),
        Spacer(1, 12),
        info,
        Spacer(1, 10),
    ]

    sections = {"overall result", "findings", "interpretation", "recommendation", "disclaimer"}
    fmt = lambda t: re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(t))

    for line in report.splitlines():
        s = line.strip()
        if not s:
            continue

        clean = s.lstrip("#").strip()
        plain = clean.replace("*", "").rstrip(":").strip()

        if s.startswith("#") or plain.lower() in sections or (clean.startswith("**") and clean.endswith("**")):
            content += [Paragraph(escape(plain), head_st),
                        HRFlowable(width="100%", thickness=0.5, color=line_c)]
        elif s[:2] in ("- ", "* "):
            content.append(Paragraph(fmt(s[2:]), body_st, bulletText="•"))
        else:
            content.append(Paragraph(fmt(s), body_st))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2*cm, 1.2*cm,
                          "Experimental ML project - not a medical device. Do not rely on this report for medical decisions.")
        canvas.drawRightString(A4[0]-2*cm, 1.2*cm, f"Page {doc.page}")
        canvas.restoreState()

    pdf = io.BytesIO()
    SimpleDocTemplate(pdf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                      topMargin=2*cm, bottomMargin=2*cm
                      ).build(content, onFirstPage=footer, onLaterPages=footer)

    st.session_state.report = report
    st.session_state.pdf = pdf.getvalue()


def show_pdf(key):
    if st.session_state.report and st.session_state.pdf:
        st.markdown(st.session_state.report)
        st.download_button("Download PDF", st.session_state.pdf, "ECG_Report.pdf",
                           mime="application/pdf", key=key)


# ---------- Live ECG component ----------

LIVE_HTML = r"""
<!DOCTYPE html>
<html>
<head>
<style>
    body { font-family: sans-serif; text-align: center; margin-top: 10px; }
    button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #FF4B4B;
             color: white; border: none; border-radius: 5px; }
    #status { margin-top: 10px; font-weight: bold; color: #555; }
    canvas { width: 100%; height: 300px; margin-top: 15px; background: white; border: 1px solid #ddd; }
</style>
</head>
<body>

<button id="connectBtn">Connect to Arduino & Read 10s</button>
<div id="status">Waiting for connection...</div>
<canvas id="ecgCanvas"></canvas>

<script>

const send = (type, data) =>
    window.parent.postMessage({isStreamlitMessage: true, type, ...data}, "*");

send("streamlit:componentReady", {apiVersion: 1});
send("streamlit:setFrameHeight", {height: 430});

let port = null, writer = null, sent = false;
let ecgData = [], displayData = [];
let lp = null, bl = null, pending = false;

const status = document.getElementById("status");
const canvas = document.getElementById("ecgCanvas");
const ctx = canvas.getContext("2d");

// Display-only filter (raw ecgData is what goes to the model)
function filterSample(x) {
    if (lp === null) { lp = x; bl = x; }
    lp += 0.25 * (x - lp);     // smoothing
    bl += 0.01 * (x - bl);     // baseline removal
    return lp - bl;
}

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    canvas.width = r.width * dpr;
    canvas.height = r.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawECG() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    if (displayData.length < 2) return;

    const data = displayData.slice(-800);
    const min = Math.min(...data), max = Math.max(...data);
    const range = Math.max(max - min, 1);

    ctx.beginPath();
    ctx.strokeStyle = "#00A86B";
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";

    data.forEach((v, i) => {
        const x = (i / (data.length - 1)) * w;
        const y = h - ((v - min) / range) * (h - 40) - 20;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });

    ctx.stroke();
}

function requestDraw() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; drawECG(); });
}

resizeCanvas();
window.addEventListener("resize", () => { resizeCanvas(); drawECG(); });


window.addEventListener("message", async (e) => {

    if (e.data.type !== "streamlit:render") return;

    const result = e.data.args.lcd_result;

    if (result && port && writer && !sent) {

        sent = true;

        try {
            await writer.write(new TextEncoder().encode(result + "\n"));
            await new Promise(r => setTimeout(r, 500));
            writer.releaseLock();
            await port.close();
            status.innerText = "Result sent to LCD: " + result;
        } catch (err) {
            status.innerText = "Could not send result to LCD.";
            console.error(err);
        }

        port = null;
        writer = null;
    }
});


document.getElementById("connectBtn").addEventListener("click", async () => {

    sent = false;
    ecgData = [];
    displayData = [];
    lp = bl = null;
    drawECG();

    try {

        port = await navigator.serial.requestPort();
        await port.open({ baudRate: 115200 });
        writer = port.writable.getWriter();

        status.innerText = "Connected! Live ECG reading...";

        const decoder = new TextDecoderStream();
        const closed = port.readable.pipeTo(decoder.writable);
        const reader = decoder.readable.getReader();

        let buffer = "";
        const start = Date.now();

        while (Date.now() - start < 10000) {

            const remaining = 10000 - (Date.now() - start);

            const readResult = await Promise.race([
                reader.read(),
                new Promise(resolve => setTimeout(() => resolve(null), remaining))
            ]);

            if (readResult === null || readResult.done) break;

            buffer += readResult.value;
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {

                const parts = line.trim().split(",");
                if (parts.length !== 3) continue;

                if (parseInt(parts[1]) === 1 || parseInt(parts[2]) === 1) {

                    status.innerText = "Error: Electrodes disconnected!";

                    reader.cancel();
                    await closed.catch(() => {});
                    writer.releaseLock();
                    await port.close();

                    port = null;
                    writer = null;

                    send("streamlit:setComponentValue", {value: [], dataType: "json"});
                    return;
                }

                const sample = parseFloat(parts[0]);

                if (!isNaN(sample)) {
                    ecgData.push(sample);
                    displayData.push(filterSample(sample));
                    if (displayData.length > 1500) displayData.shift();
                    requestDraw();
                }
            }
        }

        reader.cancel();
        await closed.catch(() => {});

        status.innerText = "Finished reading. Classifying...";

        send("streamlit:setComponentValue", {value: ecgData, dataType: "json"});

    } catch (err) {
        status.innerText = "Connection failed or cancelled.";
        console.error(err);
    }
});

</script>
</body>
</html>
"""

comp_dir = Path(tempfile.gettempdir()) / "ecg_live_component"
comp_dir.mkdir(exist_ok=True)
(comp_dir / "index.html").write_text(LIVE_HTML, encoding="utf-8")

live_component = components.declare_component("ecg_live", path=str(comp_dir))


# ---------- Tabs ----------

tab1, tab2 = st.tabs(["Upload CSV", "Live ECG"])

with tab1:

    upload = st.file_uploader("Upload ECG File", type=["csv"], key="upload")

    if upload and st.button("Classify ECG", key="classify"):
        try:
            data = pd.read_csv(upload)

            beats = (data.values if data.shape[1] == 360
                     else get_beats(pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna().values))

            if len(beats):
                st.session_state.report = None
                st.session_state.pdf = None
                make_report(*classify(beats))
            else:
                st.error("No valid ECG beats found.")

        except Exception as e:
            st.error(f"Error: {e}")

    show_pdf("upload_pdf")


with tab2:

    st.warning("Note: Web Serial only works on Google Chrome or Microsoft Edge desktop browsers.")

    data = live_component(lcd_result=st.session_state.get("live_result"), key="live")

    if data == []:
        st.error("Electrodes are not connected or reading was cancelled.")

    elif data and data != st.session_state.get("live_data"):

        st.session_state.live_data = data
        st.session_state.live_result = None

        beats = get_beats(np.array(data))

        if len(beats):
            try:
                st.session_state.report = None
                st.session_state.pdf = None

                result, normal, abnormal, types = classify(beats)

                st.session_state.live_result = result
                st.session_state.live_out = (result, normal, abnormal, dict(types))

                make_report(result, normal, abnormal, types)

            except Exception as e:
                st.error(f"Error: {e}")
            else:
                st.rerun()
        else:
            st.error("Not enough ECG data. Try again.")

    if st.session_state.get("live_out") and st.session_state.get("live_data"):

        result, normal, abnormal, types = st.session_state.live_out

        st.write("**Detected Result:**", result)
        st.write("**Normal Beats:**", normal, "| **Abnormal Beats:**", abnormal)

        if result == "ABNORMAL":
            st.write("**Detected Beat Types:**", ", ".join(f"{k}: {v}" for k, v in types.items()))

    show_pdf("live_pdf")
