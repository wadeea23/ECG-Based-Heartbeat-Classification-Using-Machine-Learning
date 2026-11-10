
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

st.title("ECG Heartbeat Classifier")

st.write("Upload an ECG CSV file or image to classify heartbeats.")

model = tf.keras.models.load_model("cnn_model.keras")

classes = {
    0: "Normal (N)",
    1: "Supraventricular (S)",
    2: "Ventricular (V)",
    3: "Fusion (F)",
    4: "Unknown (Q)"
}

descriptions = {
    0: "Normal heartbeat",
    1: "Heartbeat starts above the ventricles",
    2: "Heartbeat starts in the ventricles",
    3: "Mix of normal and ventricular beats",
    4: "Unknown heartbeat"
}

file = st.file_uploader(
    "Upload ECG File",
    type=["csv", "png", "jpg", "jpeg"]
)

if file is not None:

    if file.name.endswith(".csv"):

        data = pd.read_csv(file, header=None)
        beats = data.iloc[:, :187].apply(
            pd.to_numeric, errors="coerce"
        ).dropna().values

    else:

        img = Image.open(file)
        st.image(img, use_container_width=True)

        gray = np.array(img.convert("L"))
        signal = 255 - gray.mean(axis=0)
        signal = (signal - signal.min()) / (
            signal.max() - signal.min() + 1e-9
        )

        signal = np.interp(
            np.linspace(0, len(signal) - 1, 187),
            np.arange(len(signal)),
            signal
        )

        beats = signal.reshape(1, 187)

    if len(beats) == 0:

        st.error("No ECG signal found.")

    else:

        st.subheader("ECG Signal")
        st.line_chart(beats[0])

        if st.button("Predict"):

            beats = beats.reshape(-1, 187, 1)

            predictions = model.predict(beats, verbose=0)

            results = np.argmax(predictions, axis=1)
            confidence = predictions.max(axis=1)

            total = len(results)
            abnormal = np.sum(results != 0)

            if abnormal == 0:

                st.success(
                    f"All {total} heartbeats are Normal."
                )

            else:

                st.error(
                    f"Abnormal heartbeats: {abnormal} out of {total}"
                )

                st.write(
                    "Please consult a doctor for medical evaluation."
                )

            st.subheader("Heartbeats per Class")

            counts = pd.Series(results).value_counts()

            summary = pd.DataFrame({
                "Class": [classes[i] for i in classes],
                "Heartbeats": [
                    int(counts.get(i, 0)) for i in classes
                ],
                "Meaning": [
                    descriptions[i] for i in classes
                ]
            })

            st.dataframe(summary, hide_index=True)

            st.bar_chart(
                summary.set_index("Class")["Heartbeats"]
            )

            st.subheader("Each Heartbeat")

            result_table = pd.DataFrame({
                "Heartbeat": range(1, total + 1),
                "Class": [
                    classes[int(i)] for i in results
                ],
                "Confidence": [
                    f"{i * 100:.1f}%" for i in confidence
                ]
            })

            if abnormal > 0:
                only_abnormal = st.checkbox(
                    "Show only abnormal heartbeats"
                )

                if only_abnormal:
                    result_table = result_table[
                        results != 0
                    ]

            st.dataframe(result_table, hide_index=True)

            if file.name.endswith(".csv"):

                st.subheader("View a Heartbeat")

                number = st.number_input(
                    "Choose heartbeat",
                    min_value=1,
                    max_value=total,
                    value=1
                )

                st.line_chart(beats[int(number) - 1])

                st.write(
                    "Class:",
                    classes[int(results[int(number) - 1])]
                )

st.caption(
    "This app is for educational purposes only, not a medical diagnosis."
)
