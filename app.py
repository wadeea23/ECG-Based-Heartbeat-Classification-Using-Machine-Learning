
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.title("ECG Heartbeat Classifier")

st.write("Upload an ECG CSV file.")

model = tf.keras.models.load_model("cnn_model.keras")

classes = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)"
]

file = st.file_uploader("Upload ECG CSV", type=["csv"])

if file is not None:

    data = pd.read_csv(file, header=None)

    beats = data.iloc[:, :187].values

    st.subheader("ECG Signal")
    st.line_chart(beats[0])

    if st.button("Predict"):

        beats = beats.astype(float)
        beats = beats.reshape(-1, 187, 1)

        prediction = model.predict(beats)

        results = np.argmax(prediction, axis=1)
        confidence = np.max(prediction, axis=1)

        total = len(results)
        abnormal = np.sum(results != 0)

        if abnormal == 0:
            st.success("All heartbeats are Normal.")
        else:
            st.error(str(abnormal) + " abnormal heartbeats found.")

        st.subheader("Classification Results")

        result_data = pd.DataFrame({
            "Heartbeat": range(1, total + 1),
            "Class": [classes[i] for i in results],
            "Confidence": [str(round(c * 100, 2)) + "%" for c in confidence]
        })

        st.dataframe(result_data, hide_index=True)

        st.subheader("Heartbeats per Class")

        counts = pd.Series(results).value_counts()

        summary = pd.DataFrame({
            "Class": classes,
            "Count": [counts.get(i, 0) for i in range(5)]
        })

        st.bar_chart(summary.set_index("Class"))

        number = st.number_input(
            "Choose Heartbeat",
            min_value=1,
            max_value=total,
            value=1
        )

        st.subheader("Selected Heartbeat")

        st.line_chart(beats[int(number) - 1].flatten())

        st.write("Classification:", classes[results[int(number) - 1]])
