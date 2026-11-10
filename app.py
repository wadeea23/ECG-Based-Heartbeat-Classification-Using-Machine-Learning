import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

st.title("ECG Heartbeat Classifier")

st.write("Upload your ECG CSV file to classify heartbeats.")

model = tf.keras.models.load_model("cnn_model.keras")

classes = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)"
]

file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    data = pd.read_csv(file, header=None)

    beats = data.iloc[:, :187].values
    beats = beats.astype(float)

    beats = beats.reshape(-1, 187, 1)

    if st.button("Analyze ECG"):

        prediction = model.predict(beats)

        results = np.argmax(prediction, axis=1)

        total = len(results)
        abnormal = np.sum(results != 0)

        st.subheader("Analysis Results")

        st.write("Total Heartbeats:", total)
        st.write("Abnormal Heartbeats:", abnormal)
        st.write("Normal Heartbeats:", total - abnormal)

        st.subheader("Heartbeat Classification")

        counts = pd.Series(results).value_counts()

        summary = pd.DataFrame({
            "Heartbeat Type": classes,
            "Count": [counts.get(i, 0) for i in range(5)]
        })

        st.dataframe(summary, hide_index=True)

        st.subheader("Classification Chart")

        st.bar_chart(summary.set_index("Heartbeat Type"))

