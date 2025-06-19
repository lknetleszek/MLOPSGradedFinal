import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/predict/")

st.title("Diabetes Disease Prediction")

uploaded_file = st.file_uploader("Upload your CSV file for diabetes prediction", type=["csv"])

if uploaded_file is not None:
    # Preview uploaded data
    original_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(original_df.head())

    if st.button("Get Predictions"):
        with st.spinner("Sending data to API..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
            response = requests.post(API_URL, files=files)

            if response.status_code == 200:
                prediction_records = response.json()
                prediction_df = pd.DataFrame(prediction_records)

                # Optional: Join predictions with original data (safe join on index if no ID column)
                combined_df = original_df.copy()
                for col in prediction_df.columns:
                    if col not in combined_df.columns:
                        combined_df[col] = prediction_df[col]

                st.success("Predictions received!")
                st.subheader("Data with Predictions")
                st.dataframe(combined_df)

                st.download_button(
                    "Download Results as CSV",
                    combined_df.to_csv(index=False),
                    file_name="diabetes_predictions.csv",
                )
            else:
                st.error(f"API Error: {response.status_code}")
