import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Title
st.title("🌊 Fluid Dynamics Prediction App")

# Load model
model = load_model("https://github.com/Gaurav4541/fluid-dynamic-app/blob/main/fluid%20dynamic_model.h5")

# Upload file
uploaded_file = st.file_uploader("Upload your CSV file (like Flow.csv)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df.head())

    # Preprocessing (Adjust this based on your model training)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Predict
    predictions = model.predict(X_scaled)
    st.write("Predicted Output:")
    st.dataframe(predictions)
