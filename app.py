
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
import pickle

# Load model
model = pickle.load(open("Credit_Card_Fraud_model.pkl", "rb"))

st.title("💳 Credit Card Fraud Detection")

st.write("Enter transaction details:")

inputs = []

# Take 30 inputs
for i in range(30):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

if st.button("Predict"):
    input_data = np.array([inputs])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🚨 Fraud Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")