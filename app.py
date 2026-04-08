!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load trained model
model = joblib.load("Credit_Card_Fraud_model.pkl")

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write("Detect whether a transaction is *Fraudulent or Legitimate*")

# Sidebar
st.sidebar.header("Choose Input Method")
option = st.sidebar.radio("Select Option", ["Manual Input", "Upload CSV"])

# ===============================
# 🔹 Manual Input
# ===============================
if option == "Manual Input":
    st.subheader("Enter Transaction Details")

    amount = st.number_input("Transaction Amount", min_value=0.0)

    # Example V1–V5 (extend if needed)
    v1 = st.number_input("V1")
    v2 = st.number_input("V2")
    v3 = st.number_input("V3")
    v4 = st.number_input("V4")
    v5 = st.number_input("V5")

    if st.button("Predict"):
        input_data = np.array([[amount, v1, v2, v3, v4, v5]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("🚨 Fraud Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

# ===============================
# 🔹 CSV Upload
# ===============================
elif option == "Upload CSV":
    st.subheader("Upload CSV File")

    file = st.file_uploader("Upload your dataset", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)
        st.write("📄 Uploaded Data:", data.head())

        if st.button("Predict All"):
            try:
                predictions = model.predict(data)

                data["Prediction"] = predictions
                data["Prediction"] = data["Prediction"].map({
                    0: "Legitimate",
                    1: "Fraud"
                })

                st.write("✅ Results:", data.head())

                # Download results
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error: {e}")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.write("Built with ❤️ using Streamlit")
pip install --upgrade pip