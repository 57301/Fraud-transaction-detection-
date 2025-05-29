import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# Streamlit configuration
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
    .stTextInput>div>input {border-radius: 8px;}
    .sidebar .sidebar-content {background-color: #e0e7ff;}
    .success-box {background-color: #d4edda; padding: 10px; border-radius: 8px;}
    .error-box {background-color: #f8d7da; padding: 10px; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Fraud Detection Dashboard")
st.sidebar.markdown("Input transaction details to predict fraud using our ML model.")
st.sidebar.image("https://img.icons8.com/color/96/000000/credit-card.png", width=100)

# Main content
st.title("🔍 Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it's fraudulent.")

# API URL (update with your FastAPI URL)
API_URL = "https://fraud-detection-api.onrender.com/predict"  # Replace with your FastAPI URL

# Input form
required_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
                    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                    'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
                    'V28', 'Amount']

with st.form("transaction_form"):
    st.subheader("Transaction Features")
    cols = st.columns(5)
    data = {}
    for i, feature in enumerate(required_features):
        with cols[i % 5]:
            data[feature] = st.number_input(feature, value=0.0, format="%.6f", key=feature)
    submitted = st.form_submit_button("Predict Fraud", use_container_width=True)

# Prediction and visualization
if submitted:
    with st.spinner("Analyzing transaction..."):
        try:
            response = requests.post(API_URL, json=data)
            response.raise_for_status()
            result = response.json()
            st.subheader("Prediction Result")
            if result["fraud"]:
                st.markdown(f"<div class='error-box'>🚨 <b>Fraudulent Transaction!</b> (Probability: {result['fraud_probability']:.2%})</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='success-box'>✅ <b>Non-Fraudulent Transaction</b> (Probability of Fraud: {result['fraud_probability']:.2%})</div>", unsafe_allow_html=True)
            prob_data = pd.DataFrame({
                "Category": ["Non-Fraud", "Fraud"],
                "Probability": [1 - result["fraud_probability"], result["fraud_probability"]]
            })
            fig = px.pie(prob_data, values="Probability", names="Category", title="Fraud Probability Distribution",
                         color_discrete_sequence=["#4CAF50", "#FF5252"])
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("View Input Transaction Details"):
                st.json(data)
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
