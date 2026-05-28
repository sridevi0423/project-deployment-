import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # Standard for loading sklearn models

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    layout="wide"
)

st.title("Bank Customer Churn Prediction System")
st.write("Predicting customer churn risk based on the provided European Bank dataset features.")

# -----------------------------------
# Load Trained Model
# -----------------------------------
@st.cache_resource
def load_model():
    # joblib is more robust than pickle for sklearn versions
    return joblib.load("gradient_boosting_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure 'gradient_boosting_model.pkl' is in the same folder as this script.")
    st.stop()

# -----------------------------------
# Sidebar Inputs (Feature Collection)
# -----------------------------------
st.sidebar.header("Customer Input Features")

year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2025)
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
gender_selection = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 35)
tenure = st.sidebar.slider("Tenure", 0, 10, 5)
balance = st.sidebar.number_input("Balance", min_value=0.0, value=50000.0)
num_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_card = st.sidebar.selectbox("Has Credit Card (1=Yes, 0=No)", [1, 0])
active_member = st.sidebar.selectbox("Is Active Member (1=Yes, 0=No)", [1, 0])
salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# -----------------------------------
# Data Transformation
# -----------------------------------
gender_val = 1 if gender_selection == "Male" else 0
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# EXACT ORDER matching your dataset index
input_data = pd.DataFrame({
    'Year': [year],
    'CreditScore': [credit_score],
    'Gender': [gender_val],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_products],
    'HasCrCard': [has_card],
    'IsActiveMember': [active_member],
    'EstimatedSalary': [salary],
    'Geography_Germany': [geo_germany],
    'Geography_Spain': [geo_spain]
})

# -----------------------------------
# Prediction Display (Real-time)
# -----------------------------------
st.subheader("Analysis Results")

# Get probability for the "Positive" class (Churn)
probability = model.predict_proba(input_data)[:, 1][0]

col_metrics, col_chart = st.columns([1, 1])

with col_metrics:
    st.metric(label="Churn Probability", value=f"{probability:.2%}")
    
    if probability >= 0.75:
        st.error("Status: High Risk Customer")
    elif probability >= 0.50:
        st.warning("Status: Medium Risk Customer")
    else:
        st.success("Status: Low Risk Customer")

with col_chart:
    fig, ax = plt.subplots(figsize=(6, 2))
    color = 'red' if probability >= 0.5 else 'green'
    ax.barh(['Churn Risk'], [probability], color=color)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='black', linestyle='--')
    st.pyplot(fig)

# -----------------------------------
# Feature Importance Dashboard
# -----------------------------------
st.divider()
st.subheader("Feature Importance Dashboard")

importance = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': input_data.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=True)

st.bar_chart(importance_df.set_index('Feature'))