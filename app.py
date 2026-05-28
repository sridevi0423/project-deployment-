import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
import pickle

@st.cache_resource
def load_model():
    with open("gradient_boosting_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'gradient_boosting_model.pkl' not found. Please upload the model file to the same directory.")
    st.stop()

# -----------------------------------
# Sidebar Inputs
# -----------------------------------
st.sidebar.header("Customer Input Features")

# 1. Year
year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2019)

# 2. CreditScore
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)

# 3. Gender (Handled as Male=1, Female=0)
gender_selection = st.sidebar.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender_selection == "Male" else 0

# 4. Age
age = st.sidebar.slider("Age", 18, 100, 35)

# 5. Tenure
tenure = st.sidebar.slider("Tenure", 0, 10, 5)

# 6. Balance
balance = st.sidebar.number_input("Balance", min_value=0.0, value=50000.0)

# 7. NumOfProducts
num_products = st.sidebar.slider("Number of Products", 1, 4, 1)

# 8. HasCrCard
has_card = st.sidebar.selectbox("Has Credit Card (1=Yes, 0=No)", [1, 0])

# 9. IsActiveMember
active_member = st.sidebar.selectbox("Is Active Member (1=Yes, 0=No)", [1, 0])

# 10. EstimatedSalary
salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# 11 & 12. Geography (One-Hot Encoding)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# -----------------------------------
# Create Input DataFrame in SPECIFIC Order
# -----------------------------------
# This dictionary matches the exact index order you provided
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
# Prediction Display
# -----------------------------------

if st.button("Predict Churn Risk"):

    st.subheader("Analysis Results")

    probability = model.predict_proba(
        input_data
    )[:,1][0]

    col_metrics, col_chart = st.columns([1,1])

    with col_metrics:

        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

        if probability >= 0.75:
            st.error(
                "Status: High Risk Customer"
            )

        elif probability >= 0.50:
            st.warning(
                "Status: Medium Risk Customer"
            )

        else:
            st.success(
                "Status: Low Risk Customer"
            )

    with col_chart:

        fig, ax = plt.subplots(
            figsize=(6,2)
        )

        ax.barh(
            ['Churn Risk'],
            [probability]
        )

        ax.set_xlim(0,1)

        ax.axvline(
            0.5,
            linestyle='--'
        )

        st.pyplot(fig)
# -----------------------------------
# Feature Importance
# -----------------------------------
st.divider()
st.subheader("Model Feature Importance")
importance = model.feature_importances_
feat_importance = pd.Series(importance, index=input_data.columns).sort_values()

st.bar_chart(feat_importance)

st.caption("Note: The model considers 'Year' as a feature as per the European Bank dataset structure.")