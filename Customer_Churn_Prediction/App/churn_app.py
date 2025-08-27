import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature columns
pipeline = joblib.load("churn_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("Customer Churn Prediction app")

st.write("Enter customer details to predict churn probability.")

gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support", ["No", "Yes"])
payment_method = st.selectbox("Payment Method", ["Credit card (automatic)", "Electronic check", "Mailed check"])


# Start with empty row
input_data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

# Map inputs to encoded format
input_data["gender"] = 1 if gender == "Male" else 0
input_data["SeniorCitizen"] = 1 if senior == "Yes" else 0
input_data["Partner"] = 1 if partner == "Yes" else 0
input_data["Dependents"] = 1 if dependents == "Yes" else 0
input_data["tenure"] = tenure
input_data["PhoneService"] = 1 if phone_service == "Yes" else 0
input_data["PaperlessBilling"] = 1 if paperless == "Yes" else 0
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges

# One-hot encoded features
input_data[f"Contract_{contract}"] = 1
input_data[f"TechSupport_{tech_support}"] = 1
input_data[f"PaymentMethod_{payment_method}"] = 1

if st.button("Predict Churn"):
    prediction = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0][1]  # churn probability

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {proba:.2f})")
    else:
        st.success(f"✅ Customer likely to stay (Churn Probability: {proba:.2f})")
