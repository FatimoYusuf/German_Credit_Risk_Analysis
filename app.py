import streamlit as st
import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("ensemble_credit_model.pkl")
sex_encoder = joblib.load("sex_encoder.pkl")
housing_encoder = joblib.load("housing_encoder.pkl")
saving_encoder = joblib.load("saving_account_encoder.pkl")
checking_encoder = joblib.load("checking_account_encoder.pkl")

# App title
st.title("German Credit Risk Predictor")
st.write("""
This app predicts whether a loan applicant is low or high risk 
based on their financial and demographic information.
Built using a Random Forest and Extra Trees Ensemble model 
trained on the German Credit dataset.
""")

# Input fields
age = st.number_input("Age", min_value=18, max_value=75, value=30)

sex = st.selectbox("Sex", ["male", "female"])

job = st.selectbox("Job Type", ["unskilled_and_non-resident", "skilled", "highly skilled"])

housing = st.selectbox("Housing", ["own", "free", "rent"])

saving_account = st.selectbox("Saving Account", ["little", "moderate", "quite rich", "rich"])

checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])

credit_amount = st.number_input("Credit Amount", min_value=250, max_value=18424, value=3000)

duration = st.number_input("Loan Duration (months)", min_value=4, max_value=72, value=12)

# Predict button
if st.button("Predict Credit Risk"):

    # Encode inputs
    sex_encoded = sex_encoder.transform([sex])[0]
    job_encoded = {"unskilled_and_non-resident": 0, "skilled": 1, "highly skilled": 2}[job]
    housing_encoded = housing_encoder.transform([housing])[0]
    saving_encoded = saving_encoder.transform([saving_account])[0]
    checking_encoded = checking_encoder.transform([checking_account])[0]

    # Create input dataframe
    input_data = pd.DataFrame(
        [[age, sex_encoded, job_encoded, housing_encoded,
          saving_encoded, checking_encoded,
          credit_amount, duration]],
        columns=["age", "sex", "job", "housing",
                 "saving_account", "checking_account",
                 "credit_amount", "duration"]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Show result
    if prediction == 0:
        st.success(f"Low Risk - This applicant is likely to repay the loan. Confidence: {probability[0]*100:.1f}%")
    else:
        st.error(f"High Risk - This applicant is likely to default on the loan. Confidence: {probability[1]*100:.1f}%")
