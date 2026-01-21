import os
import pandas as pd
import joblib
import streamlit as st
# import google.generativeai as genai  # Removed as we're using SambaNova
import requests  # Added for SambaNova API calls
from dotenv import load_dotenv
# 1. Load environment variables from .env file
load_dotenv("gemini.env")  # You may want to rename this file to something like .env for SambaNova

# 2. Configure SambaNova securely
# Make sure your .env file has the line: SAMBANOVA_API_KEY=your_actual_key
api_key = os.getenv("SAMBANOVA_API_KEY")  # Changed to SambaNova API key

# ----------------------------
# SambaNova Logic (renamed from Gemini)
# ----------------------------
def sambanova_patient_explanation(patient_dict, risk_percent, pred):
    """
    Generates a patient-friendly explanation using SambaNova API.
    """
    prompt = f"""
    You are a healthcare assistant inside a clinical decision support tool.
    Write a patient-friendly explanation in SIMPLE English.

    Rules:
    - Do NOT say "you have diabetes". This is only a risk estimate.
    - Be calm and supportive, not scary.
    - Use bullet points.
    - Mention key reasons using the patient's values (e.g., BMI of {patient_dict['bmi']}).
    - Suggest next steps: confirmatory tests and lifestyle changes.
    - End with a safety note: consult a doctor.
    - Include a diet chart in tabular form (use Markdown table format).
    - Suggest exercises based on the patient's health data, and include suggestions for images (provide image URLs or descriptions that can be displayed visually).
    - Add visually appealing elements like emojis, bold text, or simple formatting to make it engaging.

    Patient data: {patient_dict}
    Model output: prediction={pred} (0=no diabetes, 1=high risk), risk={risk_percent:.2f}%

    Output format (use exactly these headings):
    Title:
    What this means:
    Why this result:
    What you can do now:
    When to see a doctor:
    Safety note:
    Diet Chart:
    Exercise Suggestions:
    """

    # Using SambaNova API
    api_url = "https://api.sambanova.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b",  # Use the appropriate model name; check SambaNova docs for latest
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,  # Adjust as needed
        "temperature": 0.7
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"

# ----------------------------
# Load saved ML model
# ----------------------------
# Ensure "diabetes_model.pkl" is in the same folder as this script
try:
    ml_model = joblib.load("diabetes_model.pkl")
except Exception as e:
    st.error(f"Could not load ML model: {e}")

st.title("Diabetes Risk Predictor (Demo)")
st.caption("This tool gives a risk estimate, not a diagnosis. Please consult a clinician for decisions.")

# ----------------------------
# Input fields (UI)
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    hypertension = st.selectbox("Hypertension (0=No, 1=Yes)", [0, 1])
    heart_disease = st.selectbox("Heart Disease (0=No, 1=Yes)", [0, 1])

with col2:
    smoking_history = st.selectbox(
        "Smoking History",
        ["never", "former", "current", "not current", "No Info", "ever"]
    )
    bmi = st.number_input("BMI", min_value=5.0, max_value=80.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
    glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=400, value=120)

# ----------------------------
# Prediction Logic
# ----------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }])

    # Get results from your trained model
    pred = int(ml_model.predict(input_df)[0])
    proba = float(ml_model.predict_proba(input_df)[0][1])
    risk_percent = proba * 100

    st.subheader("Result")
    
    if proba >= 0.80:
        st.error(f"High Risk — Probability: {risk_percent:.2f}%")
        st.write("Recommend confirmatory test (HbA1c / Fasting glucose).")
    elif proba >= 0.50:
        st.warning(f"Medium Risk — Probability: {risk_percent:.2f}%")
        st.write("Lifestyle changes + follow-up recommended.")
    else:
        st.success(f"Low Risk — Probability: {risk_percent:.2f}%")
        st.write("Continue routine monitoring.")

    st.divider()
    
    # ----------------------------
    # SambaNova Explanation Section (renamed from Gemini)
    # ----------------------------
    st.subheader("AI Explanation for Patient")
    
    patient_dict = {
        "gender": gender, "age": age, "hypertension": hypertension,
        "heart_disease": heart_disease, "smoking_history": smoking_history,
        "bmi": bmi, "HbA1c_level": hba1c, "blood_glucose_level": glucose
    }

    if not api_key:
        st.info("API key missing. Please check your .env file.")
    else:
        with st.spinner("SambaNova is analyzing the results..."):
            try:
                explanation = sambanova_patient_explanation(patient_dict, risk_percent, pred)  # Changed function call
                st.markdown(explanation)
            except Exception as e:
                st.error(f"AI Explanation failed: {e}")

st.caption("Safety note: This explanation supports understanding; it is not medical advice or a diagnosis.")
