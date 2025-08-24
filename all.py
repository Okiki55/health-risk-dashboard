import streamlit as st
import pickle
import numpy as np

# Load models
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))
heart_model = pickle.load(open("heart.pkl", "rb"))

st.set_page_config(page_title="AI-Powered Disease Risk Dashboard", layout="wide")
st.title("🩺 AI-Powered Disease Risk Dashboard")
st.write("Predict risk for **Diabetes** and **Heart Disease** using patient records.")

# Sidebar for navigation
option = st.sidebar.selectbox(
    "Select Disease to Predict",
    ("Diabetes", "Heart Disease")
)

def get_user_input(features):
    user_data = []
    for f in features:
        val = st.number_input(f"Enter {f}:", value=0.0)
        user_data.append(val)
    return np.array(user_data).reshape(1, -1)

if option == "Diabetes":
    st.subheader("Diabetes Risk Prediction")
    diabetes_features = ["Pregnancies", "Glucose", "BloodPressure", 
                         "SkinThickness", "Insulin", "BMI", 
                         "DiabetesPedigreeFunction", "Age"]
    inputs = get_user_input(diabetes_features)

    if st.button("Predict Diabetes Risk"):
        prediction = diabetes_model.predict(inputs)[0]
        if prediction == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

elif option == "Heart Disease":
    st.subheader("Heart Disease Risk Prediction")
    heart_features = ["Age", "Sex", "ChestPainType", "RestingBP", 
                      "Cholesterol", "FastingBS", "RestingECG", 
                      "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope","cp","trestbps"]
    inputs = get_user_input(heart_features)

    if st.button("Predict Heart Disease Risk"):
        prediction = heart_model.predict(inputs)[0]
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")
