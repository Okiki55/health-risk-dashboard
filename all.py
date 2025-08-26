# app.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ===============================
# Load Models + Background Data
# ===============================
diabetes_model = pickle.load(open("diabetes.pkl", "rb"))   # pipeline
heart_model = pickle.load(open("heart.pkl", "rb"))         # pipeline

dia_bg = pickle.load(open("diabetes_bg.pkl", "rb"))
heart_bg = pickle.load(open("heart_bg.pkl", "rb"))

st.set_page_config(page_title="AI-Powered Disease Risk Dashboard", layout="wide")
st.title("🩺 AI-Powered Disease Risk Dashboard")
st.write("Predict risk for **Diabetes** and **Heart Disease**, then inspect **why** with model explanations.")

# Sidebar
option = st.sidebar.selectbox("Select Disease to Predict", ("Diabetes", "Heart Disease"))

# Util: plot helper (Streamlit-friendly)
def st_shap(plot_func, height=400):
    fig = plt.figure(figsize=(8, 5))
    plot_func()
    st.pyplot(fig, clear_figure=True)

# ===============================
# Diabetes Prediction + Explainability
# ===============================
if option == "Diabetes":
    st.subheader("Diabetes Risk Prediction")

    # Inputs
    diabetes_features = {
        "Pregnancies": st.number_input("Pregnancies", min_value=0, max_value=20, step=1),
        "Glucose": st.number_input("Glucose", min_value=0, max_value=300),
        "BloodPressure": st.number_input("Blood Pressure", min_value=0, max_value=200),
        "SkinThickness": st.number_input("Skin Thickness", min_value=0, max_value=100),
        "Insulin": st.number_input("Insulin", min_value=0, max_value=1000),
        "BMI": st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1),
        "DiabetesPedigreeFunction": st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01),
        "Age": st.number_input("Age", min_value=0, max_value=120, step=1),
    }
    input_df = pd.DataFrame([diabetes_features])

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Predict Diabetes Risk"):
            pred = int(diabetes_model.predict(input_df)[0])
            prob = float(diabetes_model.predict_proba(input_df)[0][1])
            if pred == 1:
                st.error(f"⚠️ High Risk of Diabetes — Probability: {prob:.2f}")
            else:
                st.success(f"✅ Low Risk of Diabetes — Probability: {prob:.2f}")

    with c2:
        show_exp = st.checkbox("Explain Prediction (SHAP)")

    if show_exp:
        st.markdown("### 🔍 Why did the model predict that?")
        # Build explainer on the pipeline’s final estimator with a masker on background (through the pipeline)
        # We pass preprocessed background by running it through the pipeline’s first step.
        # Extract columns in the same order as training:
        dia_cols = list(dia_bg.columns)

        # Transform background through the scaler inside the pipeline so SHAP sees model inputs
        preproc = diabetes_model.named_steps["scaler"]
        model = diabetes_model.named_steps["model"]

        X_bg_trans = preproc.transform(dia_bg[dia_cols])
        X_in_trans = preproc.transform(input_df[dia_cols])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_in_trans, check_additivity=False)

        # Local waterfall (single row)
        st.markdown("**Local explanation (waterfall):**")
        fig = shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], feature_names=dia_cols, show=False)
        st.pyplot(fig, clear_figure=True)

        # Global importance (bar) using SHAP on background
        bg_shap = explainer(X_bg_trans, check_additivity=False)
        st.markdown("**Global feature importance (SHAP):**")
        fig2 = plt.figure(figsize=(8, 5))
        shap.summary_plot(bg_shap, features=X_bg_trans, feature_names=dia_cols, plot_type="bar", show=False)
        st.pyplot(fig2, clear_figure=True)
# ===============================
# Heart Disease Prediction + Explainability
# ===============================
elif option == "Heart Disease":
    st.subheader("Heart Disease Risk Prediction")

    # Dropdowns for categorical features (labels match training mappings)
    sex = st.selectbox("Sex", ["Female", "Male"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["≤120 mg/dl", ">120 mg/dl"])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])

    heart_features = {
        "Age": st.number_input("Age", min_value=0, max_value=120, step=1),
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": st.number_input("Resting BP", min_value=0, max_value=250),
        "Cholesterol": st.number_input("Cholesterol", min_value=0, max_value=700),
        "FastingBS": fasting_bs,
        "RestingECG": rest_ecg,
        "MaxHR": st.number_input("Max Heart Rate", min_value=60, max_value=250),
        "ExerciseAngina": exercise_angina,
        "Oldpeak": st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1),
        "ST_Slope": st_slope,
        "CA": st.number_input("Major Vessels (0–3)", min_value=0, max_value=3, step=1),
        "Thal": thal,
    }

    input_df = pd.DataFrame([heart_features])

    # 🔹 Apply same mappings as training
    sex_map = {"Female": 0, "Male": 1}
    cp_map = {"Typical angina": 0, "Atypical angina": 1, "Non-anginal pain": 2, "Asymptomatic": 3}
    fbs_map = {"≤120 mg/dl": 0, ">120 mg/dl": 1}
    restecg_map = {"Normal": 0, "ST-T abnormality": 1, "LV hypertrophy": 2}
    exang_map = {"No": 0, "Yes": 1}
    slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_map = {"Normal": 0, "Fixed defect": 1, "Reversible defect": 2}

    input_df = input_df.copy()
    input_df["Sex"] = input_df["Sex"].map(sex_map)
    input_df["ChestPainType"] = input_df["ChestPainType"].map(cp_map)
    input_df["FastingBS"] = input_df["FastingBS"].map(fbs_map)
    input_df["RestingECG"] = input_df["RestingECG"].map(restecg_map)
    input_df["ExerciseAngina"] = input_df["ExerciseAngina"].map(exang_map)
    input_df["ST_Slope"] = input_df["ST_Slope"].map(slope_map)
    input_df["Thal"] = input_df["Thal"].map(thal_map)

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Predict Heart Disease Risk"):
            pred = int(heart_model.predict(input_df)[0])
            prob = float(heart_model.predict_proba(input_df)[0][1])
            if pred == 1:
                st.error(f"⚠️ High Risk of Heart Disease — Probability: {prob:.2f}")
            else:
                st.success(f"✅ Low Risk of Heart Disease — Probability: {prob:.2f}")

    with c2:
        show_exp = st.checkbox("Explain Prediction (SHAP)")

    if show_exp:
        st.markdown("### 🔍 Why did the model predict that?")
        heart_cols = list(heart_bg.columns)

        # Preprocess background & input through pipeline’s scaler
        preproc = heart_model.named_steps["scaler"]
        model = heart_model.named_steps["model"]

        X_bg_trans = preproc.transform(heart_bg[heart_cols])
        X_in_trans = preproc.transform(input_df[heart_cols])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_in_trans, check_additivity=False)

        # Local explanation
        st.markdown("**Local explanation (waterfall):**")
        fig = shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            feature_names=heart_cols,
            show=False
        )
        st.pyplot(fig, clear_figure=True)

        # Global importance (SHAP bar) using background
        bg_shap = explainer(X_bg_trans, check_additivity=False)
        st.markdown("**Global feature importance (SHAP):**")
        fig2 = plt.figure(figsize=(8, 5))
        shap.summary_plot(bg_shap, features=X_bg_trans, feature_names=heart_cols, plot_type="bar", show=False)
        st.pyplot(fig2, clear_figure=True)
