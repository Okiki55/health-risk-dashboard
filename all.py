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

# background data used for SHAP global plots (must match training preprocessing)
dia_bg = pickle.load(open("diabetes_bg.pkl", "rb"))
heart_bg = pickle.load(open("heart_bg.pkl", "rb"))

# ===============================
# Page config + header (colorful)
# ===============================
st.set_page_config(page_title="AI-Powered Disease Risk Dashboard", layout="wide")

st.markdown(
    """
    <div style="background: linear-gradient(90deg,#6DD5FA,#2980B9); padding: 16px; border-radius:8px;">
      <h1 style="color:white; margin:0;">🩺 AI-Powered Disease Risk Dashboard</h1>
      <p style="color: white; margin:0;">Predict Diabetes & Heart Disease and inspect model explanations (SHAP).</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")  # spacing

# Sidebar
option = st.sidebar.selectbox("Select Disease to Predict", ("Diabetes", "Heart Disease"))

# Util: plot helper (Streamlit-friendly)
def st_shap(plot_func, height=420):
    fig = plt.figure(figsize=(8, 5))
    plot_func()
    st.pyplot(fig, clear_figure=True)

# ===============================
# Diabetes Prediction + Explainability
# ===============================
if option == "Diabetes":
    st.subheader("Diabetes Risk Prediction")

    # Inputs (two-column layout)
    left, right = st.columns(2)
    with left:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, value=0)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    with right:
        insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, step=0.01, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)

    diabetes_features = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }
    input_df = pd.DataFrame([diabetes_features])

    c1, c2 = st.columns([1, 1])
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

        # Ensure columns order matches training background
        dia_cols = list(dia_bg.columns)

        # Use pipeline's scaler + model (training pipeline: scaler -> model)
        preproc = diabetes_model.named_steps["scaler"]
        model = diabetes_model.named_steps["model"]

        # transform background and input (columns aligned)
        X_bg_trans = preproc.transform(dia_bg[dia_cols])
        X_in_trans = preproc.transform(input_df[dia_cols])

        explainer = shap.TreeExplainer(model)

        # Get shap values robustly (handle Explanation objects)
        try:
            raw_shap = explainer(X_in_trans, check_additivity=False)
            if hasattr(raw_shap, "values"):
                shap_vals = raw_shap.values
            else:
                shap_vals = np.array(raw_shap)
        except Exception as e:
            st.error(f"SHAP explainer failed: {e}")
            shap_vals = None

        if shap_vals is not None:
            # Local waterfall (single row) - ensure we pass a 1D array for the sample
            st.markdown("**Local explanation (waterfall):**")
            try:
                sample_shap = shap_vals[0]
                exp_val = explainer.expected_value
                # If expected_value is array (multiclass) pick relevant element if needed; we assume binary prob for class 1
                fig = shap.plots._waterfall.waterfall_legacy(exp_val, sample_shap, feature_names=dia_cols, show=False)
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"Waterfall plot failed: {e}")

            # Global importance
            try:
                bg_shap_raw = explainer(X_bg_trans, check_additivity=False)
                if hasattr(bg_shap_raw, "values"):
                    bg_shap_vals = bg_shap_raw.values
                else:
                    bg_shap_vals = np.array(bg_shap_raw)

                st.markdown("**Global feature importance (SHAP):**")
                fig2 = plt.figure(figsize=(8, 5))
                shap.summary_plot(bg_shap_vals, features=X_bg_trans, feature_names=dia_cols, plot_type="bar", show=False)
                st.pyplot(fig2, clear_figure=True)
            except Exception as e:
                st.error(f"Global SHAP plot failed: {e}")

# ===============================
# Heart Disease Prediction + Explainability
# ===============================
elif option == "Heart Disease":
    st.subheader("Heart Disease Risk Prediction")

    # Dropdowns for categorical features (labels match training mappings)
    left, right = st.columns(2)
    with left:
        sex = st.selectbox("Sex", ["Female", "Male"])
        chest_pain = st.selectbox("Chest Pain Type", ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"])
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["≤120 mg/dl", ">120 mg/dl"])
        rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T abnormality", "LV hypertrophy"])
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    with right:
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=50)
        resting_bp = st.number_input("Resting BP (trestbps)", min_value=0, max_value=250, value=120)
        chol = st.number_input("Cholesterol (chol)", min_value=0, max_value=700, value=200)
        max_hr = st.number_input("Max Heart Rate (thalach)", min_value=60, max_value=250, value=150)
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
        ca = st.number_input("Major Vessels (ca) 0–3", min_value=0, max_value=3, step=1, value=0)

    heart_features = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": chol,
        "FastingBS": fasting_bs,
        "RestingECG": rest_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
        "CA": ca,
        "Thal": thal,
    }

    input_df = pd.DataFrame([heart_features])

    # 🔹 Apply same mappings as training (values)
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

    # 🔹 Rename columns to match training dataset column names (lowercase)
    input_df = input_df.rename(columns={
        "Age": "age",
        "Sex": "sex",
        "ChestPainType": "cp",
        "RestingBP": "trestbps",
        "Cholesterol": "chol",
        "FastingBS": "fbs",
        "RestingECG": "restecg",
        "MaxHR": "thalach",
        "ExerciseAngina": "exang",
        "Oldpeak": "oldpeak",
        "ST_Slope": "slope",
        "CA": "ca",
        "Thal": "thal"
    })

    # Show a small summary card
    st.markdown("### Input summary")
    st.table(input_df.T.rename(columns={0: "value"}))

    c1, c2 = st.columns([1, 1])
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

        # use background columns in the same order as training
        heart_cols = list(heart_bg.columns)

        # Preprocess background & input through pipeline’s scaler (training pipeline: scaler -> model)
        preproc = heart_model.named_steps["scaler"]
        model = heart_model.named_steps["model"]

        # transform background and input (columns aligned)
        X_bg_trans = preproc.transform(heart_bg[heart_cols])
        X_in_trans = preproc.transform(input_df[heart_cols])

        explainer = shap.TreeExplainer(model)

        # Get shap values robustly
        try:
            raw_shap = explainer(X_in_trans, check_additivity=False)
            if hasattr(raw_shap, "values"):
                shap_vals = raw_shap.values
            else:
                shap_vals = np.array(raw_shap)
        except Exception as e:
            st.error(f"SHAP explainer failed: {e}")
            shap_vals = None

        if shap_vals is not None:
            # Local waterfall (single row)
            st.markdown("**Local explanation (waterfall):**")
            try:
                sample_shap = shap_vals[0]
                exp_val = explainer.expected_value
                fig = shap.plots._waterfall.waterfall_legacy(exp_val, sample_shap, feature_names=heart_cols, show=False)
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"Waterfall plot failed: {e}")

            # Global importance (SHAP bar)
            try:
                bg_shap_raw = explainer(X_bg_trans, check_additivity=False)
                if hasattr(bg_shap_raw, "values"):
                    bg_shap_vals = bg_shap_raw.values
                else:
                    bg_shap_vals = np.array(bg_shap_raw)

                st.markdown("**Global feature importance (SHAP):**")
                fig2 = plt.figure(figsize=(8, 5))
                shap.summary_plot(bg_shap_vals, features=X_bg_trans, feature_names=heart_cols, plot_type="bar", show=False)
                st.pyplot(fig2, clear_figure=True)
            except Exception as e:
                st.error(f"Global SHAP plot failed: {e}")
