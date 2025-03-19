import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Load the new model
model = joblib.load('stacking_classifier_model.pkl')

# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('X_test.csv')

Duration_of_diabetes = {
    1: '≤1year',
    2: '1-5years',
    3: '＞5years'
}


# Define feature names from the new dataset
feature_names = ["Age",'Duration_of_diabetes',"SCII",'PLT','AST_ALT',"PBG","HbAlc","VFA","METS_IR"]

# Streamlit user interface
st.title("CMM Predictor")

# Input fields in the left column
st.subheader("Input Features")
Age = st.number_input("Age:", min_value=0, max_value=120, value=41)  # 统一为浮点数
Duration_of_diabetes = st.selectbox("Duration.of.diabetes:", options=list(Duration_of_diabetes.keys()), format_func=lambda x: Duration_of_diabetes[x])
SCII = st.selectbox("SCII (NO, YES):", options=[0, 1], format_func=lambda x: 'NO' if x == 0 else 'YES')
PLT = st.number_input("PLT:", min_value=0.0, max_value=10000.0, value=157.0)  # 统一为浮点数
AST_ALT = st.number_input("AST/ALT:", min_value=0.00, max_value=1000.00, value=1.00)  # 已经是浮点数
PBG = st.number_input("PBG:", min_value=0.00, max_value=100.00, value=8.00)  # 已经是浮点数
HbAlc = st.number_input("HbAlc:", min_value=0.0, max_value=50.0, value=6.0)
VFA = st.number_input("VFA:", min_value=0.0, max_value=10000.0, value=90.0)  # 统一为浮点数
METS_IR = st.number_input("METS_IR:", min_value=0.00, max_value=500.00, value=22.00)

# Process inputs and make predictions in the right column
feature_values = [Age,Duration_of_diabetes,SCII,PLT,AST_ALT,PBG,HbAlc,VFA,METS_IR]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取类别 0 和类别 1 的概率
    prob_class_0 = predicted_proba[0] * 100
    prob_class_1 = predicted_proba[1] * 100

    # 展示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Probability of Class 0 (No Heart Disease):** {prob_class_0:.1f}%")
    st.write(f"**Probability of Class 1 (Heart Disease):** {prob_class_1:.1f}%")

    # 根据预测结果生成建议
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease. "
            f"The model predicts that your probability of having heart disease is {prob_class_1:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease. "
            f"The model predicts that your probability of not having heart disease is {prob_class_0:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

    st.write(advice)

   




































