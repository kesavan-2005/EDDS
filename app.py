import streamlit as st
import pandas as pd
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Load models ===
MODEL_DIR = r"D:\EDDS\models"

diabetes_model = load_model(os.path.join(MODEL_DIR, r"D:\EDDS\models\diabetes_model.keras"))
diabetes_scaler = joblib.load(os.path.join(MODEL_DIR, r"D:\EDDS\models\diabetes_scaler.pkl"))
retinopathy_model = load_model(os.path.join(MODEL_DIR, r"D:\EDDS\models\retinopathy_model.keras"))
xray_model = load_model(os.path.join(MODEL_DIR, r"D:\EDDS\models\xray_cnn_model.keras"))

# === Predict diabetes ===
def predict_diabetes(input_data):
    data = np.array(input_data).reshape(1, -1)
    scaled = diabetes_scaler.transform(data)
    pred = diabetes_model.predict(scaled)
    result = "Diabetes Found" if pred[0][0] > 0.5 else "Diabetes Not Found"
    return result, float(pred[0][0])

# === Predict retinopathy ===
def predict_retinopathy(img_path):
    classes = ["Mild", "Moderate", "No DR", "Proliferate", "Severe"]
    img = image.load_img(img_path, target_size=(128, 128))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = retinopathy_model.predict(img_arr)
    return classes[np.argmax(pred[0])]

# === Predict x-ray ===
def predict_xray(img_path):
    classes = [
        "abscess", "ards", "atelectasis", "atherosclerosis of the aorta", "cardiomegaly",
        "emphysema", "fracture", "hydropneumothorax", "hydrothorax", "pneumonia",
        "pneumosclerosis", "post-inflammatory changes", "post-traumatic ribs deformation",
        "sarcoidosis", "scoliosis", "tuberculosis", "venous congestion"
    ]
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    pred = xray_model.predict(img_arr)
    return classes[np.argmax(pred[0])]

# === Streamlit UI ===
st.set_page_config(page_title="AI Health Predictor", layout="wide")
st.title("🧠 AI-Powered Early Disease Detection System")

# Diabetes Section
st.header("🩺 Diabetes Prediction")
st.markdown("Enter vitals to predict diabetes:")

cols = st.columns(4)
inputs = []
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
for i, feat in enumerate(features):
    val = cols[i % 4].number_input(feat, value=0.0 if feat != "Age" else 18.0)
    inputs.append(val)

if st.button("🔍 Predict Diabetes"):
    with st.spinner("Analyzing vitals..."):
        result, confidence = predict_diabetes(inputs)
        st.success(f"Prediction: {result} (Confidence: {confidence:.2f})")

st.markdown("---")

# Retinopathy Section
st.header("👁️ Retinopathy Detection")
retina_img = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"], key="retina")

if st.button("🔬 Predict Retinopathy"):
    if retina_img:
        retina_path = "temp_retina.jpg"
        with open(retina_path, "wb") as f:
            f.write(retina_img.read())
        with st.spinner("Analyzing retina image..."):
            r_class = predict_retinopathy(retina_path)
            st.success(f"Predicted Class: {r_class}")
            os.remove(retina_path)
    else:
        st.warning("Please upload a retina image.")

st.markdown("---")

# X-ray Section
st.header("🫁 Chest X-ray Analysis")
xray_img = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"], key="xray")

if st.button("🧬 Predict Chest X-ray"):
    if xray_img:
        xray_path = "temp_xray.jpg"
        with open(xray_path, "wb") as f:
            f.write(xray_img.read())
        with st.spinner("Analyzing chest X-ray..."):
            x_class = predict_xray(xray_path)
            st.success(f"Predicted Class: {x_class}")
            os.remove(xray_path)
    else:
        st.warning("Please upload a chest X-ray image.")
