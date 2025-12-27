import streamlit as st
import numpy as np 
import joblib

# load model & scaler
model = joblib.load('pcos_model.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title='PCOS Prediction App', layout='centered')

st.title('🩺 PCOS Prediction App')

st.write('Masukkan data pasien untuk memprediksi risiko PCOS')

# input user
age = st.number_input('Age', min_value=10, max_value=60,value=25)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=23.0)

mens = st.selectbox(
    'Menstrual Irregularity',
    options=[0,1],
    format_func=lambda x: 'Tidak' if x==0 else 'Ya'
)

testosteron = st.number_input(
    'Testosterone Level (ng/dL)',
    min_value=10.0, max_value=200.0, value=50.0
)

afc = st.number_input(
    'Antral Follicle Count',
    min_value=0, max_value=50, value=8
)

st.caption(
    "⚠️ Aplikasi ini hanya untuk tujuan edukasi dan tidak menggantikan diagnosis medis."
)

# prediction
if st.button('🔍 Predict'):
    input_data = np.array([[age, bmi, mens, testosteron, afc]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Risiko PCOS TINGGI\n\nProbabilitas: {probability:.2%}")
    else:
        st.success(f"✅ Risiko PCOS RENDAH\n\nProbabilitas: {probability:.2%}")