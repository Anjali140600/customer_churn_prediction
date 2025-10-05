import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import pandas as pd

model = tf.keras.models.load_model('model.h5')
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}
input_df = pd.DataFrame([input_data])
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
input_full = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)
input_scaled = scaler.transform(input_full)
prediction = model.predict(input_scaled)
prediction_proba=prediction[0][0]

churn = prediction[0][0] > 0.5
if churn:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
st.write(f'Churn probability: {prediction[0][0]:.2f}')