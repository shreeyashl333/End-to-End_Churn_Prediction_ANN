import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle
import pandas as pd, numpy as np

# Load trained model
model = tf.keras.models.load_model('model.h5')

#load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    le_gender = pickle.load(file)

with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title("Customer Churn Prediction")
st.write("Enter customer details and click **Predict**")

# User Inputs from UI
credit_score = st.number_input("Credit Score")
geography = st.selectbox("Geography", ohe_geo.categories_[0])
gender = st.selectbox("Gender", le_gender.classes_)
age = st.slider("Age", 18, 92)
tenure = st.slider("Tenure (years)", 0, 10)
balance = st.number_input("Balance")
num_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
salary = st.number_input("Estimated Salary")

if st.button("🔮 Predict Churn"):
    # prep input data
    input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [le_gender.transform([gender])[0]],           # Label Encode 
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active],
            "EstimatedSalary": [salary]
        })

    # OHE "Geography"
    geo_encoded = ohe_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

    # combine OHE to input_df
    input_df = input_data
    input_df = input_df.join(geo_encoded_df)

    # scaling input
    input_df_scaled = scaler.transform(input_df)

    # pred
    prediction = model.predict(input_df_scaled)


    # Output
    st.subheader("Prediction Result")
    st.write(f"**Prediction Probability:** {prediction[0][0]:.2%}")

    if prediction > 0.5:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is unlikely to churn")