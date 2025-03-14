import streamlit as st
import joblib
import numpy as np

# Load the trained model, scaler, and encoders
model = joblib.load("model.pkl")
scaler = joblib.load("scalar.pkl")
label_encoder = joblib.load("label.pkl")
emp_enocoded = joblib.load("emparked.pkl")

# Streamlit UI
st.title("Titanic Survival Prediction")
st.write("Enter passenger details to predict survival.")

# User Inputs
pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, step=0.1)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, step=1)
fare = st.number_input("Fare", min_value=0.0, max_value=600.0, step=0.1)
embarked = st.selectbox("Embarked", ["C", "Q", "S"])

# Preprocessing
sex_encoded = label_encoder.transform([sex])[0]
embarked_encoded = emp_enocoded.transform([embarked])[0]

input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data_scaled)
    result = "Survived" if prediction[0] == 1 else "Did not Survive"
    st.write(f"Prediction: **{result}**")
