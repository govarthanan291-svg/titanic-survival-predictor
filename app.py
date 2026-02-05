import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained Naive Bayes model
model_file = 'naive_bayes_model_2026-02-05_12-30-15.pkl'  # Replace with your actual pickle file name

# Check if the model file exists and load it
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    st.write("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Streamlit app header
st.title("Titanic Survival Predictor")
st.write("Predict whether a passenger survived the Titanic disaster based on their information.")

# Input fields for the user
pclass = st.selectbox("Pclass (Ticket Class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("SibSp (Siblings/Spouses aboard)", min_value=0, value=0)
parch = st.number_input("Parch (Parents/Children aboard)", min_value=0, value=0)
fare = st.number_input("Fare", min_value=0.0, value=10.0)
embarked = st.selectbox("Embarked (Port of Embarkation)", ["C", "Q", "S"])

# Label encoding for 'Sex' and 'Embarked'
sex_map = {"male": 0, "female": 1}
embarked_map = {"C": 0, "Q": 1, "S": 2}

# Show the input values
st.write(f"Pclass: {pclass}, Sex: {sex}, Age: {age}, SibSp: {sibsp}, Parch: {parch}, Fare: {fare}, Embarked: {embarked}")

# Prepare the input data for prediction
input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex_map[sex]],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked": [embarked_map[embarked]],
})

# Debugging: Show input data
st.write("Input data for prediction:")
st.write(input_data)

# Make the prediction using the loaded model
try:
    prediction = model.predict(input_data)
    st.write(f"Prediction result: {prediction[0]}")
except Exception as e:
    st.error(f"Error making prediction: {e}")

# Display the prediction result
if prediction[0] == 1:
    st.write("The passenger survived.")
else:
    st.write("The passenger did not survive.")
