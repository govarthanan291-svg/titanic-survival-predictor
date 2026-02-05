import streamlit as st
import pandas as pd
import pickle
import os

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ------------------------------
# TITLE
# ------------------------------
st.markdown("<h1 style='text-align:center;color:#00BFFF;'>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;'>Naive Bayes Machine Learning App</h4>", unsafe_allow_html=True)
st.write("---")

# ------------------------------
# LOAD MODEL
# ------------------------------
MODEL_FILE = "naive_bayes_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error("❌ Model file not found! Place your .pkl file in this folder.")
    st.stop()

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

st.success("✅ Model Loaded Successfully")

# ------------------------------
# USER INPUTS
# ------------------------------
st.subheader("🧍 Passenger Details")

pclass = st.selectbox("🎫 Ticket Class", [1, 2, 3])
sex = st.radio("🧑 Gender", ["male", "female"])
age = st.number_input("🎂 Age", 0, 100, 25)
sibsp = st.number_input("👨‍👩‍👧 Siblings/Spouse", 0, 10, 0)
parch = st.number_input("👶 Parents/Children", 0, 10, 0)
fare = st.number_input("💰 Ticket Fare", 0.0, 600.0, 50.0)
embarked = st.selectbox("⚓ Port", ["C", "Q", "S"])

# ------------------------------
# ENCODING
# ------------------------------
sex_map = {"male": 0, "female": 1}
embarked_map = {"C": 0, "Q": 1, "S": 2}

# ------------------------------
# PREDICTION BUTTON
# ------------------------------
if st.button("🔮 Predict Survival"):

    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex_map[sex]],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked_map[embarked]]
    })

    prediction = model.predict(input_data)[0]

    st.write("---")

    if prediction == 1:
        st.success("🎉 Passenger **SURVIVED**")
        st.balloons()
    else:
        st.error("💀 Passenger **DID NOT SURVIVE**")

# ------------------------------
# FOOTER
# ------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center;'>Made with ❤️ using Streamlit & Naive Bayes</p>",
    unsafe_allow_html=True
)

