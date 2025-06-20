import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
# Page configuration & custom style
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="💼",
    layout="centered",
)
st.markdown(
    """
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Gradient background */
    body {
        background: linear-gradient(90deg, #e8f2ff 0%, #f9fcff 33%, #fffdfb 66%, #ffece6 100%);
    }

    /* Progress‑bar gradient */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #ff4b1f, #ff9068);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Header
st.title("✨ Customer Churn Predictor")
st.markdown("Predict the probability that a bank customer will leave — instantly!")
# Helper functions (cached)
@st.cache_resource
def load_model(path: str):
    """Load and cache the TensorFlow model."""
    return tf.keras.models.load_model(path)
@st.cache_resource
def load_pickle(path: str):
    """Generic pickle loader with caching."""
    with open(path, "rb") as f:
        return pickle.load(f)
# Load artefacts
model = load_model("model.h5")
label_encoder_gender = load_pickle("label_encoder_gender.pkl")
onehot_encoder_geo = load_pickle("onehot_encoder_geo.pkl")
scaler = load_pickle("scaler.pkl")
# Sidebar — user inputs
with st.sidebar:
    st.header("🔧 Input Parameters")
    geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 92, 35)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1, value=650)
    with col2:
        tenure = st.slider("Tenure (years)", 0, 10, 3)
        balance = st.number_input("Balance", min_value=0.0, step=100.0, value=0.0, format="%.2f")
    st.subheader("Products & Activity")
    col3, col4 = st.columns(2)
    with col3:
        num_of_products = st.slider("Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])
    with col4:
        is_active_member = st.selectbox("Active Member", ["No", "Yes"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=1000.0, value=50000.0, format="%.2f")
    st.caption("ℹ︎ Adjust the sliders & inputs, then click **Predict**!")
# Map categorical Yes/No to 0/1
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_member_val = 1 if is_active_member == "Yes" else 0
# Prediction
if st.button("🚀 Predict"):
    # Prepare the dataframe
    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card_val],
        "IsActiveMember": [is_active_member_val],
        "EstimatedSalary": [estimated_salary],
    })
    # One‑hot encode Geography and join
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
    )
    input_df = pd.concat([input_df, geo_df], axis=1)
    # Scale
    input_scaled = scaler.transform(input_df)
    # Predict
    proba = model.predict(input_scaled)[0][0]
    # Display results
    st.subheader("📊 Result")
    st.metric("Churn Probability", f"{proba:.2%}")
    st.progress(min(int(proba * 100), 100))
    if proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("🎉 The customer is **not likely to churn**.")
