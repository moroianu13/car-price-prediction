import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# Set title
st.title("🚗 Car Price Prediction")

# Custom CSS for styling buttons & inputs
st.markdown(
    """
    <style>
        /* Style the button */
        .stButton>button {
            background-color: #D90429; /* Red color */
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 10px;
            width: 100%;
            padding: 10px;
        }

        /* Style number input boxes */
        .stNumberInput>div>div>input {
            font-size: 16px;
            text-align: center;
        }

        /* Style the sidebar */
        [data-testid="stSidebar"] {
            background-color: #1B1B1B;
        }

        /* Style the success message */
        .stAlert {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# Load trained model
model_path = "models/car_price_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error("❌ Model file not found. Train and save the model first.")

# Define feature names (must match training data)
feature_names = ["curb_weight", "engine_size", "horsepower", "width"]

# Create input fields for user
st.subheader("Enter Car Details:")




with st.form("prediction_form"):
    curb_weight = st.number_input("Curb Weight (kg)", min_value=500.0, max_value=3000.0, value=1200.0, step=1.0)
    engine_size = st.number_input("Engine Size (cc)", min_value=600.0, max_value=6000.0, value=2000.0, step=10.0)
    horsepower = st.number_input("Horsepower (hp)", min_value=50.0, max_value=1000.0, value=150.0, step=5.0)
    width = st.number_input("Width (cm)", min_value=100.0, max_value=250.0, value=170.0, step=1.0)

    submit_button = st.form_submit_button("Predict Price")

# Make prediction when user submits the form
if submit_button:
    if model is not None:
        # Convert inputs to DataFrame
        input_data = pd.DataFrame([[curb_weight, engine_size, horsepower, width]], columns=feature_names)

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        st.success(f"🚗 Predicted Car Price: **${predicted_price:,.2f}**")
    else:
        st.error("❌ Model not loaded. Cannot make predictions.")
