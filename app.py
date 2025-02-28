import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
from pycaret.regression import load_model as load_regressor

st.title("Interactive PyCaret Model Deployment")

# Load the PyCaret model from the 'models' folder
model_path = "models/best_regression_model"
model = load_model(model_path)
st.success("Model loaded successfully from models/best_regression_model.pkl!")

# Dynamically ask for feature inputs
st.subheader("Enter Feature Values")

# Extract feature names from the PyCaret model
feature_names = list(model.feature_names_in_)  # Expected features
user_inputs = {}

for feature in feature_names:
    user_inputs[feature] = st.text_input(f"Enter {feature}")

# Convert input to DataFrame and predict
if st.button("Predict"):
    try:
        # Convert input values to the correct types
        input_df = pd.DataFrame([user_inputs]).apply(pd.to_numeric, errors='coerce')

        # Check for missing values
        if input_df.isnull().values.any():
            st.error("Please enter valid numerical values for all features.")
        else:
            # Predict using PyCaret
            predictions = predict_model(model, data=input_df)
            st.success(f"Prediction: {predictions['prediction_label'][0]}")
    except Exception as e:
        st.error(f"Error: {e}")
