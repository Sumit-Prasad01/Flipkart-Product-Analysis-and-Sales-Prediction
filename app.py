import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open("models/random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Flipkart Escalation Predictor", layout="centered")

st.title("📦 Flipkart Support Escalation Predictor")
st.write("Predict whether a customer issue is likely to be escalated based on agent interaction and ticket metadata.")

# Input fields
response_time = st.number_input("Response Time (in minutes)", min_value=0.0, step=1.0, value=30.0)
item_price = st.number_input("Item Price (₹)", min_value=0.0, step=10.0, value=999.0)

channel = st.selectbox("Channel", ["Inbound", "Outcall"])
shift = st.selectbox("Agent Shift", ["Morning", "Evening", "Night", "Split"])
tenure = st.selectbox("Agent Tenure", ["31-60", "61-90", ">90", "On Job Training"])

# Convert to one-hot encoding manually (same as training)
input_data = {
    'response_time_minutes': response_time,
    'Item_price': item_price,
    'channel_name_Inbound': 1 if channel == "Inbound" else 0,
    'channel_name_Outcall': 1 if channel == "Outcall" else 0,
    'Agent Shift_Morning': 1 if shift == "Morning" else 0,
    'Agent Shift_Evening': 1 if shift == "Evening" else 0,
    'Tenure Bucket_31-60': 1 if tenure == "31-60" else 0,
    'Tenure Bucket_61-90': 1 if tenure == "61-90" else 0,
    'Tenure Bucket_>90': 1 if tenure == ">90" else 0,
    'Tenure Bucket_On Job Training': 1 if tenure == "On Job Training" else 0
}

# Arrange columns
ordered_cols = [
    'response_time_minutes', 'Item_price', 
    'channel_name_Inbound', 'Tenure Bucket_>90', 
    'Agent Shift_Evening', 'channel_name_Outcall', 
    'Agent Shift_Morning', 'Tenure Bucket_31-60', 
    'Tenure Bucket_On Job Training', 'Tenure Bucket_61-90'
]

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[ordered_cols]

# Predict button
if st.button("Predict Escalation"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Escalation Likely! (Probability: {probability:.2f})")
    else:
        st.success(f"✅ No Escalation Expected. (Probability: {probability:.2f})")
