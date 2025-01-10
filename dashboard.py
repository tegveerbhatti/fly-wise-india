import streamlit as st
import pickle
from datetime import time
import pandas as pd
import numpy as np

with open('encoder.pkl', 'rb') as file:
    loaded_encoder = pickle.load(file)

def preprocess_user_data(airline, flight_class, stops, src, dest, arrival, depart, duration, days_left):
    new_data = pd.DataFrame({
        'airline': [airline],
        'source_city': [src],
        'departure_time': [depart],
        'arrival_time': [arrival],
        'destination_city': [dest],
    })
    
    encoded_data = loaded_encoder.transform(new_data)
    encoded_data_array = encoded_data.toarray()
    
    stops = np.array([[stops]])
    encoded_data_array = np.insert(encoded_data_array, 3, stops, axis=1)
    encoded_class = np.array([[1 if flight_class == 'Business' else 0]])
    encoded_data_array = np.append(encoded_data_array, encoded_class, axis=1)
    encoded_duration = np.array([[duration / 60]])
    encoded_data_array = np.append(encoded_data_array, encoded_duration, axis=1)
    days_left = np.array([[days_left]])
    encoded_data_array = np.append(encoded_data_array, days_left, axis=1)
    
    return encoded_data_array

with open('flight_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict_price(airline, flight_class, stops, src, dest, arrival_time, departure_time, duration, days_left):
    encoded_data_array = preprocess_user_data(airline, flight_class, stops, src, dest, arrival_time, departure_time, duration, days_left)
    price = model.predict(encoded_data_array)
    st.write(f"### The predicted price of the flight is: ₹{price[0]} 🛫")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .title {
        font-size: 3rem;
        font-weight: bold;
        color: #4a4e69;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #22223b;
    }
    .highlight {
        font-weight: bold;
        color: #9a8c98;
    }
    .stButton > button {
        background-color: #4a4e69;
        color: white;
        border-radius: 10px;
        font-size: 1rem;
        padding: 10px 20px;
    }
    .stButton > button:hover {
        background-color: #22223b;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">FlyWise India 🛩️</div>', unsafe_allow_html=True)

# Introduction
st.markdown("""
    <div class="subtitle">
        Welcome to <span class="highlight">FlyWise India 🇮🇳</span>! This app predicts the price 💰 of a flight in India based on various features such as:
        <ul>
            <li>The Airline 🛄</li>
            <li>Source and Destination Cities 🏙️</li>
            <li>Arrival and Departure Times ⏰</li>
            <li>Number of Stops and More 📦</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Fill in the details below to get your flight price prediction:</div>", unsafe_allow_html=True)

# Layout improvements with columns
st.markdown("---")
col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1.5])

with col1:
    airline = st.selectbox("✈️ Airline:", ['SpiceJet', 'Air_India', 'Vistara', 'GO_FIRST', 'Indigo'])
    src = st.selectbox("🌆 Source City:", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    duration = st.number_input("🕒 Duration (in minutes):", min_value=0, max_value=1440, step=1)

with col2:
    flight_class = st.selectbox("🎫 Class:", ['Economy', 'Business'])
    dest = st.selectbox("🌆 Destination City:", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    stops = st.number_input("🛑 Number of Stops:", min_value=0, max_value=5, step=1)

with col3:
    arrival_time = st.selectbox("⏰ Arrival Time:", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
    days_left = st.number_input("📅 Days Left for Flight:", min_value=1, max_value=365, step=1)

with col4:
    departure_time = st.selectbox("⏰ Departure Time:", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])

# Predict button
st.markdown("---")
center_button = st.columns([1, 1, 1])
with center_button[1]:
    if st.button("🚀 Predict Price 💸"):
        predict_price(airline, flight_class, stops, src, dest, arrival_time, departure_time, duration, days_left)
