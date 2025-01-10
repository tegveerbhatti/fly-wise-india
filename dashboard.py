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

st.title("FlyWise India 🛩️")

st.write("""### Welcome to FlyWise India 🇮🇳! This app predicts the price 💰 of a flight in India based on various features such as:
         \n#### - The Airline 🛄
         \n#### - Source and Destination Cities 🏙️
         \n#### - Arrival and Departure Times ⏰
         \n#### - And More 📦""")
st.write("#### Please fill in the details below to get your flight price prediction.")

col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

airline = col1.selectbox("Airline:", ['SpiceJet', 'Air_India', 'Vistara', 'GO_FIRST', 'Indigo'])
flight_class = col2.selectbox("Class:", ['Economy', 'Business'])
stops = col3.number_input("Number of Stops:", min_value=0, max_value=5, step=1)
src = col1.selectbox("Source City:", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
dest = col2.selectbox("Destination City:", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
arrival_time = col3.selectbox("Arrival Time:", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
departure_time = col4.selectbox("Departure Time:", ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
duration = col1.number_input("Duration (in minutes):", min_value=0, max_value=1440, step=1)
days_left = col4.number_input("Days Left for Flight:", min_value=1, max_value=365, step=1)

if st.button("Predict Price 💸"):
    predict_price(airline, flight_class, stops, src, dest, arrival_time, departure_time, duration, days_left)
