# FlyWise India - Flight Price Prediction Web App 🛩️

FlyWise India is an interactive web application that predicts flight prices in India based on user inputs such as airline, flight class, number of stops, departure and arrival times, and more. The app is built using **Streamlit** for the frontend and leverages a pre-trained machine learning model to predict flight prices in real time.

## Features

- **User-friendly interface**: Built with Streamlit, the dashboard allows users to easily input flight details.
- **Real-time flight price prediction**: Uses a machine learning model trained on historical flight data to predict prices.
- **Various input options**: Users can specify airline, flight class, stops, source city, destination city, duration, and days left before the flight.

## How It Works

1. **User Input**: The app collects flight details through a user-friendly form on the dashboard.
2. **Data Preprocessing**: Inputs are preprocessed using an encoder to transform categorical data into numerical format.
3. **Price Prediction**: A pre-trained machine learning model (Random Forest Regressor) predicts the price based on the preprocessed input data.
4. **Results**: The predicted flight price is displayed to the user on the dashboard.

## File Structure

- **`dashboard.py`**: Contains the Streamlit code for the frontend, including the user input form and logic to display the predicted price.
- **`main.py`**: Contains the core logic for preprocessing the data, loading the machine learning model, and generating predictions.
- **`Clean_Dataset.csv`**: The dataset containing historical flight information used to train the model. It includes features like airline, source and destination cities, stops, departure and arrival times, flight duration, days left until flight, and the actual price.

## Dataset

The dataset (`Clean_Dataset.csv`) includes the following features:

- `airline`: The airline operating the flight (e.g., SpiceJet, Air_India, Vistara).
- `flight`: The flight number.
- `source_city`: The city from which the flight departs.
- `departure_time`: The time of day the flight departs (e.g., Morning, Evening).
- `stops`: Number of stops during the flight.
- `arrival_time`: The time of day the flight arrives.
- `destination_city`: The city where the flight lands.
- `class`: The class of the flight (Economy or Business).
- `duration`: The duration of the flight in hours.
- `days_left`: Number of days left before the flight.
- `price`: The actual price of the flight.

## Installation

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command:
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app
   ```bash
    streamlit run dashboard.py

## Usage

Once the app is running, fill in the details of your flight (such as airline, flight class, source and destination cities, etc.) and click the "Predict Price" button. The predicted price will be displayed on the screen.

## Model Details

The machine learning model used for prediction is a Random Forest Regressor trained using the scikit-learn library. The model achieved an R² score of 98.49% on the training data, making it highly accurate for flight price predictions.

## Technologies Used

- **Streamlit**: For building the interactive web application.
- **pandas**: For data manipulation and preprocessing.
- **scikit-learn**: For building and training the machine learning model.
- **NumPy**: For numerical computations.
- **Matplotlib**: For visualizing prediction accuracy and model performance.
