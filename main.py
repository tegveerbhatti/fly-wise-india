import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import math
import matplotlib.pyplot as plt
import streamlit as st
import pickle

df = pd.read_csv('Clean_Dataset.csv')
df = df.drop(['Unnamed: 0', 'flight'], axis=1) 

df['class'] = df['class'].apply(lambda x: 1 if x == 'Business' else 0)
df['stops'] = pd.factorize(df['stops'])[0]

encoder = OneHotEncoder(handle_unknown='ignore')
encoder = encoder.fit(df[['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city']])
with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

df = df.join(pd.get_dummies(df.airline, prefix='airline')).drop(['airline'], axis=1)
df = df.join(pd.get_dummies(df.source_city, prefix='source')).drop(['source_city'], axis=1)
df = df.join(pd.get_dummies(df.destination_city, prefix='dest')).drop(['destination_city'], axis=1)
df = df.join(pd.get_dummies(df.arrival_time, prefix='arrival')).drop(['arrival_time'], axis=1)
df = df.join(pd.get_dummies(df.departure_time, prefix='departure')).drop(['departure_time'], axis=1)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor(n_jobs=-1)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# print('R2 Score:', r2_score(y_test, y_pred))
# print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
# print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
# print('Root Mean Squared Error:', math.sqrt(mean_squared_error(y_test, y_pred)))

with open('flight_price_model.pkl', 'wb') as f:
    pickle.dump(reg, f)

# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, alpha=0.5, color='b')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3) 
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual vs Predicted Flight Prices')
# plt.tight_layout()
# plt.show()





