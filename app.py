import streamlit as st
import pandas as pd 
import numpy as np
import joblib

### load the model
filename = 'Final_Linear_Regression.pkl'
loaded_model = joblib.load(filename)

st.title('House Price Predictor Model Using Linear Regression')

## Input values from the user
income = st.number_input('Enter Average Area Income', min_value=16000, max_value=110000, value=68000)
house_age = st.number_input('Enter House Age', min_value=1, max_value=12, value=6)
rooms = st.number_input('Enter Average Number of Rooms', min_value=3, max_value=11, value=7)
bedrooms = st.number_input('Enter Average Number of Bedrooms', min_value=1, max_value=7, value=4)
population = st.number_input('Enter Average Area Population', min_value=150, max_value=70000, value=36000)


input_dict = {'income':income, 'house_age':house_age, 'rooms':rooms, 'bedrooms':bedrooms, 'population':population}
input_df = pd.DataFrame(input_dict, index=[0])
## predict button
predict_button = st.button('Predict Price')

if predict_button:

    price = np.abs(loaded_model.predict(input_df))
    st.metric('Estimated Price', price.round(0))
