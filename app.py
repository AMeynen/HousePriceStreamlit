import streamlit as st
import numpy as np
import pandas as pd
from pycaret.regression import *


# Load trained ML model
house_price_model = load_model('house_price_model')

# Test trained ML model
# prediction = predict_model(house_price_model,data=test_data)

# Title
st.title('House prediction app :house:')
# st.text(prediction['prediction_label'].values[0])

# Inputs

with st.form("prediction_form"):
    area = st.slider('Area in sq meters:', 100, 15000)
    bedrooms = st.slider('Number of Bedrooms:', 1, 5)
    bathrooms = st.slider('Number of Bathrooms:', 1, 5)
    stories = st.slider('Number of building stories:', 1, 4)
    mainroad = st.toggle("Near Mainroad?")
    if mainroad:
        mainroad = 'yes'
    else: 
        mainroad = 'no'
    guestroom = st.toggle("Has guest room?")
    if guestroom:
        guestroom = 'yes'
    else: 
        guestroom = 'no'
    basement = st.toggle("Has basement?")
    if basement:
        basement = 'yes'
    else: 
        basement = 'no'
    hotwaterheating = st.toggle("Has hot water heating?")
    if hotwaterheating:
        hotwaterheating = 'yes'
    else: 
        hotwaterheating = 'no'
    airconditioning = st.toggle("Has airconditioning?")
    if airconditioning:
        airconditioning = 'yes'
    else: 
        airconditioning = 'no'
    parking = st.slider('Number of parking spots:', 0, 4)
    prefarea = st.toggle("Located in preferential area?")
    if prefarea:
        prefarea = 'yes'
    else:
        prefarea ='no'
    furnishingstatus = st.select_slider('Furnishing status:', ['unfurnished', 'semi-furnished', 'furnished'])
    submitted = st.form_submit_button("Predict :house: price!")
if submitted: # Check when template is submitted.
    prediction_data = pd.DataFrame({"area":area, "bedrooms":bedrooms, "bathrooms":bathrooms,
                                    "stories":stories, "mainroad":mainroad, "guestroom":guestroom,
                                    "basement":basement, "hotwaterheating":hotwaterheating, "airconditioning":airconditioning,
                                    "parking":parking, "prefarea":prefarea, "furnishingstatus":furnishingstatus}, index= [0])
    prediction = predict_model(house_price_model,data=prediction_data)
    st.header("Predicted house price is: {:.0f} $".format(prediction['prediction_label'].values[0]))
