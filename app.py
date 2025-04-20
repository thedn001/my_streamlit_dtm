# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import numpy as np
import pickle

# load model
with open('dtm_trained_model.pkl', 'rb') as f:
    dtm_model = pickle.load(f)

# application title
st.title("Iris flower classification")
st.write("Enter the feature of the iris flower: ")

# Input fields
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)

petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.7)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = dtm_model.predict(input_data)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"The Predicted Species is: **{species[prediction[0]]}**")