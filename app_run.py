#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Shop data Prediction App

This app predicts the **Shop data price**!
""")
st.write('---')


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('age', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider('income', X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider('gender', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider('m_status', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
    
    data = {'age': AGE,
            'income': INCOME,
            'gender': GENDER,
            'm_status': M_STATUS}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)

# Reads in saved classification model
load_clf = pickle.load(open('data.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.header('Prediction of Product')
st.write(prediction)
st.write('---')



st.subheader('Prediction')


shop_species = np.array(['yes','no'])
st.write(shop_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:




