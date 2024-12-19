#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Helper function for cleaning LinkedIn usage column
def clean_sm(column):
    # Convert column to binary: 1 for LinkedIn users, 0 otherwise
    return np.where(column == 1, 1, 0)

# Prediction function
def predict_linkedin_usage(model, income, education, parent, married, female, age):
    individual = pd.DataFrame({
        'income': [income],
        'education': [education],
        'parent': [parent],
        'married': [married],
        'female': [female],
        'age': [age]
    })
    probability = model.predict_proba(individual)[:, 1][0]
    prediction = model.predict(individual)[0]
    return prediction, round(probability, 3)

# Streamlit app layout
st.title("LinkedIn Usage Predictor")
st.write("Predict whether a person uses LinkedIn and the probability of usage based on personal attributes.")

# Human-readable options for income and education
income_labels = {
    1: "Less than $10,000",
    2: "$10,000 - $19,999",
    3: "$20,000 - $29,999",
    4: "$30,000 - $39,999",
    5: "$40,000 - $49,999",
    6: "$50,000 - $74,999",
    7: "$75,000 - $99,999",
    8: "$100,000 - $149,999",
    9: "$150,000 or more"
}

education_labels = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year bachelor’s degree",
    7: "Some postgraduate education",
    8: "Postgraduate or professional degree"
}

# Load the social media usage data (no user upload required)
try:
    s = pd.read_csv("social_media_usage.csv")

    # Filter and clean data
    filtered_data = s[(s['income'] <= 9) & (s['educ2'] <= 8) & (s['age'] <= 98)]
    ss = filtered_data[['income', 'educ2', 'par', 'marital', 'sex', 'age']].copy()
    ss.rename(columns={'educ2': 'education', 'par': 'parent', 'marital': 'married', 'sex': 'female'}, inplace=True)
    ss['female'] = np.where(ss['female'] == 2, 1, 0)
    ss['sm_li'] = clean_sm(filtered_data['web1h'])

    # Features and target
    y = ss['sm_li']
    X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression Model
    log_reg = LogisticRegression(class_weight='balanced', random_state=42)
    log_reg.fit(X_train, y_train)

    # User inputs
    income = st.selectbox("Household Income", options=income_labels.keys(), format_func=lambda x: income_labels[x])
    education = st.selectbox("Education Level", options=education_labels.keys(), format_func=lambda x: education_labels[x])
    parent = st.selectbox("Are you a parent?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    married = st.selectbox("Marital Status", options=[0, 1], format_func=lambda x: "Married" if x == 1 else "Not Married")
    female = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 1 else "Male")
    age = st.number_input("Age", min_value=1, max_value=98, value=42)

    # Prediction button
    if st.button("Predict LinkedIn Usage"):
        prediction, probability = predict_linkedin_usage(log_reg, income, education, parent, married, female, age)
        result = "LinkedIn User" if prediction == 1 else "Not a LinkedIn User"
        st.write(f"**Prediction:** {result}")
        st.write(f"**Probability of LinkedIn Usage:** {probability * 100:.1f}%")
        
except Exception as e:
    st.error(f"An error occurred while loading or processing the data: {e}")
