import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from tensorflow.keras.models import load_model


model_tf = load_model("model_tf.h5")
loaded_preprocessor = joblib.load("preprocessor.joblib")
geography = loaded_preprocessor.named_transformers_["cat"].named_steps["cat_encoder"].categories_[0].tolist()
gender = loaded_preprocessor.named_transformers_["cat"].named_steps["cat_encoder"].categories_[1].tolist()

# Streamlit app
st.title("Customer Churn Prediction")

# User Input
geography   = st.selectbox("Geography", geography)
gender      = st.selectbox("Gender", gender)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
creditscore = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active member".title(), [0, 1])

# prepare the input data
input_data = pd.DataFrame(
    [{
        "CreditScore" : creditscore,
        "Geography" : geography,
        "Gender" : gender,
        "Age" : age,
        "Tenure" : tenure,
        "Balance" : balance,
        "NumOfProducts" : num_of_products,
        "HasCrCard" : has_cr_card,
        "IsActiveMember" : is_active_member,
        "EstimatedSalary" : estimated_salary
    }]
)

# preprocess data
preprocessed_arr = loaded_preprocessor.transform(input_data)
st.write("Customer not likely to churn" if model_tf.predict(preprocessed_arr)[0][0] <=0.5 else "Customer likely to churn")

"Introduction to python collections part I (List_set_tuples_dictionaries extended)"