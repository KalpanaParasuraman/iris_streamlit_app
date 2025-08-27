# app.py
import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
#from sklearn.metrics import accuracy_score

# Load iris dataset (for labels)
iris = load_iris()

# Load trained model
model = joblib.load(r"C:\Users\admin\.conda\envs\myenv\iris_model.pkl")

st.title("🌸 Iris Flower Predictor")

# User input sliders
st.sidebar.write("Inputs:::::")
sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 1.0)

features = [[sepal_length, sepal_width, petal_length, petal_width]]
if st.sidebar.button("Predict"):
	# Prediction
	prediction = model.predict(features)[0]
	st.write("### Predicted Species:")
	st.write(iris.target_names[prediction])

