import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
target_names = iris.target_names

# Load the trained model
model_file = "svc_model.pkl"
loaded_model = joblib.load(model_file)

# App Title
st.title("Iris Flower Prediction App 🌸")
st.markdown("""
This app predicts the **Iris flower species** based on the input features.
Use the sliders below to set the flower's measurements.
""")

# Sliders for input features
st.sidebar.header("Input Features")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(iris.data[:, 0].min()), float(iris.data[:, 0].max()), 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(iris.data[:, 1].min()), float(iris.data[:, 1].max()), 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", float(iris.data[:, 2].min()), float(iris.data[:, 2].max()), 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", float(iris.data[:, 3].min()), float(iris.data[:, 3].max()), 0.2)

# Prediction Button
if st.sidebar.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = loaded_model.predict(input_data)
    predicted_class = target_names[prediction[0]]
    
    # Display prediction result
    st.success(f"The predicted Iris species is **{predicted_class.capitalize()}**.")
else:
    st.info("Adjust the sliders and click 'Predict' to get the result.")

# App Footer
st.markdown("""
---
Developed using **Streamlit** and **Scikit-learn**.  
Trained on the Iris dataset.
""")
