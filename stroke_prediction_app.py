import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import requests
from io import BytesIO

# Display the image at the start of the page
image_url = "https://raw.githubusercontent.com/lalit0801/Datasets/main/latest707.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
st.image(image, caption="Latest Image")

# Load the dataset
url = "https://raw.githubusercontent.com/lalit0801/Datasets/main/healthcare-dataset-stroke-data.csv"
data = pd.read_csv(url)

# Prepare the data
data = data.dropna()  # Remove rows with missing values
features = ['age', 'hypertension', 'heart_disease', 'bmi', 'avg_glucose_level']
target = 'stroke'
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Create the Streamlit app
st.title("Stroke Prediction")
st.write("Enter the following information to predict stroke probability:")

# Get user input
age = st.number_input("Age", min_value=0, max_value=120, step=1)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1)
glucose_level = st.number_input("Glucose Level", min_value=0.0, step=0.1)

# Make prediction
input_data = [[age, hypertension, heart_disease, bmi, glucose_level]]
prediction = model.predict(input_data)[0]

# Display prediction percentage
st.write("Prediction Percentage:", round(prediction * 100, 2), "%")

