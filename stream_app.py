import streamlit as st 
import pandas as pd
import numpy as np
import joblib
st.write("streamlit=",st.__version__)
st.write("pandas=",st.__version__)
st.write("joblib=",st.__version__)
st.write("numpy=",st.__version__)


st.title("California House Price Predictor")

# Load Model
model = joblib.load("model_Ml.pkl")

st.write("Enter House Deatails Below")

# Input Field
longitude = st.number_input("Longitube", value=122.33)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.number_input("Housing Median Age", value=41)

total_rooms = st.number_input("Total Rooms", value=880)
total_bedrooms = st.number_input("Total Bedrooms", value=129)
population = st.number_input("Population", value=322)
households = st.number_input("Households", value=126)

median_income = st.number_input("Median Income", value=8.32)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "INLAND", "NEAR OCEAN", "ISLAND", "<1H OCEAN"]
)

# Feature Engineering (IMPORTANT)
rooms_per_household = total_rooms / households
bedrooms_per_room = total_bedrooms / total_rooms
population_per_household = population / households


# Prediction
if st.button("Predict Price"):
    data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
        "ocean_proximity": ocean_proximity
    }])

    pred_log = model.predict(data)
    pred = np.expm1(pred_log)

    st.success(f"Predicted House Price: ${pred[0]:,.2f}")
