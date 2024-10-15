import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('bhutan_tourism_data.csv', parse_dates=['date'], index_col='date')
model = ARIMA(data['tourists'], order=(5, 1, 0))
model_fit = model.fit()

# User input for prediction
st.title('Tourist Arrival Prediction in Bhutan')
months = st.number_input('Months to Predict:', min_value=1, max_value=12)

# Forecasting
forecast = model_fit.forecast(steps=months)
st.line_chart(forecast)

# Display results
st.write('Predicted Tourist Arrivals:', forecast)