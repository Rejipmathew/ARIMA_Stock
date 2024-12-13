import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf

# Streamlit Configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# --- Functions ---

# Data Loading (yfinance) with caching
@st.cache_data  
def data_loader(ticker="AAPL"):
    data = yf.download(ticker)
    data = data.dropna()
    st.write(f"**Downloaded data for {ticker}:**")
    st.write(data.head())
    return data

# Plotting Functions (Streamlit) - Modified to use .values
def plot_predictions(train, predictions, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train.values, label='Actual')  # Use .values
    ax.plot(train.index, predictions, label='Predicted', color='red')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close-Price')
    st.pyplot(fig)

def plot_raw_data(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["Adj Close"].values, label='Close Price')  # Use .values
    ax.set_title('Raw Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

def plot_train_test(train, test):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train.values, label='Train Set')  # Use .values
    ax.plot(test.index, test.values, label='Test Set', color='orange')  # Use .values
    ax.set_title('Train and Test Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    st.pyplot(fig)

def plot_prediction_errors(errors):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(errors, label='Prediction Errors') 
    ax.set_title('Prediction Errors over Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Error')
    ax.legend()
    st.pyplot(fig)

def plot_final_predictions(test, final_predictions):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test.values, label='Actual')  # Use .values
    ax.plot(test.index, final_predictions, label='Corrected Prediction', color='green')
    ax.set_title('Final Predictions with Error Correction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

def plot_accuracy(mse, rmse, mae):
    # ... (no changes needed here) ...

def plot_arima_accuracy(mse, rmse, mae):
    # ... (no changes needed here) ...

# Data Allocation (returns Series for train and test)
def data_allocation(data, days):
    # ... (no changes needed here) ...

# Data Transformation for LSTM (handles different index types)
def apply_transform(data, n: int):
    # ... (no changes needed here) ...

# LSTM Model Training
def LSTM(train, n: int, number_nodes, learning_rate, epochs, batch_size):
    # ... (no changes needed here) ...

# Calculate Accuracy
def calculate_accuracy(true_values, predictions):
    # ... (no changes needed here) ...

# Error Evaluation from LSTM Predictions - Modified to use .iloc
def Error_Evaluation(train_data, predict_train_data, n: int):
    errors = []
    for i in range(len(predict_train_data)):
        err = train_data.iloc[n + i] - predict_train_data[i]  # Use .iloc
        errors.append(err)
    return errors

# ARIMA Parameter Calculation
def Parameter_calculation(data, lag):
    # ... (no changes needed here) ...

# ARIMA Model for Error Prediction
def ARIMA_Model(train, len_test, ord):
    # ... (no changes needed here) ...

# Final Predictions (LSTM + ARIMA)
def Final_Predictions(predictions_errors, predictions, days):
    # ... (no changes needed here) ...

# --- Streamlit App ---
def main():
    # ... (no changes needed here) ...

if __name__ == '__main__':
    main()
