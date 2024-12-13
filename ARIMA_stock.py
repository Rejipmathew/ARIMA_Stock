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

# Plotting Functions (Streamlit)
def plot_predictions(train, predictions, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train, label='Actual')
    ax.plot(train.index, predictions, label='Predicted', color='red')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close-Price')
    st.pyplot(fig)

def plot_raw_data(data):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data.index, data["Adj Close"], label='Close Price')
    ax.set_title('Raw Time Series Data')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

def plot_train_test(train, test):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index, train, label='Train Set')
    ax.plot(test.index, test, label='Test Set', color='orange')
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
    ax.plot(test.index, test, label='Actual')
    ax.plot(test.index, final_predictions, label='Corrected Prediction', color='green')
    ax.set_title('Final Predictions with Error Correction')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig)

def plot_accuracy(mse, rmse, mae):
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics, values, color=['blue', 'orange', 'green'])
    ax.set_title('Model Accuracy Metrics')
    st.pyplot(fig)

def plot_arima_accuracy(mse, rmse, mae):
    metrics = ['MSE', 'RMSE', 'MAE']
    values = [mse, rmse, mae]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(metrics, values, color=['blue', 'orange', 'green'])
    ax.set_title('ARIMA Model Accuracy Metrics')
    st.pyplot(fig)

# Data Allocation
def data_allocation(data, days):
    train_len_val = len(data) - days
    train, test = data["Adj Close"].iloc[0:train_len_val], data["Adj Close"].iloc[train_len_val:]
    st.write("**Training Set:**")
    st.write(train)
    st.write(f"Number of Entries: {len(train)}")
    st.write("**Testing Set:**")
    st.write(test)
    st.write(f"Number of Entries: {len(test)}")
    return train, test

# Data Transformation for LSTM
def apply_transform(data, n: int):
    middle_data = []
    target_data = []
    for i in range(n, len(data)):
        input_sequence = data[i-n:i]
        middle_data.append(input_sequence)
        target_data.append(data[i])
    middle_data = np.array(middle_data).reshape((len(middle_data), n, 1))
    target_data = np.array(target_data)
    return middle_data, target_data

# LSTM Model Training
def LSTM(train, n: int, number_nodes, learning_rate, epochs, batch_size):
    middle_data, target_data = apply_transform(train, n)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((n,1)),
        tf.keras.layers.LSTM(number_nodes,input_shape=(n, 1)),
        tf.keras.layers.Dense(units=number_nodes, activation="relu"),
        tf.keras.layers.Dense(units=number_nodes, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["mean_absolute_error"])
    history = model.fit(middle_data, target_data, epochs=epochs, batch_size=batch_size, verbose=0)
    full_predictions = model.predict(middle_data).flatten()
    return model, history, full_predictions

# Calculate Accuracy
def calculate_accuracy(true_values, predictions):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    return mse, rmse, mae

# Error Evaluation from LSTM Predictions
def Error_Evaluation(train_data, predict_train_data, n: int):
    errors = []
    for i in range(len(predict_train_data)):
        err = train_data[n + i] - predict_train_data[i]
        errors.append(err)
    return errors

# ARIMA Parameter Calculation
def Parameter_calculation(data, lag):
    finding = auto_arima(data, trace=True, suppress_warnings=True)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(data, lags=lag, ax=ax[0])
    plot_pacf(data, lags=lag, ax=ax[1])
    st.pyplot(fig)
    ord = finding.order
    return ord

# ARIMA Model for Error Prediction
def ARIMA_Model(train, len_test, ord):
    model = ARIMA(train, order=ord)
    model = model.fit()
    # Predict an extra day for the forecast
    predictions = model.predict(start=len(train), end=len(train) + len_test, type='levels')  
    full_predictions = model.predict(start=0, end=len(train)-1, type='levels')
    return model, predictions, full_predictions

# Final Predictions (LSTM + ARIMA)
def Final_Predictions(predictions_errors, predictions, days):
    final_values = []
    for i in range(days):
        final_values.append(predictions_errors[i] + predictions[i])
    return final_values

# --- Streamlit App ---
def main():
    st.title("Stock Price Prediction with LSTM and ARIMA")

    # Sidebar Inputs
    ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
    n = st.sidebar.slider("Lag Value for Neural Network", min_value=1, max_value=100, value=20)
    days = st.sidebar.slider("Prediction Days", min_value=1, max_value=30, value=10)
    lag = st.sidebar.slider("Lag Value for ACF and PACF Plots", min_value=1, max_value=100, value=20)

    # Model Parameters
    epochs = st.sidebar.slider("Epochs", min_value=10, max_value=500, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
    batch_size = st.sidebar.slider("Batch Size", min_value=8, max_value=128, value=32)
    number_nodes = st.sidebar.slider("Number of Nodes (LSTM)", min_value=10, max_value=100, value=50)

    # Data Loading
    data = data_loader(ticker)
    plot_raw_data(data)

    # Data Allocation
    train, test = data_allocation(data, days)
    plot_train_test(train, test)

    # LSTM Model 
    st1 = time.time()
    # Pass the "Adj Close" column from the training set to the LSTM function
    model, history, full_predictions = LSTM(train, n, number_nodes, learning_rate, epochs, batch_size) 
    plot_predictions(train[n:], full_predictions, "LSTM PREDICTIONS VS ACTUAL Values For TRAIN Data Set")

    # Prediction and Evaluation
    last_sequence = train[-n:].values.reshape((1, n, 1))
    predictions = []
    for i in range(days + 1):  # Predict one extra day for the forecast
        next_prediction = model.predict(last_sequence).flatten()[0]
        predictions.append(next_prediction)
        if i < len(test):
            actual_value = test.iloc[i]
            new_row = np.append(last_sequence[:, 1:, :], np.array([[[actual_value]]]), axis=1)
        else:
            new_row = np.append(last_sequence[:, 1:, :], np.array([[[next_prediction]]]), axis=1)
        last_sequence = new_row.reshape((1, n, 1))

    plot_predictions(test, predictions[:-1], "LSTM Predictions VS Actual Values")
    errors_data = Error_Evaluation(train, full_predictions, n)
    plot_prediction_errors(errors_data)
    st.write(f"\n\n**The {days} prediction values from LSTM:**\n")
    for i in range(days):
        actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
        st.write(f"Day {i+1} => ACTUAL VALUE: {actual_value:.2f} | PREDICTED VALUE: {predictions[i]:.2f}")

    # Calculate and display LSTM accuracy
    mse, rmse, mae = calculate_accuracy(test[:days], predictions[:days])
    plot_accuracy(mse, rmse, mae)
    st.write("**LSTM Model Accuracy:**")
    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

    # ARIMA Model
    ord = Parameter_calculation(errors_data, lag)
    Arima_Model, predictions_errors, full_predictions_errors = ARIMA_Model(errors_data, len(test), ord)
    st.write(f"\n\n**ARIMA Model {days} Predictions:**\n")
    for i, pred in enumerate(predictions_errors):
        st.write(f"{i+1}: {pred:.2f}")

    # Calculate and display ARIMA accuracy
    arima_mse, arima_rmse, arima_mae = calculate_accuracy(errors_data, full_predictions_errors)
    plot_arima_accuracy(arima_mse, arima_rmse, arima_mae)
    st.write("**ARIMA Model Accuracy (on LSTM errors):**")
    st.write(f"Mean Squared Error (MSE): {arima_mse:.4f}")
    st.write(f"Root Mean Squared Error (RMSE): {arima_rmse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {arima_mae:.4f}")

    # Final Predictions
    final_predictions = Final_Predictions(predictions_errors, predictions, days)
    plot_final_predictions(test[:days], final_predictions)
    st.write("\n**Final Predictions (LSTM + ARIMA):**\n")
    for i in range(days):
        actual_value = test.iloc[i] if i < len(test) else "No actual value (out of range)"
        st.write(f"Day {i+1} => ACTUAL VALUE: {actual_value:.2f} | PREDICTED VALUE: {final_predictions[i]:.2f}")

    # Forecast for the next data point (using the last predicted error)
    forecast_value = predictions[days] + predictions_errors[-1]  
    st.write(f"\n**The forecast value for the next data point is:** {forecast_value:.2f}")

    end1 = time.time()
    st.write(f"\n**Time taken for model training and predictions:** {end1 - st1:.2f} seconds")

if __name__ == '__main__':
    main()
