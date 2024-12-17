import pandas_datareader.data as web
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
import time
from memory_profiler import memory_usage

# Define the date range
start_date = '1990-01-01'
end_date = '2024-01-01'

def run_lstm_analysis():
    # Fetch the data
    bond_yield = web.DataReader('DGS5', 'fred', start_date, end_date)  # 10-Year Treasury Constant Maturity Rate
    cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)      # Consumer Price Index for All Urban Consumers
    vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)        # CBOE Volatility Index
    fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)  # Federal Funds Rate
    brent = web.DataReader('POILBREUSDM', 'fred', start_date, end_date)  # Brent Oil Prices
    unemp = web.DataReader('UNRATE', 'fred', start_date, end_date)      # Unemployment Rate

    # Combine the data into a single DataFrame
    data = pd.concat([bond_yield, cpi, vix, fed_funds_rate, brent, unemp], axis=1)
    data.columns = ['Bond_Yield', 'CPI', 'VIX', 'Fed_Funds_Rate', 'GDP', 'Unemployment']
    data.dropna(inplace=True)

    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(data[['CPI', 'VIX', 'Fed_Funds_Rate', 'GDP', 'Unemployment']])
    y_scaled = data['Bond_Yield'].values  # We keep the target variable as it is

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.25, random_state=42)

    # Reshape data for LSTM
    X_train_lstm = np.expand_dims(X_train, axis=1)
    X_test_lstm = np.expand_dims(X_test, axis=1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(3, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(3, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_lstm, y_train, epochs=3000, batch_size=32, validation_split=0.25, verbose=2)

    # Make predictions
    y_pred = model.predict(X_test_lstm).flatten()
    y_fit = model.predict(X_train_lstm).flatten()

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print('\nLSTM Regression Results:')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')
    print(f'Mean Absolute Percentage Error: {mape}%')

    # Plotting
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    offset = len(X_train)
    train_seq = data.index[:len(X_train)]
    test_seq = data.index[len(X_train):len(X_train) + len(X_test)]

    # Plot Bond Yield Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_seq, y_train, '-b', label='Train')
    ax1.plot(train_seq, y_fit, '-r', label='Fit')
    ax1.plot(test_seq, y_test, '--b', label='Test')
    ax1.plot(test_seq, y_pred, '--r', label='Pred')
    ax1.set_title('Complete Time Series of Bond Yield')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Bond Yield')
    ax1.legend()
    ax1.grid(True)

    # Plot Actual vs Predicted Bond Yield
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(test_seq, y_test, '--b', label='Actual Bond Yield')
    ax2.plot(test_seq, y_pred, '--r', label='Predicted Bond Yield')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Bond Yield')
    ax2.set_title('Actual vs Predicted (Testing Period)')
    ax2.grid(True)

    # Plot Actual vs Predicted Scatter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=1)
    ax3.set_xlabel('Actual')
    ax3.set_ylabel('Predicted')
    ax3.set_title('Actual vs Predicted: Residuals')
    ax3.grid(True)

    # Plot Training and Validation Loss Over Epochs
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(history.epoch, history.history['loss'], '-k', label="Training Loss")
    ax4.plot(history.epoch, history.history['val_loss'], '-b', label="Validation Loss")
    ax4.set_yscale('log')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training and Validation Loss Over Epochs')
    ax4.legend()
    ax4.grid(True)

    # Adjust layout and show plot
    plt.tight_layout()
    plt.savefig("./combined_plots_lstm.pdf")
    # plt.show()
    model.summary()

if __name__ == "__main__":
    # Measure memory usage and execution time
    mem_usage = memory_usage(run_lstm_analysis)
    start_time = time.time()
    run_lstm_analysis()
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Memory Usage: {max(mem_usage) - min(mem_usage)} MiB")
    print(f"Execution Time: {execution_time} seconds")
