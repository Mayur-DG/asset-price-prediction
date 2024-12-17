import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.gridspec as gridspec
import time
from memory_profiler import memory_usage

# Define the date range
start_date = '1990-01-01'
end_date = '2024-08-01'

def run_analysis():
    # Fetch data
    bond_yield = web.DataReader('DGS10', 'fred', start_date, end_date)  # 10-Year Treasury Constant Maturity Rate
    cpi = web.DataReader('CORESTICKM159SFRBATL', 'fred', start_date, end_date)  # Consumer Price Index for All Urban Consumers
    vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)  # CBOE Volatility Index
    fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)  # Federal Funds Rate
    interest = web.DataReader('POILBREUSDM', 'fred', start_date, end_date)  # Interest Rate
    unemp = web.DataReader('UNRATE', 'fred', start_date, end_date)  # Unemployment Rate

    # Combine data into a single DataFrame
    data = pd.concat([bond_yield, cpi, vix, fed_funds_rate, interest, unemp], axis=1)
    data.columns = ['Bond_Yield', 'CPI', 'VIX', 'Fed_Funds_Rate', 'Interest', 'Unemployment']
    data.dropna(inplace=True)

    # Ensure the index is a DateTimeIndex and sort by date
    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    # Display the first few rows of the dataset
    print(data.head())

    # Split the data into features (X) and target (y)
    X = data[['CPI', 'VIX', 'Fed_Funds_Rate', 'Interest', 'Unemployment']]
    y = data['Bond_Yield']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Reshape data for RNN: (samples, timesteps, features)
    X_train_rnn = np.expand_dims(X_train.values, axis=1)
    X_test_rnn = np.expand_dims(X_test.values, axis=1)

    # Create and fit the Simple RNN model using Keras
    rnn_model = Sequential()
    rnn_model.add(SimpleRNN(units=10, activation='tanh', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])))
    rnn_model.add(Dropout(0.2))
    rnn_model.add(Dense(1))
    rnn_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the RNN model with validation split
    history = rnn_model.fit(X_train_rnn, y_train, epochs=5000, batch_size=32, verbose=2, validation_split=0.25)

    # Predict on the test data with RNN
    y_pred_rnn = rnn_model.predict(X_test_rnn).flatten()
    y_fit_rnn = rnn_model.predict(X_train_rnn).flatten()

    # Define the mean_absolute_percentage_error function
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Calculate metrics for RNN
    mse_rnn = mean_squared_error(y_test, y_pred_rnn)
    rmse_rnn = np.sqrt(mse_rnn)
    r2_rnn = r2_score(y_test, y_pred_rnn)
    mape_rnn = mean_absolute_percentage_error(y_test, y_pred_rnn)

    print('\nRNN Regression:')
    print(f'Mean Squared Error: {mse_rnn}')
    print(f'Root Mean Squared Error: {rmse_rnn}')
    print(f'R-squared: {r2_rnn}')
    print(f'Mean Absolute Percentage Error: {mape_rnn}%')

    rnn_model.summary()

    # Plotting
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
    offset = len(X_train)
    train_seq = data.index[:len(X_train)]
    test_seq = data.index[len(X_train):len(X_train) + len(X_test)]

    # Plot Bond Yield Over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_seq, y_train, '-b', label='Train')
    ax1.plot(train_seq, y_fit_rnn, '-r', label='Fit')
    ax1.plot(test_seq, y_test, '--b', label='Test')
    ax1.plot(test_seq, y_pred_rnn, '--r', label='Pred')
    ax1.set_title('Complete Time Series of Bond Yield')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Bond Yield')
    ax1.legend()
    ax1.grid(True)

    # Plot Actual vs Predicted Bond Yield
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(test_seq, y_test.values, '--b', label='Actual Bond Yield')
    ax2.plot(test_seq, y_pred_rnn, '--r', label='Predicted Bond Yield')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Bond Yield')
    ax2.set_title('Actual vs Predicted (Testing Period)')
    ax2.grid(True)

    # Plot Actual vs Predicted Scatter
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_test, y_pred_rnn, color='blue', edgecolor='k', alpha=0.7)
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

    # Adjust layout and save plot
    plt.tight_layout()
    plt.savefig("./combined_plots_rnn.pdf")


if __name__ == "__main__":
    # Measure memory usage and execution time
    mem_usage = memory_usage(run_analysis)
    start_time = time.time()
    run_analysis()
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Memory Usage: {max(mem_usage) - min(mem_usage)} MiB")
    print(f"Execution Time: {execution_time} seconds")
