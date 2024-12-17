import pandas as pd
import pandas_datareader as web
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
from memory_profiler import memory_usage

def run_analysis():
    # Define date range
    start_date = '1990-01-01'
    end_date = '2024-01-08'

    # Fetch data from FRED
    bond_yield = web.DataReader('DGS10', 'fred', start_date, end_date)
    cpi = web.DataReader('CORESTICKM159SFRBATL', 'fred', start_date, end_date)
    vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)
    fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)
    unemp = web.DataReader('UNRATE', 'fred', start_date, end_date)
    interest = web.DataReader('POILBREUSDM', 'fred', start_date, end_date)

    # Merge all data into a single DataFrame
    data = pd.concat([bond_yield, cpi, vix, fed_funds_rate, unemp, interest], axis=1)

    # Rename columns for clarity
    data.columns = ['BondYield', 'CPI', 'VIX', 'FedFundsRate', 'Unemployment', 'Interest']

    # Drop rows with missing values
    data = data.dropna()

    # Define features (X) and target (y)
    X = data[['CPI', 'VIX', 'FedFundsRate', 'Unemployment', 'Interest']]
    y = data['BondYield']

    # Normalize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # Start timing
    start_time = time.time()

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")

    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    mape_rf = mean_absolute_percentage_error(y_test, y_pred)
    print(f'Mean Absolute Percentage Error: {mape_rf}%')

    # Display model coefficients
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    print("\nModel Coefficients:")
    print(coefficients)
    beta_0 = model.intercept_
    print(f"The intercept (β0) is: {beta_0}")

    # Print training time
    print(f"Training Time: {training_time} seconds")

    # Plotting code
    # plt.figure(figsize=(12, 6))
    # plt.plot(data.index[-len(y_test):], y_test.values, 'b', label='Actual Bond Yield')
    # plt.plot(data.index[-len(y_test):], y_pred, 'r--', label='Predicted Bond Yield')
    # plt.xlabel('Time')
    # plt.ylabel('Bond Yield')
    # plt.title('Actual vs Predicted Bond Yield (Testing Period)')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(4,4))
    # plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    # plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=1)
    # plt.xlabel('Actual')
    # plt.ylabel('Predicted')
    # plt.title('Linear Regression: Actual vs Predicted')
    # plt.grid(True)
    # plt.show()

    # plt.figure(figsize=(15, 5))
    train_seq = data.index[:len(y_train)]
    test_seq = data.index[len(y_train):]

    # plt.plot(train_seq, y_train, '-b', label='Train')
    # plt.plot(train_seq, model.predict(X_train), '-r', label='Fit')
    # plt.plot(test_seq, y_test, '--b', label='Test')
    # plt.plot(test_seq, y_pred, '--r', label='Pred')
    # plt.legend(ncols=2)
    # plt.xlabel('Time')
    # plt.ylabel('Normalized Bond Yield')
    # plt.title('Time Series of Bond Yield')
    # plt.savefig("./output_time_series.svg")
    # plt.show()
    # plt.close()

    # plt.figure(figsize=(5, 2))
    # plt.plot(test_seq, y_test, '--b', label='Actual')
    # plt.plot(test_seq, y_pred, '--r', label='Predicted')
    # plt.xlabel('Time')
    # plt.ylabel('Bond Yield')
    # plt.legend()
    # plt.title('Actual vs Predicted')
    # plt.savefig("./output_actual_vs_predicted.svg")
    # plt.show()
    # plt.close()

    residuals = y_test - y_pred

    # Create the figure and gridspec to combine separate plots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3, 1])

# Top plot: Time series (spanning both columns)
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(train_seq, y_train, '-b', label='Train')
    ax0.plot(train_seq, y_train + 0.1, '-r', label='Fit')  # Simulate fit data
    ax0.plot(test_seq, y_test, '--b', label='Test')
    ax0.plot(test_seq, y_pred, '--r', label='Pred')
    ax0.set_title('Complete Time Series of Bond Yield')
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Bond Yield')
    ax0.legend()
    ax0.grid(True)

# Bottom right plot: Residuals (Zoomed-in subplot, 75% width)
    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(test_seq, y_test, '--b', label='Actual')
    ax1.plot(test_seq, y_pred, '--r', label='Predicted')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Zoomed-in Residuals')
    ax1.grid(True)

# Bottom left plot: Actual vs Predicted with square axes and same range (25% width)
    ax2 = plt.subplot(gs[1, :])
    ax2.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
    ax2.plot([0, 10], [0, 10], color='red', lw=1)  # y=x line for reference
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Actual vs Predicted')
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 10])
    ax2.set_xticks(np.arange(0, 11, 2))
    ax2.set_yticks(np.arange(0, 11, 2))
    ax2.set_aspect('equal', adjustable='box')  # Make the plot square
    ax2.grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.savefig("./combined_plots_lr.pdf")
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Measure memory usage and execution time
    mem_usage = memory_usage(run_analysis)
    start_time = time.time()
    run_analysis()
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Memory Usage: {max(mem_usage) - min(mem_usage)} MiB")
    print(f"Execution Time: {execution_time} seconds")
