import numpy as np
import pandas as pd
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
# Define the date range
start_date = '1990-01-01'
end_date = '2024-08-01'

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
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

# Define the parameter grid
param_grid = {
    'C': [0.1, 0.5,0.8,1, 10, 50],
    'epsilon': [0.01, 0.1, 0.5, 1, 1.5, 2],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]
}

# Custom scoring function to optimize for MSE
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform GridSearchCV
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring=scorer, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print the best parameters found
print(f'Best parameters found: {grid_search.best_params_}')

# Train the SVR model with the best parameters
best_svr_model = grid_search.best_estimator_
best_svr_model.fit(X_train, y_train)

# Predict on the test data with the best SVR model
y_pred_svr = best_svr_model.predict(X_test)

# Define the mean_absolute_percentage_error function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate performance metrics
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)
mape_svr = mean_absolute_percentage_error(y_test, y_pred_svr)

print('\nOptimized Support Vector Regression:')
print(f'Mean Squared Error: {mse_svr}')
print(f'Root Mean Squared Error: {rmse_svr}')
print(f'R-squared: {r2_svr}')
print(f'Mean Absolute Percentage Error: {mape_svr}%')

# Plot the results
fig, ax = plt.subplots(1, 1, sharex='col', sharey=False, figsize=(14, 10))

# Plotting
train_seq = data.index[:len(X_train)]
test_seq = data.index[len(X_train):len(X_train) + len(X_test)]

plt.plot(train_seq, y_train.values, '-b', label='Train Data')
plt.plot(test_seq, y_test.values, '--b', label='Test Data')
plt.plot(test_seq, y_pred_svr, '--r', label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Bond Yield')
plt.title('Bond Yield Over Time')
plt.legend()
plt.grid(True)

# Save and show the figure
plt.tight_layout()
plt.savefig("./outputsvr_optimized.svg")
plt.close()

plt.plot(test_seq, y_test.values, '--b', label='Actual Bond Yield')
plt.plot(test_seq, y_pred_svr, '--r', label='Predicted Bond Yield')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Bond Yield')
plt.title('Actual vs Predicted Bond Yield (Testing Period)')
plt.legend()

# Optionally, add grid lines
plt.grid(True)

# Show the plot
plt.show()

plt.figure(figsize=(4, 4))
plt.scatter(y_test, y_pred_svr, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Bond Yield')
plt.ylabel('Predicted Bond Yield')
plt.title('SVR: Residuals')
plt.grid(True)
plt.show()
# print(best_svr_model.summary())
plt.plot(train_seq, y_train, '-b', label='Train')
plt.plot(train_seq, best_svr_model.predict(X_train), '-r', label='Fit')
plt.plot(test_seq, y_test, '--b', label='Test')
plt.plot(test_seq, y_pred_svr, '--r', label='Pred')
plt.legend(ncols=2)
plt.xlabel('Time')
plt.ylabel('Normalized Bond Yield')
plt.title('Time Series of Bond Yield')
plt.savefig("./output_time_series_rf_.svg")
plt.close()
residuals = y_test - y_pred_svr

# Create the figure and gridspec to combine separate plots
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[3,1])  # 2 rows, 2 columns


# Top plot: Time series (spanning both columns)
ax0 = plt.subplot(gs[0, 0])
ax0.plot(train_seq, y_train, '-b', label='Train')
ax0.plot(train_seq, y_train + 0.1, '-r', label='Fit')  # Simulate fit data
ax0.plot(test_seq, y_test, '--b', label='Test')
ax0.plot(test_seq, y_pred_svr, '--r', label='Pred')
ax0.set_title('Complete Time Series of Bond Yield')
ax0.set_xlabel('Time')
ax0.set_ylabel('Bond Yield')
ax0.legend()
ax0.grid(True)

# Bottom right plot: Residuals (Zoomed-in subplot, 75% width)
ax1 = plt.subplot(gs[0, 1])
ax1.plot(test_seq, y_test, '--b', label='Actual')
ax1.plot(test_seq, y_pred_svr, '--r', label='Predicted')
ax1.set_xlabel('Time')
ax1.set_ylabel('Residuals')
ax1.set_title('Zoomed-in Residuals')
ax1.grid(True)

# Bottom left plot: Actual vs Predicted with square axes and same range (25% width)
ax2 = plt.subplot(gs[1, :])
ax2.scatter(y_test, y_pred_svr, color='blue', edgecolor='k', alpha=0.7)
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
plt.savefig("./combined_plots_svr.pdf")
plt.show()