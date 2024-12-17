import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas_datareader.data as web
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Fetch and prepare data
start_date = '1990-01-01'
end_date = '2024-08-01'

# Fetch data from FRED
bond_yield = web.DataReader('DGS10', 'fred', start_date, end_date)  # 10-Year Treasury Constant Maturity Rate
cpi = web.DataReader('CORESTICKM159SFRBATL', 'fred', start_date, end_date)  # Consumer Price Index for All Urban Consumers
vix = web.DataReader('VIXCLS', 'fred', start_date, end_date)  # CBOE Volatility Index
fed_funds_rate = web.DataReader('FEDFUNDS', 'fred', start_date, end_date)  # Federal Funds Rate
interest = web.DataReader('POILBREUSDM', 'fred', start_date, end_date)  # Oil Prices
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

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Define and compile the MLP model
model = Sequential()
model.add(Dense(17, activation='tanh', input_shape=(X_train.shape[1],)))  # Hidden layer with 17 nodes
model.add(Dense(1))  # Output layer for regression

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.25, verbose=1)

# Predict on the test set
y_pred = model.predict(X_test).flatten()
y_fit = model.predict(X_train).flatten()
# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
model.summary()
# Custom MAPE function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_mlp = mean_absolute_percentage_error(y_test, y_pred)

# Print model evaluation metrics
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
print("Mean Absolute Percentage Error:", mape_mlp)

# Plot training and validation loss
plt.figure(figsize=(10,5))
plt.plot(history.epoch, history.history['loss'], '-k', label = "Training Loss")
plt.plot(history.epoch, history.history['val_loss'], '-b', label="Validation Loss")
plt.yscale('log')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("./lossmlp.pdf")
plt.show()
# plt.close()
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1])
offset = len(X_train)
train_seq = range(0, offset)
test_seq = range(offset, offset + len(X_test))
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

# Plot Actual vs Predicted Bond Yield (Top-right, spans one grid space)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(test_seq, y_test.values, '--b', label='Actual Bond Yield')
ax2.plot(test_seq, y_pred, '--r', label='Predicted Bond Yield')
ax2.set_xlabel('Time')
ax2.set_ylabel('Bond Yield')
ax2.set_title('Actual vs Predicted (Testing Period)')
ax2.grid(True)

# Plot Actual vs Predicted Scatter (Bottom-left, spans one grid space)
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=1)
ax3.set_xlabel('Actual')
ax3.set_ylabel('Predicted')
ax3.set_title('Actual vs Predicted: Residuals')
# ax2.set_aspect('equal')
ax3.grid(True)

# Plot Training and Validation Loss Over Epochs (Bottom-right, spans one grid space)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(history.epoch, history.history['loss'], '-k', label="Training Loss")
ax4.plot(history.epoch, history.history['val_loss'], '-b', label="Validation Loss")
ax4.set_yscale('log')  # Set y-axis to logarithmic scale
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.set_title('Training and Validation Loss Over Epochs')
ax4.legend()
ax4.grid(True)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig("./combined_plots_mlp.pdf")
plt.show()

plt.close()
model.summary()