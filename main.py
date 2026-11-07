import pandas as pd
import matplotlib
matplotlib.use('Agg')  # saves charts to files instead of popup windows
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load data
df = pd.read_csv('azerbaijan_crude_oil_extraction_1871_2022.csv')
print(df.head())

# Step 2: Plot full history
plt.figure(figsize=(10,5))
plt.plot(df['year'], df['crude_oil_extraction_thousand_tonnes'])
plt.title('Azerbaijan Crude Oil Extraction 1871-2025')
plt.xlabel('Year')
plt.ylabel('Thousand tonnes')
plt.savefig('full_history.png')
print("Saved full_history.png")

# Step 3: Filter to modern era
modern = df[df['year'] >= 1994].reset_index(drop=True)

# Step 4: Train/test split
train = modern[modern['year'] <= 2017]
test = modern[modern['year'] > 2017]
print("Train shape:", train.shape, "Test shape:", test.shape)

# Step 5: Naive baseline
last_value = train['crude_oil_extraction_thousand_tonnes'].iloc[-1]
naive_forecast = [last_value] * len(test)

# Step 6: Linear trend model
X_train = train['year'].values.reshape(-1, 1)
y_train = train['crude_oil_extraction_thousand_tonnes'].values
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
X_test = test['year'].values.reshape(-1, 1)
linear_forecast = lin_model.predict(X_test)

# Step 7: ARIMA model
arima_model = ARIMA(train['crude_oil_extraction_thousand_tonnes'], order=(1,1,0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(test))

# Step 8: Compare errors
actual = test['crude_oil_extraction_thousand_tonnes'].values
mae_naive = mean_absolute_error(actual, naive_forecast)
mae_linear = mean_absolute_error(actual, linear_forecast)
mae_arima = mean_absolute_error(actual, arima_forecast)

print(f"\nNaive MAE:  {mae_naive:.1f}")
print(f"Linear MAE: {mae_linear:.1f}")
print(f"ARIMA MAE:  {mae_arima:.1f}")

# Step 9: Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(test['year'], actual, label='Actual', marker='o')
plt.plot(test['year'], naive_forecast, label='Naive', linestyle='--')
plt.plot(test['year'], linear_forecast, label='Linear Trend', linestyle='--')
plt.plot(test['year'], arima_forecast, label='ARIMA', linestyle='--')
plt.legend()
plt.title('Actual vs Predicted Oil Extraction (Test Period)')
plt.savefig('forecast_comparison.png')
print("Saved forecast_comparison.png")
