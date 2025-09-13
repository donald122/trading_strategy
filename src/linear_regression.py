#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt

load_dotenv()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# Paramenter
GLASSNODE_API_KEY = os.getenv('GLASSNODE_API_KEY')
ASSET = 'BTC'
INTERVAL = '10m'

# Glassnode fetching BTC price
def fetch_asset_price(asset, interval, api_key):
    url = "https://api.glassnode.com/v1/metrics/market/price_usd_close"
    params = {
        'a': asset,
                's': '1577836800', # since 2020-01-01 UTC
        'i': interval,
        'api_key': api_key,
                'timestamp_format': 'unix'
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        #df = df.convert_dtypes(dtype_backend='pyarrow') # might increase performance
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], unit='s')
        return df
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

def plot_regression_results(y_true, y_pred, price_factors_df):
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Actual vs Predicted Plot
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Actual vs Predicted Prices')

    # 2. Residuals Plot
    residuals = y_true - y_pred
    plt.subplot(2, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')

    # 3. Time Series Plot
    plt.subplot(2, 1, 2)
    dates = price_factors_df['Date'].loc[y_true.index]
    plt.plot(dates, y_true, label='Actual Price', alpha=0.7)
    plt.plot(dates, y_pred, label='Predicted Price', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices Over Time')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('regression_analysis.png')
    plt.close()


# Data
btc_price_df = fetch_asset_price(ASSET, INTERVAL, GLASSNODE_API_KEY)
btc_price_df.columns = ['Date', 'Price']
btc_price_df["Date"] = pd.to_datetime(btc_price_df["Date"])
btc_price_df['Changes'] = btc_price_df['Price'].pct_change(fill_method=None)

# Factors
directory = '../data/linear_regression_factors'

price_factors_df = btc_price_df

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        factor_df = pd.read_csv(file_path)
        factor_df.columns = ['Date', filename]
        factor_df["Date"] = pd.to_datetime(factor_df["Date"])
        factor_df[filename] = factor_df[filename].shift(1) # shift data to avoid bias (24H: Shift = 1; 1h: Shift = 2; 10m: Shift = 5)
        price_factors_df = pd.merge(price_factors_df, factor_df, how='inner', on='Date')

X = price_factors_df.drop(columns=['Price', 'Changes', 'Date'])
y = price_factors_df['Price']

X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = y.loc[X.index]

# Calculate correlation matrix
correlation_matrix = X.corr()

# Find highly correlated pairs
CORRELATION_THRESHOLD = 0.8
high_correlation_pairs = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) >= CORRELATION_THRESHOLD:
            high_correlation_pairs.append({
                'Variable1': correlation_matrix.columns[i],
                'Variable2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

# Sort pairs by absolute correlation
high_correlation_pairs = sorted(high_correlation_pairs,
                              key=lambda x: abs(x['Correlation']),
                              reverse=True)
for pair in high_correlation_pairs:
    print(f"{pair['Variable1']} - {pair['Variable2']}: {pair['Correlation']:.3f}")

# Function to calculate VIF for a single feature
def calculate_vif(index):
    return variance_inflation_factor(X.values, index)

with tqdm_joblib(tqdm(desc="Calculating VIF", total=X.shape[1])) as progress_bar:
        vif_values = Parallel(n_jobs=-1)(
            delayed(calculate_vif)(i) for i in range(X.shape[1])
            )

vif = pd.DataFrame({
    'Variable': X.columns,
    'VIF': vif_values
}).sort_values(by='VIF', ascending=False)



print(vif)

model = LinearRegression()
model.fit(X, y)

# Get the coefficients and intercept
coefficients = pd.Series(model.coef_, index=X.columns)
coefficients = coefficients.sort_values(ascending=False)
print("Coefficients for each factor:")
print(coefficients)
print(f"Intercept: {model.intercept_}")

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"R² Score: {r2}")

predictions_df = pd.DataFrame({
        'Date': price_factors_df['Date'].loc[X.index],
        'Predicted_Price': y_pred
        })

predictions_df.to_csv('predictions.csv', index=False)

plot_regression_results(y, y_pred, price_factors_df)

