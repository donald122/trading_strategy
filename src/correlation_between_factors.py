#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv


load_dotenv()

# Parameter
folder_path = "../data/1h_factors/filter_high_correlation_metrics/1"
filepaths = glob.glob(f"{folder_path}/*.csv")
GLASSNODE_API_KEY = os.getenv('GLASSNODE_API_KEY')
ASSET = 'BTC'
INTERVAL = '1h'

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

# Data
btc_price_df = fetch_asset_price(ASSET, INTERVAL, GLASSNODE_API_KEY)
btc_price_df.columns = ['Date', 'Price']
btc_price_df["Date"] = pd.to_datetime(btc_price_df["Date"])

# Combine all CSVs into one DataFrame
data_frames = [pd.read_csv(filepath) for filepath in filepaths]

for i, df in enumerate(data_frames):
    data_frames[i]['Date'] = pd.to_datetime(data_frames[i]['Date'])
    data_frames[i] = data_frames[i].set_index('Date')

combined_data = pd.concat(data_frames, axis=1, join="outer")

# Filter numerical columns
numerical_data = combined_data.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
correlation_matrix = numerical_data.corr()
filtered_correlation_matrix = correlation_matrix.where((correlation_matrix > 0.999) | (correlation_matrix < -0.999))
np.fill_diagonal(filtered_correlation_matrix.values, np.nan)
filtered_correlation_matrix = filtered_correlation_matrix.dropna(how="all", axis=0)
filtered_correlation_matrix = filtered_correlation_matrix.dropna(how="all", axis=1)
print(filtered_correlation_matrix)

# Display the correlation matrix
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(filtered_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 8})
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=5)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=5)
plt.title("Correlation Matrix")
plt.show()


