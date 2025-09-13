#!/usr/bin/env python
import numpy as np
import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from strategy.strategy import Strategy
import lightgbm as lgb
from lightgbm import LGBMClassifier
from dotenv import load_dotenv


load_dotenv()

pd.set_option('display.min_rows', 200)
pd.set_option('display.max_rows', 600)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
# Parameter
folder_path = "../data/1h_factors/filtered"
#folder_path = "../data/24h_factors/random_forest"
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
btc_price_df['Price'] = btc_price_df['Price'].shift(-1)

btc_price_df = btc_price_df.dropna()

# Combine all CSVs into one DataFrame
factors = [pd.read_csv(filepath) for filepath in filepaths]

for i, df in enumerate(factors):
    factors[i]['Date'] = pd.to_datetime(factors[i]['Date'])
    factors[i] = factors[i].set_index('Date')
    factors[i] = factors[i].apply(pd.to_numeric, errors='coerce')

factors_df = pd.concat(factors, axis=1, join="outer")
price_factors_df = pd.merge(btc_price_df, factors_df, how='inner', on='Date')

##### add lagged BTC features
num_lags = 3  # Example: Create 3 lagged features

# Add lagged BTC prices
for lag in range(1, num_lags + 1):
    price_factors_df[f'btc_price_lag_{lag}'] = price_factors_df['Price'].shift(lag)

##### add polynomial features
#imputer = SimpleImputer(strategy='mean')
#price_factors_df_imputed = imputer.fit_transform(price_factors_df.drop(columns=['Price', 'Date']))
#poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#poly_features = poly.fit_transform(price_factors_df_imputed)
#poly_features_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(price_factors_df.columns))
#print(poly_features_df)


##### add delta features
def add_delta_features(features_df):
    df = features_df.copy()
    delta_features = {}
    for col in df.columns:
            delta_features[f'{col}_abs_change'] = df[col].diff()
            delta_features[f'{col}_pct_change'] = df[col].ffill().pct_change() * 100
            delta_features[f'{col}_rolling_avg_change'] = df[col].rolling(window=7).mean().diff()
    new_features = pd.DataFrame(delta_features)
    df = pd.concat([df, new_features], axis=1)
    # pct_change will have inf sometime
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


##### add moving average features
def add_ma_features(features_df, windows=5):
    df = features_df.copy()
    # Calculate moving averages for all features
    moving_averages = df.rolling(window=windows).mean()

    # Rename columns to indicate moving averages
    moving_averages.columns = [f'{col}_MA' for col in df.columns]

    # Concatenate original DataFrame with the moving averages
    df = pd.concat([df, moving_averages], axis=1)
    return df


price_factors_df['future_price'] = price_factors_df['Price'].shift(-1)
price_factors_df['price_change'] = (price_factors_df['future_price'] - price_factors_df['Price']) / price_factors_df['Price']
price_factors_df['smoothed_price_change'] = price_factors_df['price_change'].rolling(window=4).mean().shift(-3)
price_factors_df = price_factors_df.dropna(subset=['smoothed_price_change'])


buy_threshold = 0.001
sell_threshold = -0.001

price_factors_df['signal'] = price_factors_df['smoothed_price_change'].apply(
    #lambda x: 1 if x > buy_threshold else (-1 if x < sell_threshold else 0)
    lambda x: 1 if x > 0 else -1
)

def rolling_mode(series):
    return series.mode().iloc[0] if not series.mode().empty else None

price_factors_df['smoothed_signal'] = price_factors_df['signal'].rolling(window=5).apply(rolling_mode, raw=False).shift(-4)
price_factors_df = price_factors_df.dropna(subset=['smoothed_signal'])
print(price_factors_df['smoothed_signal'].value_counts())


X = price_factors_df.drop(columns=['Price', 'Date', 'signal', 'future_price', 'price_change', 'smoothed_price_change', 'smoothed_signal'])

y = price_factors_df['smoothed_signal']
#y = price_factors_df['Price']
price_date = price_factors_df[['Date', 'Price']]
X_train, X_test, y_train, y_test, train_price_date, test_price_date = train_test_split(X, y, price_date, test_size=0.2, random_state=42, shuffle=False)

# add features
X_train = add_delta_features(X_train)
X_test = add_delta_features(X_test)
X_train = add_ma_features(X_train)
X_test = add_ma_features(X_test)

# drop features in the list
with open(f"{folder_path}/backward_features_select_lightgbm_list", 'r') as file:
    selected_features = [line.strip() for line in file]
X_train = X_train.drop(columns=[col for col in selected_features if col in X.columns])
X_test = X_test.drop(columns=[col for col in selected_features if col in X.columns])

X_train = X_train.clip(lower=-1e9, upper=1e9)
X_test = X_test.clip(lower=-1e9, upper=1e9)

lgb_model = LGBMClassifier(max_depth=10, n_jobs=-1)
lgb_model.fit(X_train, y_train)

importances = lgb_model.feature_importances_
feature_names = X_train.columns

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

non_zero_importances = feature_importance_df[feature_importance_df['Importance'] > -1]
print("\nFeature Importances (non-zero):\n", non_zero_importances)

accuracy = lgb_model.score(X_test, y_test)
train_accuracy = lgb_model.score(X_train, y_train)
print("Accuracy train:", train_accuracy)
print("Accuracy test:", accuracy)

selected_features = non_zero_importances['Feature'].tolist()
with open(f"{folder_path}/features_select_lightgbm_list", 'w') as f:
    for feature in selected_features:
        f.write(f"{feature}\n")
