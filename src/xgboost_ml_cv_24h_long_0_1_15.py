#!/usr/bin/env python
import shap
import numpy as np
import os
import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import requests
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from strategy.strategy import Strategy
import lightgbm as lgb
from lightgbm import LGBMClassifier
from dotenv import load_dotenv
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
from tqdm import tqdm
from multiprocessing import set_start_method
set_start_method('fork', force=True)

load_dotenv()

pd.set_option('display.min_rows', 200)
pd.set_option('display.max_rows', 600)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option("display.max_columns", None)
# Parameter
#folder_path = "../data/1h_factors/filtered"
folder_path = "../data/24h_factors/filtered"
filepaths = glob.glob(f"{folder_path}/*.csv")
GLASSNODE_API_KEY = os.getenv('GLASSNODE_API_KEY')
ASSET = 'BTC'
INTERVAL = '24h'
DELAY = 0
# Look aheahd and MA has to be at least 1
LOOK_AHEAD = 1
MA = 1

def create_target(price_change_pct):
    if price_change_pct > 0.0005:
        return 1    # up
    else:
        return 0    # down

# Glassnode fetching BTC price
def fetch_asset_price(asset, interval, api_key):
    url = "https://api.glassnode.com/v1/metrics/market/price_usd_close"
    params = {
        'a': asset,
		's': '1577836800', # since 2020-01-01 UTC
        'u': '1750780800', # until 2025-06-24
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
#btc_price_df =  pd.read_csv("./_v1_metrics_market_price_usd_close.csv")
btc_price_df.columns = ['Date', 'Price']
btc_price_df["Date"] = pd.to_datetime(btc_price_df["Date"])
btc_price_df['Price'] = btc_price_df['Price'].shift(-DELAY)
btc_price_df = btc_price_df.set_index('Date')

btc_price_df = btc_price_df.dropna()

# Combine all CSVs into one DataFrame
factors = [pd.read_csv(filepath) for filepath in filepaths]

for i, df in enumerate(factors):
    factors[i]['Date'] = pd.to_datetime(factors[i]['Date'])
    factors[i] = factors[i].set_index('Date')
    factors[i] = factors[i].apply(pd.to_numeric, errors='coerce')

factors_df = pd.concat(factors, axis=1, join="outer")
#factors_df = factors_df[['/v1/metrics/distribution/balance_mtgox_trustee', '/v1/metrics/distribution/balance_us_government', '/v1/metrics/transactions/transfers_exchanges_to_whales_count']]
price_factors_df = btc_price_df.join(factors_df, how='left')
# defragmentation
price_factors_df = price_factors_df.copy()



##### add lagged BTC features
#num_lags = 3  # Example: Create 3 lagged features

# Add lagged BTC prices
#for lag in range(1, num_lags + 1):
#    price_factors_df[f'btc_price_lag_{lag}'] = price_factors_df['Price'].shift(lag)

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
            delta_features[f'{col}_pct_change'] = df[col].ffill().pct_change(fill_method=None) * 100
            delta_features[f'{col}_pct_change'] = delta_features[f'{col}_pct_change'].clip(lower=-10000, upper=10000)
            delta_features[f'{col}_rolling_std16'] = df[col].rolling(window=16).std()
            delta_features[f'{col}_rolling_avg_change30'] = df[col].rolling(window=30).mean().diff()
            delta_features[f'{col}_rolling_avg_change15'] = df[col].rolling(window=15).mean().diff()
            delta_features[f'{col}_rolling_avg_change7'] = df[col].rolling(window=7).mean().diff()
    new_features = pd.DataFrame(delta_features)
    df = pd.concat([df, new_features], axis=1)
    # pct_change will have inf sometime
    return df.copy()


##### add moving average features
def add_ma_features(features_df):
    df = features_df.copy()
    # Calculate moving averages for all features
    moving_averages_3 = df.rolling(window=3).mean()
    moving_averages_7 = df.rolling(window=7).mean()
    moving_averages_15 = df.rolling(window=15).mean()
    moving_averages_30 = df.rolling(window=30).mean()

    # Rename columns to indicate moving averages
    moving_averages_3.columns = [f'{col}_MA3' for col in df.columns]
    moving_averages_7.columns = [f'{col}_MA7' for col in df.columns]
    moving_averages_15.columns = [f'{col}_MA15' for col in df.columns]
    moving_averages_30.columns = [f'{col}_MA30' for col in df.columns]

    # Concatenate original DataFrame with the moving averages
    df = pd.concat([df, moving_averages_3], axis=1)
    df = pd.concat([df, moving_averages_7], axis=1)
    df = pd.concat([df, moving_averages_15], axis=1)
    df = pd.concat([df, moving_averages_30], axis=1)
    return df

price_factors_df['price_ma'] = price_factors_df['Price'].rolling(window=MA).mean().shift(-MA+1)
price_factors_df['future_price'] = price_factors_df['price_ma'].shift(-LOOK_AHEAD)
price_factors_df['price_change'] = (price_factors_df['future_price'] - price_factors_df['price_ma']) / price_factors_df['price_ma']
price_factors_df['target'] = price_factors_df['price_change'].apply(create_target)




X = price_factors_df.drop(columns=['Price', 'target', 'future_price', 'price_change', 'price_ma'])
X = X.ffill()





y = price_factors_df['target']
#y = price_factors_df['Price']
price_date = price_factors_df[['Price']]
X_train, X_test, y_train, y_test, train_price_date, test_price_date = train_test_split(X, y, price_date, test_size=0.2, shuffle=False)
X_train_99, X_test_99, y_train_99, y_test_99, train_price_date_99, test_price_date_99 = train_test_split(X, y, price_date, test_size=0.05, shuffle=False)

# add features
X_train = add_delta_features(X_train)
X_test = add_delta_features(X_test)
X_train = add_ma_features(X_train)
X_test = add_ma_features(X_test)
X_train_99 = add_delta_features(X_train_99)
X_test_99 = add_delta_features(X_test_99)
X_train_99 = add_ma_features(X_train_99)
X_test_99 = add_ma_features(X_test_99)
X_train_100 = X.copy()
X_train_100 = add_delta_features(X_train_100)
X_train_100 = add_ma_features(X_train_100)

# add features in the list
with open(f"{folder_path}/forward_features_selection_list_24h_long_0_1_1", 'r') as file:
    selected_features = [line.strip() for line in file]
#with open(f"{folder_path}/remove_features_selection_list_1h_2_1_0", 'r') as file:
#    removed_features = [line.strip() for line in file]
removed_features = []
try:
    X_train_list = X_train[selected_features]
    X_test_list = X_test[selected_features]
    X_train_list_99 = X_train_99[selected_features]
    X_test_list_99 = X_test_99[selected_features]
    X_train_list_100 = X_train_100[selected_features]
except:
    X_list = pd.DataFrame()
    X_train_list = pd.DataFrame()
    X_test_list = pd.DataFrame()
    X_train_list_99 = pd.DataFrame()
    X_test_list_99 = pd.DataFrame()
    X_train_list_100 = pd.DataFrame()
X = X.drop(columns=[col for col in removed_features if col in X.columns])
X_train = X_train.drop(columns=[col for col in removed_features if col in X_train.columns])
X_test = X_test.drop(columns=[col for col in removed_features if col in X_test.columns])
X_train_99 = X_train_99.drop(columns=[col for col in removed_features if col in X_train_99.columns])
X_test_99 = X_test_99.drop(columns=[col for col in removed_features if col in X_test_99.columns])
X_train_100 = X_train_100.drop(columns=[col for col in removed_features if col in X_train_100.columns])

neg, pos = y_train.value_counts()
scale_pos_weight = (neg / pos) * 1
print(scale_pos_weight)
xgb_default_params = {
    'objective': 'binary:logistic',  # For binary classification
    'grow_policy': 'lossguide',
    'eval_metric': 'logloss',       # Evaluation metric
    'eta': 0.005,                     # Learning rate
    'max_depth': 3,                 # Max depth of trees
    #'subsample': 0.75,               # Row sampling
    'colsample_bytree': 0.75,         # Feature sampling
    #'colsample_bylevel': 0.75,
    #'n_jobs': -1,
    #'alpha': 1,
    #'lambda': 2,
    'min_child_weight': 0,
    #'scale_pos_weight': scale_pos_weight,
    'seed': 42,
    'gamma': 0
}

# Create expanding window folds
def create_expanding_window_folds(data_length, initial_train_ratio=0.5, n_folds=5):
    """
    Create expanding window folds for time series validation.

    Parameters:
    -----------
    data_length : int
        Total number of samples in the dataset
    initial_train_ratio : float
        Proportion of data to use for initial training
    n_folds : int
        Number of folds to create

    Returns:
    --------
    list of tuples
        Each tuple contains (train_indices, test_indices) for a fold
    """
    # Calculate initial training size
    initial_train_size = int(data_length * initial_train_ratio)

    # Calculate size of each test fold
    remaining = data_length - initial_train_size
    test_fold_size = remaining // n_folds

    folds = []
    for i in range(n_folds):
        # Test set is the next chunk after current training data
        test_start = initial_train_size + i * test_fold_size
        test_end = test_start + test_fold_size if i < n_folds - 1 else data_length

        # Training set is everything before test set
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)

        folds.append((train_idx, test_idx))

    return folds


def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label == 1)
    if ratio > 2.5:
        param["scale_pos_weight"] = ratio
    return(dtrain, dtest, param)

def run_cv(X_train, y_train):
    folds = create_expanding_window_folds(len(X_train), initial_train_ratio=0.6, n_folds=4)

    dtrain = xgb.DMatrix(X_train, label=y_train)

    cv_results = xgb.cv(
            params = xgb_default_params,
            dtrain=dtrain,
            num_boost_round=1000000,
            nfold=1,
            early_stopping_rounds=200,
            metrics=['aucpr', 'auc', 'logloss', 'error'],
            fpreproc=fpreproc,
            folds=folds
    )
    if len(cv_results) < 2:
        raise ValueError("cv_results less than 2")
        return None
    best_round_logloss = cv_results['test-logloss-mean'].idxmin()
    best_logloss = cv_results.loc[best_round_logloss, 'test-logloss-mean']
    best_round_auc = cv_results['test-auc-mean'].idxmax()
    best_auc = cv_results.loc[best_round_auc, 'test-auc-mean']
    best_round_aucpr = cv_results['test-aucpr-mean'].idxmax()
    best_aucpr = cv_results.loc[best_round_aucpr, 'test-aucpr-mean']

    # remove -std
    mean_cols = [col for col in cv_results.columns if not col.endswith('-std')]
    cv_results = cv_results[mean_cols]


    auc_gap = abs(cv_results['train-auc-mean'] - cv_results['test-auc-mean'])
    aucpr_gap = abs(cv_results['train-aucpr-mean'] - cv_results['test-aucpr-mean'])
    logloss_gap = abs(cv_results['train-logloss-mean'] - cv_results['test-logloss-mean'])
    error_gap = abs(cv_results['train-error-mean'] - cv_results['test-error-mean'])
    cv_results['auc_gap'] = auc_gap
    cv_results['aucpr_gap'] = aucpr_gap
    cv_results['logloss_gap'] = logloss_gap
    cv_results['error_gap'] = error_gap
    return cv_results, best_round_aucpr, best_aucpr

def run_model(X_train, X_test, y_train, y_test, train_price_date, test_price_date, num_boost_round=10000):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, 'train'), (dtest, 'eval')]  # For monitoring performance
    _, _, params = fpreproc(dtrain, dtest, xgb_default_params)

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=evals, verbose_eval=False)

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_df = pd.DataFrame(shap_values, columns=X_train.columns)
    shap_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "SHAP_Importance": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="SHAP_Importance", ascending=False)
    #shap_importance.to_csv('shape_value_24h_shift_3.csv', index=False, float_format="%.16f")
    #print(shap_importance)
    #shap.summary_plot(shap_values, X_train, max_display=50, show=False)
    #plt.yticks(fontsize=6)
    #plt.show()
    #interaction_values = explainer.shap_interaction_values(X_train)
    #shap.summary_plot(interaction_values, X_train, max_display=20, show=False)
    #plt.show()


    y_pred = model.predict(dtest)
    y_train_pred = model.predict(dtrain)

    test_precision, test_recall, test_thresholds = precision_recall_curve(y_test, y_pred)
    train_precision, train_recall, train_thresholds = precision_recall_curve(y_train, y_train_pred)
    test_aucpr = auc(test_recall, test_precision)
    train_aucpr = auc(train_recall, train_precision)
    # Convert probabilities to binary predictions (for classification)
    y_train_pred_binary = [1 if y > 0.5 else 0 for y in y_train_pred]
    y_train_pred_signal = [1 if y > 0.5 else 0 for y in y_train_pred]
    y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
    y_pred_signal = [1 if y > 0.5 else 0 for y in y_pred]

    train_accuracy = accuracy_score(y_train, y_train_pred_binary)
    accuracy = accuracy_score(y_test, y_pred_binary)
    #print(classification_report(y_test, y_pred_binary, target_names=["0", "1"]))

    # Retrieve feature importance
    importance = model.get_score(importance_type='weight')  # Default: 'weight'
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Print importance
    #for feature, score in sorted_importance:
    #    print(f"Feature: {feature}, Importance: {score}")

    #xgb.plot_importance(model, max_num_features=30)
    #xgb.plot_tree(model, num_trees=0, max_num_features=10)
    #plt.show()

    test_price_date['Position'] = y_pred_signal
    train_price_date['Position'] = y_train_pred_signal
    report_test = classification_report(y_test, y_pred_binary, target_names=['0', '1'], output_dict=True, zero_division=0)
    report_train = classification_report(y_train, y_train_pred_binary, target_names=['0', '1'], output_dict=True, zero_division=0)
    report = [report_train, report_test]

    output = {}
    xgboost_test = Strategy('All Metrics', test_price_date, 0, 0.99)
    xgboost_train = Strategy('All Metrics', train_price_date, 1, 0.99)
    xgboost_test.result_df = test_price_date.copy()
    xgboost_train.result_df = train_price_date.copy()
    xgboost_test.add_profit(xgboost_test.result_df)
    xgboost_train.add_profit(xgboost_train.result_df)
    output_test = xgboost_test.dump_data()
    output_train = xgboost_train.dump_data()
    output['Train Sharpe'] = output_train['Sharpe']
    output['Test Sharpe'] = output_test['Sharpe']
    output['Train Return'] = output_train['Annual_Return']
    output['Test Return'] = output_test['Annual_Return']
    output['Train MDD'] = output_train['MDD']
    output['Test MDD'] = output_test['MDD']
    output['Train score'] = report_train['1']['f1-score']
    output['Test score'] = report_test['1']['f1-score']
    output['Train aucpr'] = train_aucpr
    output['Test aucpr'] = test_aucpr
    output['Test precision'] = report_test['1']['precision']
    #output['Test score'] = report_test['1']['precision']
    #output['Shap'] = shap_importance
    #xgboost.plot_graph()
    return output, report, model

##### Test result for removing single features
#output_list = []
#for drop_col_name in X_train_list.columns:
#    X_train_100_dropped =X_train_list_100.drop(columns=[drop_col_name])
#    X_train_dropped = X_train_list.drop(columns=[drop_col_name])
#    X_test_dropped = X_test_list.drop(columns=[drop_col_name])
#    cv_results, num_boost_round, score  = run_cv(X_train_100_dropped, y)
#    if num_boost_round == 0:
#        continue
#    output, _, _ = run_model(X_train_dropped, X_test_dropped, y_train, y_test, train_price_date, test_price_date, num_boost_round)
#    output['CV score'] = score
#    output['Dropped feature'] = drop_col_name
#    output['aucpr_gap'] = cv_results.loc[num_boost_round, 'aucpr_gap']
#    output_list.append(output)
#sorted_output_list = pd.DataFrame(sorted(output_list, key=lambda x: x["Test score"], reverse=True))
#print(sorted_output_list)


##### Test runing 1 features at a time
if __name__ == '__main__':
    output_single_feature_list = []
    poor_perform_feature_list = []
    X_train.to_csv('X_train_test', index=False)
    def evaluate_feature(feature_name):
        X_train_combine = pd.concat([X_train_list, X_train[[feature_name]]], axis=1)
        #print(X_train_combine.columns)
        X_test_combine = pd.concat([X_test_list, X_test[[feature_name]]], axis=1)
        X_train_combine_100 = pd.concat([X_train_list_100, X_train_100[[feature_name]]], axis=1)
        X_test_combine_99 = pd.concat([X_test_list_99, X_test_99[[feature_name]]], axis=1)
        X_train_combine_99 = pd.concat([X_train_list_99, X_train_99[[feature_name]]], axis=1)
        try:
            cv_results, num_boost_round, score  = run_cv(X_train_combine_100, y)
        except Exception as e:
            #print(f"{e}: {feature_name}")
            score = 0
            num_boost_round = 0

        if score > 0.69597 and num_boost_round > 0:
            if cv_results.loc[num_boost_round, 'aucpr_gap'] < 0.079:
                output, _, _ = run_model(X_train_combine, X_test_combine, y_train, y_test, train_price_date, test_price_date, num_boost_round)
                output['CV score'] = score
                output['nround'] = num_boost_round
                output['Feature name'] = feature_name
                output['aucpr_gap'] = cv_results.loc[num_boost_round, 'aucpr_gap']
                if output['Train Return'] > 0.1 and output['Test Return'] > 0.1 and output['nround'] > 20 and output['Test precision'] > 0.68 and output['Test score'] > 0.658:
                    print(output)
                    # Save a full model
                    output_combine_99, _, model_combine_99 = run_model(X_train_combine_99, X_test_combine_99, y_train_99, y_test_99, train_price_date_99, test_price_date_99, num_boost_round)
                    feature_name_replace = feature_name.replace('/', '_')
                    model_combine_99.save_model(f"24h_long_models/{feature_name_replace}.ubj")
                    return 'good', output
        #    if output_1['Test score'] > 0.639:
        #        output_2, _, _ = run_model(X_train_combine_2, X_test_combine_2, y_train_2, y_test_2, train_price_date_2, test_price_date_2)
        #        if output_2['Test score'] > 0.627:
        #            return 'good', output
        #if output['Test score'] < 0.511:
        #    output['Feature name'] = feature_name
        #    return 'poor', output

        return None, None
    features = list(X_train.columns)
    #results = Parallel(n_jobs=8)(delayed(evaluate_feature)(col_name) for col_name in tqdm(features, desc="Processing Features", unit="feature"))
    #for category, result in results:
    #    if category == 'good':
    #        output_single_feature_list.append(result)
    #    elif category == 'poor':
    #        poor_perform_feature_list.append(result)
    sorted_output_single_feature_list = pd.DataFrame(sorted(output_single_feature_list, key=lambda x: x["CV score"], reverse=True))
    #print(pd.DataFrame(poor_perform_feature_list))
    print(sorted_output_single_feature_list)
    cv_results, num_boost_round, score  = run_cv(X_train_list_100, y)
    output, report, model = run_model(X_train_list, X_test_list, y_train, y_test, train_price_date, test_price_date, num_boost_round)
    print(pd.DataFrame(report[0]))
    print(pd.DataFrame(report[1]))
    output['CV score'] = score
    output['nround'] = num_boost_round
    output['aucpr_gap'] = cv_results.loc[num_boost_round, 'aucpr_gap']
    print(output)
    print(cv_results.iloc[::10])
    output_99, _, model_99 = run_model(X_train_list_99, X_test_list_99, y_train_99, y_test_99, train_price_date_99, test_price_date_99, num_boost_round)
    model_99.save_model("20250226_xgboost_24h_0_8_0.ubj")
    print(output_99)
    os.system('say "your program has finished"')
