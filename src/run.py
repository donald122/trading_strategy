#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
from strategy.moving_average import MovingAverage
from strategy.z_score import ZScore
from strategy.rsi import RSI
from strategy.roc import ROC
from strategy.percentile import Percentile
from strategy.min_max import MinMax
from strategy.robust import Robust
from joblib import Parallel, delayed
from optimization import Optimization

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Data
btc_price_df = pd.read_csv("../data/BTC-price_usd_close-NATIVE-2020_01_01-2024_07_12-24h.csv")[['timestamp', 'value']]
btc_price_df.columns = ['Date', 'Price']
btc_price_df["Date"] = pd.to_datetime(btc_price_df["Date"])

# Factor
directory = '../data/factors'
price_factor_dict = {}
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        factor_df = pd.read_csv(file_path)[['timestamp', 'value']]

        factor_df.columns = ['Date', 'Target']
        factor_df["Date"] = pd.to_datetime(factor_df["Date"])
        price_factor_df = pd.merge(btc_price_df, factor_df, how='inner', on='Date')

        price_factor_dict[filename] = price_factor_df

window_size_list = np.arange(2, 100, 5)
threshold_params = {
        'ZScore': np.round(np.linspace(0, 3, 42), 3),
        'MovingAverage': np.round(np.linspace(0, 5, 32), 3),
        'RSI': np.round(np.linspace(0.2, 0.8, 32), 4),
        'ROC': np.round(np.linspace(0, 0.8, 32), 4),
        'MinMax': np.round(np.linspace(0.1, 0.9, 42), 4),
        'Robust': np.round(np.linspace(0, 5, 32), 4),
        'Percentile': np.round(np.linspace(0.1, 0.9, 32), 4)
    }
strategy_list = ['ZScore', 'MovingAverage', 'RSI', 'ROC', 'MinMax', 'Robust', 'Percentile']
long_short_params = ['long', 'short', 'both']
condition_params = ['lower', 'higher']

running_list = []
for filename in price_factor_dict:
    for strategy in strategy_list:
        for long_short in long_short_params:
            for condition in condition_params:
                test = {
                        'Metric': filename,
                        'Strategy': strategy,
                        'Strategy Type': long_short,
                        'Condition': condition
                    }
                running_list.append(test)



overall_result = pd.DataFrame()

def running_single_strategy(strategy, metric, price_factor_df, long_short, condition):

    test = Optimization(strategy, price_factor_df, window_size_list, threshold_params[strategy], target="Target", price='Price', long_short=long_short, condition=condition)
    test.run()


    test_output = {
            'Metric': metric,
            'Output': test.get_best()
            }
    return test_output


if __name__ == "__main__":

    results_list = []

    parallel_results =  Parallel(n_jobs=-1)(delayed(running_single_strategy)(run['Strategy'], run['Metric'], price_factor_dict[run['Metric']], run['Strategy Type'], run['Condition'])for run in running_list)

    for result in parallel_results:
        ouput = result['Output']

        results_list.append({ "Metric": result['Metric'] } | ouput.dump_data())


    results_list_df = pd.DataFrame(results_list)
    results_list_df.to_csv("btc_factor_test2.csv")


    print('done')
    #print(result.calmar)
    #print(result.sharpe)
    #result.result_df.to_csv("btc_liq_test.csv")
    #print(result.result_df)
    #result.plot_graph()

