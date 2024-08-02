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
btc_price_df = pd.read_csv("../data/btc-hourly.csv")[['Date', 'Price']]
btc_price_df.columns = ['Date', 'Price']
btc_price_df["Date"] = pd.to_datetime(btc_price_df["Date"])

# Factor
filename = '_v1_metrics_supply_loss_sum.csv'
factor_df = pd.read_csv("../data/" + filename)[['Date', 'Value']]
factor_df.columns = ['Date', 'Target']
factor_df["Date"] = pd.to_datetime(factor_df["Date"])

price_factor_df = pd.merge(btc_price_df, factor_df, how='inner', on='Date')

window_size_list = np.arange(1000, 5000, 100)
threshold_params = {
        'ZScore': np.round(np.linspace(0, 6, 42), 3),
        'MovingAverage': np.round(np.linspace(0, 1, 32), 3),
        'RSI': np.round(np.linspace(0.2, 0.8, 32), 4),
        'ROC': np.round(np.linspace(0.03, 0.4, 30), 4),
        'MinMax': np.round(np.linspace(0.1, 0.9, 42), 4),
        'Robust': np.round(np.linspace(0, 5, 32), 4),
        'Percentile': np.round(np.linspace(0.1, 0.9, 32), 4)
}

overall_result = pd.DataFrame()

def running_single_strategy(strategy, price_factor_df, long_short, condition):

    test = Optimization(strategy, price_factor_df, window_size_list, threshold_params[strategy], target="Target", price='Price', long_short=long_short, condition=condition)
    test.run()

    optimization_params = pd.DataFrame({
        'Metric': filename,
        'Strategy': strategy,
        'Strategy Type': test.long_short,
        'Condition': test.condition
        }, index=[0])

    return test



if __name__ == "__main__":

    #roc_strategy = ROC(price_factor_df, 3000, 0.014, target='Target', long_short='long', condition='lower')
    #roc_strategy.plot_graph()
    for i in threshold_params:
        test = running_single_strategy(i, price_factor_df, 'long', 'lower')
        test.get_best().result_df.to_csv(i + "_test_logic.csv")
        print(type(test.get_best()).__name__ + ": " + str(test.get_best().window_size) + " " + str(test.get_best().threshold))
    #test.plot_heat_map()
    print('done')
    #print(result.calmar)
    #print(result.sharpe)
    #result.result_df.to_csv("btc_liq_test.csv")
    #print(result.result_df)
    #result.plot_graph()

