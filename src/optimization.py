import gc
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import os
from strategy.moving_average import MovingAverage
from strategy.z_score import ZScore
from strategy.rsi import RSI
from strategy.roc import ROC
from strategy.percentile import Percentile
from strategy.min_max import MinMax
from strategy.robust import Robust
from joblib import Parallel, delayed
from memory_profiler import profile
from matplotlib.patches import Rectangle

class Optimization():

    def __init__(self, metric, strategy_class,  train_df, test_df, window_size_list, threshold_list, target, price='Price', long_short='long', condition='higher'):
        self.metric = metric
        self.strategy_class = strategy_class
        self.train_df = train_df
        self.test_df = test_df
        self.window_size_list = window_size_list
        self.threshold_list = threshold_list
        self.target = target
        self.price = price
        self.long_short = long_short
        self.condition = condition

        self.train_results_data_df = pd.DataFrame()
        self.test_results_data_df = pd.DataFrame()

        # remove all the negative threshold if it is running both long short
        if long_short == "both":
            self.threshold_list = [x for x in self.threshold_list if x >= 0]


    def _run_strategy(self, window_size, threshold):

        train_result = self.strategy_class(self.metric, self.train_df, window_size, threshold, target=self.target, price=self.price, long_short=self.long_short, condition=self.condition)
        test_result = self.strategy_class(self.metric, self.test_df, window_size, threshold, target=self.target, price=self.price, long_short=self.long_short, condition=self.condition)

        train_result_data = train_result.dump_data()
        test_result_data = test_result.dump_data()

        return train_result_data, test_result_data

    def run(self):
        #results = []
        train_results_data = []
        test_results_data = []
        #results = Parallel(n_jobs=-1, backend='threading')(delayed(self._run_strategy)(window_size, threshold) for window_size in self.window_size_list for threshold in self.threshold_list)


        for window_size in self.window_size_list:
            for threshold in self.threshold_list:
                train_result, test_result = self._run_strategy(window_size, threshold)
                train_results_data.append(train_result)
                test_results_data.append(test_result)

        self.train_results_data_df = pd.DataFrame(train_results_data)
        self.train_results_data_df = self.train_results_data_df.sort_values(by='Sharpe', ascending=False)

        self.test_results_data_df = pd.DataFrame(test_results_data)
        self.test_results_data_df = self.test_results_data_df.sort_values(by='Sharpe', ascending=False)

    def _create_optimization_result(self):
        # get best train windows size and threshold
        train_best_window_size = self.train_results_data_df.iloc[0]['Window']
        train_best_threshold = self.train_results_data_df.iloc[0]['Threshold']
        train_best_sharpe = self.train_results_data_df.iloc[0]['Sharpe']
        test_best_sharpe = self.test_results_data_df.iloc[0]['Sharpe']

        train_result_data_pivot = self.train_results_data_df.pivot(index='Window', columns='Threshold', values='Sharpe')
        test_result_data_pivot = self.test_results_data_df.pivot(index='Window', columns='Threshold', values='Sharpe')




        # create subpot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(self.strategy_class.__name__ + " " + self.long_short.capitalize() + " " + self.condition.capitalize())

        sns.heatmap(train_result_data_pivot, ax=ax1, cmap="Greens",  annot=True, annot_kws={"size": 6}, fmt=".1f").set_title("Train")
        sns.heatmap(test_result_data_pivot, ax=ax2, cmap="Greens", annot=True, annot_kws={"size": 6}, fmt=".1f").set_title("Test")

        #### add red box
        # Find the positions of the specified value
        train_pivot_positions = np.argwhere(np.isclose(train_result_data_pivot.values, train_best_sharpe, atol=1e-6))
        test_pivot_positions = np.argwhere(np.isclose(test_result_data_pivot.values, test_best_sharpe, atol=1e-6))

        for row, col in train_pivot_positions:
            ax1.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor='red', lw=2))
        for row, col in test_pivot_positions:
            ax2.add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor='red', lw=2))

        # plot the best strategy
        train_best, _ = self.get_best()
        ax3.set_title(
                f"Equity Curve - {self.strategy_class.__name__} (Window: {train_best.window_size}, Threshold: {train_best.threshold:.3f})")
        ax3.set_ylabel('Cumulative Profit')
        ax3.set_xlabel('Date')
        train_best.result_df.plot(x='Date', y=['Cumulative_Profit', 'Cumulative_Bnh'], color=['blue', 'green'], ax=ax3, label=['Train Strategy', 'Train Buy and Hold'])


        test_result = self.strategy_class(self.metric, self.test_df, train_best.window_size, train_best.threshold, target=self.target, price=self.price, long_short=self.long_short, condition=self.condition)
        test_result_df = test_result.result_df
        test_result.result_df.plot(x='Date', y=['Cumulative_Profit', 'Cumulative_Bnh'], color=['blue', 'green'], ax=ax3, label=['Test Strategy', 'Test Buy And Hold'], linestyle='--')

       # Add text box with key metrics
        train_metrics = (f"Train - Annual Return: {train_best.annual_return * 100:.2f}%, "
                         f"Sharpe: {train_best.sharpe:.2f}, MDD: {train_best.mdd * 100:.2f}%, "
                         f"Calmar: {train_best.calmar:.2f} "
                         f"Correlation: {train_best.correlation:.2f}")

        test_metrics = (f"Test - Annual Return: {test_result.annual_return * 100:.2f}%, "
                        f"Sharpe: {test_result.sharpe:.2f}, MDD: {test_result.mdd * 100:.2f}%, "
                        f"Calmar: {test_result.calmar:.2f}")

        metrics_text = (f"{train_metrics}\n{test_metrics}\n"
                        f"Best Window: {train_best.window_size}, Best Threshold: {train_best.threshold:.3f}")
        ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, verticalalignment='top',
                 fontsize=7, bbox=dict(facecolor='white', alpha=0.8))
        plt.tight_layout()


        return fig, (ax1, ax2)

    def plot_optimization_result(self):

        fig, _ = self._create_optimization_result()
        plt.show()

    def save_optimization_result(self, directory):
        fig, _ = self._create_optimization_result()
        train_best, test_best = self.get_best()

        # do not save if position transition less than a value
        train_best_transitions = (train_best.result_df['Position'] != train_best.result_df['Position'].shift()).sum() - 1
        if train_best_transitions > 10:

            if train_best.sharpe > 1.1 or (self.long_short == 'short' and train_best.calmar > 0.5):
                if train_best.correlation < 1:
                    # Save graph
                    os.makedirs(directory, exist_ok=True)
                    result_strategy_name = f"{self.strategy_class.__name__}_{self.long_short}_{self.condition}"
                    fig.savefig(directory + result_strategy_name + ".png", dpi=300)
                    # Save csv
                    train_best.result_df.to_csv(directory + result_strategy_name + ".csv")

        plt.close()

    def get_best(self):

        train_best_data = self.train_results_data_df.iloc[0]
        train_best_object = self.strategy_class(self.metric, self.train_df, train_best_data['Window'], train_best_data['Threshold'], target=self.target, price=self.price, long_short=self.long_short, condition=self.condition)

        test_best_data =  self.test_results_data_df.iloc[0]
        test_best_object = self.strategy_class(self.metric, self.test_df, test_best_data['Window'], test_best_data['Threshold'], target=self.target, price=self.price, long_short=self.long_short, condition=self.condition)

        return train_best_object, test_best_object


