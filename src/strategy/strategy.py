import numpy as np
#import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt

class Strategy:

    bps = 5

    def __init__(self, metric, source_df, window_size, threshold, target='Price', price='Price', long_short="long", condition="higher"):
        self.result_df = pd.DataFrame()
        self.metric = metric
        self.commission = 0.0001 * Strategy.bps
        self.source_df = source_df
        self.source_df[target] = pd.to_numeric(self.source_df[target], errors='coerce')
        self.source_df['Changes'] = self.source_df['Price'].pct_change(fill_method=None)
        self.source_df['Cumulative_Bnh'] = self.source_df['Changes'].cumsum()
        self.window_size = window_size
        self.threshold = threshold
        if long_short == "both":
            self.threshold = abs(threshold)
        self.target = target

        self.annual_return = 0
        self.mdd = 0
        self.calmar = 0
        self.sharpe = 0
        self.correlation = 0

    @staticmethod
    def get_annual_return(df, profit='Profit'):
        multiplier = Strategy.return_timeframe_multiplier(df)
        return df[profit].mean() * multiplier

    @staticmethod
    def return_moving_average(df, col_name, window_size):
        return df[col_name].rolling(int(window_size)).mean()

    @staticmethod
    def return_moving_average_sd(df, col_name, window_size):
        return df[col_name].rolling(int(window_size)).std()

    @staticmethod
    def get_sharpe(df, profit='Profit'):
        if df[profit].std() != 0 :
            sharpe = (df[profit].mean() / df[profit].std()) * np.sqrt(Strategy.return_timeframe_multiplier(df)).item()
        else:
            sharpe = 0
        return sharpe

    @staticmethod
    def return_mdds(df):
        roll_max = df.cummax()
        drawdown = df - roll_max
        mdds = drawdown.cummin()
        mdds.replace(0, np.inf, inplace=True)
        return mdds

    @staticmethod
    def return_timeframe_multiplier(df):
        diff = df.index.to_series().diff()
        most_common_interval = diff.mode()[0]
        multiplier = 0
        if most_common_interval >= pd.Timedelta(days=1):
            multiplier = 365
        elif most_common_interval >= pd.Timedelta(hours=1):
            multiplier = 365 * 24
        elif most_common_interval >= pd.Timedelta(minutes=10):
            multiplier = 365 * 24 * 6
        elif most_common_interval >= pd.Timedelta(minutes=1):
            multiplier = 365 * 24 * 60
        elif most_common_interval >= pd.Timedelta(seconds=1):
            multiplier = 365 * 24 * 60 * 60

        return multiplier

    def _add_position(self, df, signal, signal_type, threshold, long_short, condition):
        match signal_type:
            # signal can be positive and negative
            case "diff":
                match long_short:
                    case "long":
                        if condition == "higher":
                            # buy when price is higher than moving average
                            df['Position'] = (df[signal].astype(float) > threshold).astype(int)
                        if condition == "lower":
                            # buy  when price is lower than moving average
                            df['Position'] = (df[signal].astype(float) < threshold).astype(int)

                    case "short":
                        if condition == "higher":
                            # short when price is higher than moving average
                            df['Position'] = (df[signal].astype(float) > threshold).astype(int) * - 1
                        if condition == "lower":
                            # short when price is lower than moving average
                            df['Position'] = (df[signal].astype(float) < threshold).astype(int) * -1

                    case "both":
                        if condition == "higher":
                            df['Position'] = (df[signal].astype(float)  > threshold).astype(int) + \
                                            (df[signal].astype(float) < (threshold * - 1)).astype(int) * -1
                        if condition == "lower":
                            df['Position'] = (df[signal].astype(float) < (threshold * -1)).astype(int) + \
                                            (df[signal].astype(float)  > threshold).astype(int) * -1

            # signal is between 0 and 1
            case "bounded":
                match long_short:
                    case "long":
                        if condition == "higher":
                            df['Position'] = (df[signal].astype(float) > threshold).astype(int)
                        if condition == "lower":
                            df['Position'] = (df[signal].astype(float) < threshold).astype(int)

                    case "short":
                        if condition == "higher":
                            df['Position'] = (df[signal].astype(float) > threshold).astype(int) * -1
                        if condition == "lower":
                            df['Position'] = (df[signal].astype(float) < threshold).astype(int) * -1
                    case "both":
                        if condition == "higher":
                            df['Position'] = ((df[signal].astype(float) > threshold).astype(int)) + \
                                            ((df[signal].astype(float) < (1 - threshold)).astype(int) * -1)
                        if condition == "lower":
                            df['Position'] = ((df[signal].astype(float) < threshold).astype(int)) + \
                                            ((df[signal].astype(float) > (1 - threshold)).astype(int) * -1)

        self.add_profit(df)

    def add_profit(self, df):

        df['Profit'] = df['Position'].shift(1) * df['Changes'] - abs(df['Position'].diff()).fillna(0) * self.commission
        #df['Profit'] = df['Position'].shift(1) * df['Changes']
        df['Cumulative_Profit'] = df['Profit'].cumsum()
        self.correlation = self._get_correlation(df)
        self.annual_return = self.get_annual_return(df)
        mdds = self.return_mdds(df['Cumulative_Profit'])
        self.mdd = mdds[mdds.last_valid_index()]
        self.calmar = self.annual_return/abs(self.mdd)
        self.sharpe = self.get_sharpe(df)

    def _get_correlation(self, df):
        correlation = 0
        df_cleaned = df.loc[:, ['Cumulative_Profit', 'Cumulative_Bnh']]
        df_cleaned[['Cumulative_Profit', 'Cumulative_Bnh']] = df_cleaned[['Cumulative_Profit', 'Cumulative_Bnh']].replace([np.inf, -np.inf, 0], np.nan)

        if len(df_cleaned) > 10:
            try:
                correlation = df_cleaned['Cumulative_Profit'].corr(df_cleaned['Cumulative_Bnh'])
            except:
                print('Fail to get correlation: ' + metric)
        return correlation


    def dump_data(self):
        data = {
            'Window': self.window_size,
            'Threshold': self.threshold,
            'Sharpe': self.sharpe,
            'Annual_Return': self.annual_return,
            'MDD': self.mdd,
            'Calmar': self.calmar,
            'Correlation': self.correlation
        }

        return data

    def create_graph(self):
        fig, ax = plt.subplots(figsize=(12, 8))
    	# Plot both lines
        self.result_df.plot(x='Date',
                           y=['Cumulative_Profit', 'Cumulative_Bnh'],
                           ax=ax)
        # Set title
        ax.set_title(f"{type(self).__name__} {self.window_size} {self.threshold}")
        # Add grid
        ax.grid(True)
        # Adjust layout
        plt.tight_layout()

        return fig, ax

    def plot_graph(self):
        fig, _ = self.create_graph()
        plt.show()
