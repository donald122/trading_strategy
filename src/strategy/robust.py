import numpy as np
#import pandas as pd
import modin.pandas as pd
import plotly.express as px
from  strategy.moving_average import MovingAverage
from strategy.strategy import Strategy

class Robust(Strategy):

    def __init__(self, metric, source_df, window_size, threshold, target='Price', price='Price', long_short="long", condition="higher"):
        super().__init__(metric, source_df, window_size, threshold, target=target, price=price, long_short=long_short, condition=condition)
        self.result_df = self._robust_strategy(source_df.copy(), window_size, threshold, target, long_short, condition)

    def _robust_strategy(self, df, window_size, threshold, target, long_short, condition):
        df['rolling_median'] = df[target].rolling(window=int(window_size)).median()
        df['rolling_q75'] = df[target].rolling(window=int(window_size)).quantile(0.75)
        df['rolling_q25'] = df[target].rolling(window=int(window_size)).quantile(0.25)

        df['Robust_Scaled'] = (df[target] - df['rolling_median']) / (df['rolling_q75'] - df['rolling_q25'])

        self._add_position(df, "Robust_Scaled", "diff", threshold, long_short, condition)

        return df

