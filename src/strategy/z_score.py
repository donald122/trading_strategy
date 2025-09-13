import numpy as np
#import pandas as pd
import modin.pandas as pd
import plotly.express as px
from  strategy.moving_average import MovingAverage
from strategy.strategy import Strategy

class ZScore(Strategy):

    def __init__(self, metric, source_df, window_size, threshold, target='Price', price='Price', long_short="long", condition="higher"):
        super().__init__(metric, source_df, window_size, threshold, target=target, price=price, long_short=long_short, condition=condition)
        self.result_df = self._z_score_strategy(source_df.copy(), window_size, threshold, target, long_short, condition)

    def _z_score_strategy(self, df, window_size, threshold, target, long_short, condition):
        df['Moving_Average'] = Strategy.return_moving_average(df, target, window_size)
        df['SD'] = Strategy.return_moving_average_sd(df, target, window_size)
        df['Z_Score'] = (df[target] - df['Moving_Average'])/df['SD']

        self._add_position(df, "Z_Score", "diff", threshold, long_short, condition)
        return df

