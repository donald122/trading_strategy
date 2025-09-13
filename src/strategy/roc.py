import numpy as np
#import pandas as pd
import modin.pandas as pd
import plotly.express as px
from  strategy.moving_average import MovingAverage
from strategy.strategy import Strategy

class ROC(Strategy):

    def __init__(self, metric, source_df, window_size, threshold, target='Price', price='Price', long_short="long", condition="higher"):
        super().__init__(metric, source_df, window_size, threshold, target=target, price=price, long_short=long_short, condition=condition)
        self.result_df = self._roc_strategy(source_df.copy(), window_size, threshold, target, long_short, condition)

    def _roc_strategy(self, df, window_size, threshold, target, long_short, condition):
        df['ROC'] = (df[target] - df[target].shift(int(window_size)))/df[target].shift(int(window_size))
        self._add_position(df, "ROC", "diff", threshold, long_short, condition)

        return df


