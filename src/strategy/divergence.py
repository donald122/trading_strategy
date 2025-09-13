import numpy as np
import pandas as pd
import plotly.express as px
from strategy.strategy import Strategy
import logging

logger = logging.getLogger(__name__)


class Divergence(Strategy):

    def __init__(self, metric, source_df, window_size, threshold, target='Price', price='Price', long_short="long",
                 condition="higher"):
        logger.debug(
            f"Initializing Divergence strategy with window_size={window_size}, threshold={threshold}, target={target}, price={price}, long_short={long_short}, condition={condition}")
        logger.debug(f"Source DataFrame shape: {source_df.shape}")
        logger.debug(f"Source DataFrame columns: {source_df.columns}")
        logger.debug(f"First few rows of source DataFrame:\n{source_df.head()}")

        super().__init__(metric, source_df,window_size, threshold, target=target, price=price, long_short=long_short,
                         condition=condition)
        self.result_df = self._divergence_strategy(source_df.copy(), window_size, threshold, target, price, long_short,
                                                   condition)

    def _divergence_strategy(self, df, window_size, threshold, target, price, long_short, condition):
        logger.debug(f"Executing divergence strategy")
        logger.debug(f"DataFrame shape: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns}")

        # Calculate Z-scores for both price and target
        df['Price_MA'] = Strategy.return_moving_average(df, price, window_size)
        df['Price_SD'] = Strategy.return_moving_average_sd(df, price, window_size)
        df['Price_Z_Score'] = (df[price] - df['Price_MA']) / df['Price_SD']

        df['Target_MA'] = Strategy.return_moving_average(df, target, window_size)
        df['Target_SD'] = Strategy.return_moving_average_sd(df, target, window_size)
        df['Target_Z_Score'] = (df[target] - df['Target_MA']) / df['Target_SD']

        # Calculate divergence
        df['Divergence'] = df['Price_Z_Score'] - df['Target_Z_Score']

        logger.debug(f"Divergence calculation complete")
        logger.debug(f"DataFrame columns after calculation: {df.columns}")
        logger.debug(f"First few rows after calculation:\n{df.head()}")

        self._add_position(df, "Divergence", "diff", threshold, long_short, condition)
        return df
