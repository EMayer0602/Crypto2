"""
Compatibility layer for ta library using finta.

Provides AverageTrueRange class compatible with ta.volatility.AverageTrueRange
using finta's ATR implementation.
"""
import pandas as pd
from finta import TA


class AverageTrueRange:
    """
    Wrapper for finta's ATR to match ta.volatility.AverageTrueRange interface.
    """

    def __init__(self, high, low, close, window=14):
        """
        Args:
            high: High prices series
            low: Low prices series
            close: Close prices series
            window: ATR period (default 14)
        """
        self.high = high
        self.low = low
        self.close = close
        self.window = window
        self._atr = None

    def average_true_range(self):
        """
        Calculate ATR using finta.

        Returns:
            pandas.Series: ATR values
        """
        if self._atr is None:
            # Create DataFrame for finta
            df = pd.DataFrame({
                'high': self.high,
                'low': self.low,
                'close': self.close
            })

            # Calculate ATR using finta
            self._atr = TA.ATR(df, period=self.window)

        return self._atr
