"""Base strategy class for backtesting."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base class for all backtest strategies.

    A strategy receives historical data for a given date and returns
    a ranked DataFrame of stocks to trade.
    """

    def __init__(
        self,
        top_n: int = 20,
        hold_days: int = 1,
        target_pct: float = None,   # Profit target (e.g. 0.03 = 3%)
        stop_pct: float = None,     # Stop loss (e.g. 0.02 = 2%)
    ):
        """
        Args:
            top_n:      Number of stocks to hold in portfolio each day
            hold_days:  Maximum days to hold (adaptive exit triggers earlier)
            target_pct: Exit when return exceeds this threshold (e.g. 0.03)
            stop_pct:   Exit when loss exceeds this threshold (e.g. 0.02)
        """
        self.top_n = top_n
        self.hold_days = hold_days
        self.target_pct = target_pct
        self.stop_pct = stop_pct

    @abstractmethod
    def generate_signals(self, target_date: date, data: dict) -> pd.DataFrame:
        """
        Generate ranked stock signals for target_date.

        Args:
            target_date: Date to generate signals for
            data:        Dict of pre-fetched data (quotes, disclosures, etc.)

        Returns:
            DataFrame with columns: code, score, [other signal columns]
            Sorted by score descending.
        """
        pass

    def get_entry_price(self, code: str, target_date: date, quotes: pd.DataFrame) -> float:
        """Return entry price (next day's open) for a stock."""
        mask = (quotes["code"].astype(str).str[:4] == str(code)[:4]) & \
               (quotes["date"].dt.date == target_date)
        row = quotes[mask]
        if not row.empty:
            return float(row["open"].iloc[0])
        return None

    def get_exit_price(self, code: str, exit_date: date, quotes: pd.DataFrame) -> float:
        """Return exit price for a stock."""
        mask = (quotes["code"].astype(str).str[:4] == str(code)[:4]) & \
               (quotes["date"].dt.date == exit_date)
        row = quotes[mask]
        if not row.empty:
            return float(row["open"].iloc[0])  # Exit at open of exit_date
        return None
