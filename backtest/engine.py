"""Backtesting engine for Japan daily stock strategies.

Entry: Buy top_n stocks at next-day OPEN (after signal date close)
Exit:  Sell at next-day OPEN (1-day overnight hold, default)
       OR same-day CLOSE (intraday hold)

Assumption:
- Equal-weighted portfolio across top_n stocks
- No transaction costs by default (add slippage if needed)
- No look-ahead bias: signals use ONLY data available before market open
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Dict, Callable

import pandas as pd
import numpy as np

from backtest.metrics import compute_metrics, print_metrics
from backtest.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

JP_TRADING_DAYS_PER_YEAR = 245


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name:    str
    start_date:       date
    end_date:         date
    top_n:            int
    daily_pnl:        pd.Series          # Date → average daily return
    trade_log:        pd.DataFrame       # All individual trades
    metrics:          dict               # Performance metrics
    daily_scores:     Dict[date, pd.DataFrame] = field(default_factory=dict)  # Scored stocks per day

    def print_summary(self):
        print_metrics(self.metrics, self.strategy_name)

    def get_cum_returns(self) -> pd.Series:
        return (1 + self.daily_pnl).cumprod()


class BacktestEngine:
    """
    Runs a strategy over a historical date range and computes performance.

    Usage:
        engine = BacktestEngine(all_quotes_df)
        result = engine.run(ShockRecoveryStrategy(), date(2024,1,1), date(2024,12,31))
        result.print_summary()
    """

    def __init__(
        self,
        all_quotes: pd.DataFrame,
        slippage_bps: float = 5.0,    # One-way slippage in basis points
        data_loader: Optional[Callable] = None,
    ):
        """
        Args:
            all_quotes:   All daily OHLCV data (multi-stock, multi-date)
                          Required columns: code, date, open, high, low, close, volume
            slippage_bps: One-way slippage in basis points (5bps = 0.05%)
            data_loader:  Optional function(date) → dict of data for that date
                          (disclosures, US data, etc.) for strategies that need it
        """
        self.all_quotes = all_quotes.copy()
        self.all_quotes["code"] = self.all_quotes["code"].astype(str).str[:4]
        self.all_quotes["date"] = pd.to_datetime(self.all_quotes["date"])
        self.all_quotes["date_only"] = self.all_quotes["date"].dt.date
        self.slippage = slippage_bps / 10000  # Convert bps to decimal
        self.data_loader = data_loader

    # ------------------------------------------------------------------ #
    #  Main backtest loop                                                  #
    # ------------------------------------------------------------------ #
    def run(
        self,
        strategy: BaseStrategy,
        start_date: date,
        end_date: date,
        progress: bool = True,
    ) -> BacktestResult:
        """
        Run backtest for strategy over [start_date, end_date].

        Signal day = T (yesterday's data)
        Entry day  = T+1 at open
        Exit day   = T+2 at open (for 1-day hold) OR T+1 at close
        """
        logger.info("Starting backtest: %s to %s", start_date, end_date)

        trading_dates = self._get_trading_dates(start_date, end_date)
        logger.info("Trading dates: %d days", len(trading_dates))

        daily_pnl = {}
        trade_records = []
        daily_scores_cache = {}

        for i, signal_date in enumerate(trading_dates[:-1]):
            entry_date = trading_dates[i + 1]  # T+1
            if strategy.hold_days == 1:
                exit_date = trading_dates[i + 2] if i + 2 < len(trading_dates) else None
            else:
                exit_idx = i + 1 + strategy.hold_days
                exit_date = trading_dates[exit_idx] if exit_idx < len(trading_dates) else None

            if exit_date is None:
                continue

            # Load data for signal date (no look-ahead: only data up to signal_date)
            data = self._load_data(signal_date, strategy)

            # Generate signals for signal_date (picks for entry on entry_date)
            try:
                signals = strategy.generate_signals(entry_date, data)
            except Exception as e:
                logger.warning("Signal generation failed for %s: %s", signal_date, e)
                continue

            if signals is None or signals.empty:
                daily_pnl[entry_date] = 0.0
                continue

            daily_scores_cache[signal_date] = signals

            # Execute trades with adaptive exit (target/stop/max-hold)
            day_returns = []
            for _, sig in signals.head(strategy.top_n).iterrows():
                code = str(sig["code"])[:4]
                entry_price = self._get_price(code, entry_date, "open")
                if entry_price is None:
                    continue

                entry_adj = entry_price * (1 + self.slippage)
                actual_exit_date, exit_price, exit_reason = self._adaptive_exit(
                    code, entry_date, entry_adj, trading_dates, i + 1, strategy
                )
                if exit_price is None:
                    continue

                exit_adj = exit_price * (1 - self.slippage)
                ret = (exit_adj - entry_adj) / entry_adj
                day_returns.append(ret)

                trade_records.append({
                    "signal_date":  signal_date,
                    "entry_date":   entry_date,
                    "exit_date":    actual_exit_date,
                    "code":         code,
                    "entry_price":  round(entry_price, 2),
                    "exit_price":   round(exit_price, 2),
                    "return_pct":   round(ret * 100, 4),
                    "score":        sig.get("score", sig.get("composite_score", None)),
                    "is_win":       ret > 0,
                    "exit_reason":  exit_reason,
                    "hold_days":    (actual_exit_date - entry_date).days if actual_exit_date else 0,
                })

            daily_pnl[entry_date] = np.mean(day_returns) if day_returns else 0.0

            if progress and (i + 1) % 20 == 0:
                running_wins = sum(1 for v in daily_pnl.values() if v > 0)
                running_total = len(daily_pnl)
                logger.info("Progress: %d/%d | Win ratio: %.1f%%",
                            i + 1, len(trading_dates),
                            running_wins / running_total * 100 if running_total > 0 else 0)

        pnl_series = pd.Series(daily_pnl)
        trade_df   = pd.DataFrame(trade_records)
        metrics    = compute_metrics(pnl_series)

        return BacktestResult(
            strategy_name=strategy.__class__.__name__,
            start_date=start_date,
            end_date=end_date,
            top_n=strategy.top_n,
            daily_pnl=pnl_series,
            trade_log=trade_df,
            metrics=metrics,
            daily_scores=daily_scores_cache,
        )

    # ------------------------------------------------------------------ #
    #  Parameter sweep / optimization                                     #
    # ------------------------------------------------------------------ #
    def sweep_parameters(
        self,
        strategy_class,
        param_grid: dict,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Sweep strategy parameters and return performance table.

        Args:
            strategy_class: Strategy class to instantiate
            param_grid:     Dict of param_name → list of values to try
            start_date/end_date: Backtest period

        Returns:
            DataFrame of parameter combinations × metrics, sorted by Sharpe
        """
        import itertools
        results = []
        keys = list(param_grid.keys())
        for values in itertools.product(*param_grid.values()):
            params = dict(zip(keys, values))
            strategy = strategy_class(**params)
            try:
                result = self.run(strategy, start_date, end_date, progress=False)
                row = {**params, **result.metrics}
                results.append(row)
                logger.info("Params %s → Sharpe %.3f, WinRate %.1f%%",
                             params, result.metrics.get("sharpe_ratio", 0),
                             result.metrics.get("win_ratio", 0))
            except Exception as e:
                logger.warning("Sweep failed for %s: %s", params, e)

        df = pd.DataFrame(results)
        if not df.empty and "sharpe_ratio" in df.columns:
            df = df.sort_values("sharpe_ratio", ascending=False)
        return df

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _get_trading_dates(self, start_date: date, end_date: date) -> List[date]:
        """Get list of trading dates from the quotes data."""
        mask = (self.all_quotes["date_only"] >= start_date) & \
               (self.all_quotes["date_only"] <= end_date)
        dates = sorted(self.all_quotes[mask]["date_only"].unique())
        return [d for d in dates if d.weekday() < 5]  # Weekdays only

    def _get_price(self, code: str, target_date: date, price_type: str) -> Optional[float]:
        """Get open/close price for a code on a date."""
        mask = (self.all_quotes["code"] == code) & \
               (self.all_quotes["date_only"] == target_date)
        row = self.all_quotes[mask]
        if row.empty:
            return None
        try:
            return float(row[price_type].iloc[0])
        except (KeyError, ValueError, IndexError):
            return None

    def _adaptive_exit(
        self,
        code: str,
        entry_date: date,
        entry_price_adj: float,  # Already slippage-adjusted entry
        trading_dates: list,
        entry_idx: int,
        strategy: BaseStrategy,
    ):
        """
        Adaptive exit: sell when target hit, stop-loss hit, or max_hold reached.

        For each hold day after entry:
        - Check intraday high: if high > entry * (1 + target) → exit at target (stop profit)
        - Check intraday low:  if low  < entry * (1 - stop)  → exit at stop (stop loss)
        - Otherwise: hold until max_hold, exit at next open

        Returns: (exit_date, exit_price, exit_reason)
        """
        target = getattr(strategy, "target_pct", None)
        stop   = getattr(strategy, "stop_pct", None)
        max_hold = getattr(strategy, "hold_days", 1)

        for hold_day in range(1, max_hold + 1):
            hold_idx = entry_idx + hold_day
            if hold_idx >= len(trading_dates):
                break
            hold_date = trading_dates[hold_idx]

            # Get intraday data for this hold day
            high  = self._get_price(code, hold_date, "high")
            low   = self._get_price(code, hold_date, "low")
            close = self._get_price(code, hold_date, "close")
            open_p = self._get_price(code, hold_date, "open")

            if open_p is None:
                continue

            # Check target hit (use intraday high)
            if target and high is not None:
                target_price = entry_price_adj * (1 + target)
                if high >= target_price:
                    return hold_date, target_price, "target"

            # Check stop-loss hit (use intraday low)
            if stop and low is not None:
                stop_price = entry_price_adj * (1 - stop)
                if low <= stop_price:
                    return hold_date, stop_price, "stop"

            # Last hold day: exit at open
            if hold_day == max_hold:
                return hold_date, open_p, "max_hold"

        # Fallback: exit at entry_date + 1 open
        fallback_idx = entry_idx + 1
        if fallback_idx < len(trading_dates):
            fb_date = trading_dates[fallback_idx]
            fb_price = self._get_price(code, fb_date, "open")
            return fb_date, fb_price, "fallback"
        return None, None, "no_exit"

    def _load_data(self, signal_date: date, strategy: BaseStrategy) -> dict:
        """Load all data needed for signal generation on signal_date."""
        # Quotes up to and including signal_date (no look-ahead)
        hist_start = signal_date - timedelta(days=40)
        quotes_hist = self.all_quotes[
            (self.all_quotes["date_only"] >= hist_start) &
            (self.all_quotes["date_only"] <= signal_date)
        ].copy()

        data = {"quotes": quotes_hist}

        # Load additional data (disclosures, US overnight, etc.)
        if self.data_loader:
            try:
                extra = self.data_loader(signal_date)
                data.update(extra)
            except Exception as e:
                logger.debug("Data loader failed for %s: %s", signal_date, e)

        return data
