"""Performance metrics for backtesting."""

import numpy as np
import pandas as pd
from typing import Optional


def compute_metrics(pnl_series: pd.Series, annual_trading_days: int = 245) -> dict:
    """
    Compute comprehensive performance metrics from a daily PnL series.

    Args:
        pnl_series: Daily returns as decimal (e.g. 0.01 = 1%)
        annual_trading_days: Trading days per year (JP market ≈ 245)

    Returns:
        dict with all performance metrics
    """
    pnl = pnl_series.dropna()
    if pnl.empty:
        return {}

    n = len(pnl)
    wins  = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    # Basic stats
    win_ratio   = len(wins) / n if n > 0 else 0
    avg_return  = pnl.mean()
    avg_win     = wins.mean()    if len(wins)   > 0 else 0
    avg_loss    = losses.mean()  if len(losses) > 0 else 0
    max_win     = pnl.max()
    max_loss    = pnl.min()

    # Cumulative returns
    cum_returns = (1 + pnl).cumprod()
    total_return = cum_returns.iloc[-1] - 1 if n > 0 else 0
    annual_return = (1 + total_return) ** (annual_trading_days / n) - 1 if n > 0 else 0

    # Drawdown
    rolling_max = cum_returns.cummax()
    drawdowns   = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Sharpe ratio (annualized, assuming 0 risk-free rate in Japan near-zero env)
    std = pnl.std()
    sharpe = (avg_return / std * np.sqrt(annual_trading_days)) if std > 0 else 0

    # Sortino ratio (downside deviation only)
    downside = pnl[pnl < 0].std()
    sortino = (avg_return / downside * np.sqrt(annual_trading_days)) if downside > 0 else 0

    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss   = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Streak analysis
    is_win = (pnl > 0).astype(int)
    max_win_streak  = _max_streak(is_win, 1)
    max_loss_streak = _max_streak(is_win, 0)

    return {
        "n_trades":          n,
        "win_ratio":         round(win_ratio * 100, 1),        # %
        "avg_daily_return":  round(avg_return * 100, 4),       # %
        "avg_win":           round(avg_win * 100, 4),          # %
        "avg_loss":          round(avg_loss * 100, 4),         # %
        "max_win":           round(max_win * 100, 2),          # %
        "max_loss":          round(max_loss * 100, 2),         # %
        "total_return":      round(total_return * 100, 2),     # %
        "annual_return":     round(annual_return * 100, 2),    # %
        "max_drawdown":      round(max_drawdown * 100, 2),     # %
        "sharpe_ratio":      round(sharpe, 3),
        "sortino_ratio":     round(sortino, 3),
        "calmar_ratio":      round(calmar, 3),
        "profit_factor":     round(profit_factor, 3),
        "max_win_streak":    max_win_streak,
        "max_loss_streak":   max_loss_streak,
        "gross_profit_pct":  round(gross_profit * 100, 2),
        "gross_loss_pct":    round(gross_loss * 100, 2),
    }


def _max_streak(series: pd.Series, val: int) -> int:
    """Find maximum consecutive run of `val` in binary series."""
    max_s = 0
    curr_s = 0
    for v in series:
        if v == val:
            curr_s += 1
            max_s = max(max_s, curr_s)
        else:
            curr_s = 0
    return max_s


def compute_monthly_returns(pnl_series: pd.Series) -> pd.DataFrame:
    """Aggregate daily PnL into monthly returns table."""
    df = pnl_series.to_frame("daily_return")
    df.index = pd.to_datetime(df.index)
    monthly = (1 + df["daily_return"]).resample("ME").prod() - 1
    return monthly.to_frame("monthly_return")


def print_metrics(metrics: dict, strategy_name: str = "Strategy"):
    """Print formatted metrics table."""
    print(f"\n{'='*50}")
    print(f" {strategy_name} Performance Summary")
    print(f"{'='*50}")
    print(f" Trades:           {metrics.get('n_trades', 0):>8}")
    print(f" Win Ratio:        {metrics.get('win_ratio', 0):>7.1f}%")
    print(f" Avg Daily Return: {metrics.get('avg_daily_return', 0):>7.4f}%")
    print(f" Avg Win:          {metrics.get('avg_win', 0):>7.4f}%")
    print(f" Avg Loss:         {metrics.get('avg_loss', 0):>7.4f}%")
    print(f" Total Return:     {metrics.get('total_return', 0):>7.2f}%")
    print(f" Annual Return:    {metrics.get('annual_return', 0):>7.2f}%")
    print(f" Max Drawdown:     {metrics.get('max_drawdown', 0):>7.2f}%")
    print(f" Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):>8.3f}")
    print(f" Sortino Ratio:    {metrics.get('sortino_ratio', 0):>8.3f}")
    print(f" Calmar Ratio:     {metrics.get('calmar_ratio', 0):>8.3f}")
    print(f" Profit Factor:    {metrics.get('profit_factor', 0):>8.3f}")
    print(f" Max Win Streak:   {metrics.get('max_win_streak', 0):>8}")
    print(f" Max Loss Streak:  {metrics.get('max_loss_streak', 0):>8}")
    print(f"{'='*50}\n")
