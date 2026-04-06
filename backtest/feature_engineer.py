"""Feature engineering for LightGBM-based shock recovery strategy.

Extracts raw, un-thresholded features from each trade candidate so that
LightGBM can learn the optimal decision boundaries and feature interactions
itself — rather than relying on hand-crafted rules.

Features are computed using ONLY data available on signal_date (no look-ahead).
"""

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  Main feature builder
# ─────────────────────────────────────────────────────────────────────────────

def build_candidate_features(
    code: str,
    signal_date: date,
    quotes: pd.DataFrame,         # Full multi-stock/multi-date OHLCV
    topix_quotes: pd.DataFrame,   # TOPIX daily OHLCV
    us_data: dict,                # US overnight performance dict
    tdnet_df: pd.DataFrame,       # TDnet disclosures for signal_date
    edinet_df: pd.DataFrame,      # EDINET disclosures for signal_date
    short_df: pd.DataFrame,       # J-Quants short positions
    margin_df: pd.DataFrame,      # J-Quants weekly margin interest
    universe_df: pd.DataFrame,    # Listed companies with sector codes
) -> Optional[dict]:
    """
    Build feature dict for a single (code, signal_date) candidate.

    Returns None if insufficient data.

    Feature groups:
      price_*     Price/return features
      vol_*       Volume features
      tech_*      Technical indicators (RSI, MA ratios)
      shock_*     Shock characterisation
      macro_*     US overnight + TOPIX context
      supply_*    Margin / short selling data
      disc_*      Disclosure flags
    """
    code4 = str(code)[:4]

    # ── Slice historical window (signal_date and 60 prior days) ──
    hist_start = signal_date - timedelta(days=90)
    code_hist = (
        quotes[
            (quotes["code"].astype(str).str[:4] == code4) &
            (pd.to_datetime(quotes["date"]).dt.date >= hist_start) &
            (pd.to_datetime(quotes["date"]).dt.date <= signal_date)
        ]
        .sort_values("date")
        .copy()
    )

    if len(code_hist) < 5:
        return None

    today = code_hist.iloc[-1]
    close   = float(today.get("close", 0) or 0)
    open_p  = float(today.get("open",  0) or 0)
    high    = float(today.get("high",  0) or 0)
    low     = float(today.get("low",   0) or 0)
    volume  = float(today.get("volume", 0) or 0)

    if close <= 0 or open_p <= 0 or volume <= 0:
        return None

    closes  = code_hist["close"].astype(float).values
    volumes = code_hist["volume"].astype(float).values

    # ── Price features ──
    intraday_ret = (close - open_p) / open_p
    prev_close   = float(code_hist["close"].iloc[-2]) if len(code_hist) >= 2 else close
    gap_ret      = (close - prev_close) / prev_close if prev_close > 0 else 0
    total_ret    = min(intraday_ret, gap_ret)           # Worst of intraday / gap

    ret_5d  = (close / closes[-6]  - 1) if len(closes) >= 6  else 0
    ret_20d = (close / closes[-21] - 1) if len(closes) >= 21 else 0

    intraday_range = (high - low) / open_p if open_p > 0 else 0   # Volatility proxy

    # ── Volume features ──
    vol_avg_5d  = volumes[:-1][-5:].mean()  if len(volumes) >= 6  else volume
    vol_avg_20d = volumes[:-1][-20:].mean() if len(volumes) >= 21 else volume
    vol_ratio_5d  = volume / vol_avg_5d  if vol_avg_5d  > 0 else 1.0
    vol_ratio_20d = volume / vol_avg_20d if vol_avg_20d > 0 else 1.0
    turnover      = volume * close                # ¥ turnover on shock day

    # ── Technical indicators ──
    ma5  = closes[-5:].mean()  if len(closes) >= 5  else close
    ma20 = closes[-20:].mean() if len(closes) >= 20 else close
    ma60 = closes[-60:].mean() if len(closes) >= 60 else close

    price_vs_ma5  = (close - ma5)  / ma5  if ma5  > 0 else 0
    price_vs_ma20 = (close - ma20) / ma20 if ma20 > 0 else 0
    price_vs_ma60 = (close - ma60) / ma60 if ma60 > 0 else 0

    # 20-day high/low position (0 = at 20d low, 1 = at 20d high)
    hi20 = closes[-20:].max() if len(closes) >= 20 else close
    lo20 = closes[-20:].min() if len(closes) >= 20 else close
    range20 = hi20 - lo20
    price_position_20d = (close - lo20) / range20 if range20 > 0 else 0.5

    # RSI-14
    rsi = _compute_rsi(closes, period=14)

    # Consecutive down days (falling-knife indicator)
    downtrend_days = _count_downtrend_days(closes)

    # ── Shock characterisation ──
    # Relative drop vs TOPIX (excess weakness → more recovery upside)
    topix_ret = _get_topix_ret(topix_quotes, signal_date)
    excess_drop = total_ret - (topix_ret or 0)  # Negative = dropped more than market
    is_macro_shock = int(topix_ret is not None and topix_ret < -0.005)

    # Wick ratio: (open - low) / (high - low) → large lower wick = buyers stepped in
    wick_range = high - low
    lower_wick = (open_p - low)  / wick_range if wick_range > 0 else 0
    upper_wick = (high  - close) / wick_range if wick_range > 0 else 0

    # ── US overnight context ──
    sp500_overnight  = us_data.get("sp500_overnight",  0) or 0
    nasdaq_overnight = us_data.get("nasdaq_overnight", 0) or 0
    vix              = us_data.get("vix",              20) or 20
    usd_jpy_chg      = us_data.get("usd_jpy_change",   0) or 0

    # Sector-specific US ETF recovery
    sector_etf_ret = _get_sector_etf_ret(code4, universe_df, us_data)

    # ── Supply / demand ──
    short_ratio   = _get_short_ratio(code4, short_df)
    margin_ratio  = _get_margin_ratio(code4, margin_df)   # 信用倍率

    # ── Disclosure flags ──
    has_negative_disc = int(_has_negative_disclosure(code4, tdnet_df, edinet_df))
    has_positive_disc = int(_has_positive_disclosure(code4, tdnet_df, edinet_df))

    # ── Sector dummy (encoded as int from TSE-33 code) ──
    sector_code = _get_sector_code(code4, universe_df)

    return {
        # Price
        "price_intraday_ret":    intraday_ret,
        "price_gap_ret":         gap_ret,
        "price_total_ret":       total_ret,
        "price_ret_5d":          ret_5d,
        "price_ret_20d":         ret_20d,
        "price_intraday_range":  intraday_range,
        "price_level":           close,
        # Volume
        "vol_ratio_5d":          vol_ratio_5d,
        "vol_ratio_20d":         vol_ratio_20d,
        "vol_turnover":          turnover,
        # Technical
        "tech_price_vs_ma5":     price_vs_ma5,
        "tech_price_vs_ma20":    price_vs_ma20,
        "tech_price_vs_ma60":    price_vs_ma60,
        "tech_price_position_20d": price_position_20d,
        "tech_rsi14":            rsi,
        "tech_downtrend_days":   downtrend_days,
        "tech_lower_wick":       lower_wick,
        "tech_upper_wick":       upper_wick,
        # Shock
        "shock_excess_drop":     excess_drop,
        "shock_is_macro":        is_macro_shock,
        "shock_topix_ret":       topix_ret if topix_ret is not None else 0,
        # Macro / US overnight
        "macro_sp500":           sp500_overnight,
        "macro_nasdaq":          nasdaq_overnight,
        "macro_vix":             vix,
        "macro_usd_jpy_chg":     usd_jpy_chg,
        "macro_sector_etf":      sector_etf_ret if sector_etf_ret is not None else 0,
        # Supply/demand
        "supply_short_ratio":    short_ratio  if short_ratio  is not None else 0,
        "supply_margin_ratio":   margin_ratio if margin_ratio is not None else 0,
        # Disclosure flags (binary)
        "disc_has_negative":     has_negative_disc,
        "disc_has_positive":     has_positive_disc,
        # Sector (categorical)
        "sector_code":           sector_code if sector_code is not None else -1,
        # Metadata (not used as features, needed for joining)
        "_code":                 code4,
        "_signal_date":          signal_date,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Feature matrix builder (for a full date range)
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "price_intraday_ret", "price_gap_ret", "price_total_ret",
    "price_ret_5d", "price_ret_20d", "price_intraday_range", "price_level",
    "vol_ratio_5d", "vol_ratio_20d", "vol_turnover",
    "tech_price_vs_ma5", "tech_price_vs_ma20", "tech_price_vs_ma60",
    "tech_price_position_20d", "tech_rsi14", "tech_downtrend_days",
    "tech_lower_wick", "tech_upper_wick",
    "shock_excess_drop", "shock_is_macro", "shock_topix_ret",
    "macro_sp500", "macro_nasdaq", "macro_vix", "macro_usd_jpy_chg",
    "macro_sector_etf",
    "supply_short_ratio", "supply_margin_ratio",
    "disc_has_negative", "disc_has_positive",
    "sector_code",
]

CATEGORICAL_COLS = ["sector_code", "shock_is_macro",
                    "disc_has_negative", "disc_has_positive"]


def build_feature_matrix(
    trade_log: pd.DataFrame,
    quotes: pd.DataFrame,
    topix_quotes: pd.DataFrame,
    us_data_by_date: dict,     # date → us_overnight dict
    tdnet_by_date: dict,       # date → DataFrame
    edinet_by_date: dict,      # date → DataFrame
    short_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature matrix from a trade_log produced by BacktestEngine.

    Each row = one trade. Returns DataFrame with FEATURE_COLS + target columns.
    Automatically labels each trade with:
      - target_return:    actual return_pct achieved
      - target_win:       1 if return > 0, else 0
      - target_win_50bps: 1 if return > 0.5%, else 0  (more meaningful threshold)
    """
    rows = []
    for _, trade in trade_log.iterrows():
        signal_date = pd.to_datetime(trade["signal_date"]).date()
        code = str(trade["code"])[:4]

        feats = build_candidate_features(
            code=code,
            signal_date=signal_date,
            quotes=quotes,
            topix_quotes=topix_quotes,
            us_data=us_data_by_date.get(signal_date, {}),
            tdnet_df=tdnet_by_date.get(signal_date, pd.DataFrame()),
            edinet_df=edinet_by_date.get(signal_date, pd.DataFrame()),
            short_df=short_df,
            margin_df=margin_df,
            universe_df=universe_df,
        )
        if feats is None:
            continue

        feats["target_return"]    = trade["return_pct"] / 100   # Convert % to decimal
        feats["target_win"]       = int(trade["return_pct"] > 0)
        feats["target_win_50bps"] = int(trade["return_pct"] > 0.5)
        feats["exit_reason"]      = trade.get("exit_reason", "")
        feats["hold_days"]        = trade.get("hold_days", 1)

        rows.append(feats)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Wilder RSI over the last `period+1` closes."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _count_downtrend_days(closes: np.ndarray) -> int:
    """Count consecutive down closes at tail of array."""
    count = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] < closes[i - 1]:
            count += 1
        else:
            break
    return count


def _get_topix_ret(topix_df: pd.DataFrame, signal_date: date) -> Optional[float]:
    if topix_df is None or topix_df.empty:
        return None
    try:
        tdf = topix_df.copy()
        tdf["date_only"] = pd.to_datetime(tdf.get("date", tdf.index)).dt.date
        day = tdf[tdf["date_only"] == signal_date]
        if day.empty:
            return None
        o = float(day.get("open",  day.get("Open",  pd.Series([0]))).iloc[0])
        c = float(day.get("close", day.get("Close", pd.Series([0]))).iloc[0])
        return (c - o) / o if o > 0 else None
    except Exception:
        return None


def _get_sector_etf_ret(
    code4: str,
    universe_df: Optional[pd.DataFrame],
    us_data: dict,
) -> Optional[float]:
    if universe_df is None or universe_df.empty:
        return None
    try:
        from japan_stock_daily.config import JP_SECTOR_TO_US_ETF
        row = universe_df[universe_df["code"].astype(str).str[:4] == code4]
        if row.empty:
            return None
        sector = row["sector33"].iloc[0] if "sector33" in row.columns else ""
        etf = JP_SECTOR_TO_US_ETF.get(sector)
        if not etf:
            return None
        val = us_data.get("sector_signals", {}).get(etf)
        return val / 100 if val is not None else None
    except Exception:
        return None


def _get_short_ratio(code4: str, short_df: pd.DataFrame) -> Optional[float]:
    if short_df is None or short_df.empty:
        return None
    try:
        df = short_df.copy()
        df["code"] = df.get("code", df.get("Code", pd.Series(dtype=str))).astype(str).str[:4]
        row = df[df["code"] == code4]
        if row.empty:
            return None
        for col in ("ShortSellingRatio", "short_ratio"):
            if col in row.columns:
                return float(row[col].iloc[0])
    except Exception:
        pass
    return None


def _get_margin_ratio(code4: str, margin_df: pd.DataFrame) -> Optional[float]:
    """信用倍率 = margin long balance / margin short balance."""
    if margin_df is None or margin_df.empty:
        return None
    try:
        df = margin_df.copy()
        df["code"] = df.get("code", df.get("Code", pd.Series(dtype=str))).astype(str).str[:4]
        row = df[df["code"] == code4]
        if row.empty:
            return None
        # J-Quants field names
        for col in ("MarginRatio", "margin_ratio", "信用倍率"):
            if col in row.columns:
                return float(row[col].iloc[0])
        # Compute from long/short balances if available
        long_col  = next((c for c in row.columns if "Long"  in c or "買残" in c), None)
        short_col = next((c for c in row.columns if "Short" in c or "売残" in c), None)
        if long_col and short_col:
            long_val  = float(row[long_col].iloc[0])
            short_val = float(row[short_col].iloc[0])
            return long_val / short_val if short_val > 0 else None
    except Exception:
        pass
    return None


_NEGATIVE_TDNET_KW = [
    "下方修正", "業績悪化", "損失", "不正", "調査", "訴訟",
    "第三者割当", "公募増資", "上場廃止",
]
_POSITIVE_TDNET_KW = [
    "上方修正", "増配", "特別配当", "株式取得", "業務提携",
    "自己株式取得", "MBO", "TOB",
]


def _has_negative_disclosure(
    code4: str,
    tdnet_df: pd.DataFrame,
    edinet_df: pd.DataFrame,
) -> bool:
    if tdnet_df is not None and not tdnet_df.empty:
        for _, row in tdnet_df[
            tdnet_df.get("code", tdnet_df.get("Code", pd.Series(dtype=str)))
            .astype(str).str[:4] == code4
        ].iterrows():
            title = row.get("title", "")
            if any(kw in title for kw in _NEGATIVE_TDNET_KW):
                return True
    return False


def _has_positive_disclosure(
    code4: str,
    tdnet_df: pd.DataFrame,
    edinet_df: pd.DataFrame,
) -> bool:
    if tdnet_df is not None and not tdnet_df.empty:
        for _, row in tdnet_df[
            tdnet_df.get("code", tdnet_df.get("Code", pd.Series(dtype=str)))
            .astype(str).str[:4] == code4
        ].iterrows():
            title = row.get("title", "")
            if any(kw in title for kw in _POSITIVE_TDNET_KW):
                return True
    return False


def _get_sector_code(code4: str, universe_df: Optional[pd.DataFrame]) -> Optional[int]:
    if universe_df is None or universe_df.empty:
        return None
    try:
        row = universe_df[universe_df["code"].astype(str).str[:4] == code4]
        if row.empty:
            return None
        for col in ("sector33code", "Sector33Code", "sector_code"):
            if col in row.columns:
                return int(row[col].iloc[0])
    except Exception:
        pass
    return None
