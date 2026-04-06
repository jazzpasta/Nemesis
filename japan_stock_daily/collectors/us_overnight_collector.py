"""US overnight market collector using yfinance.

Fetches after-hours/pre-market US market data to predict
intraday direction of Japanese stocks when TSE opens at 9AM JST.

Key insight: US market closes 5-6AM JST; JP market opens 9AM JST.
S&P 500 / Nasdaq futures trade 24/7 → best continuous signal.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, Optional

import pandas as pd
import yfinance as yf
import pytz

from japan_stock_daily.config import US_SYMBOLS, JP_SECTOR_TO_US_ETF

logger = logging.getLogger(__name__)

JST = pytz.timezone("Asia/Tokyo")
ET  = pytz.timezone("America/New_York")


class USOverNightCollector:
    """
    Fetches overnight US market performance for JP stock prediction.

    Covers:
    - S&P 500 / Nasdaq / Dow futures (24/7, best signal)
    - VIX (risk-off indicator)
    - USD/JPY (yen strength → exporter impact)
    - US sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLB)
    """

    def __init__(self):
        self._cache: Optional[dict] = None
        self._cache_time: Optional[datetime] = None

    def _cache_valid(self) -> bool:
        if self._cache is None or self._cache_time is None:
            return False
        age = (datetime.now(tz=JST) - self._cache_time).total_seconds()
        return age < 1800  # 30 min cache

    # ------------------------------------------------------------------ #
    #  Core: fetch latest price for a ticker                              #
    # ------------------------------------------------------------------ #
    def _get_latest(self, symbol: str, prepost: bool = True) -> Optional[float]:
        """Return the most recent available price for symbol."""
        try:
            ticker = yf.Ticker(symbol)
            # Use 2-day 5-min data to get most recent price including after-hours
            df = ticker.history(period="2d", interval="5m", prepost=prepost)
            if df.empty:
                return None
            return float(df["Close"].iloc[-1])
        except Exception as e:
            logger.debug("yfinance fetch failed for %s: %s", symbol, e)
            return None

    def _get_regular_close(self, symbol: str) -> Optional[float]:
        """Return yesterday's regular session close price."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="1d", prepost=False)
            if len(df) < 1:
                return None
            return float(df["Close"].iloc[-1])
        except Exception as e:
            logger.debug("yfinance regular close failed for %s: %s", symbol, e)
            return None

    def _pct_change(self, current: Optional[float], base: Optional[float]) -> Optional[float]:
        if current is None or base is None or base == 0:
            return None
        return round((current - base) / base * 100, 3)

    # ------------------------------------------------------------------ #
    #  Main: get overnight performance dict                               #
    # ------------------------------------------------------------------ #
    def get_us_overnight_performance(self) -> Dict:
        """
        Fetch all US overnight indicators.

        Returns dict with:
            sp500_futures_chg:  % change in ES=F (S&P 500 futures)
            nasdaq_futures_chg: % change in NQ=F (Nasdaq futures)
            vix:                current VIX level
            usd_jpy:            current USD/JPY rate
            sp500_overnight:    % change in SPY after regular close (after-hours)
            sector_signals:     {sector_etf → overnight_pct_change}
            market_bias:        'bullish' | 'bearish' | 'neutral'
            yen_direction:      'weakening' | 'strengthening' | 'stable'
        """
        if self._cache_valid():
            return self._cache

        logger.info("Fetching US overnight market data")
        result = {}

        # --- Futures (best real-time signal, 24/7) ---
        try:
            es_ticker = yf.Ticker("ES=F")
            es_df = es_ticker.history(period="2d", interval="5m", prepost=True)
            if not es_df.empty and len(es_df) >= 2:
                latest_es = float(es_df["Close"].iloc[-1])
                # Find yesterday's US regular close (approx 4PM ET = prior session)
                es_daily = es_ticker.history(period="5d", interval="1d", prepost=False)
                base_es = float(es_daily["Close"].iloc[-2]) if len(es_daily) >= 2 else None
                result["sp500_futures_chg"] = self._pct_change(latest_es, base_es)
                result["sp500_futures_price"] = latest_es
        except Exception as e:
            logger.warning("ES=F fetch failed: %s", e)
            result["sp500_futures_chg"] = None

        try:
            nq_ticker = yf.Ticker("NQ=F")
            nq_daily = nq_ticker.history(period="5d", interval="1d", prepost=False)
            nq_5m = nq_ticker.history(period="2d", interval="5m", prepost=True)
            if not nq_5m.empty and len(nq_daily) >= 2:
                result["nasdaq_futures_chg"] = self._pct_change(
                    float(nq_5m["Close"].iloc[-1]),
                    float(nq_daily["Close"].iloc[-2])
                )
        except Exception as e:
            logger.warning("NQ=F fetch failed: %s", e)
            result["nasdaq_futures_chg"] = None

        # --- VIX ---
        result["vix"] = self._get_latest("^VIX", prepost=False)

        # --- USD/JPY ---
        usd_jpy = self._get_latest("JPY=X", prepost=False)
        result["usd_jpy"] = usd_jpy
        prev_usd_jpy = self._get_regular_close("JPY=X")
        result["usd_jpy_chg"] = self._pct_change(usd_jpy, prev_usd_jpy)

        # Yen direction (JPY=X is USD/JPY: higher = yen weaker)
        yen_chg = result.get("usd_jpy_chg", 0) or 0
        if yen_chg > 0.3:
            result["yen_direction"] = "weakening"   # Good for exporters
        elif yen_chg < -0.3:
            result["yen_direction"] = "strengthening"  # Bad for exporters
        else:
            result["yen_direction"] = "stable"

        # --- Sector ETFs (after-hours) ---
        sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLB", "XLU"]
        sector_signals = {}
        for etf in sector_etfs:
            try:
                ticker = yf.Ticker(etf)
                daily = ticker.history(period="5d", interval="1d", prepost=False)
                intraday = ticker.history(period="2d", interval="5m", prepost=True)
                if not intraday.empty and len(daily) >= 1:
                    latest = float(intraday["Close"].iloc[-1])
                    reg_close = float(daily["Close"].iloc[-1])
                    sector_signals[etf] = self._pct_change(latest, reg_close)
            except Exception as e:
                logger.debug("Sector ETF fetch failed for %s: %s", etf, e)
                sector_signals[etf] = None
        result["sector_signals"] = sector_signals

        # --- Overall market bias ---
        sp_chg = result.get("sp500_futures_chg", 0) or 0
        nq_chg = result.get("nasdaq_futures_chg", 0) or 0
        vix = result.get("vix", 20) or 20
        composite = (sp_chg + nq_chg) / 2
        if composite > 0.5 and vix < 25:
            result["market_bias"] = "bullish"
        elif composite < -0.5 or vix > 30:
            result["market_bias"] = "bearish"
        else:
            result["market_bias"] = "neutral"

        # --- SPY after-hours change (alternative to futures) ---
        try:
            spy = yf.Ticker("SPY")
            spy_daily = spy.history(period="5d", interval="1d", prepost=False)
            spy_5m = spy.history(period="2d", interval="5m", prepost=True)
            if not spy_5m.empty and len(spy_daily) >= 1:
                result["sp500_overnight"] = self._pct_change(
                    float(spy_5m["Close"].iloc[-1]),
                    float(spy_daily["Close"].iloc[-1])
                )
        except Exception:
            result["sp500_overnight"] = result.get("sp500_futures_chg")

        result["fetched_at"] = datetime.now(tz=JST).isoformat()
        self._cache = result
        self._cache_time = datetime.now(tz=JST)
        return result

    # ------------------------------------------------------------------ #
    #  Sector-level signal for a JP sector name                          #
    # ------------------------------------------------------------------ #
    def get_sector_overnight_signal(self, jp_sector: str) -> Optional[float]:
        """
        Return overnight % change for the US ETF mapped to this JP sector.
        Returns None if no mapping exists.
        """
        etf = JP_SECTOR_TO_US_ETF.get(jp_sector)
        if not etf:
            return None
        overnight = self.get_us_overnight_performance()
        return overnight.get("sector_signals", {}).get(etf)

    # ------------------------------------------------------------------ #
    #  Historical: for backtesting                                        #
    # ------------------------------------------------------------------ #
    def get_historical_overnight(self, target_date: date) -> Dict:
        """
        Reconstruct overnight US performance for a historical date.
        Uses daily OHLC data (next-open vs prev-close approximation).
        """
        date_str = target_date.strftime("%Y-%m-%d")
        next_date = target_date + timedelta(days=1)

        result = {}
        for symbol_key, symbol in [("sp500", "^GSPC"), ("nasdaq", "^IXIC"), ("vix", "^VIX")]:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=(target_date - timedelta(days=5)).strftime("%Y-%m-%d"),
                    end=(target_date + timedelta(days=5)).strftime("%Y-%m-%d"),
                    interval="1d"
                )
                df.index = pd.to_datetime(df.index).tz_localize(None)
                dates = df.index.strftime("%Y-%m-%d").tolist()
                if date_str in dates:
                    idx = dates.index(date_str)
                    close = float(df["Close"].iloc[idx])
                    if idx + 1 < len(df):
                        next_open = float(df["Open"].iloc[idx + 1])
                        result[f"{symbol_key}_overnight"] = self._pct_change(next_open, close)
                    result[f"{symbol_key}_close"] = close
            except Exception as e:
                logger.debug("Historical fetch failed %s on %s: %s", symbol, date_str, e)

        # Market bias from historical data
        sp_chg = result.get("sp500_overnight", 0) or 0
        vix = result.get("vix_close", 20) or 20
        result["market_bias"] = (
            "bearish" if sp_chg < -0.01 or vix > 30
            else "bullish" if sp_chg > 0.005 and vix < 20
            else "neutral"
        )
        result["sp500_futures_chg"] = result.get("sp500_overnight")
        result["usd_jpy_chg"] = None  # Not easily available historically

        # Sector ETFs historical
        sector_signals = {}
        for etf in ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY"]:
            try:
                ticker = yf.Ticker(etf)
                df = ticker.history(
                    start=(target_date - timedelta(days=3)).strftime("%Y-%m-%d"),
                    end=(target_date + timedelta(days=3)).strftime("%Y-%m-%d"),
                    interval="1d"
                )
                df.index = pd.to_datetime(df.index).tz_localize(None)
                dates = df.index.strftime("%Y-%m-%d").tolist()
                if date_str in dates and len(df) > 0:
                    idx = dates.index(date_str)
                    if idx + 1 < len(df):
                        sector_signals[etf] = self._pct_change(
                            float(df["Open"].iloc[idx + 1]),
                            float(df["Close"].iloc[idx])
                        )
            except Exception:
                pass
        result["sector_signals"] = sector_signals

        # Yen direction
        result["yen_direction"] = "stable"

        return result
