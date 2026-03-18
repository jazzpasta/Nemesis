"""Negative Shock Recovery Strategy.

Captures mean-reversion after unjustified negative shocks.

Core insight:
  Stocks that drop sharply due to macro/sector events (NOT company-specific
  bad news) tend to recover by next open when JP investors buy the dip.

Signal conditions (all must be met):
  1. Stock dropped > SHOCK_THRESHOLD yesterday (intraday or vs prior close)
  2. Drop is macro-driven (US was also weak OR whole sector was down)
     → Confirmed by: no negative TDnet/EDINET disclosure for this stock
  3. Volume spike on down day (panic selling = washout, smart money buys)
  4. Stock is NOT in multi-day downtrend (avoid falling knives)
  5. Sector ETF is recovering in US after-hours (confirmation)

Ranking:
  Shock score = drop_magnitude × volume_spike × (1 + short_ratio) × sector_recovery

Entry: Buy at next-day open
Exit:  Sell at next-day close OR day+1 open (configurable)
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from backtest.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

# Tunable parameters
SHOCK_THRESHOLD  = 0.030   # -3% intraday drop triggers shock condition
VOLUME_SPIKE_MIN = 1.8     # Volume must be at least 1.8x 20-day average
MAX_DOWNTREND_DAYS = 3     # Exclude stocks in downtrend > this many days
MIN_PRICE = 200            # Exclude penny stocks (< ¥200)
MIN_VOLUME = 50_000        # Minimum daily volume (shares)


class ShockRecoveryStrategy(BaseStrategy):
    """
    Shock recovery: buy stocks that dropped on macro fear, not fundamentals.

    Best conditions:
    - US market dropped 1-3% overnight (mild-to-moderate shock)
    - VIX is elevated but not catastrophic (20-35 range)
    - Individual stocks dropped more than TOPIX (relative weakness → stronger recovery)
    - No negative company-specific news (clean TDnet/EDINET)
    """

    def __init__(
        self,
        top_n: int = 20,
        shock_threshold: float = SHOCK_THRESHOLD,
        volume_spike_min: float = VOLUME_SPIKE_MIN,
        exit_type: str = "next_open",   # 'next_open' or 'same_close'
    ):
        super().__init__(top_n=top_n, hold_days=1)
        self.shock_threshold  = shock_threshold
        self.volume_spike_min = volume_spike_min
        self.exit_type        = exit_type

    def generate_signals(self, target_date: date, data: dict) -> pd.DataFrame:
        """
        Generate shock recovery candidates for stocks entering on target_date's open.

        Args:
            target_date: Date of ENTRY (signal day = target_date - 1)
            data: Dict with keys:
                - quotes:     DataFrame with all daily quotes (multi-date)
                - topix:      TOPIX daily data
                - tdnet:      TDnet disclosures for signal day
                - edinet:     EDINET disclosures for signal day
                - us_overnight: US overnight performance dict
                - margin:     Weekly margin data
                - short:      Short positions data

        Returns:
            DataFrame with: code, shock_score, drop_pct, volume_ratio,
                            short_ratio, sector_recovery, in_downtrend
        """
        signal_date = target_date - timedelta(days=1)  # Yesterday's data
        quotes = data.get("quotes", pd.DataFrame())
        us_data = data.get("us_overnight", {})
        tdnet_df = data.get("tdnet", pd.DataFrame())
        edinet_df = data.get("edinet", pd.DataFrame())

        if quotes.empty:
            return pd.DataFrame()

        quotes = quotes.copy()
        quotes["code"] = quotes["code"].astype(str).str[:4]
        quotes["date_only"] = pd.to_datetime(quotes["date"]).dt.date

        # Build set of codes with negative disclosures (exclude these)
        negative_codes = self._get_negative_disclosure_codes(tdnet_df, edinet_df, signal_date)

        # Get signal day quotes (yesterday)
        sig_quotes = quotes[quotes["date_only"] == signal_date].copy()
        if sig_quotes.empty:
            logger.warning("No quotes found for signal date %s", signal_date)
            return pd.DataFrame()

        # Get 20-day historical quotes for volume average + trend analysis
        hist_start = signal_date - timedelta(days=30)
        hist_quotes = quotes[
            (quotes["date_only"] >= hist_start) &
            (quotes["date_only"] <= signal_date)
        ].copy()

        # TOPIX signal day performance (to classify macro vs company shock)
        topix = data.get("topix", pd.DataFrame())
        topix_drop = self._get_topix_drop(topix, signal_date)

        records = []
        for _, row in sig_quotes.iterrows():
            code = str(row["code"])[:4]

            # Skip stocks with negative disclosures
            if code in negative_codes:
                continue

            # Skip illiquid stocks
            vol = row.get("volume", 0) or 0
            close = row.get("close", 0) or 0
            if vol < MIN_VOLUME or close < MIN_PRICE:
                continue

            # Calculate intraday drop (close vs open)
            open_p = row.get("open", 0) or 0
            if open_p <= 0:
                continue
            intraday_drop = (close - open_p) / open_p  # Negative = drop

            # Also compare to previous close (gap down)
            code_hist = hist_quotes[hist_quotes["code"] == code].sort_values("date")
            if len(code_hist) >= 2:
                prev_close = float(code_hist["close"].iloc[-2])
                gap_drop = (close - prev_close) / prev_close if prev_close > 0 else 0
                # Use worst of intraday or gap
                total_drop = min(intraday_drop, gap_drop)
            else:
                total_drop = intraday_drop

            # Must meet shock threshold
            if total_drop > -self.shock_threshold:
                continue

            # Volume spike check
            if len(code_hist) >= 10:
                vol_avg_20d = code_hist["volume"].iloc[:-1].tail(20).mean()
                volume_ratio = vol / vol_avg_20d if vol_avg_20d > 0 else 1.0
            else:
                volume_ratio = 1.0

            if volume_ratio < self.volume_spike_min:
                continue

            # Downtrend check (avoid falling knives)
            downtrend_days = self._count_downtrend_days(code_hist)
            if downtrend_days > MAX_DOWNTREND_DAYS:
                continue

            # Is this macro-driven? (TOPIX also dropped on same day)
            is_macro_shock = topix_drop is not None and topix_drop < -0.005
            stock_drop_excess = abs(total_drop) - abs(topix_drop or 0)

            # Short ratio (from margin/short data)
            short_ratio = self._get_short_ratio(code, data.get("short", pd.DataFrame()))

            # US sector recovery signal
            sector_recovery = self._get_sector_recovery(code, data.get("universe"), us_data)

            # Compute shock score
            shock_score = self._compute_shock_score(
                drop_pct=abs(total_drop),
                volume_ratio=volume_ratio,
                short_ratio=short_ratio or 0,
                sector_recovery=sector_recovery or 0,
                is_macro_shock=is_macro_shock,
                stock_drop_excess=stock_drop_excess,
            )

            records.append({
                "code":             code,
                "shock_score":      round(shock_score, 2),
                "drop_pct":         round(total_drop * 100, 2),
                "volume_ratio":     round(volume_ratio, 2),
                "short_ratio":      short_ratio,
                "sector_recovery":  sector_recovery,
                "is_macro_shock":   is_macro_shock,
                "downtrend_days":   downtrend_days,
                "signal_date":      signal_date,
            })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # Boost macro-confirmed shocks
        df.loc[df["is_macro_shock"] == True, "shock_score"] *= 1.2
        df = df.sort_values("shock_score", ascending=False).head(self.top_n)
        df["score"] = df["shock_score"]  # Unified interface
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Helper methods                                                      #
    # ------------------------------------------------------------------ #
    def _compute_shock_score(
        self,
        drop_pct: float,
        volume_ratio: float,
        short_ratio: float,
        sector_recovery: float,
        is_macro_shock: bool,
        stock_drop_excess: float,
    ) -> float:
        """
        Score = drop_magnitude × volume_spike × short_squeeze_fuel × sector_tailwind

        Higher = better recovery candidate.
        """
        # Base: magnitude of drop (bigger drop → more recovery room)
        base = min(drop_pct * 10, 40)  # Cap at 40 pts for 4%+ drop

        # Volume spike (washout confirmation)
        vol_bonus = min((volume_ratio - 1.0) * 8, 20)  # Cap at 20 pts

        # Short squeeze fuel (high shorts = more covering on recovery)
        short_bonus = min(short_ratio * 1.5, 15) if short_ratio else 0

        # Sector recovery in US after-hours
        sector_bonus = min(sector_recovery * 10, 15) if sector_recovery and sector_recovery > 0 else 0

        # Macro-confirmed bonus (whole market dropped, not just this stock)
        macro_bonus = 10 if is_macro_shock else 0

        # Relative weakness bonus (dropped more than market → more recovery upside)
        rel_bonus = min(stock_drop_excess * 8, 10)

        return base + vol_bonus + short_bonus + sector_bonus + macro_bonus + rel_bonus

    def _get_negative_disclosure_codes(
        self,
        tdnet_df: Optional[pd.DataFrame],
        edinet_df: Optional[pd.DataFrame],
        signal_date: date,
    ) -> set:
        """Build set of codes that have negative company-specific disclosures."""
        negative_codes = set()
        NEGATIVE_CATEGORIES = {
            "earnings_rev_down", "capital_raise", "legal_risk",
            "large_shareholder_decrease", "m_and_a_target_hostile"
        }
        # Also exclude management changes (often negative signal)
        NEGATIVE_TDNET_KEYWORDS = [
            "下方修正", "業績悪化", "損失", "不正", "調査", "訴訟",
            "第三者割当", "公募増資", "上場廃止"
        ]

        if tdnet_df is not None and not tdnet_df.empty:
            for _, row in tdnet_df.iterrows():
                cat = row.get("category", "")
                title = row.get("title", "")
                if cat in NEGATIVE_CATEGORIES:
                    negative_codes.add(str(row.get("code", ""))[:4])
                elif any(kw in title for kw in NEGATIVE_TDNET_KEYWORDS):
                    negative_codes.add(str(row.get("code", ""))[:4])

        if edinet_df is not None and not edinet_df.empty:
            for _, row in edinet_df.iterrows():
                cat = row.get("event_category", "")
                direction = row.get("direction", "")
                if cat in ("capital_raise", "legal"):
                    negative_codes.add(str(row.get("code", ""))[:4])
                if cat == "large_shareholder" and direction == "decrease":
                    # Large shareholder selling is bearish
                    change = row.get("ownership_change", 0)
                    try:
                        if float(change) < -2.0:  # Significant decrease
                            negative_codes.add(str(row.get("code", ""))[:4])
                    except (TypeError, ValueError):
                        pass

        return negative_codes

    def _get_topix_drop(self, topix_df: pd.DataFrame, signal_date: date) -> Optional[float]:
        """Get TOPIX return for signal_date."""
        if topix_df is None or topix_df.empty:
            return None
        try:
            topix_df = topix_df.copy()
            topix_df["date_only"] = pd.to_datetime(topix_df.get("date", topix_df.index)).dt.date
            day_data = topix_df[topix_df["date_only"] == signal_date]
            if day_data.empty:
                return None
            close_col = "close" if "close" in day_data.columns else "Close"
            open_col  = "open"  if "open"  in day_data.columns else "Open"
            if close_col in day_data.columns and open_col in day_data.columns:
                o = float(day_data[open_col].iloc[0])
                c = float(day_data[close_col].iloc[0])
                return (c - o) / o if o > 0 else None
        except Exception:
            pass
        return None

    def _count_downtrend_days(self, code_hist: pd.DataFrame) -> int:
        """Count consecutive down days at end of history."""
        if code_hist.empty:
            return 0
        closes = code_hist["close"].values
        count = 0
        for i in range(len(closes) - 1, 0, -1):
            if closes[i] < closes[i-1]:
                count += 1
            else:
                break
        return count

    def _get_short_ratio(self, code: str, short_df: pd.DataFrame) -> Optional[float]:
        """Get short selling ratio for a code."""
        if short_df is None or short_df.empty:
            return None
        short_df = short_df.copy()
        short_df["code"] = short_df.get("code", short_df.get("Code", "")).astype(str).str[:4]
        row = short_df[short_df["code"] == code]
        if row.empty:
            return None
        for col in ("ShortSellingRatio", "short_ratio"):
            if col in row.columns:
                try:
                    return float(row[col].iloc[0])
                except (ValueError, TypeError):
                    pass
        return None

    def _get_sector_recovery(
        self,
        code: str,
        universe_df: Optional[pd.DataFrame],
        us_data: dict,
    ) -> Optional[float]:
        """Get US sector ETF overnight change for this stock's sector."""
        if universe_df is None or universe_df.empty:
            return None
        from japan_stock_daily.config import JP_SECTOR_TO_US_ETF
        row = universe_df[universe_df["code"].astype(str).str[:4] == code]
        if row.empty:
            return None
        sector = row["sector33"].iloc[0] if "sector33" in row.columns else ""
        etf = JP_SECTOR_TO_US_ETF.get(sector)
        if not etf:
            return None
        sector_signals = us_data.get("sector_signals", {})
        val = sector_signals.get(etf)
        if val is None:
            return None
        return val / 100  # Convert % to decimal
