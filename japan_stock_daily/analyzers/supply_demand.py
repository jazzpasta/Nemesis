"""Supply/demand analyzer using J-Quants Pro data.

Converts raw J-Quants Pro API data into 0-100 supply/demand scores
for each stock using:
- Weekly margin trading outstanding (信用倍率)
- Outstanding short selling positions
- Investor-type trading breakdown (外国人, 投信, 個人)
- Volume and price momentum
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SupplyDemandAnalyzer:
    """
    Scores stocks on supply/demand dynamics using J-Quants Pro signals.

    Signal contributions (total = 100 pts max):
    - Margin ratio (信用倍率) squeeze potential:  25 pts
    - Short squeeze potential:                    20 pts
    - Foreign investor flow:                      20 pts
    - Volume anomaly:                             15 pts
    - Price vs TOPIX relative strength:           20 pts
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    #  Per-stock scoring                                                   #
    # ------------------------------------------------------------------ #
    def score_ticker(
        self,
        code: str,
        quotes_30d: pd.DataFrame,
        topix_30d: pd.DataFrame,
        margin_row: Optional[pd.Series] = None,
        short_row: Optional[pd.Series] = None,
        breakdown_row: Optional[pd.Series] = None,
    ) -> dict:
        """
        Compute supply/demand score for a single ticker.

        Args:
            code:          Stock code (4-digit)
            quotes_30d:    Last 30 days of OHLCV for this stock
            topix_30d:     Last 30 days of TOPIX index data
            margin_row:    Latest row from weekly_margin_interest
            short_row:     Latest row from short_selling_positions
            breakdown_row: Latest row from breakdown (investor flows)

        Returns:
            dict with keys: score (0-100), components, signals
        """
        components = {}
        signals = []

        # --- 1. Margin ratio (信用倍率) --- 25 pts max
        margin_score, margin_signals = self._score_margin(margin_row)
        components["margin_squeeze"] = margin_score
        signals.extend(margin_signals)

        # --- 2. Short squeeze potential --- 20 pts max
        short_score, short_signals = self._score_short_squeeze(short_row, quotes_30d)
        components["short_squeeze"] = short_score
        signals.extend(short_signals)

        # --- 3. Foreign investor flow --- 20 pts max
        flow_score, flow_signals = self._score_investor_flow(breakdown_row)
        components["investor_flow"] = flow_score
        signals.extend(flow_signals)

        # --- 4. Volume anomaly --- 15 pts max
        vol_score, vol_signals = self._score_volume(quotes_30d)
        components["volume_anomaly"] = vol_score
        signals.extend(vol_signals)

        # --- 5. Relative strength vs TOPIX --- 20 pts max
        rs_score, rs_signals = self._score_relative_strength(quotes_30d, topix_30d)
        components["relative_strength"] = rs_score
        signals.extend(rs_signals)

        total = sum(components.values())
        total = min(max(total, 0), 100)

        return {
            "code":       code,
            "score":      round(total, 1),
            "components": components,
            "signals":    signals,
        }

    # ------------------------------------------------------------------ #
    #  Margin ratio scoring                                               #
    # ------------------------------------------------------------------ #
    def _score_margin(self, margin_row: Optional[pd.Series]) -> tuple:
        """Score based on 信用倍率 (margin long/short ratio)."""
        if margin_row is None:
            return 0, []

        score = 0
        signals = []

        # Margin ratio = 信用買い残 / 信用売り残
        ratio = None
        for col in ("margin_ratio", "MarginRatio", "信用倍率"):
            if col in margin_row.index:
                try:
                    ratio = float(margin_row[col])
                    break
                except (ValueError, TypeError):
                    pass

        if ratio is not None:
            if ratio < 1.0:
                # More shorts than longs → extreme short-squeeze potential
                score = 25
                signals.append(f"信用倍率 {ratio:.2f}x (< 1.0: 強い踏み上げ候補)")
            elif ratio < 1.5:
                score = 20
                signals.append(f"信用倍率 {ratio:.2f}x (< 1.5: 踏み上げ候補)")
            elif ratio < 2.5:
                score = 10
                signals.append(f"信用倍率 {ratio:.2f}x (普通)")
            else:
                score = 2
                signals.append(f"信用倍率 {ratio:.2f}x (> 2.5: 信用買い過多・注意)")

        # Also check if margin longs are declining (supply reduction)
        long_col = None
        for col in ("LongMarginTradeVolume", "信用買い残"):
            if col in (margin_row.index if margin_row is not None else []):
                long_col = col
                break

        return score, signals

    # ------------------------------------------------------------------ #
    #  Short squeeze scoring                                              #
    # ------------------------------------------------------------------ #
    def _score_short_squeeze(
        self,
        short_row: Optional[pd.Series],
        quotes_30d: pd.DataFrame
    ) -> tuple:
        """Score based on short position outstanding and price action."""
        if short_row is None:
            return 0, []

        score = 0
        signals = []

        short_ratio = None
        for col in ("ShortSellingRatio", "short_ratio", "空売り比率"):
            if short_row is not None and col in short_row.index:
                try:
                    short_ratio = float(short_row[col])
                    break
                except (ValueError, TypeError):
                    pass

        if short_ratio is not None:
            # High short ratio + rising price = short squeeze fuel
            if not quotes_30d.empty and len(quotes_30d) >= 5:
                recent_return = _pct_change(
                    quotes_30d["close"].iloc[-1],
                    quotes_30d["close"].iloc[-5]
                )
                if short_ratio > 10.0 and recent_return > 0:
                    score = 20
                    signals.append(f"空売り比率 {short_ratio:.1f}% + 株価上昇: 踏み上げ発生中")
                elif short_ratio > 5.0 and recent_return > 0:
                    score = 12
                    signals.append(f"空売り比率 {short_ratio:.1f}% (高い)")
                elif short_ratio > 10.0:
                    score = 8
                    signals.append(f"空売り比率 {short_ratio:.1f}% (高い・価格反発待ち)")

        return score, signals

    # ------------------------------------------------------------------ #
    #  Investor flow scoring                                              #
    # ------------------------------------------------------------------ #
    def _score_investor_flow(self, breakdown_row: Optional[pd.Series]) -> tuple:
        """Score based on investor-type buy/sell flows from breakdown data."""
        if breakdown_row is None:
            return 0, []

        score = 0
        signals = []

        # Try to extract foreign investor net buy/sell
        foreign_net = _try_float(breakdown_row, [
            "NetBuyForeigner", "ForeignBuyValue", "外国人_net",
            "Foreigner_net", "ForeignNetBuy"
        ])

        trust_net = _try_float(breakdown_row, [
            "NetBuyInvestmentTrust", "TrustBuyValue", "投信_net",
            "InvestmentTrust_net"
        ])

        retail_net = _try_float(breakdown_row, [
            "NetBuyIndividual", "IndividualBuyValue", "個人_net",
            "Individual_net"
        ])

        if foreign_net is not None:
            if foreign_net > 0:
                score += 12
                signals.append("外国人: 買い越し (機関投資家の買い)")
            elif foreign_net < 0:
                score -= 5
                signals.append("外国人: 売り越し")

        if trust_net is not None and trust_net > 0:
            score += 8
            signals.append("投信: 買い越し (インデックス採用候補の可能性)")

        # Retail selling = bullish contrarian signal
        if retail_net is not None and retail_net < 0:
            score += 5
            signals.append("個人: 売り越し (逆張り買いシグナル)")

        score = min(max(score, 0), 20)
        return score, signals

    # ------------------------------------------------------------------ #
    #  Volume anomaly scoring                                             #
    # ------------------------------------------------------------------ #
    def _score_volume(self, quotes_30d: pd.DataFrame) -> tuple:
        """Score based on volume vs 20-day average."""
        if quotes_30d.empty or len(quotes_30d) < 5:
            return 0, []

        score = 0
        signals = []

        try:
            vol_20d_avg = quotes_30d["volume"].iloc[:-1].mean()
            latest_vol  = quotes_30d["volume"].iloc[-1]
            if vol_20d_avg > 0:
                ratio = latest_vol / vol_20d_avg
                if ratio >= 5.0:
                    score = 15
                    signals.append(f"出来高 {ratio:.1f}x (異常急増: 強いシグナル)")
                elif ratio >= 3.0:
                    score = 12
                    signals.append(f"出来高 {ratio:.1f}x (急増)")
                elif ratio >= 2.0:
                    score = 8
                    signals.append(f"出来高 {ratio:.1f}x (増加)")
                elif ratio >= 1.3:
                    score = 4
                    signals.append(f"出来高 {ratio:.1f}x (やや増加)")
        except Exception:
            pass

        return score, signals

    # ------------------------------------------------------------------ #
    #  Relative strength vs TOPIX                                        #
    # ------------------------------------------------------------------ #
    def _score_relative_strength(
        self,
        quotes_30d: pd.DataFrame,
        topix_30d: pd.DataFrame
    ) -> tuple:
        """Score based on relative performance vs TOPIX over 5/20 days."""
        if quotes_30d.empty or topix_30d.empty:
            return 10, []  # Neutral when data missing

        score = 0
        signals = []

        try:
            # 5-day relative strength
            stock_5d = _pct_change(
                quotes_30d["close"].iloc[-1],
                quotes_30d["close"].iloc[-min(5, len(quotes_30d))]
            )
            topix_5d = _pct_change(
                topix_30d["close"].iloc[-1],
                topix_30d["close"].iloc[-min(5, len(topix_30d))]
            )
            if stock_5d is not None and topix_5d is not None:
                rs_5d = stock_5d - topix_5d
                if rs_5d > 3.0:
                    score += 12
                    signals.append(f"5日相対強度 +{rs_5d:.1f}% (TOPIXアウトパフォーム)")
                elif rs_5d > 1.0:
                    score += 8
                    signals.append(f"5日相対強度 +{rs_5d:.1f}%")
                elif rs_5d > 0:
                    score += 5
                elif rs_5d < -3.0:
                    signals.append(f"5日相対強度 {rs_5d:.1f}% (アンダーパフォーム)")

            # 20-day relative strength
            if len(quotes_30d) >= 20 and len(topix_30d) >= 20:
                stock_20d = _pct_change(quotes_30d["close"].iloc[-1], quotes_30d["close"].iloc[-20])
                topix_20d = _pct_change(topix_30d["close"].iloc[-1], topix_30d["close"].iloc[-20])
                if stock_20d is not None and topix_20d is not None:
                    rs_20d = stock_20d - topix_20d
                    if rs_20d > 5.0:
                        score += 8
                        signals.append(f"20日相対強度 +{rs_20d:.1f}%")
                    elif rs_20d > 2.0:
                        score += 4

        except Exception as e:
            logger.debug("Relative strength calculation failed: %s", e)

        score = min(max(score, 0), 20)
        return score, signals

    # ------------------------------------------------------------------ #
    #  Batch scoring                                                      #
    # ------------------------------------------------------------------ #
    def score_all(
        self,
        codes: list,
        quotes_by_code: dict,
        topix_quotes: pd.DataFrame,
        margin_df: pd.DataFrame,
        short_df: pd.DataFrame,
        breakdown_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Score all stocks and return a sorted DataFrame.

        Args:
            codes:          List of stock codes
            quotes_by_code: Dict of code → last 30 days OHLCV DataFrame
            topix_quotes:   TOPIX last 30 days OHLCV
            margin_df:      Weekly margin data (all stocks)
            short_df:       Short selling positions (all stocks)
            breakdown_df:   Daily breakdown (all stocks)
        """
        records = []
        margin_idx = _index_by_code(margin_df)
        short_idx  = _index_by_code(short_df)
        break_idx  = _index_by_code(breakdown_df)

        for code in codes:
            result = self.score_ticker(
                code=code,
                quotes_30d=quotes_by_code.get(code, pd.DataFrame()),
                topix_30d=topix_quotes,
                margin_row=margin_idx.get(code),
                short_row=short_idx.get(code),
                breakdown_row=break_idx.get(code),
            )
            records.append({
                "code":             code,
                "sd_score":         result["score"],
                "sd_signals":       "; ".join(result["signals"]),
                **{f"sd_{k}": v for k, v in result["components"].items()},
            })

        df = pd.DataFrame(records)
        return df.sort_values("sd_score", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def _pct_change(current, base) -> Optional[float]:
    try:
        if base == 0:
            return None
        return (float(current) - float(base)) / float(base) * 100
    except (TypeError, ValueError):
        return None


def _try_float(row: pd.Series, col_names: list) -> Optional[float]:
    for col in col_names:
        if col in row.index:
            try:
                return float(row[col])
            except (ValueError, TypeError):
                pass
    return None


def _index_by_code(df: pd.DataFrame) -> dict:
    """Return dict of code → row Series for fast lookup."""
    if df is None or df.empty:
        return {}
    code_col = None
    for c in ("code", "Code"):
        if c in df.columns:
            code_col = c
            break
    if code_col is None:
        return {}
    return {str(row[code_col])[:4]: row for _, row in df.iterrows()}
