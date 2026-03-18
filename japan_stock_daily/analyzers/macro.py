"""Macro analyzers: BOJ + e-Stats + METI sector context + US overnight signals."""

import logging
from typing import Dict, Optional, List

import pandas as pd

from japan_stock_daily.config import JP_SECTOR_TO_US_ETF

logger = logging.getLogger(__name__)


class MacroAnalyzer:
    """
    Converts BOJ, e-Stats, and METI data into sector-level macro scores (0-100).

    Slower-moving signals that provide fundamental tailwind/headwind context.
    """

    def get_sector_macro_score(
        self,
        sector_name: str,
        boj_data: dict,
        estats_data: dict,
        meti_trend: str,
    ) -> dict:
        """
        Score the macro environment for a given TSE-33 sector.

        Returns: {score: 0-100, signals: [...]}
        """
        score = 50  # Neutral baseline
        signals = []

        # --- Rate environment (BOJ) ---
        rate_env = boj_data.get("rate_environment", "stable")
        policy_rate = boj_data.get("policy_rate")

        if sector_name in ("銀行業", "保険業", "証券・商品先物取引業"):
            # Financials benefit from rate hikes
            if rate_env == "hiking":
                score += 15
                signals.append("BOJ利上げ局面: 金融セクター有利")
            elif rate_env == "easing":
                score -= 10
                signals.append("BOJ緩和局面: 金融セクター不利")

        elif sector_name in ("電気機器", "輸送用機器", "機械"):
            # Exporters: yen weakness is bullish
            usd_jpy = boj_data.get("usd_jpy", 150)
            if usd_jpy and usd_jpy > 145:
                score += 12
                signals.append(f"円安 ({usd_jpy:.1f}円): 輸出企業有利")
            elif usd_jpy and usd_jpy < 130:
                score -= 8
                signals.append(f"円高 ({usd_jpy:.1f}円): 輸出企業不利")

        elif sector_name in ("小売業", "サービス業", "外食"):
            # Consumer sectors: CPI and retail sales
            retail_yoy = estats_data.get("retail_sales_yoy")
            if retail_yoy is not None:
                if retail_yoy > 3.0:
                    score += 12
                    signals.append(f"小売販売額 前年比 +{retail_yoy:.1f}%: 消費旺盛")
                elif retail_yoy < 0:
                    score -= 8
                    signals.append(f"小売販売額 前年比 {retail_yoy:.1f}%: 消費低迷")

        elif sector_name in ("不動産業",):
            # Real estate: sensitive to rates
            jgb_10y = boj_data.get("jgb_10y")
            if jgb_10y is not None:
                if jgb_10y > 1.5:
                    score -= 10
                    signals.append(f"長期金利 {jgb_10y:.2f}%上昇: 不動産セクター不利")
                elif jgb_10y < 0.5:
                    score += 8
                    signals.append(f"長期金利 {jgb_10y:.2f}%: 不動産セクター有利")

        # --- METI IIP trend ---
        if meti_trend == "expanding":
            score += 10
            signals.append("IIP産業生産 拡大トレンド")
        elif meti_trend == "contracting":
            score -= 8
            signals.append("IIP産業生産 縮小トレンド")

        # --- Unemployment (general) ---
        unemployment = estats_data.get("unemployment_latest")
        if unemployment is not None:
            if unemployment < 2.5:
                score += 5
                signals.append(f"完全失業率 {unemployment:.1f}% (低水準: 消費サポート)")
            elif unemployment > 3.5:
                score -= 5
                signals.append(f"完全失業率 {unemployment:.1f}% (高水準)")

        score = min(max(score, 0), 100)
        return {"score": round(score, 1), "signals": signals}

    def score_all_sectors(
        self,
        sector_names: List[str],
        boj_data: dict,
        estats_data: dict,
        meti_collector,
    ) -> Dict[str, dict]:
        """Return macro scores for all sectors."""
        results = {}
        for sector in sector_names:
            meti_trend = meti_collector.get_sector_trend(sector) if meti_collector else "unknown"
            results[sector] = self.get_sector_macro_score(
                sector_name=sector,
                boj_data=boj_data,
                estats_data=estats_data,
                meti_trend=meti_trend,
            )
        return results


class USOverNightAnalyzer:
    """
    Converts US overnight market data into per-stock scores.

    Uses S&P 500/Nasdaq futures, VIX, USD/JPY, and sector ETF performance
    to predict intraday direction of JP stocks at TSE open.
    """

    def score_ticker(
        self,
        sector_name: str,
        us_overnight: dict,
        yen_sensitive: bool = False,
    ) -> dict:
        """
        Score a single stock's expected move based on US overnight data.

        Args:
            sector_name:   TSE-33 sector name
            us_overnight:  Output of USOverNightCollector.get_us_overnight_performance()
            yen_sensitive: True for exporters (自動車, 電機, 精密機器)

        Returns:
            dict with: score (0-100), signals, expected_direction
        """
        score = 50  # Neutral baseline
        signals = []

        market_bias = us_overnight.get("market_bias", "neutral")
        sp_chg = us_overnight.get("sp500_futures_chg", 0) or 0
        nq_chg = us_overnight.get("nasdaq_futures_chg", 0) or 0
        vix = us_overnight.get("vix", 20) or 20
        yen_direction = us_overnight.get("yen_direction", "stable")
        yen_chg = us_overnight.get("usd_jpy_chg", 0) or 0

        # --- Overall market bias ---
        if market_bias == "bullish":
            score += 15
            signals.append(f"米国市場↑ S&P500先物 +{sp_chg:.1f}%")
        elif market_bias == "bearish":
            score -= 15
            signals.append(f"米国市場↓ S&P500先物 {sp_chg:.1f}%")

        # --- VIX risk-off ---
        if vix > 30:
            score -= 20
            signals.append(f"VIX {vix:.0f} リスクオフ: 全面的な売り圧力")
        elif vix > 25:
            score -= 10
            signals.append(f"VIX {vix:.0f} やや高め")
        elif vix < 15:
            score += 5
            signals.append(f"VIX {vix:.0f} 低位安定")

        # --- Sector ETF signal ---
        etf = JP_SECTOR_TO_US_ETF.get(sector_name)
        if etf:
            sector_chg = (us_overnight.get("sector_signals") or {}).get(etf)
            if sector_chg is not None:
                if sector_chg > 1.5:
                    score += 20
                    signals.append(f"{etf} (米国セクター) +{sector_chg:.1f}%: セクター追い風")
                elif sector_chg > 0.5:
                    score += 12
                    signals.append(f"{etf} +{sector_chg:.1f}%")
                elif sector_chg < -1.5:
                    score -= 18
                    signals.append(f"{etf} {sector_chg:.1f}%: セクター向かい風")
                elif sector_chg < -0.5:
                    score -= 8
                    signals.append(f"{etf} {sector_chg:.1f}%")

        # --- Yen impact on exporters ---
        if yen_sensitive:
            if yen_direction == "weakening":
                score += 12
                signals.append(f"円安 +{yen_chg:.1f}%: 輸出企業恩恵")
            elif yen_direction == "strengthening":
                score -= 10
                signals.append(f"円高 {yen_chg:.1f}%: 輸出企業逆風")

        # Nasdaq-heavy sectors (電気機器, 情報通信)
        if sector_name in ("電気機器", "情報・通信業") and nq_chg != 0:
            if nq_chg > 1.0:
                score += 8
                signals.append(f"Nasdaq先物 +{nq_chg:.1f}% (テック追い風)")
            elif nq_chg < -1.0:
                score -= 8
                signals.append(f"Nasdaq先物 {nq_chg:.1f}%")

        score = min(max(score, 0), 100)
        expected = "up" if score > 60 else "down" if score < 40 else "neutral"

        return {
            "score":              round(score, 1),
            "signals":            signals,
            "expected_direction": expected,
        }

    def score_all(
        self,
        stocks_df: pd.DataFrame,
        us_overnight: dict,
    ) -> pd.DataFrame:
        """
        Score all stocks using US overnight data.

        Args:
            stocks_df:   DataFrame with columns: code, sector33, name
            us_overnight: Output of USOverNightCollector

        Returns:
            DataFrame with: code, us_score, us_signals, expected_direction
        """
        YEN_SENSITIVE_SECTORS = {
            "輸送用機器", "電気機器", "機械", "精密機器", "ゴム製品"
        }

        records = []
        for _, row in stocks_df.iterrows():
            sector = row.get("sector33", "")
            is_yen_sensitive = sector in YEN_SENSITIVE_SECTORS
            result = self.score_ticker(
                sector_name=sector,
                us_overnight=us_overnight,
                yen_sensitive=is_yen_sensitive,
            )
            records.append({
                "code":               str(row.get("code", ""))[:4],
                "us_score":           result["score"],
                "us_signals":         " | ".join(result["signals"]),
                "us_expected_dir":    result["expected_direction"],
            })

        return pd.DataFrame(records)
