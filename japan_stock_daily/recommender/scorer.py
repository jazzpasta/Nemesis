"""Daily stock scorer: combines all signals into composite recommendation.

Two strategies:
- MultiFactor: normal days (TDnet events + supply/demand + macro + US overnight)
- ShockRecovery: days after US market drops >1% (focus on oversold JP stocks)

Strategy selection is automatic based on US overnight performance.
"""

import logging
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np

from japan_stock_daily.config import SCORE_WEIGHTS, SHOCK_RECOVERY_THRESHOLD
from japan_stock_daily.collectors.jquants_collector import JQuantsCollector
from japan_stock_daily.collectors.edinet_collector import EdinetCollector
from japan_stock_daily.collectors.tdnet_collector import TDnetCollector
from japan_stock_daily.collectors.boj_collector import BOJCollector
from japan_stock_daily.collectors.estats_collector import EStatsCollector
from japan_stock_daily.collectors.meti_collector import METICollector
from japan_stock_daily.collectors.us_overnight_collector import USOverNightCollector
from japan_stock_daily.analyzers.supply_demand import SupplyDemandAnalyzer
from japan_stock_daily.analyzers.disclosure import DisclosureAnalyzer
from japan_stock_daily.analyzers.macro import MacroAnalyzer, USOverNightAnalyzer

logger = logging.getLogger(__name__)


class DailyScorer:
    """
    Orchestrates all data collection and scoring for a given trading date.

    Usage:
        scorer = DailyScorer()
        recommendations = scorer.score_all(date(2025, 3, 18))
        top20 = scorer.get_top_recommendations(recommendations, n=20)
    """

    def __init__(self):
        self.jquants    = JQuantsCollector()
        self.edinet     = EdinetCollector()
        self.tdnet      = TDnetCollector()
        self.boj        = BOJCollector()
        self.estats     = EStatsCollector()
        self.meti       = METICollector()
        self.us_overnight = USOverNightCollector()

        self.sd_analyzer   = SupplyDemandAnalyzer()
        self.disc_analyzer = DisclosureAnalyzer()
        self.macro_analyzer = MacroAnalyzer()
        self.us_analyzer   = USOverNightAnalyzer()

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def score_all(self, target_date: date, universe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run full pipeline for target_date.

        Args:
            target_date: Date to generate recommendations for
            universe:    Optional DataFrame of stocks to score (code, name, sector33)
                         If None, fetches from J-Quants listed companies.

        Returns:
            DataFrame sorted by composite_score descending with all signal columns.
        """
        logger.info("=== Daily Scoring: %s ===", target_date)

        # --- Step 1: Determine strategy (before any JP market data needed) ---
        logger.info("Fetching US overnight data...")
        us_data = self._get_us_overnight_safe(target_date)
        strategy = self._select_strategy(us_data)
        logger.info("Strategy selected: %s (US overnight: %s%%)",
                    strategy,
                    us_data.get("sp500_futures_chg", "N/A"))

        # --- Step 2: Universe of stocks ---
        if universe is None:
            logger.info("Fetching J-Quants listed companies...")
            universe = self.jquants.get_listed_companies()
        codes = universe["code"].astype(str).str[:4].tolist()
        logger.info("Universe: %d stocks", len(codes))

        # --- Step 3: Fetch all data for target_date ---
        logger.info("Fetching J-Quants daily quotes...")
        quotes_today = self.jquants.get_daily_quotes(target_date)

        # For 30-day historical quotes (needed for volume/RS analysis)
        quotes_30d_by_code = self._fetch_historical_quotes_batch(
            codes[:200],  # Limit for speed; extend as needed
            start=target_date - timedelta(days=40),
            end=target_date,
        )

        # TOPIX for relative strength
        topix_df = self._get_topix_30d(target_date)

        logger.info("Fetching J-Quants Pro supply/demand data...")
        margin_df    = self._safe_fetch(self.jquants.get_weekly_margin_interest, target_date)
        short_df     = self._safe_fetch(self.jquants.get_short_selling_positions, target_date)
        breakdown_df = self._safe_fetch(self.jquants.get_breakdown, target_date)

        logger.info("Fetching TDnet disclosures...")
        tdnet_df = self._safe_fetch(self.tdnet.get_daily_disclosures, target_date)

        logger.info("Fetching EDINET disclosures...")
        edinet_df = self._safe_fetch(self.edinet.get_daily_signals, target_date)

        logger.info("Fetching macro data...")
        boj_data   = self.boj.get_all_indicators()
        boj_data["rate_environment"] = self.boj.get_rate_environment()
        estats_data = self.estats.get_all_indicators()
        macro_scores = self.macro_analyzer.score_all_sectors(
            sector_names=universe["sector33"].dropna().unique().tolist(),
            boj_data=boj_data,
            estats_data=estats_data,
            meti_collector=self.meti,
        )

        # --- Step 4: Score each stock ---
        logger.info("Computing disclosure scores...")
        disc_scores = self.disc_analyzer.score_all(
            tdnet_df=tdnet_df if tdnet_df is not None else pd.DataFrame(),
            edinet_df=edinet_df if edinet_df is not None else pd.DataFrame(),
            universe_codes=codes,
        )

        logger.info("Computing supply/demand scores...")
        sd_scores = self.sd_analyzer.score_all(
            codes=codes,
            quotes_by_code=quotes_30d_by_code,
            topix_quotes=topix_df,
            margin_df=margin_df if margin_df is not None else pd.DataFrame(),
            short_df=short_df if short_df is not None else pd.DataFrame(),
            breakdown_df=breakdown_df if breakdown_df is not None else pd.DataFrame(),
        )

        logger.info("Computing US overnight scores...")
        us_scores = self.us_analyzer.score_all(
            stocks_df=universe[["code", "sector33", "name"]],
            us_overnight=us_data,
        )

        # --- Step 5: Compute price momentum scores ---
        momentum_scores = self._compute_momentum_scores(codes, quotes_30d_by_code, topix_df)

        # --- Step 6: Add macro scores to stocks ---
        universe["macro_score"] = universe["sector33"].map(
            lambda s: macro_scores.get(s, {}).get("score", 50)
        )
        universe["macro_signals"] = universe["sector33"].map(
            lambda s: " | ".join(macro_scores.get(s, {}).get("signals", []))
        )

        # --- Step 7: Merge all scores ---
        result = universe[["code", "name", "sector33", "market",
                            "macro_score", "macro_signals"]].copy()
        result["code"] = result["code"].astype(str).str[:4]

        result = result.merge(disc_scores[["code", "disc_score", "disc_category",
                                            "disc_signals", "has_negative"]],
                               on="code", how="left")
        result = result.merge(sd_scores[["code", "sd_score", "sd_signals"]],
                               on="code", how="left")
        result = result.merge(us_scores[["code", "us_score", "us_signals", "us_expected_dir"]],
                               on="code", how="left")
        result = result.merge(momentum_scores[["code", "momentum_score"]],
                               on="code", how="left")

        # Fill missing scores with neutral
        for col, neutral in [("disc_score", 30), ("sd_score", 30),
                               ("us_score", 50), ("macro_score", 50), ("momentum_score", 40)]:
            result[col] = result[col].fillna(neutral)

        # --- Step 8: Composite score ---
        if strategy == "shock_recovery":
            result["composite_score"] = self._compute_shock_recovery_score(result, us_data)
            result["strategy"] = "shock_recovery"
        else:
            w = SCORE_WEIGHTS
            result["composite_score"] = (
                result["disc_score"]     * w["disclosure"]     +
                result["sd_score"]       * w["supply_demand"]  +
                result["us_score"]       * w["us_overnight"]   +
                result["momentum_score"] * w["price_momentum"] +
                result["macro_score"]    * w["macro"]
            )
            result["strategy"] = "multi_factor"

        # Penalize stocks with negative disclosures
        result.loc[result["has_negative"] == True, "composite_score"] *= 0.7

        # --- Step 9: Add today's price info ---
        if quotes_today is not None and not quotes_today.empty:
            price_cols = quotes_today[["code", "open", "high", "low", "close", "volume"]].copy()
            price_cols["code"] = price_cols["code"].astype(str).str[:4]
            result = result.merge(price_cols, on="code", how="left")

        result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)
        result["rank"] = result.index + 1
        result["date"] = target_date

        logger.info("Scoring complete. Top 5:")
        for _, row in result.head(5).iterrows():
            logger.info("  #%d %s (%s) %.1f [%s]",
                        row["rank"], row["code"], row.get("name", ""),
                        row["composite_score"], row.get("disc_category", ""))

        return result

    def get_top_recommendations(self, scored_df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """Return top N recommendations with all scores and signal rationale."""
        return scored_df.head(n)[["rank", "code", "name", "sector33",
                                   "composite_score", "disc_score", "sd_score",
                                   "us_score", "momentum_score", "macro_score",
                                   "disc_category", "disc_signals", "sd_signals",
                                   "us_signals", "us_expected_dir", "macro_signals",
                                   "strategy", "date"]].copy()

    # ------------------------------------------------------------------ #
    #  Strategy selection                                                  #
    # ------------------------------------------------------------------ #
    def _select_strategy(self, us_data: dict) -> str:
        """Select strategy based on overnight US market performance."""
        sp_chg = us_data.get("sp500_futures_chg", 0) or 0
        vix = us_data.get("vix", 20) or 20
        # Shock recovery: US dropped significantly OR VIX spiked
        if sp_chg < SHOCK_RECOVERY_THRESHOLD or vix > 28:
            return "shock_recovery"
        return "multi_factor"

    def _compute_shock_recovery_score(self, result: pd.DataFrame, us_data: dict) -> pd.Series:
        """
        Compute shock recovery scores.
        Prioritizes stocks that:
        1. Dropped the most yesterday (large negative momentum = recovery fuel)
        2. Have NO negative disclosure (not company-specific bad news)
        3. Have high short ratio (short covering fuel)
        4. Are in sectors where US sector ETF is recovering
        """
        # Invert momentum: bigger recent drop → higher recovery potential
        # (momentum_score < 40 means stock underperformed)
        recovery_potential = (50 - result["momentum_score"]).clip(lower=0) * 1.5

        # Boost for high short ratio (from sd_score components)
        short_boost = result["sd_score"].clip(upper=40) * 0.5

        # Sector recovery from US ETF
        sector_boost = result["us_score"].apply(lambda s: max(s - 50, 0))

        # Penalize any negative disclosure
        neg_penalty = result["has_negative"].fillna(False).map({True: 0.3, False: 1.0})

        score = (recovery_potential * 0.4 +
                 short_boost * 0.25 +
                 sector_boost * 0.25 +
                 result["macro_score"] * 0.10) * neg_penalty

        return score.clip(0, 100).round(1)

    # ------------------------------------------------------------------ #
    #  Price momentum scoring                                             #
    # ------------------------------------------------------------------ #
    def _compute_momentum_scores(
        self,
        codes: list,
        quotes_by_code: dict,
        topix_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute 5-day and 20-day price momentum scores."""
        records = []
        for code in codes:
            df = quotes_by_code.get(code, pd.DataFrame())
            score = 40  # Neutral default
            if not df.empty and len(df) >= 5:
                try:
                    # 5-day return
                    ret_5d = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100
                    # 20-day return
                    ret_20d = None
                    if len(df) >= 20:
                        ret_20d = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100

                    # Price above 5-day MA
                    ma5 = df["close"].tail(5).mean()
                    above_ma5 = df["close"].iloc[-1] > ma5

                    score = 40
                    if ret_5d > 5:
                        score += 25
                    elif ret_5d > 2:
                        score += 15
                    elif ret_5d > 0:
                        score += 8
                    elif ret_5d < -5:
                        score -= 15
                    elif ret_5d < -2:
                        score -= 8

                    if ret_20d is not None:
                        if ret_20d > 10:
                            score += 15
                        elif ret_20d > 5:
                            score += 8
                        elif ret_20d < -10:
                            score -= 12

                    if above_ma5:
                        score += 5

                    score = min(max(score, 0), 100)
                except Exception:
                    pass
            records.append({"code": code, "momentum_score": score})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    #  Data fetching helpers                                               #
    # ------------------------------------------------------------------ #
    def _get_us_overnight_safe(self, target_date: date) -> dict:
        """Fetch US overnight data; return neutral dict on failure."""
        try:
            # For today: use live data
            from datetime import date as date_cls
            if target_date >= date_cls.today():
                return self.us_overnight.get_us_overnight_performance()
            else:
                # Historical backtest mode
                return self.us_overnight.get_historical_overnight(target_date)
        except Exception as e:
            logger.warning("US overnight fetch failed: %s", e)
            return {
                "sp500_futures_chg": None,
                "nasdaq_futures_chg": None,
                "vix": None,
                "usd_jpy_chg": None,
                "market_bias": "neutral",
                "yen_direction": "stable",
                "sector_signals": {},
                "sp500_overnight": None,
            }

    def _safe_fetch(self, fn, *args, **kwargs):
        """Call a data fetcher, returning None on failure."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning("Data fetch failed (%s): %s", fn.__name__, e)
            return None

    def _fetch_historical_quotes_batch(
        self,
        codes: list,
        start: date,
        end: date,
    ) -> dict:
        """Fetch 30-day historical quotes for a batch of codes."""
        result = {}
        # For large universes, fetch by date and pivot
        try:
            # Fetch all daily quotes for each date in range
            current = start
            all_rows = []
            while current <= end:
                try:
                    daily = self.jquants.get_daily_quotes(current)
                    if daily is not None and not daily.empty:
                        all_rows.append(daily)
                except Exception:
                    pass
                current += timedelta(days=1)

            if all_rows:
                full_df = pd.concat(all_rows, ignore_index=True)
                full_df["code"] = full_df["code"].astype(str).str[:4]
                for code in codes:
                    code_df = full_df[full_df["code"] == code].sort_values("date")
                    result[code] = code_df
        except Exception as e:
            logger.warning("Batch historical fetch failed: %s", e)
        return result

    def _get_topix_30d(self, target_date: date) -> pd.DataFrame:
        """Fetch TOPIX index data for last 30 days."""
        try:
            all_rows = []
            current = target_date - timedelta(days=40)
            while current <= target_date:
                try:
                    indices = self.jquants.get_indices(current)
                    if indices is not None and not indices.empty:
                        topix_row = indices[indices.get("Index", indices.columns[0]) == "TOPIX"]
                        if not topix_row.empty:
                            all_rows.append(topix_row)
                except Exception:
                    pass
                current += timedelta(days=1)
            if all_rows:
                df = pd.concat(all_rows, ignore_index=True)
                # Normalize column names
                for close_col in ("Close", "close", "IndexClose"):
                    if close_col in df.columns:
                        df = df.rename(columns={close_col: "close"})
                        break
                return df
        except Exception as e:
            logger.debug("TOPIX fetch failed: %s", e)
        return pd.DataFrame()
