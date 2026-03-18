"""Multi-factor strategy: wraps DailyScorer for backtesting."""

import logging
from datetime import date

import pandas as pd

from backtest.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class MultiFactorStrategy(BaseStrategy):
    """
    Wraps the DailyScorer for backtesting in multi-factor mode.

    Uses pre-computed scored DataFrames (from DailyScorer.score_all)
    rather than fetching data again, to allow fast iteration.
    """

    def __init__(self, top_n: int = 20, score_cache: dict = None):
        """
        Args:
            top_n:        Number of top stocks to select each day
            score_cache:  Dict of date → pre-computed scored DataFrame
                         (pass None to compute on the fly)
        """
        super().__init__(top_n=top_n, hold_days=1)
        self.score_cache = score_cache or {}

    def generate_signals(self, target_date: date, data: dict) -> pd.DataFrame:
        """
        Return top_n stocks by composite score for target_date.

        If pre-computed scores are available, uses those.
        Otherwise, delegates to DailyScorer.
        """
        if target_date in self.score_cache:
            df = self.score_cache[target_date]
            return df.head(self.top_n)[["code", "composite_score"]].rename(
                columns={"composite_score": "score"}
            )

        # Compute on the fly
        scorer = data.get("scorer")
        if scorer is None:
            from japan_stock_daily.recommender.scorer import DailyScorer
            scorer = DailyScorer()
        scored = scorer.score_all(target_date, universe=data.get("universe"))
        self.score_cache[target_date] = scored
        return scored.head(self.top_n)[["code", "composite_score"]].rename(
            columns={"composite_score": "score"}
        )
