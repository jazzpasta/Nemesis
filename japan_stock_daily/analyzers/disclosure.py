"""Disclosure analyzer for TDnet + EDINET signals.

Converts raw disclosure events into 0-100 scores per stock.
"""

import logging
from typing import List, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Event category → base score (0-100)
CATEGORY_SCORES: Dict[str, float] = {
    # Strong positive
    "m_and_a":                          95,
    "earnings_rev_up":                  80,
    "large_shareholder_increase":       75,
    "large_shareholder_new":            72,
    "buyback":                          70,
    "special_dividend":                 68,
    # Moderate positive
    "alliance":                         60,
    "dividend":                         50,
    "stock_split":                      45,
    "restructuring":                    42,
    # Weak positive
    "new_listing":                      35,
    "other":                            30,
    # Negative (below neutral 50 → penalizes score)
    "earnings_rev_down":                10,
    "capital_raise":                    15,
    "legal_risk":                        5,
    "large_shareholder_decrease":       20,
    "management_change":                30,
}

# Weight modifier for EDINET vs TDnet signals (EDINET often more actionable)
EDINET_WEIGHT = 1.1
TDNET_WEIGHT  = 1.0

# Neutral baseline (stocks with no events get this)
NEUTRAL_SCORE = 30


class DisclosureAnalyzer:
    """
    Converts TDnet and EDINET events into per-stock disclosure scores.

    Scoring logic:
    - Multiple events for same stock: take maximum positive + penalize negatives
    - EDINET large shareholder increase with >1% change: +10% bonus
    - Both TDnet and EDINET positive for same stock: +5 synergy bonus
    """

    def score_ticker(
        self,
        code: str,
        tdnet_events: List[dict],
        edinet_events: List[dict],
    ) -> dict:
        """
        Score a single stock based on its disclosure events.

        Args:
            code:          Stock code
            tdnet_events:  List of TDnet event dicts (category, title, ...)
            edinet_events: List of EDINET event dicts (event_category, direction, ...)

        Returns:
            dict with: score, category, signals, has_negative
        """
        if not tdnet_events and not edinet_events:
            return {"code": code, "score": NEUTRAL_SCORE, "category": "none",
                    "signals": [], "has_negative": False}

        all_scores = []
        signals = []
        has_negative = False
        best_category = "other"

        # --- TDnet events ---
        for event in tdnet_events:
            cat = event.get("category", "other")
            base = CATEGORY_SCORES.get(cat, 30)
            adj = base * TDNET_WEIGHT

            if base < 20:
                has_negative = True
                all_scores.append(adj * -1)  # Penalize
                signals.append(f"[TDnet-] {event.get('title', cat)}")
            else:
                all_scores.append(adj)
                signals.append(f"[TDnet+] {event.get('title', cat)}")
                if base > CATEGORY_SCORES.get(best_category, 0):
                    best_category = cat

        # --- EDINET events ---
        for event in edinet_events:
            cat = event.get("event_category", "other")
            direction = event.get("direction", "")

            # Map EDINET categories to scoring categories
            if cat == "large_shareholder":
                if direction == "increase":
                    cat = "large_shareholder_increase"
                elif direction == "new":
                    cat = "large_shareholder_new"
                elif direction == "decrease":
                    cat = "large_shareholder_decrease"

            elif cat == "m_and_a":
                cat = "m_and_a"
            elif cat == "capital_raise":
                cat = "capital_raise"
            elif cat == "legal":
                cat = "legal_risk"

            base = CATEGORY_SCORES.get(cat, 30)
            adj = base * EDINET_WEIGHT

            # Bonus for large ownership change
            change = event.get("ownership_change")
            if change and direction == "increase":
                try:
                    if float(change) >= 2.0:
                        adj *= 1.15
                        signals.append(f"[EDINET+] {event.get('filer_name', '')} 持株比率 +{change}% (大幅増加)")
                    else:
                        signals.append(f"[EDINET+] {event.get('filer_name', '')} 大量保有({direction})")
                except (ValueError, TypeError):
                    signals.append(f"[EDINET+] {event.get('filer_name', '')} {cat}")
            else:
                signals.append(f"[EDINET] {event.get('filer_name', '')} {cat}")

            if base < 20:
                has_negative = True
                all_scores.append(adj * -1)
            else:
                all_scores.append(adj)
                if base > CATEGORY_SCORES.get(best_category, 0):
                    best_category = cat

        if not all_scores:
            final_score = NEUTRAL_SCORE
        else:
            # Take the highest positive score; reduce by negatives
            positive_scores = [s for s in all_scores if s > 0]
            negative_scores = [s for s in all_scores if s < 0]
            max_positive = max(positive_scores) if positive_scores else NEUTRAL_SCORE
            neg_penalty  = abs(sum(negative_scores)) * 0.5 if negative_scores else 0

            # Synergy bonus: both TDnet and EDINET positive
            if tdnet_events and edinet_events and max_positive > 50:
                max_positive = min(max_positive + 5, 100)
                signals.append("[Synergy] TDnet + EDINET 両方でポジティブ")

            final_score = max(max_positive - neg_penalty, 5)
            final_score = min(final_score, 100)

        return {
            "code":         code,
            "score":        round(final_score, 1),
            "category":     best_category,
            "signals":      signals,
            "has_negative": has_negative,
        }

    # ------------------------------------------------------------------ #
    #  Batch scoring                                                      #
    # ------------------------------------------------------------------ #
    def score_all(
        self,
        tdnet_df: pd.DataFrame,
        edinet_df: pd.DataFrame,
        universe_codes: List[str],
    ) -> pd.DataFrame:
        """
        Score all stocks in universe and return sorted DataFrame.

        Args:
            tdnet_df:       Output of TDnetCollector.get_daily_disclosures()
            edinet_df:      Output of EdinetCollector.get_daily_signals()
            universe_codes: Full list of stock codes to score

        Returns:
            DataFrame with columns: code, disc_score, disc_category, disc_signals, has_negative
        """
        # Build per-code event lists
        tdnet_by_code: Dict[str, list] = {}
        if not tdnet_df.empty:
            for _, row in tdnet_df.iterrows():
                code = str(row.get("code", ""))[:4]
                tdnet_by_code.setdefault(code, []).append(row.to_dict())

        edinet_by_code: Dict[str, list] = {}
        if not edinet_df.empty:
            for _, row in edinet_df.iterrows():
                code = str(row.get("code", ""))[:4]
                edinet_by_code.setdefault(code, []).append(row.to_dict())

        records = []
        for code in universe_codes:
            result = self.score_ticker(
                code=code,
                tdnet_events=tdnet_by_code.get(code, []),
                edinet_events=edinet_by_code.get(code, []),
            )
            records.append({
                "code":           code,
                "disc_score":     result["score"],
                "disc_category":  result["category"],
                "disc_signals":   " | ".join(result["signals"]),
                "has_negative":   result["has_negative"],
            })

        df = pd.DataFrame(records)
        return df.sort_values("disc_score", ascending=False).reset_index(drop=True)
