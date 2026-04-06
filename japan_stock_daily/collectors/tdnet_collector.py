"""TDnet collector using Yanoshin's free API (https://webapi.yanoshin.jp/tdnet/).

Fetches daily 適時開示 (timely disclosures) from TSE-listed companies.
No authentication required. Data syncs with TSE every few minutes.
"""

import time
import logging
import re
from datetime import date
from typing import List, Dict

import requests
import pandas as pd

from japan_stock_daily.config import TDNET_LIST_URL, TDNET_DETAIL_URL, REQUEST_DELAY

logger = logging.getLogger(__name__)

# Keyword → event category mapping (priority order: first match wins)
CATEGORY_PATTERNS: List[tuple] = [
    # Negative signals first (to avoid mis-classifying as positive)
    ("capital_raise",       ["第三者割当", "公募増資", "新株予約権"]),
    ("legal_risk",          ["訴訟", "課徴金", "行政処分", "不正", "横領", "調査委員会"]),
    ("earnings_rev_down",   ["業績予想の修正", "下方修正", "業績の修正"]),
    # Positive signals
    ("m_and_a",             ["TOB", "公開買付", "株式取得", "子会社化", "買収", "合併"]),
    ("alliance",            ["業務提携", "資本業務提携", "資本提携", "戦略的提携"]),
    ("earnings_rev_up",     ["上方修正", "業績予想の上方"]),
    ("special_dividend",    ["特別配当", "記念配当", "配当予想の修正"]),
    ("dividend",            ["配当"]),
    ("buyback",             ["自己株式取得", "自社株買い", "自己株式の取得"]),
    ("stock_split",         ["株式分割"]),
    ("restructuring",       ["事業再編", "会社分割", "事業譲渡", "事業売却"]),
    ("new_listing",         ["上場"]),
]

# Scores for each category (0–100 scale)
CATEGORY_SCORES: Dict[str, float] = {
    "m_and_a":          95,
    "earnings_rev_up":  80,
    "large_shareholder_increase": 75,  # shared with EDINET
    "buyback":          70,
    "special_dividend": 68,
    "alliance":         60,
    "dividend":         50,
    "stock_split":      45,
    "restructuring":    40,
    "new_listing":      35,
    # Negative
    "earnings_rev_down": 10,
    "capital_raise":    15,
    "legal_risk":       5,
    "other":            30,
}


def _classify(title: str) -> str:
    """Return the event category for a disclosure title string."""
    for category, keywords in CATEGORY_PATTERNS:
        if any(kw in title for kw in keywords):
            return category
    return "other"


class TDnetCollector:
    """
    Fetches 適時開示 from Yanoshin's free TDnet JSON API.

    Endpoint: https://webapi.yanoshin.jp/webapi/tdnet/list/{YYYYMMDD}.json2
    Returns JSON array with fields: code, company, time, title, url, etc.
    """

    BASE_LIST_URL = "https://webapi.yanoshin.jp/webapi/tdnet/list/{date}.json2"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NemesisBot/1.0"})

    def _fetch(self, url: str) -> list:
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        data = resp.json()
        # Yanoshin wraps response in {"items": [...]} or returns list directly
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "data", "list", "contents"):
                if key in data:
                    return data[key]
            # Some endpoints return {"tdnet": [...]}
            for v in data.values():
                if isinstance(v, list):
                    return v
        return []

    # ------------------------------------------------------------------ #
    #  Daily disclosure list                                               #
    # ------------------------------------------------------------------ #
    def get_daily_disclosures(self, target_date: date) -> pd.DataFrame:
        """
        Fetch all TDnet disclosures for target_date.

        Returns DataFrame columns:
            code, company, time, title, category, category_score, pdf_url
        """
        date_str = target_date.strftime("%Y%m%d")
        url = self.BASE_LIST_URL.format(date=date_str)
        logger.info("Fetching TDnet disclosures for %s from Yanoshin", date_str)

        try:
            items = self._fetch(url)
        except Exception as e:
            logger.error("TDnet fetch failed for %s: %s", date_str, e)
            return pd.DataFrame()

        if not items:
            logger.warning("No TDnet disclosures found for %s", date_str)
            return pd.DataFrame()

        records = []
        for item in items:
            title = item.get("title", item.get("Title", ""))
            category = _classify(title)
            records.append({
                "code":           _extract_code(item),
                "company":        item.get("company", item.get("Company", "")),
                "time":           item.get("time", item.get("Time", "")),
                "title":          title,
                "category":       category,
                "category_score": CATEGORY_SCORES.get(category, 30),
                "pdf_url":        item.get("url", item.get("Url", item.get("PDF_URL", ""))),
                "disclosure_id":  item.get("id", item.get("disclosure_no", "")),
            })

        df = pd.DataFrame(records)
        df = df[df["code"] != ""].copy()
        df = df.sort_values("category_score", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    #  Per-stock signals                                                   #
    # ------------------------------------------------------------------ #
    def get_signals_for_codes(self, target_date: date, codes: List[str]) -> pd.DataFrame:
        """Return TDnet disclosures filtered to a specific list of stock codes."""
        all_disc = self.get_daily_disclosures(target_date)
        if all_disc.empty:
            return pd.DataFrame()
        return all_disc[all_disc["code"].isin(codes)].reset_index(drop=True)

    def get_best_signal_per_code(self, target_date: date) -> pd.DataFrame:
        """
        Return the highest-scoring disclosure per stock code for target_date.
        Used for merging with the scoring system.
        """
        df = self.get_daily_disclosures(target_date)
        if df.empty:
            return pd.DataFrame()
        # Keep highest score per stock
        best = (df.sort_values("category_score", ascending=False)
                  .groupby("code", as_index=False)
                  .first())
        return best


def _extract_code(item: dict) -> str:
    """Extract 4-digit stock code from a Yanoshin item dict."""
    for key in ("code", "Code", "security_code", "stock_code"):
        val = item.get(key, "")
        if val:
            # Strip trailing zeros (e.g. "72030" → "7203")
            val = str(val).strip()
            if len(val) == 5 and val.endswith("0"):
                return val[:4]
            return val[:4]
    return ""
