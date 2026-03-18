"""e-Stats API collector for Japanese government statistics.

Official portal: https://www.e-stat.go.jp/api/en
Requires free registration and Application ID.
"""

import time
import logging
from datetime import datetime
from typing import Optional, Dict

import requests
import pandas as pd

from japan_stock_daily.config import ESTATS_APP_ID, ESTATS_BASE_URL, ESTATS_SERIES, REQUEST_DELAY

logger = logging.getLogger(__name__)

_cache: Dict[str, tuple] = {}
CACHE_TTL = 86400  # 24 hours (government stats update monthly)


def _is_cached(key: str) -> bool:
    if key not in _cache:
        return False
    ts, _ = _cache[key]
    return (datetime.now() - ts).total_seconds() < CACHE_TTL


class EStatsCollector:
    """
    Fetches Japanese government statistics from e-Stat API.

    Key indicators used for sector macro scoring:
    - CPI (Consumer Price Index) → consumption environment
    - Unemployment rate → labor market health
    - Retail sales → consumer sector tailwind
    - Industrial production index → manufacturing sector
    """

    def __init__(self, app_id: str = ESTATS_APP_ID):
        self.app_id = app_id
        self.session = requests.Session()

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{ESTATS_BASE_URL}/{endpoint}"
        params = params or {}
        params["appId"] = self.app_id
        params["lang"] = "J"  # Japanese language for category names
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()

    def get_stats_data(self, stats_data_id: str, n_latest: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch the latest n_latest data points for a stats series.

        Returns DataFrame with columns: time_period, value, unit, category
        """
        cache_key = f"{stats_data_id}_{n_latest}"
        if _is_cached(cache_key):
            return _cache[cache_key][1]

        if not self.app_id:
            logger.warning("ESTATS_APP_ID not set; skipping e-Stats fetch")
            return None

        try:
            data = self._get("getStatsData", {
                "statsDataId": stats_data_id,
                "metaGetFlg": "N",
                "cntGetFlg": "N",
                "explanationGetFlg": "N",
                "annotationGetFlg": "N",
                "sectionHeaderFlg": "1",
            })

            stat_data = (data.get("GET_STATS_DATA", {})
                             .get("STATISTICAL_DATA", {})
                             .get("DATA_INF", {})
                             .get("VALUE", []))

            if not stat_data:
                return None

            if isinstance(stat_data, dict):
                stat_data = [stat_data]

            records = []
            for item in stat_data:
                records.append({
                    "time_period": item.get("@time", ""),
                    "value":       _safe_float(item.get("$", "")),
                    "area":        item.get("@area", ""),
                    "cat01":       item.get("@cat01", ""),
                })

            df = pd.DataFrame(records)
            df = df.dropna(subset=["value"])
            df = df.sort_values("time_period").tail(n_latest)
            _cache[cache_key] = (datetime.now(), df)
            return df

        except Exception as e:
            logger.warning("e-Stats fetch failed for %s: %s", stats_data_id, e)
            return None

    def get_latest_value(self, stats_data_id: str) -> Optional[float]:
        """Return the single latest value for a stats series."""
        df = self.get_stats_data(stats_data_id, n_latest=1)
        if df is None or df.empty:
            return None
        return df["value"].iloc[-1]

    def get_yoy_change(self, stats_data_id: str) -> Optional[float]:
        """Return year-over-year change for a stats series."""
        df = self.get_stats_data(stats_data_id, n_latest=13)
        if df is None or len(df) < 13:
            return None
        latest = df["value"].iloc[-1]
        year_ago = df["value"].iloc[0]
        if year_ago == 0:
            return None
        return round((latest - year_ago) / year_ago * 100, 2)

    def get_all_indicators(self) -> Dict[str, Optional[float]]:
        """Return all key e-Stats macro indicators."""
        return {
            "cpi_latest":           self.get_latest_value(ESTATS_SERIES["cpi"]),
            "unemployment_latest":  self.get_latest_value(ESTATS_SERIES["unemployment"]),
            "retail_sales_yoy":     self.get_yoy_change(ESTATS_SERIES["retail_sales"]),
            "iip_yoy":              self.get_yoy_change(ESTATS_SERIES["industrial_production"]),
        }


def _safe_float(val) -> Optional[float]:
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, TypeError):
        return None
