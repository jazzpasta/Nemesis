"""Bank of Japan REST API collector for macroeconomic indicators.

BOJ Time-Series Data Search API:
  https://www.stat-search.boj.or.jp
  No authentication required.
  Returns 200,000+ time-series data points.
"""

import time
import json
import logging
import functools
from datetime import datetime, timedelta
from typing import Optional, Dict

import requests
import pandas as pd

from japan_stock_daily.config import REQUEST_DELAY

logger = logging.getLogger(__name__)

BOJ_API_BASE = "https://www.stat-search.boj.or.jp/ssi/api/v1"

# Key series for daily macro context
# Series codes from BOJ Time-Series Data Search
BOJ_KEY_SERIES = {
    "policy_rate":  "FM01'MAADMOCG@CO'",
    "jgb_10y":      "FM08'MAADMZGBMX/M'",
    "usd_jpy":      "FM08'MAADMFXGD@US'",
    "cpi_core":     "PR01'MAADMOCG@CO'",
    "m2":           "MD01'MAADMOCG@CO'",
}

# Simpler direct-download series IDs (used if API unavailable)
BOJ_SERIES_ALT = {
    "policy_rate": "SS23",   # Policy rate (BOJ uncollateralized overnight)
    "jgb_10y":     "SS24",   # 10-year JGB yield
    "usd_jpy":     "SS21",   # USD/JPY spot rate
    "cpi_core":    "SS35",   # CPI (excl. fresh food)
    "m2":          "SS31",   # M2
}

_cache: Dict[str, tuple] = {}  # series_name → (timestamp, value)
CACHE_TTL = 3600  # 1 hour


def _is_cached(key: str) -> bool:
    if key not in _cache:
        return False
    ts, _ = _cache[key]
    return (datetime.now() - ts).total_seconds() < CACHE_TTL


class BOJCollector:
    """
    Fetches macroeconomic time-series data from the Bank of Japan.
    Results are cached for 1 hour to avoid rate limiting.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NemesisBot/1.0"})

    def _get_series(self, series_code: str, n_periods: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch last n_periods of a BOJ time-series by series code.
        Uses the stat-search.boj.or.jp API.
        """
        cache_key = f"{series_code}_{n_periods}"
        if _is_cached(cache_key):
            return _cache[cache_key][1]

        try:
            # BOJ API endpoint for time-series data
            url = f"{BOJ_API_BASE}/series"
            params = {
                "seriesId": series_code,
                "obsDimensionId": "TIME_PERIOD",
                "startPeriod": (datetime.now() - timedelta(days=365)).strftime("%Y-%m"),
                "endPeriod": datetime.now().strftime("%Y-%m"),
                "output": "json",
            }
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            time.sleep(REQUEST_DELAY)

            obs = data.get("dataSets", [{}])[0].get("series", {})
            if not obs:
                return None

            # Parse observations
            records = []
            for period, vals in obs.items():
                for obs_key, obs_val in (vals.get("observations") or {}).items():
                    records.append({"period": period, "value": obs_val[0] if obs_val else None})
            df = pd.DataFrame(records).tail(n_periods)
            _cache[cache_key] = (datetime.now(), df)
            return df

        except Exception as e:
            logger.warning("BOJ API failed for %s: %s", series_code, e)
            return None

    def _get_via_direct_download(self, indicator: str) -> Optional[float]:
        """
        Fallback: fetch from BOJ stat-search HTML endpoint and parse latest value.
        """
        try:
            from bs4 import BeautifulSoup
            url = f"https://www.stat-search.boj.or.jp/ssi/mtshtml/{indicator}_m_en.html"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            # Find the last data row in the table
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                for row in reversed(rows):
                    cells = row.find_all("td")
                    if len(cells) >= 2:
                        val_text = cells[-1].get_text(strip=True).replace(",", "")
                        try:
                            return float(val_text)
                        except ValueError:
                            continue
        except Exception as e:
            logger.debug("BOJ direct download failed for %s: %s", indicator, e)
        return None

    # ------------------------------------------------------------------ #
    #  Public methods                                                      #
    # ------------------------------------------------------------------ #
    def get_policy_rate(self) -> Optional[float]:
        """Return latest BOJ policy rate (%)."""
        cache_key = "policy_rate"
        if _is_cached(cache_key):
            return _cache[cache_key][1]
        val = self._fetch_latest("policy_rate")
        _cache[cache_key] = (datetime.now(), val)
        return val

    def get_jgb_10y(self) -> Optional[float]:
        """Return latest 10-year JGB yield (%)."""
        cache_key = "jgb_10y"
        if _is_cached(cache_key):
            return _cache[cache_key][1]
        val = self._fetch_latest("jgb_10y")
        _cache[cache_key] = (datetime.now(), val)
        return val

    def get_usd_jpy(self) -> Optional[float]:
        """Return latest USD/JPY rate."""
        cache_key = "usd_jpy"
        if _is_cached(cache_key):
            return _cache[cache_key][1]
        val = self._fetch_latest("usd_jpy")
        _cache[cache_key] = (datetime.now(), val)
        return val

    def get_cpi(self) -> Optional[float]:
        """Return latest CPI core (excl. fresh food) YoY %."""
        cache_key = "cpi_core"
        if _is_cached(cache_key):
            return _cache[cache_key][1]
        val = self._fetch_latest("cpi_core")
        _cache[cache_key] = (datetime.now(), val)
        return val

    def _fetch_latest(self, indicator: str) -> Optional[float]:
        """Try API first, fallback to direct download."""
        # Try direct download (more reliable for BOJ)
        alt_code = BOJ_SERIES_ALT.get(indicator)
        if alt_code:
            val = self._get_via_direct_download(alt_code)
            if val is not None:
                return val
        logger.warning("Could not fetch BOJ indicator: %s", indicator)
        return None

    def get_all_indicators(self) -> Dict[str, Optional[float]]:
        """Return all key macro indicators as a dict."""
        return {
            "policy_rate":  self.get_policy_rate(),
            "jgb_10y":      self.get_jgb_10y(),
            "usd_jpy":      self.get_usd_jpy(),
            "cpi_core":     self.get_cpi(),
        }

    def get_rate_environment(self) -> str:
        """
        Classify current BOJ rate environment for sector scoring.
        Returns: 'hiking' | 'stable' | 'easing'
        """
        rate = self.get_policy_rate()
        if rate is None:
            return "unknown"
        if rate >= 0.5:
            return "hiking"
        if rate <= 0.0:
            return "easing"
        return "stable"
