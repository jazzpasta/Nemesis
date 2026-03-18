"""METI Industrial Production Index (IIP) collector.

Uses DBnomics API to fetch METI industrial production data.
Data is monthly; cached locally to avoid repeated downloads.
"""

import os
import json
import logging
import time
from datetime import datetime, date
from typing import Dict, Optional

import requests
import pandas as pd

from japan_stock_daily.config import DBNOMICS_BASE_URL, CACHE_DIR, REQUEST_DELAY

logger = logging.getLogger(__name__)

# Local cache file for monthly IIP data
IIP_CACHE_FILE = os.path.join(CACHE_DIR, "meti_iip.json")

# TSE-33 sector → METI IIP industry code mapping
# METI uses Japanese Standard Industrial Classification
SECTOR_TO_IIP = {
    "食料品":           "C09",   # Food
    "繊維製品":         "C10",   # Textiles
    "化学":             "C16",   # Chemicals
    "医薬品":           "C17",   # Pharmaceuticals
    "石油・石炭製品":   "C19",   # Petroleum/coal products
    "鉄鋼":             "C24",   # Iron and steel
    "非鉄金属":         "C25",   # Non-ferrous metals
    "機械":             "C28",   # General machinery
    "電気機器":         "C27",   # Electrical machinery
    "輸送用機器":       "C31",   # Transportation equipment
    "精密機器":         "C30",   # Precision instruments
}


class METICollector:
    """
    Fetches METI Indices of Industrial Production (IIP) via DBnomics.
    Caches results monthly since data is updated monthly.
    """

    DBNOMICS_URL = f"{DBNOMICS_BASE_URL}/series/METI/IIP"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NemesisBot/1.0"})
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _load_cache(self) -> Optional[dict]:
        if not os.path.exists(IIP_CACHE_FILE):
            return None
        try:
            with open(IIP_CACHE_FILE) as f:
                data = json.load(f)
            # Invalidate if older than 25 days
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if (datetime.now() - cached_at).days > 25:
                return None
            return data
        except Exception:
            return None

    def _save_cache(self, data: dict):
        data["cached_at"] = datetime.now().isoformat()
        with open(IIP_CACHE_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_iip_by_industry(self) -> Dict[str, dict]:
        """
        Return latest IIP index and MoM change by industry code.

        Returns: {industry_code: {"latest": 102.5, "mom_pct": 1.2, "yoy_pct": 3.4}}
        """
        cached = self._load_cache()
        if cached:
            logger.debug("Using cached METI IIP data from %s", cached.get("cached_at"))
            return cached.get("iip", {})

        logger.info("Fetching METI IIP from DBnomics")
        result = {}

        try:
            resp = self.session.get(
                self.DBNOMICS_URL,
                params={"limit": 1000, "offset": 0},
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            time.sleep(REQUEST_DELAY)

            series_list = data.get("series", {}).get("docs", [])
            for series in series_list:
                series_code = series.get("series_code", "")
                periods = series.get("period", [])
                values = series.get("value", [])

                if not periods or not values:
                    continue

                # Get last 13 months of data
                pairs = [(p, v) for p, v in zip(periods, values)
                         if v is not None][-13:]
                if len(pairs) < 2:
                    continue

                latest_val = pairs[-1][1]
                prev_val   = pairs[-2][1]
                year_ago_val = pairs[0][1] if len(pairs) >= 13 else None

                result[series_code] = {
                    "latest":   latest_val,
                    "period":   pairs[-1][0],
                    "mom_pct":  round((latest_val - prev_val) / prev_val * 100, 2)
                                if prev_val else None,
                    "yoy_pct":  round((latest_val - year_ago_val) / year_ago_val * 100, 2)
                                if year_ago_val else None,
                }

        except Exception as e:
            logger.warning("DBnomics METI fetch failed: %s", e)

        if result:
            self._save_cache({"iip": result})
        return result

    def get_sector_iip(self, sector_name: str) -> Optional[dict]:
        """Return IIP data for a given TSE-33 sector name."""
        iip_code = SECTOR_TO_IIP.get(sector_name)
        if not iip_code:
            return None
        all_data = self.get_iip_by_industry()
        return all_data.get(iip_code)

    def get_sector_trend(self, sector_name: str) -> str:
        """
        Return trend direction for a sector: 'expanding' | 'contracting' | 'stable' | 'unknown'
        """
        data = self.get_sector_iip(sector_name)
        if data is None:
            return "unknown"
        mom = data.get("mom_pct", 0) or 0
        yoy = data.get("yoy_pct", 0) or 0
        if mom > 1.0 and yoy > 3.0:
            return "expanding"
        if mom < -1.0 and yoy < -3.0:
            return "contracting"
        return "stable"
