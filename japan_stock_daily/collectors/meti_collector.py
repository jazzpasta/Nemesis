"""METI Industrial Production Index (IIP) and Tertiary Industry Activity Index collector.

Two monthly datasets:
  IIP  (鉱工業指数)            - mining + manufacturing production by sector
  3AI  (第三次産業活動指数)     - service sector activity by sector

Both via DBnomics METI dataset or e-Stats API.  Data is monthly; cached locally.
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

IIP_CACHE_FILE = os.path.join(CACHE_DIR, "meti_iip.json")
TAI_CACHE_FILE = os.path.join(CACHE_DIR, "meti_3ai.json")

# TSE-33 sector → METI IIP industry code mapping
SECTOR_TO_IIP = {
    "食料品":           "C09",
    "繊維製品":         "C10",
    "化学":             "C16",
    "医薬品":           "C17",
    "石油・石炭製品":   "C19",
    "鉄鋼":             "C24",
    "非鉄金属":         "C25",
    "機械":             "C28",
    "電気機器":         "C27",
    "輸送用機器":       "C31",
    "精密機器":         "C30",
}

# TSE-33 sector → 第三次産業活動指数 category code
SECTOR_TO_3AI = {
    "情報・通信業":     "H",
    "卸売業":           "I50",
    "小売業":           "I51",
    "銀行業":           "J61",
    "保険業":           "J66",
    "証券・商品先物":   "J63",
    "不動産業":         "L",
    "陸運業":           "G53",
    "海運業":           "G55",
    "空運業":           "G56",
    "倉庫・運輸関連業": "G57",
    "サービス業":       "S",
    "電力・ガス業":     "E",
}


class METICollector:
    """
    Fetches:
      - 鉱工業指数 (IIP) — monthly mining + manufacturing production
      - 第三次産業活動指数 (3AI) — monthly service sector activity

    Both via DBnomics METI dataset.  Results cached for 25 days.
    """

    IIP_URL = f"{DBNOMICS_BASE_URL}/series/METI/IIP"
    # 第三次産業活動指数 is in METI dataset "TSAI" on DBnomics
    TAI_URL = f"{DBNOMICS_BASE_URL}/series/METI/TSAI"
    # Fallback: direct METI download URL (Excel/CSV)
    TAI_METI_URL = "https://www.meti.go.jp/statistics/tyo/sanzi/result-2.html"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NemesisBot/1.0"})
        os.makedirs(CACHE_DIR, exist_ok=True)

    def _load_cache(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if (datetime.now() - cached_at).days > 25:
                return None
            return data
        except Exception:
            return None

    def _save_cache(self, data: dict, path: str):
        data["cached_at"] = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_iip_by_industry(self) -> Dict[str, dict]:
        """
        Return latest IIP (鉱工業指数) by industry code.

        Returns: {industry_code: {"latest": 102.5, "mom_pct": 1.2, "yoy_pct": 3.4}}
        """
        cached = self._load_cache(IIP_CACHE_FILE)
        if cached:
            logger.debug("Using cached IIP from %s", cached.get("cached_at"))
            return cached.get("iip", {})

        result = self._fetch_dbnomics(self.IIP_URL)
        if result:
            self._save_cache({"iip": result}, IIP_CACHE_FILE)
        return result

    def get_tertiary_activity_index(self) -> Dict[str, dict]:
        """
        Return latest 第三次産業活動指数 (Tertiary Industry Activity Index) by sector.

        Covers service sectors not in IIP: finance, real estate, ICT services,
        transport, wholesale/retail, healthcare, education etc.

        Returns: {sector_code: {"latest": 104.2, "mom_pct": 0.5, "yoy_pct": 2.1}}
        """
        cached = self._load_cache(TAI_CACHE_FILE)
        if cached:
            logger.debug("Using cached 3AI from %s", cached.get("cached_at"))
            return cached.get("tai", {})

        result = self._fetch_dbnomics(self.TAI_URL)
        if not result:
            # Fallback: try e-Stats API for tertiary industry index
            result = self._fetch_estats_3ai()

        if result:
            self._save_cache({"tai": result}, TAI_CACHE_FILE)
        return result

    def _fetch_dbnomics(self, url: str) -> Dict[str, dict]:
        """Generic DBnomics series fetch; returns {series_code: activity_dict}."""
        result = {}
        try:
            resp = self.session.get(url, params={"limit": 1000}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            time.sleep(REQUEST_DELAY)

            for series in data.get("series", {}).get("docs", []):
                code   = series.get("series_code", "")
                periods = series.get("period", [])
                values  = series.get("value", [])
                if not periods or not values:
                    continue
                pairs = [(p, v) for p, v in zip(periods, values) if v is not None][-13:]
                if len(pairs) < 2:
                    continue
                latest_val   = pairs[-1][1]
                prev_val     = pairs[-2][1]
                year_ago_val = pairs[0][1] if len(pairs) >= 13 else None
                result[code] = {
                    "latest":  latest_val,
                    "period":  pairs[-1][0],
                    "mom_pct": round((latest_val - prev_val)       / prev_val       * 100, 2) if prev_val       else None,
                    "yoy_pct": round((latest_val - year_ago_val)   / year_ago_val   * 100, 2) if year_ago_val   else None,
                }
        except Exception as e:
            logger.warning("DBnomics fetch from %s failed: %s", url, e)
        return result

    def _fetch_estats_3ai(self) -> Dict[str, dict]:
        """
        Fallback: fetch 第三次産業活動指数 from e-Stats API.
        Uses statsDataId for tertiary industry activity (サービス産業動向調査).
        """
        from japan_stock_daily.config import ESTATS_BASE_URL, ESTATS_APP_ID
        if not ESTATS_APP_ID:
            return {}
        result = {}
        # statsDataId for 第三次産業活動指数 (total index)
        STATS_ID = "0003116017"
        try:
            resp = self.session.get(
                f"{ESTATS_BASE_URL}/getStatsData",
                params={
                    "appId":       ESTATS_APP_ID,
                    "statsDataId": STATS_ID,
                    "metaGetFlg":  "Y",
                    "cntGetFlg":   "N",
                    "explanationGetFlg": "N",
                    "annotationGetFlg": "N",
                    "sectionHeaderFlg": "1",
                    "replaceSpChars": "0",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            time.sleep(REQUEST_DELAY)
            # Parse e-Stats response into {sector_code: activity_dict}
            result = self._parse_estats_response(data)
        except Exception as e:
            logger.warning("e-Stats 3AI fetch failed: %s", e)
        return result

    @staticmethod
    def _parse_estats_response(data: dict) -> Dict[str, dict]:
        """Parse e-Stats getStatsData JSON response into activity dict."""
        result = {}
        try:
            values = (data.get("GET_STATS_DATA", {})
                          .get("STATISTICAL_DATA", {})
                          .get("DATA_INF", {})
                          .get("VALUE", []))
            # Group by category code ($@cat01) and time ($@time)
            from collections import defaultdict
            series_data: dict = defaultdict(list)
            for v in values:
                cat = v.get("@cat01", "")
                time_val = v.get("@time", "")
                val_str = v.get("$", "")
                try:
                    val = float(val_str)
                    series_data[cat].append((time_val, val))
                except (ValueError, TypeError):
                    pass
            for cat, pairs in series_data.items():
                pairs_sorted = sorted(pairs, key=lambda x: x[0])[-13:]
                if len(pairs_sorted) < 2:
                    continue
                latest_val   = pairs_sorted[-1][1]
                prev_val     = pairs_sorted[-2][1]
                year_ago_val = pairs_sorted[0][1] if len(pairs_sorted) >= 13 else None
                result[cat] = {
                    "latest":  latest_val,
                    "period":  pairs_sorted[-1][0],
                    "mom_pct": round((latest_val - prev_val)     / prev_val     * 100, 2) if prev_val     else None,
                    "yoy_pct": round((latest_val - year_ago_val) / year_ago_val * 100, 2) if year_ago_val else None,
                }
        except Exception as e:
            logger.warning("e-Stats response parse failed: %s", e)
        return result

    def get_sector_iip(self, sector_name: str) -> Optional[dict]:
        """Return IIP data for a given TSE-33 sector name."""
        iip_code = SECTOR_TO_IIP.get(sector_name)
        if not iip_code:
            return None
        return self.get_iip_by_industry().get(iip_code)

    def get_sector_3ai(self, sector_name: str) -> Optional[dict]:
        """Return 第三次産業活動指数 data for a given TSE-33 sector name."""
        tai_code = SECTOR_TO_3AI.get(sector_name)
        if not tai_code:
            return None
        return self.get_tertiary_activity_index().get(tai_code)

    def get_sector_activity(self, sector_name: str) -> Optional[dict]:
        """Return the relevant monthly activity index for any TSE-33 sector.

        Checks IIP first (manufacturing), then 3AI (services).
        """
        return self.get_sector_iip(sector_name) or self.get_sector_3ai(sector_name)

    def get_sector_trend(self, sector_name: str) -> str:
        """Return trend: 'expanding' | 'contracting' | 'stable' | 'unknown'"""
        data = self.get_sector_activity(sector_name)
        if data is None:
            return "unknown"
        mom = data.get("mom_pct", 0) or 0
        yoy = data.get("yoy_pct", 0) or 0
        if mom > 1.0 and yoy > 3.0:
            return "expanding"
        if mom < -1.0 and yoy < -3.0:
            return "contracting"
        return "stable"
