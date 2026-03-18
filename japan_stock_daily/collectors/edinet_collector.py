"""EDINET API collector for daily disclosure signals.

Focuses on:
- 大量保有報告書 (docType 160/161): Large shareholder ≥5% filings
- 臨時報告書 (docType 140): Extraordinary reports (M&A, restructuring)
"""

import time
import logging
import zipfile
import io
import re
from datetime import date
from typing import Optional
from xml.etree import ElementTree as ET

import requests
import pandas as pd

from japan_stock_daily.config import EDINET_API_KEY, EDINET_BASE_URL, EDINET_DOC_TYPES, REQUEST_DELAY

logger = logging.getLogger(__name__)

EDINET_DOCS_URL = f"{EDINET_BASE_URL}/documents.json"
EDINET_DOC_URL  = f"{EDINET_BASE_URL}/documents/{{doc_id}}"


class EdinetCollector:
    """
    Fetches and parses EDINET daily filings for stock signals.

    Key signals:
    - Large shareholder report (160/161): new activist/institutional entry
    - Extraordinary report (140): M&A, capital events, business alliances
    """

    def __init__(self, api_key: str = EDINET_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NemesisBot/1.0"})

    def _get(self, url: str, params: dict = None) -> dict:
        params = params or {}
        params["Subscription-Key"] = self.api_key
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()

    # ------------------------------------------------------------------ #
    #  Document list for a date                                            #
    # ------------------------------------------------------------------ #
    def get_daily_documents(self, target_date: date) -> pd.DataFrame:
        """
        Fetch all documents submitted to EDINET on target_date.
        Returns filtered DataFrame of high-signal document types (160, 161, 140).
        """
        date_str = target_date.strftime("%Y-%m-%d")
        logger.info("Fetching EDINET document list for %s", date_str)
        data = self._get(EDINET_DOCS_URL, {"date": date_str, "type": 2})
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        # Keep only high-value doc types
        target_types = list(EDINET_DOC_TYPES.keys())
        df = df[df["docTypeCode"].isin(target_types)].copy()
        df = df.rename(columns={
            "docID": "doc_id",
            "edinetCode": "edinet_code",
            "secCode": "sec_code",
            "filerName": "filer_name",
            "docTypeCode": "doc_type_code",
            "docDescription": "description",
            "submitDateTime": "submitted_at",
            "periodStart": "period_start",
            "periodEnd": "period_end",
        })
        df["doc_type_name"] = df["doc_type_code"].map(EDINET_DOC_TYPES)
        # Normalize security code (4-digit stock code)
        if "sec_code" in df.columns:
            df["code"] = df["sec_code"].str[:4]
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Large shareholder report (docType 160/161)                        #
    # ------------------------------------------------------------------ #
    def parse_large_shareholder_report(self, doc_id: str) -> Optional[dict]:
        """
        Download and parse 大量保有報告書 to extract:
        - holder_name: name of the large shareholder
        - ownership_pct: ownership percentage after filing
        - ownership_change: change from previous report
        - direction: 'increase' | 'decrease' | 'new'
        - holder_type: fund / corporation / individual
        """
        try:
            content = self._download_document(doc_id, doc_type=2)  # type=2: XBRL/CSV
            if content is None:
                return None
            # Parse XBRL ZIP for key fields
            return self._parse_large_holder_xbrl(content, doc_id)
        except Exception as e:
            logger.warning("Failed to parse large shareholder report %s: %s", doc_id, e)
            return None

    def _download_document(self, doc_id: str, doc_type: int = 2) -> Optional[bytes]:
        """Download document ZIP from EDINET."""
        url = EDINET_DOC_URL.format(doc_id=doc_id)
        params = {"type": doc_type, "Subscription-Key": self.api_key}
        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY)
            return resp.content
        except Exception as e:
            logger.warning("Download failed for doc %s: %s", doc_id, e)
            return None

    def _parse_large_holder_xbrl(self, zip_content: bytes, doc_id: str) -> Optional[dict]:
        """Parse XBRL from large shareholder ZIP to extract ownership data."""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
                # Find the primary XBRL instance document
                xbrl_files = [f for f in zf.namelist() if f.endswith(".xbrl") or f.endswith(".xml")]
                if not xbrl_files:
                    return None
                # Take the largest XML file (likely the main instance)
                main_file = max(xbrl_files, key=lambda f: zf.getinfo(f).file_size)
                with zf.open(main_file) as xf:
                    tree = ET.parse(xf)
                    root = tree.getroot()

            ns = {k: v for k, v in re.findall(r'xmlns(?::(\w+))?="([^"]+)"',
                                               ET.tostring(root, encoding="unicode")[:2000])}

            result = {"doc_id": doc_id}

            # Try common XBRL element names for large shareholder reports
            search_tags = {
                "holder_name": [
                    "HoldersName",
                    "ReportingPersonName",
                    "NameOfLargeHolder",
                ],
                "ownership_pct": [
                    "HoldingRatioAfterChange",
                    "OwnershipRatioAfterChange",
                    "RatioOfSharesHeld",
                ],
                "ownership_prev_pct": [
                    "HoldingRatioBeforeChange",
                    "OwnershipRatioBeforeChange",
                ],
            }

            for field, tag_candidates in search_tags.items():
                for tag in tag_candidates:
                    # Search with and without namespace
                    found = root.find(f".//{tag}")
                    if found is None:
                        for prefix, uri in ns.items():
                            found = root.find(f".//{{{uri}}}{tag}")
                            if found is not None:
                                break
                    if found is not None and found.text:
                        result[field] = found.text.strip()
                        break

            # Derive direction
            try:
                curr = float(result.get("ownership_pct", 0))
                prev = float(result.get("ownership_prev_pct", 0))
                if prev == 0:
                    result["direction"] = "new"
                elif curr > prev:
                    result["direction"] = "increase"
                else:
                    result["direction"] = "decrease"
                result["ownership_change"] = round(curr - prev, 2)
            except (ValueError, TypeError):
                result["direction"] = "unknown"

            return result

        except Exception as e:
            logger.warning("XBRL parse failed for %s: %s", doc_id, e)
            return None

    # ------------------------------------------------------------------ #
    #  Extraordinary report (docType 140)                                 #
    # ------------------------------------------------------------------ #
    def classify_rinji_report(self, description: str) -> str:
        """
        Classify a 臨時報告書 by its description text.
        Returns event category string.
        """
        desc = description or ""
        patterns = {
            "m_and_a":          ["合併", "買収", "株式取得", "子会社化", "TOB", "公開買付"],
            "alliance":         ["業務提携", "資本業務提携", "資本提携"],
            "capital_raise":    ["第三者割当", "公募増資", "新株発行"],
            "restructuring":    ["事業再編", "会社分割", "事業譲渡", "事業売却"],
            "management_change":["代表取締役", "役員変更", "経営者変更"],
            "earnings_update":  ["業績予想", "配当予想"],
            "legal":            ["訴訟", "行政処分", "課徴金"],
        }
        for category, keywords in patterns.items():
            if any(kw in desc for kw in keywords):
                return category
        return "other"

    # ------------------------------------------------------------------ #
    #  High-level: get all signals for a date                             #
    # ------------------------------------------------------------------ #
    def get_daily_signals(self, target_date: date) -> pd.DataFrame:
        """
        Return a DataFrame of EDINET signals for target_date.

        Columns: code, filer_name, doc_type, event_category,
                 direction (for large holder), description, doc_id
        """
        docs = self.get_daily_documents(target_date)
        if docs.empty:
            return pd.DataFrame()

        records = []
        for _, row in docs.iterrows():
            record = {
                "code":         row.get("code", ""),
                "filer_name":   row.get("filer_name", ""),
                "doc_type":     row.get("doc_type_name", ""),
                "doc_type_code": row.get("doc_type_code", ""),
                "description":  row.get("description", ""),
                "doc_id":       row.get("doc_id", ""),
                "submitted_at": row.get("submitted_at", ""),
                "event_category": "unknown",
                "direction":    None,
                "ownership_pct": None,
                "ownership_change": None,
            }

            if row.get("doc_type_code") in ("160", "161"):
                parsed = self.parse_large_shareholder_report(row["doc_id"])
                if parsed:
                    record["event_category"] = "large_shareholder"
                    record["direction"]       = parsed.get("direction")
                    record["ownership_pct"]   = parsed.get("ownership_pct")
                    record["ownership_change"] = parsed.get("ownership_change")
                else:
                    record["event_category"] = "large_shareholder"

            elif row.get("doc_type_code") == "140":
                record["event_category"] = self.classify_rinji_report(row.get("description", ""))

            records.append(record)

        return pd.DataFrame(records)
