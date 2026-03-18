"""J-Quants Pro API collector for supply/demand signals."""

import time
import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from japan_stock_daily.config import JQUANTS_API_KEY, REQUEST_DELAY

logger = logging.getLogger(__name__)


class JQuantsCollector:
    """
    Wraps J-Quants API V2 (Pro tier) for daily stock data collection.

    Fetches:
    - Daily OHLCV quotes
    - Breakdown data (investor-type buy/sell flows)
    - Weekly margin trading outstanding (信用残高)
    - Outstanding short selling positions
    - Investor-type trading summary (投資主体別売買動向)
    - Buyback announcements from TDnet
    - Listed company master data
    """

    def __init__(self, api_key: str = JQUANTS_API_KEY):
        self.api_key = api_key
        self._client = None
        self._listed_cache: Optional[pd.DataFrame] = None

    def _get_client(self):
        if self._client is None:
            try:
                from jquantsapi import Client
                self._client = Client(mail_address=None, password=None, refresh_token=None)
                # For V2, set api_key directly on the session if supported
                self._client.headers = {"x-api-key": self.api_key}
            except ImportError:
                raise ImportError("Install jquants-api-client: pip install jquants-api-client")
        return self._client

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Low-level GET against J-Quants API V2."""
        import requests
        url = f"https://api.jpx-jquants.com/v1{endpoint}"
        headers = {"x-api-key": self.api_key}
        resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()

    # ------------------------------------------------------------------ #
    #  Company master                                                       #
    # ------------------------------------------------------------------ #
    def get_listed_companies(self) -> pd.DataFrame:
        """Return listed company master (code, name, sector, market).
        Result is cached for the session."""
        if self._listed_cache is not None:
            return self._listed_cache
        logger.info("Fetching listed company master from J-Quants")
        data = self._get("/listed/info")
        df = pd.DataFrame(data.get("info", []))
        if not df.empty:
            df = df.rename(columns={
                "Code": "code",
                "CompanyName": "name",
                "CompanyNameEnglish": "name_en",
                "Sector17Code": "sector17_code",
                "Sector17CodeName": "sector17",
                "Sector33Code": "sector33_code",
                "Sector33CodeName": "sector33",
                "MarketCode": "market_code",
                "MarketCodeName": "market",
                "ScaleCategory": "scale",
            })
        self._listed_cache = df
        return df

    # ------------------------------------------------------------------ #
    #  Daily quotes                                                        #
    # ------------------------------------------------------------------ #
    def get_daily_quotes(self, target_date: date) -> pd.DataFrame:
        """Fetch OHLCV for all stocks on target_date.

        Returns DataFrame with columns:
            code, open, high, low, close, volume, turnover_value, adj_close
        """
        date_str = target_date.strftime("%Y%m%d")
        logger.info("Fetching J-Quants daily quotes for %s", date_str)
        data = self._get("/prices/daily_quotes", {"date": date_str})
        df = pd.DataFrame(data.get("daily_quotes", []))
        if df.empty:
            return df
        df = df.rename(columns={
            "Code": "code",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "TurnoverValue": "turnover_value",
            "AdjustmentClose": "adj_close",
        })
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_historical_quotes(self, code: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch OHLCV for a single stock over a date range."""
        data = self._get("/prices/daily_quotes", {
            "code": code,
            "from": start_date.strftime("%Y%m%d"),
            "to": end_date.strftime("%Y%m%d"),
        })
        df = pd.DataFrame(data.get("daily_quotes", []))
        if df.empty:
            return df
        df = df.rename(columns={
            "Code": "code", "Date": "date",
            "Open": "open", "High": "high", "Low": "low", "Close": "close",
            "Volume": "volume", "TurnoverValue": "turnover_value",
            "AdjustmentClose": "adj_close",
        })
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date")

    # ------------------------------------------------------------------ #
    #  Supply/demand: breakdown (Pro)                                      #
    # ------------------------------------------------------------------ #
    def get_breakdown(self, target_date: date) -> pd.DataFrame:
        """
        Fetch daily trading breakdown by investor type (Pro endpoint).

        Columns include buy/sell values for:
        - 外国人 (foreigners), 投信 (investment trusts), 事業法人 (corporations),
          個人 (retail), 自己 (proprietary)
        Also includes margin and short-selling flags.
        """
        date_str = target_date.strftime("%Y%m%d")
        logger.info("Fetching J-Quants breakdown for %s", date_str)
        data = self._get("/markets/breakdown", {"date": date_str})
        df = pd.DataFrame(data.get("breakdown", []))
        if not df.empty:
            df["date"] = pd.to_datetime(df.get("Date", date_str))
            df = df.rename(columns={"Code": "code"})
        return df

    # ------------------------------------------------------------------ #
    #  Supply/demand: weekly margin interest (Pro)                        #
    # ------------------------------------------------------------------ #
    def get_weekly_margin_interest(self, target_date: date) -> pd.DataFrame:
        """
        Weekly margin trading outstanding (信用取引残高).
        Published every Friday (or Monday for the prior week).

        Key columns:
        - LongMarginTradeVolume: 信用買い残 (outstanding margin longs)
        - ShortMarginTradeVolume: 信用売り残 (outstanding margin shorts)
        Derived: margin_ratio = LongMarginTradeVolume / ShortMarginTradeVolume
        """
        date_str = target_date.strftime("%Y%m%d")
        logger.info("Fetching J-Quants weekly margin interest for %s", date_str)
        data = self._get("/markets/weekly_margin_interest", {"date": date_str})
        df = pd.DataFrame(data.get("weekly_margin_interest", []))
        if df.empty:
            return df
        df = df.rename(columns={"Code": "code", "Date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        # Compute margin ratio (信用倍率)
        long_col = "LongMarginTradeVolume"
        short_col = "ShortMarginTradeVolume"
        if long_col in df.columns and short_col in df.columns:
            df["margin_ratio"] = df[long_col] / df[short_col].replace(0, float("nan"))
        return df

    # ------------------------------------------------------------------ #
    #  Supply/demand: short selling positions (Pro)                       #
    # ------------------------------------------------------------------ #
    def get_short_selling_positions(self, target_date: date) -> pd.DataFrame:
        """
        Outstanding short selling positions (空売り残高).
        Published bi-weekly.

        Key columns:
        - ShortSellingVolume, ShortSellingValue, ShortSellingRatio
        """
        date_str = target_date.strftime("%Y%m%d")
        logger.info("Fetching J-Quants short selling positions for %s", date_str)
        data = self._get("/markets/short_selling_positions", {"date": date_str})
        df = pd.DataFrame(data.get("short_selling_positions", []))
        if not df.empty:
            df = df.rename(columns={"Code": "code", "Date": "date"})
            df["date"] = pd.to_datetime(df["date"])
        return df

    # ------------------------------------------------------------------ #
    #  Supply/demand: trades by investor type (Pro)                       #
    # ------------------------------------------------------------------ #
    def get_trades_spec(self, target_date: date) -> pd.DataFrame:
        """
        投資主体別売買動向: weekly trading by investor category.
        Published weekly.

        Includes: foreigners, investment trusts, pension funds, corporations, retail.
        """
        date_str = target_date.strftime("%Y%m%d")
        logger.info("Fetching J-Quants trades_spec for %s", date_str)
        data = self._get("/markets/trades_spec", {
            "section": "TSEPrime",
            "from": date_str,
            "to": date_str,
        })
        df = pd.DataFrame(data.get("trades_spec", []))
        if not df.empty:
            df["date"] = pd.to_datetime(df.get("PublishedDate", date_str))
        return df

    # ------------------------------------------------------------------ #
    #  Buybacks from TDnet (Pro)                                          #
    # ------------------------------------------------------------------ #
    def get_share_buyback_tdnet(self, target_date: date) -> pd.DataFrame:
        """
        Share buyback announcements sourced from TDnet (Pro endpoint).
        Avoids need to parse TDnet separately for buybacks.
        """
        date_str = target_date.strftime("%Y%m%d")
        logger.info("Fetching J-Quants share buyback TDnet for %s", date_str)
        data = self._get("/markets/share_buyback_tdnet", {
            "from": date_str,
            "to": date_str,
        })
        df = pd.DataFrame(data.get("share_buyback_tdnet", []))
        if not df.empty:
            df = df.rename(columns={"Code": "code", "DisclosedDate": "date"})
        return df

    # ------------------------------------------------------------------ #
    #  TOPIX index for relative strength                                  #
    # ------------------------------------------------------------------ #
    def get_indices(self, target_date: date) -> pd.DataFrame:
        """Fetch index prices (TOPIX, Nikkei 225) for market context."""
        date_str = target_date.strftime("%Y%m%d")
        data = self._get("/indices", {"date": date_str})
        df = pd.DataFrame(data.get("indices", []))
        return df
