"""Centralized configuration for the Japan stock recommendation system."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
JQUANTS_API_KEY = os.getenv("JQUANTS_API_KEY", "")
EDINET_API_KEY = os.getenv("EDINET_API_KEY", "")
ESTATS_APP_ID = os.getenv("ESTATS_APP_ID", "")
TDNET_API_KEY = os.getenv("TDNET_API_KEY", "")  # optional paid key

# --- J-Quants ---
JQUANTS_BASE_URL = "https://api.jpx-jquants.com/v1"

# --- EDINET ---
EDINET_BASE_URL = "https://api.edinet-fsa.go.jp/api/v2"
# Document types for daily signals
EDINET_DOC_TYPES = {
    "160": "大量保有報告書",       # Large shareholder ≥5%
    "161": "大量保有報告書（変更）", # Amended large shareholder
    "140": "臨時報告書",            # Extraordinary report
}

# --- TDnet (Yanoshin free service) ---
TDNET_BASE_URL = "https://webapi.yanoshin.jp/webapi/tdnet"
TDNET_LIST_URL = TDNET_BASE_URL + "/list/{date}.json2"
TDNET_DETAIL_URL = TDNET_BASE_URL + "/detail/{disclosure_id}.json2"

# --- Bank of Japan ---
BOJ_BASE_URL = "https://www.stat-search.boj.or.jp/ssi/mtshtml"
BOJ_API_URL = "https://www.stat-search.boj.or.jp/ssi/api/v1"
BOJ_SERIES = {
    "policy_rate":  "FM01'MAADMOCG@CO'",   # 無担保コール翌日物 (Uncollateralized overnight call rate)
    "jgb_10y":      "FM08'MAADMZGBMX/M'",  # 10-year JGB yield
    "cpi":          "CP01'MAADMOCG@CO'",    # Consumer Price Index
    "m2":           "MD01'MAADMOCG@CO'",    # M2 money supply
    "usd_jpy":      "FM08'MAADMFXGD@US'",  # USD/JPY rate
}

# --- e-Stats ---
ESTATS_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"
ESTATS_SERIES = {
    "cpi":          "0003000795",   # Consumer Price Index
    "unemployment": "0003215843",   # Unemployment rate (完全失業率)
    "retail_sales": "0003412197",   # Retail sales (小売業販売額)
    "industrial_production": "0003103532",  # Index of Industrial Production
}

# --- METI (via DBnomics) ---
DBNOMICS_BASE_URL = "https://api.db.nomics.world/v22"
METI_PROVIDER = "METI"
METI_DATASET_IIP = "IIP"  # Indices of Industrial Production

# --- US Overnight ---
US_SYMBOLS = {
    "sp500_futures":    "ES=F",
    "nasdaq_futures":   "NQ=F",
    "sp500":            "^GSPC",
    "nasdaq":           "^IXIC",
    "dow":              "^DJI",
    "vix":              "^VIX",
    "usd_jpy":          "JPY=X",
    # Sector ETFs
    "tech":             "XLK",
    "financials":       "XLF",
    "energy":           "XLE",
    "healthcare":       "XLV",
    "industrials":      "XLI",
    "consumer_disc":    "XLY",
    "materials":        "XLB",
    "utilities":        "XLU",
}

# JP TSE-33 sector code → US sector ETF mapping
JP_SECTOR_TO_US_ETF = {
    "電気機器":         "XLK",
    "情報・通信業":     "XLK",
    "銀行業":           "XLF",
    "保険業":           "XLF",
    "証券・商品先物取引業": "XLF",
    "石油・石炭製品":   "XLE",
    "医薬品":           "XLV",
    "機械":             "XLI",
    "輸送用機器":       "XLI",
    "空運業":           "XLI",
    "陸運業":           "XLI",
    "海運業":           "XLI",
    "小売業":           "XLY",
    "サービス業":       "XLY",
    "化学":             "XLB",
    "鉄鋼":             "XLB",
    "非鉄金属":         "XLB",
    "電力・ガス業":     "XLU",
}

# --- Scoring Weights ---
SCORE_WEIGHTS = {
    "disclosure":       0.30,  # TDnet + EDINET event signals
    "supply_demand":    0.25,  # J-Quants Pro supply/demand
    "us_overnight":     0.20,  # US after-hours market signals
    "price_momentum":   0.15,  # Technical price/volume momentum
    "macro":            0.10,  # BOJ + e-Stats + METI (slow-moving)
}

# Shock recovery mode threshold: if US overnight < this, switch strategy
SHOCK_RECOVERY_THRESHOLD = -0.01  # -1%

# --- General ---
REQUEST_TIMEOUT = 30     # seconds
REQUEST_DELAY = 1.0      # seconds between API calls
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "daily")
