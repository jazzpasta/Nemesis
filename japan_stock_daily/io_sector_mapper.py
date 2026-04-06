"""Japanese Input-Output Sector Mapper.

Links listed companies (via TSE-33 sector codes or JSIC codes from EDINET)
to the 2020 IO Table's 107-sector integrated middle classification (統合中分類).

Monthly activity data for each IO sector comes from two METI series:
  - 鉱工業指数 (IIP): manufacturing sectors (C09, C10, ... C31)
  - 第三次産業活動指数 (3AI): service sectors (H, I50, J, L, S …)

The core idea:
  For each company, identify its IO sector(s).
  Then look at:
    upstream  → are input costs rising?  (upstream sectors growing = cost pressure)
    downstream → is demand growing?       (downstream sectors growing = revenue tailwind)

  score = downstream_growth - upstream_growth_penalty
          (+macro bonus if cross-sector data confirms expansion)

IO Table Source: 2020 Japan IO Table (総務省), 統合中分類107部門, published June 2024
Monthly Activity: METI IIP + 第三次産業活動指数 via e-Stat / DBnomics / direct download
"""

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Paths to data files
_DATA_DIR = Path(__file__).parent.parent / "data"
_TSE33_IO_FILE      = _DATA_DIR / "tse33_to_io.csv"
_SUPPLY_CHAIN_FILE  = _DATA_DIR / "io_supply_chain.csv"


# ─────────────────────────────────────────────────────────────────────────────
#  Static mapping: TSE-33 sector name → IO sector IDs
#  (redundant with CSV, kept inline for fast lookup without I/O)
# ─────────────────────────────────────────────────────────────────────────────

# TSE-33 sector name → primary IO sector ID (2020 107-sector classification)
TSE33_NAME_TO_IO: dict[str, list[str]] = {
    "水産業":               ["003"],
    "鉱業":                 ["004"],
    "建設業":               ["025"],
    "食料品":               ["005"],
    "繊維製品":             ["006"],
    "パルプ・紙":           ["007"],
    "化学":                 ["009", "011"],
    "医薬品":               ["010"],          # 010 = pharma within chemicals in 107-sector
    "石油・石炭製品":       ["010"],          # 010 = petroleum/coal
    "ゴム製品":             ["012"],
    "窯業・土石製品":       ["013"],
    "鉄鋼":                 ["014"],
    "非鉄金属":             ["015"],
    "金属製品":             ["016"],
    "機械":                 ["017", "018"],
    "電気機器":             ["020", "021", "022"],
    "輸送用機器":           ["023"],
    "精密機器":             ["019"],
    "その他製品":           ["024"],
    "電力・ガス業":         ["026"],
    "陸運業":               ["031"],
    "海運業":               ["031"],
    "空運業":               ["031"],
    "倉庫・運輸関連業":     ["031"],
    "情報・通信業":         ["032"],
    "卸売業":               ["028"],
    "小売業":               ["028"],
    "銀行業":               ["029"],
    "証券・商品先物取引業": ["029"],
    "保険業":               ["029"],
    "不動産業":             ["030"],
    "サービス業":           ["037"],
    "その他":               ["037"],
}

# TSE-33 numeric code → sector name
TSE33_CODE_TO_NAME: dict[str, str] = {
    "0050": "水産業",
    "0100": "鉱業",
    "0200": "建設業",
    "0300": "食料品",
    "0400": "繊維製品",
    "0500": "パルプ・紙",
    "0600": "化学",
    "0700": "医薬品",
    "0800": "石油・石炭製品",
    "0900": "ゴム製品",
    "1000": "窯業・土石製品",
    "1100": "鉄鋼",
    "1200": "非鉄金属",
    "1300": "金属製品",
    "1400": "機械",
    "1500": "電気機器",
    "1600": "輸送用機器",
    "1700": "精密機器",
    "1800": "その他製品",
    "1900": "電力・ガス業",
    "2000": "陸運業",
    "2100": "海運業",
    "2200": "空運業",
    "2300": "倉庫・運輸関連業",
    "2400": "情報・通信業",
    "2500": "卸売業",
    "2600": "小売業",
    "2700": "銀行業",
    "2800": "証券・商品先物取引業",
    "2900": "保険業",
    "3000": "不動産業",
    "3100": "サービス業",
    "3200": "その他",
    "3300": "水産業",
}

# IO sector ID → METI monthly data code
# Manufacturing sectors → IIP series codes (via DBnomics METI/IIP)
# Service sectors      → 第三次産業活動指数 series codes
IO_TO_METI_CODE: dict[str, dict] = {
    "001": {"series": "IIP", "code": "C01",  "name": "農業"},
    "003": {"series": "IIP", "code": "A03",  "name": "漁業"},
    "004": {"series": "IIP", "code": "B",    "name": "鉱業"},
    "005": {"series": "IIP", "code": "C09",  "name": "飲食料品"},
    "006": {"series": "IIP", "code": "C10",  "name": "繊維"},
    "007": {"series": "IIP", "code": "C12",  "name": "パルプ・紙"},
    "008": {"series": "IIP", "code": "C13",  "name": "印刷"},
    "009": {"series": "IIP", "code": "C16",  "name": "化学製品"},
    "010": {"series": "IIP", "code": "C17",  "name": "石油・石炭/医薬品"},
    "011": {"series": "IIP", "code": "C18",  "name": "プラスチック"},
    "012": {"series": "IIP", "code": "C20",  "name": "ゴム"},
    "013": {"series": "IIP", "code": "C21",  "name": "窯業・土石"},
    "014": {"series": "IIP", "code": "C24",  "name": "鉄鋼"},
    "015": {"series": "IIP", "code": "C25",  "name": "非鉄金属"},
    "016": {"series": "IIP", "code": "C26",  "name": "金属製品"},
    "017": {"series": "IIP", "code": "C28",  "name": "はん用機械"},
    "018": {"series": "IIP", "code": "C28",  "name": "生産用機械"},
    "019": {"series": "IIP", "code": "C30",  "name": "業務用機械"},
    "020": {"series": "IIP", "code": "C27",  "name": "電子部品・デバイス"},
    "021": {"series": "IIP", "code": "C27",  "name": "電気機械"},
    "022": {"series": "IIP", "code": "C27",  "name": "情報・通信機器"},
    "023": {"series": "IIP", "code": "C31",  "name": "輸送機械"},
    "024": {"series": "IIP", "code": "C32",  "name": "その他製造"},
    "025": {"series": "3AI", "code": "D",    "name": "建設"},
    "026": {"series": "3AI", "code": "E",    "name": "電力・ガス"},
    "028": {"series": "3AI", "code": "I",    "name": "商業"},
    "029": {"series": "3AI", "code": "J",    "name": "金融・保険"},
    "030": {"series": "3AI", "code": "L",    "name": "不動産"},
    "031": {"series": "3AI", "code": "G",    "name": "運輸・郵便"},
    "032": {"series": "3AI", "code": "H",    "name": "情報通信"},
    "034": {"series": "3AI", "code": "P",    "name": "教育・学習支援"},
    "035": {"series": "3AI", "code": "Q",    "name": "医療・福祉"},
    "037": {"series": "3AI", "code": "S",    "name": "その他サービス"},
}

# Simplified upstream/downstream links (derived from 2020 IO table)
# key = IO sector ID, value = list of (upstream_id, weight) tuples
# weights are approximate input coefficients from 2020 IO table
IO_UPSTREAM: dict[str, list[tuple[str, float]]] = {
    "003": [("001", 0.35), ("005", 0.25), ("009", 0.10)],           # Fisheries ← Agriculture, Food, Chemicals
    "005": [("001", 0.30), ("003", 0.15), ("009", 0.15)],           # Food ← Agriculture, Fisheries, Chemicals
    "006": [("001", 0.25), ("009", 0.20)],                          # Textiles ← Agriculture, Chemicals
    "009": [("004", 0.25), ("001", 0.15), ("014", 0.20)],           # Chemicals ← Mining, Agriculture, Steel
    "010": [("004", 0.60), ("009", 0.20)],                          # Petroleum ← Mining, Chemicals
    "011": [("009", 0.40), ("004", 0.20)],                          # Plastics ← Chemicals, Mining
    "012": [("009", 0.40), ("011", 0.20)],                          # Rubber ← Chemicals, Plastics
    "013": [("004", 0.25), ("009", 0.20), ("014", 0.15)],           # Ceramics ← Mining, Chemicals, Steel
    "014": [("004", 0.35), ("015", 0.15), ("010", 0.15)],           # Steel ← Mining, Non-ferrous, Petroleum
    "015": [("004", 0.35), ("014", 0.15), ("009", 0.10)],           # Non-ferrous ← Mining, Steel, Chemicals
    "016": [("014", 0.45), ("015", 0.15), ("009", 0.10)],           # Metal fab ← Steel, Non-ferrous, Chemicals
    "017": [("016", 0.30), ("014", 0.20), ("009", 0.15)],           # Gen machinery ← Metal, Steel, Chemicals
    "018": [("016", 0.30), ("014", 0.20), ("020", 0.15)],           # Prod machinery ← Metal, Steel, Electronics
    "019": [("016", 0.25), ("020", 0.20), ("021", 0.15)],           # Biz machinery ← Metal, Electronics, Elec mach
    "020": [("015", 0.30), ("009", 0.20), ("021", 0.15)],           # Electronics ← Non-ferrous, Chemicals, Elec mach
    "021": [("016", 0.25), ("020", 0.20), ("015", 0.15)],           # Elec mach ← Metal, Electronics, Non-ferrous
    "022": [("020", 0.35), ("016", 0.20), ("021", 0.15)],           # ICT equip ← Electronics, Metal, Elec mach
    "023": [("014", 0.35), ("016", 0.20), ("012", 0.10)],           # Transport equip ← Steel, Metal, Rubber
    "024": [("016", 0.25), ("009", 0.20), ("007", 0.15)],           # Other mfg ← Metal, Chemicals, Paper
    "025": [("013", 0.15), ("014", 0.30), ("021", 0.10)],           # Construction ← Ceramics, Steel, Elec mach
    "026": [("004", 0.30), ("010", 0.30), ("009", 0.15)],           # Utilities ← Mining, Petroleum, Chemicals
    "028": [("031", 0.15), ("032", 0.10), ("029", 0.10)],           # Commerce ← Transport, ICT, Finance
    "029": [("032", 0.20), ("037", 0.15), ("030", 0.10)],           # Finance ← ICT, Services, Real estate
    "030": [("025", 0.40), ("037", 0.20)],                          # Real estate ← Construction, Services
    "031": [("010", 0.30), ("026", 0.20), ("021", 0.10)],           # Transport ← Petroleum, Utilities, Elec mach
    "032": [("022", 0.15), ("032", 0.25), ("020", 0.10)],           # ICT services ← ICT equip, ICT services, Electronics
    "035": [("019", 0.20), ("009", 0.15), ("037", 0.20)],           # Healthcare ← Biz mach, Chemicals, Services
    "037": [("032", 0.15), ("031", 0.15), ("029", 0.10)],           # Other services ← ICT, Transport, Finance
}

# Downstream: for each IO sector, which sectors consume its output most
IO_DOWNSTREAM: dict[str, list[tuple[str, float]]] = {
    "001": [("005", 0.45), ("003", 0.15), ("001", 0.10)],           # Agriculture → Food, Fisheries
    "003": [("005", 0.60), ("028", 0.20)],                          # Fisheries → Food, Commerce
    "004": [("009", 0.25), ("010", 0.20), ("014", 0.25), ("015", 0.15)],  # Mining → Chemicals, Petroleum, Steel, Non-ferrous
    "005": [("028", 0.50), ("031", 0.10), ("037", 0.20)],           # Food → Commerce, Transport, Services
    "006": [("028", 0.40), ("037", 0.15)],                          # Textiles → Commerce, Services
    "007": [("008", 0.25), ("028", 0.35), ("037", 0.20)],           # Paper → Printing, Commerce, Services
    "009": [("011", 0.20), ("012", 0.10), ("005", 0.15), ("006", 0.10)],  # Chemicals → Plastics, Rubber, Food, Textiles
    "010": [("009", 0.10), ("017", 0.15), ("023", 0.20), ("031", 0.25)],  # Petroleum → Chemicals, Machinery, Transport equip, Transport
    "011": [("016", 0.15), ("017", 0.20), ("022", 0.15), ("023", 0.20)],  # Plastics → Metal fab, Machinery, ICT equip, Transport equip
    "012": [("023", 0.60), ("017", 0.20)],                          # Rubber → Transport equip, Machinery
    "013": [("025", 0.35), ("017", 0.20), ("013", 0.15)],           # Ceramics → Construction, Machinery
    "014": [("016", 0.25), ("017", 0.20), ("023", 0.25), ("025", 0.15)],  # Steel → Metal fab, Machinery, Transport equip, Construction
    "015": [("016", 0.20), ("020", 0.30), ("021", 0.20), ("023", 0.15)],  # Non-ferrous → Metal fab, Electronics, Elec mach, Transport equip
    "016": [("017", 0.25), ("018", 0.20), ("025", 0.25), ("023", 0.15)],  # Metal fab → Machinery, Prod mach, Construction, Transport equip
    "017": [("018", 0.20), ("023", 0.25), ("025", 0.20), ("031", 0.15)],  # Gen mach → Prod mach, Transport equip, Construction, Transport
    "018": [("017", 0.15), ("023", 0.30), ("025", 0.25)],           # Prod mach → Gen mach, Transport equip, Construction
    "019": [("034", 0.30), ("035", 0.30), ("037", 0.20)],           # Biz mach → Education, Healthcare, Services
    "020": [("021", 0.25), ("022", 0.30), ("019", 0.15), ("023", 0.15)],  # Electronics → Elec mach, ICT equip, Biz mach, Transport equip
    "021": [("025", 0.20), ("026", 0.15), ("023", 0.20), ("037", 0.20)],  # Elec mach → Construction, Utilities, Transport equip, Services
    "022": [("032", 0.35), ("034", 0.20), ("037", 0.25)],           # ICT equip → ICT services, Education, Services
    "023": [("028", 0.45), ("031", 0.15), ("037", 0.20)],           # Transport equip → Commerce, Transport, Services (final demand cars)
    "024": [("028", 0.50), ("037", 0.30)],                          # Other mfg → Commerce, Services
    "025": [("030", 0.60), ("037", 0.20)],                          # Construction → Real estate, Services
    "026": [("全産業", 1.0)],                                        # Utilities → all industries (cross-cutting)
    "028": [("家計", 0.60), ("037", 0.20)],                         # Commerce → Household final demand
    "029": [("全産業", 1.0)],                                        # Finance → all industries (cross-cutting)
    "030": [("家計", 0.55), ("037", 0.20), ("029", 0.10)],          # Real estate → Household, Services, Finance
    "031": [("028", 0.30), ("全産業", 0.40)],                       # Transport → Commerce, all industries
    "032": [("全産業", 0.60), ("034", 0.15)],                       # ICT services → all industries (DX demand)
    "035": [("家計", 0.80)],                                         # Healthcare → Household final demand
    "037": [("家計", 0.50), ("全産業", 0.30)],                      # Services → Household, all industries
}


# ─────────────────────────────────────────────────────────────────────────────
#  Main mapper class
# ─────────────────────────────────────────────────────────────────────────────

class IOSectorMapper:
    """
    Maps companies to IO sectors and scores them based on upstream/downstream
    monthly activity from METI statistical data.

    Primary use:
        mapper = IOSectorMapper(meti_collector)
        score = mapper.score_company(tse33_code="1500")
        # Returns IOScore with upstream_pressure, downstream_tailwind, net_score
    """

    def __init__(self, meti_collector=None):
        """
        Args:
            meti_collector: METICollector instance (for live monthly data).
                            If None, uses last-cached data only.
        """
        self._meti = meti_collector
        self._tse33_io_df: Optional[pd.DataFrame] = None
        self._supply_chain_df: Optional[pd.DataFrame] = None
        self._iip_data: dict = {}
        self._3ai_data: dict = {}

        self._load_crosswalk()

    # ─────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────

    def get_io_sectors(self, tse33_code: str = None, tse33_name: str = None) -> list[str]:
        """
        Return list of IO sector IDs for a given TSE-33 code or name.

        Args:
            tse33_code: numeric TSE-33 code (e.g. "1500") OR 4-digit string
            tse33_name: sector name in Japanese (e.g. "電気機器")

        Returns:
            list of IO sector IDs (e.g. ["020", "021", "022"])
        """
        if tse33_code:
            code_str = str(tse33_code).zfill(4)
            name = TSE33_CODE_TO_NAME.get(code_str)
            if not name:
                # Try sector33code from J-Quants (may be integer like 6030)
                # Map via the CSV crosswalk
                return self._csv_lookup(tse33_code=tse33_code)
        elif tse33_name:
            name = tse33_name
        else:
            return []

        return TSE33_NAME_TO_IO.get(name, [])

    def get_upstream_activity(self, io_sector_ids: list[str]) -> dict:
        """
        Compute weighted-average upstream sector monthly activity.

        Returns:
            {"mom_pct": float, "yoy_pct": float, "interpretation": str}
            mom_pct > 0 → upstream costs rising (input price pressure)
        """
        if not io_sector_ids:
            return {}

        self._refresh_meti_data()
        total_weight = 0.0
        mom_sum = 0.0
        yoy_sum = 0.0
        n_upstream = 0

        for sector_id in io_sector_ids:
            upstream_links = IO_UPSTREAM.get(sector_id, [])
            for upstream_id, weight in upstream_links:
                activity = self._get_sector_activity(upstream_id)
                if activity:
                    mom_sum += (activity.get("mom_pct") or 0) * weight
                    yoy_sum += (activity.get("yoy_pct") or 0) * weight
                    total_weight += weight
                    n_upstream += 1

        if total_weight == 0:
            return {"mom_pct": 0, "yoy_pct": 0, "n_linked": 0, "interpretation": "unknown"}

        mom = mom_sum / total_weight
        yoy = yoy_sum / total_weight

        return {
            "mom_pct": round(mom, 2),
            "yoy_pct": round(yoy, 2),
            "n_linked": n_upstream,
            "interpretation": self._interpret_upstream(mom, yoy),
        }

    def get_downstream_activity(self, io_sector_ids: list[str]) -> dict:
        """
        Compute weighted-average downstream sector monthly activity.

        Returns:
            {"mom_pct": float, "yoy_pct": float, "interpretation": str}
            mom_pct > 0 → downstream demand expanding (revenue tailwind)
        """
        if not io_sector_ids:
            return {}

        self._refresh_meti_data()
        total_weight = 0.0
        mom_sum = 0.0
        yoy_sum = 0.0
        n_downstream = 0

        for sector_id in io_sector_ids:
            downstream_links = IO_DOWNSTREAM.get(sector_id, [])
            for downstream_id, weight in downstream_links:
                if downstream_id in ("全産業", "家計"):
                    # Use composite macro proxy: average IIP + 3AI
                    activity = self._get_macro_composite()
                else:
                    activity = self._get_sector_activity(downstream_id)
                if activity:
                    mom_sum += (activity.get("mom_pct") or 0) * weight
                    yoy_sum += (activity.get("yoy_pct") or 0) * weight
                    total_weight += weight
                    n_downstream += 1

        if total_weight == 0:
            return {"mom_pct": 0, "yoy_pct": 0, "n_linked": 0, "interpretation": "unknown"}

        mom = mom_sum / total_weight
        yoy = yoy_sum / total_weight

        return {
            "mom_pct": round(mom, 2),
            "yoy_pct": round(yoy, 2),
            "n_linked": n_downstream,
            "interpretation": self._interpret_downstream(mom, yoy),
        }

    def score_company(
        self,
        tse33_code: str = None,
        tse33_name: str = None,
        jsic_code: str = None,
    ) -> dict:
        """
        Compute IO-based fundamental score for a company.

        Returns:
            {
              "io_sectors":          ["020", "021"],     # IO sector IDs
              "upstream":            {...},              # upstream activity
              "downstream":          {...},              # downstream activity
              "net_score":           15.0,               # -100 to +100
              "interpretation":      "Downstream demand expanding; input costs stable",
              "io_sector_names":     ["電子部品", "電気機械"],
            }
        """
        # Resolve IO sectors
        if jsic_code:
            io_sectors = self._jsic_to_io(jsic_code)
        else:
            io_sectors = self.get_io_sectors(tse33_code=tse33_code, tse33_name=tse33_name)

        if not io_sectors:
            return {"net_score": 0, "interpretation": "No IO sector mapping found"}

        upstream   = self.get_upstream_activity(io_sectors)
        downstream = self.get_downstream_activity(io_sectors)

        # Net score:
        #   downstream tailwind (+) minus upstream cost pressure (+)
        #   Both measured as YoY % change (more stable than MoM)
        downstream_yoy = downstream.get("yoy_pct") or 0
        upstream_yoy   = upstream.get("yoy_pct")   or 0

        # Downstream growth contributes positively (demand = revenue)
        # Upstream growth is ambiguous: moderate = pricing power, high = margin squeeze
        # Use a nonlinear penalty: upstream > 5% YoY starts hurting margins
        upstream_penalty = max(0, upstream_yoy - 3.0) * 1.5  # Grace: first 3% is fine

        raw_score = downstream_yoy * 2.0 - upstream_penalty
        net_score = max(-100, min(100, raw_score * 5))  # scale to -100..+100

        io_names = [IO_TO_METI_CODE.get(s, {}).get("name", s) for s in io_sectors]

        interpretation = self._build_interpretation(upstream, downstream, net_score)

        return {
            "io_sectors":       io_sectors,
            "io_sector_names":  io_names,
            "upstream":         upstream,
            "downstream":       downstream,
            "net_score":        round(net_score, 1),
            "interpretation":   interpretation,
        }

    def get_sector_activity_summary(self) -> pd.DataFrame:
        """
        Return a DataFrame showing all IO sectors with their latest monthly activity.
        Useful for macro dashboard / report generation.
        """
        self._refresh_meti_data()
        rows = []
        for io_id, meta in IO_TO_METI_CODE.items():
            activity = self._get_sector_activity(io_id)
            rows.append({
                "io_sector_id":   io_id,
                "io_sector_name": meta["name"],
                "series_type":    meta["series"],
                "meti_code":      meta["code"],
                "latest_index":   activity.get("latest")  if activity else None,
                "period":         activity.get("period")   if activity else None,
                "mom_pct":        activity.get("mom_pct")  if activity else None,
                "yoy_pct":        activity.get("yoy_pct")  if activity else None,
                "trend":          self._interpret_downstream(
                                      activity.get("mom_pct") or 0,
                                      activity.get("yoy_pct") or 0
                                  ) if activity else "unknown",
            })
        return pd.DataFrame(rows).sort_values("io_sector_id").reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_crosswalk(self):
        """Load CSV crosswalk files if they exist."""
        if _TSE33_IO_FILE.exists():
            try:
                self._tse33_io_df = pd.read_csv(_TSE33_IO_FILE, dtype=str)
            except Exception as e:
                logger.warning("Failed to load TSE33→IO crosswalk: %s", e)
        if _SUPPLY_CHAIN_FILE.exists():
            try:
                self._supply_chain_df = pd.read_csv(_SUPPLY_CHAIN_FILE, dtype=str)
            except Exception as e:
                logger.warning("Failed to load IO supply chain: %s", e)

    def _csv_lookup(self, tse33_code: str) -> list[str]:
        """Look up IO sectors from CSV crosswalk by TSE-33 code."""
        if self._tse33_io_df is None:
            return []
        mask = self._tse33_io_df["tse33_code"] == str(tse33_code).zfill(4)
        rows = self._tse33_io_df[mask]
        return rows["io_sector_id"].dropna().tolist()

    def _jsic_to_io(self, jsic_code: str) -> list[str]:
        """Map a JSIC code to IO sector IDs using the crosswalk CSV."""
        if self._tse33_io_df is None:
            return []
        # JSIC major code (first letter + digits)
        jsic_major = jsic_code[:3] if len(jsic_code) >= 3 else jsic_code
        if "jsic_major" not in self._tse33_io_df.columns:
            return []
        mask = self._tse33_io_df["jsic_major"].str.startswith(
            jsic_code[0], na=False
        )
        rows = self._tse33_io_df[mask]
        return rows["io_sector_id"].dropna().unique().tolist()

    def _refresh_meti_data(self):
        """Pull latest monthly IIP and 3AI data from METI collector if available."""
        if self._meti and not self._iip_data:
            try:
                self._iip_data = self._meti.get_iip_by_industry()
            except Exception as e:
                logger.warning("IIP refresh failed: %s", e)
        if self._meti and not self._3ai_data:
            try:
                self._3ai_data = self._meti.get_tertiary_activity_index()
            except Exception as e:
                logger.warning("3AI refresh failed: %s", e)

    def _get_sector_activity(self, io_sector_id: str) -> Optional[dict]:
        """Return monthly activity dict for a single IO sector."""
        meta = IO_TO_METI_CODE.get(io_sector_id)
        if not meta:
            return None
        code = meta["code"]
        if meta["series"] == "IIP":
            return self._iip_data.get(code)
        else:
            return self._3ai_data.get(code)

    def _get_macro_composite(self) -> Optional[dict]:
        """Composite macro activity: average of all available series."""
        all_series = list(self._iip_data.values()) + list(self._3ai_data.values())
        if not all_series:
            return None
        moms = [s["mom_pct"] for s in all_series if s.get("mom_pct") is not None]
        yoys = [s["yoy_pct"] for s in all_series if s.get("yoy_pct") is not None]
        if not moms:
            return None
        return {
            "mom_pct": round(sum(moms) / len(moms), 2),
            "yoy_pct": round(sum(yoys) / len(yoys), 2) if yoys else 0,
        }

    @staticmethod
    def _interpret_upstream(mom: float, yoy: float) -> str:
        if yoy > 5:
            return "input costs rising sharply (margin pressure)"
        if yoy > 2:
            return "input costs rising moderately"
        if yoy < -3:
            return "input costs falling (margin benefit)"
        return "input costs stable"

    @staticmethod
    def _interpret_downstream(mom: float, yoy: float) -> str:
        if yoy > 5:
            return "demand expanding strongly"
        if yoy > 2:
            return "demand growing"
        if mom > 1 and yoy > 0:
            return "demand recovering"
        if yoy < -3:
            return "demand contracting"
        if yoy < -1:
            return "demand softening"
        return "demand stable"

    @staticmethod
    def _build_interpretation(upstream: dict, downstream: dict, net_score: float) -> str:
        up_str   = upstream.get("interpretation",   "unknown")
        down_str = downstream.get("interpretation", "unknown")
        if net_score >= 20:
            prefix = "Positive IO outlook: "
        elif net_score <= -20:
            prefix = "Negative IO outlook: "
        else:
            prefix = "Neutral IO outlook: "
        return f"{prefix}{down_str}; {up_str}"
