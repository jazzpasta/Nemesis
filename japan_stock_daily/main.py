"""Main CLI runner for daily Japan stock recommendation system.

Usage:
    python -m japan_stock_daily.main
    python -m japan_stock_daily.main --date 2025-03-18
    python -m japan_stock_daily.main --date today --top 30
"""

import argparse
import logging
import os
from datetime import date, datetime

from japan_stock_daily.config import REPORTS_DIR
from japan_stock_daily.recommender.scorer import DailyScorer
from japan_stock_daily.collectors.boj_collector import BOJCollector
from japan_stock_daily.collectors.estats_collector import EStatsCollector
from japan_stock_daily.collectors.tdnet_collector import TDnetCollector
from japan_stock_daily.collectors.edinet_collector import EdinetCollector
from japan_stock_daily.reports.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nemesis")


def run(target_date: date, top_n: int = 30) -> str:
    """
    Run full daily recommendation pipeline.

    Returns path to the generated HTML report.
    """
    logger.info("=" * 60)
    logger.info("Nemesis Japan Stock Recommendation: %s", target_date)
    logger.info("=" * 60)

    scorer = DailyScorer()
    boj    = BOJCollector()
    estats = EStatsCollector()
    tdnet  = TDnetCollector()
    edinet = EdinetCollector()
    report = ReportGenerator()

    # Run scoring pipeline
    recommendations = scorer.score_all(target_date)

    # Fetch macro data for report
    boj_data    = boj.get_all_indicators()
    estats_data = estats.get_all_indicators()

    # Fetch disclosures for report highlights
    try:
        tdnet_df = tdnet.get_daily_disclosures(target_date)
    except Exception as e:
        logger.warning("TDnet fetch failed: %s", e)
        import pandas as pd
        tdnet_df = pd.DataFrame()

    try:
        edinet_df = edinet.get_daily_signals(target_date)
    except Exception as e:
        logger.warning("EDINET fetch failed: %s", e)
        import pandas as pd
        edinet_df = pd.DataFrame()

    # US overnight from scorer
    us_data = scorer.us_overnight.get_us_overnight_performance()

    # Generate report
    html_path = report.generate(
        target_date=target_date,
        recommendations=recommendations,
        us_data=us_data,
        boj_data=boj_data,
        estats_data=estats_data,
        tdnet_df=tdnet_df,
        edinet_df=edinet_df,
        top_n=top_n,
    )

    top5 = recommendations.head(5)
    logger.info("\nTop 5 Recommendations:")
    for _, row in top5.iterrows():
        logger.info("  #%d %s %s (score=%.1f, strategy=%s)",
                    row["rank"], row["code"], row.get("name", ""),
                    row["composite_score"], row.get("strategy", ""))

    logger.info("\nReport: %s", html_path)
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Nemesis Japan Stock Daily Recommendation")
    parser.add_argument("--date", default="today",
                        help="Date to run (YYYY-MM-DD or 'today')")
    parser.add_argument("--top", type=int, default=30,
                        help="Number of top recommendations to show")
    args = parser.parse_args()

    if args.date == "today":
        target_date = date.today()
    else:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    run(target_date, top_n=args.top)


if __name__ == "__main__":
    main()
