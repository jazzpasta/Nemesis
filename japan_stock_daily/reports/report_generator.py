"""HTML and CSV report generator for daily stock recommendations."""

import os
import logging
from datetime import date
from typing import Optional

import pandas as pd
from jinja2 import Template

from japan_stock_daily.config import REPORTS_DIR

logger = logging.getLogger(__name__)

# Inline Jinja2 HTML template (no external file needed)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>日本株デイリー推奨 {{ date_str }}</title>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
  <style>
    body { font-family: 'Helvetica Neue', Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }
    .container { max-width: 1400px; margin: 0 auto; }
    h1 { color: #1a1a2e; border-bottom: 3px solid #e94560; padding-bottom: 10px; }
    h2 { color: #16213e; margin-top: 30px; }
    .header-bar { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
    .kpi-card { background: white; border-radius: 8px; padding: 15px 20px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); min-width: 160px; }
    .kpi-label { font-size: 12px; color: #888; }
    .kpi-value { font-size: 22px; font-weight: bold; color: #1a1a2e; }
    .kpi-value.up { color: #e94560; }
    .kpi-value.down { color: #0f3460; }
    .strategy-badge { display: inline-block; padding: 4px 12px; border-radius: 20px;
                      font-size: 13px; font-weight: bold; margin-bottom: 15px; }
    .multi_factor { background: #e8f4fd; color: #0f3460; }
    .shock_recovery { background: #fde8e8; color: #e94560; }
    table.dataTable { width: 100% !important; }
    .score-bar { display: inline-block; height: 8px; background: linear-gradient(90deg, #e94560, #0f3460);
                 border-radius: 4px; margin-left: 8px; vertical-align: middle; }
    .disc-positive { color: #27ae60; font-weight: bold; }
    .disc-negative { color: #e74c3c; }
    .signal-text { font-size: 11px; color: #666; max-width: 300px; white-space: pre-wrap; }
    .section-card { background: white; border-radius: 8px; padding: 20px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .macro-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
    .macro-item { background: #f8f9fa; border-radius: 6px; padding: 12px; }
    .tdnet-item { border-left: 4px solid #e94560; padding: 8px 12px; margin-bottom: 8px;
                  background: #fff9f9; border-radius: 0 6px 6px 0; }
    .tdnet-positive { border-color: #27ae60; background: #f0fff4; }
    .footer { text-align: center; color: #888; font-size: 12px; margin-top: 30px; }
  </style>
</head>
<body>
<div class="container">
  <h1>🇯🇵 日本株デイリー推奨レポート</h1>
  <p><strong>{{ date_str }}</strong> ({{ weekday }})</p>

  <!-- Strategy Badge -->
  <div class="strategy-badge {{ strategy }}">
    {% if strategy == 'shock_recovery' %}
    ⚡ ショック回復戦略 (US市場下落後)
    {% else %}
    📊 マルチファクター戦略
    {% endif %}
  </div>

  <!-- Market Context KPIs -->
  <h2>マーケット概況</h2>
  <div class="header-bar">
    {% if us_data.sp500_futures_chg is not none %}
    <div class="kpi-card">
      <div class="kpi-label">S&P500先物</div>
      <div class="kpi-value {{ 'up' if us_data.sp500_futures_chg > 0 else 'down' }}">
        {{ '%+.2f' % us_data.sp500_futures_chg }}%
      </div>
    </div>
    {% endif %}
    {% if us_data.vix is not none %}
    <div class="kpi-card">
      <div class="kpi-label">VIX</div>
      <div class="kpi-value {{ 'down' if us_data.vix > 25 else '' }}">{{ '%.1f' % us_data.vix }}</div>
    </div>
    {% endif %}
    {% if us_data.usd_jpy is not none %}
    <div class="kpi-card">
      <div class="kpi-label">USD/JPY</div>
      <div class="kpi-value">{{ '%.2f' % us_data.usd_jpy }}</div>
    </div>
    {% endif %}
    {% if boj_data.policy_rate is not none %}
    <div class="kpi-card">
      <div class="kpi-label">BOJ政策金利</div>
      <div class="kpi-value">{{ '%.2f' % boj_data.policy_rate }}%</div>
    </div>
    {% endif %}
    {% if boj_data.jgb_10y is not none %}
    <div class="kpi-card">
      <div class="kpi-label">10年JGB利回り</div>
      <div class="kpi-value">{{ '%.2f' % boj_data.jgb_10y }}%</div>
    </div>
    {% endif %}
  </div>

  <!-- Top Recommendations -->
  <div class="section-card">
    <h2>📈 トップ推奨銘柄 Top {{ recommendations|length }}</h2>
    <table id="recommendations-table" class="display">
      <thead>
        <tr>
          <th>ランク</th>
          <th>コード</th>
          <th>銘柄名</th>
          <th>セクター</th>
          <th>総合スコア</th>
          <th>開示</th>
          <th>需給</th>
          <th>米国夜間</th>
          <th>モメンタム</th>
          <th>マクロ</th>
          <th>主要シグナル</th>
          <th>戦略</th>
        </tr>
      </thead>
      <tbody>
        {% for r in recommendations %}
        <tr>
          <td>{{ r.rank }}</td>
          <td><strong>{{ r.code }}</strong></td>
          <td>{{ r.name }}</td>
          <td>{{ r.sector33 }}</td>
          <td>
            <strong>{{ '%.1f' % r.composite_score }}</strong>
            <span class="score-bar" style="width: {{ r.composite_score * 0.8 }}px"></span>
          </td>
          <td class="{{ 'disc-positive' if r.disc_score > 60 else ('disc-negative' if r.disc_score < 25 else '') }}">
            {{ '%.0f' % r.disc_score }}
          </td>
          <td>{{ '%.0f' % r.sd_score }}</td>
          <td>{{ '%.0f' % r.us_score }}</td>
          <td>{{ '%.0f' % r.momentum_score }}</td>
          <td>{{ '%.0f' % r.macro_score }}</td>
          <td class="signal-text">
            {% if r.disc_signals %}{{ r.disc_signals[:120] }}{% endif %}
            {% if r.us_signals %}<br>{{ r.us_signals[:80] }}{% endif %}
          </td>
          <td>{{ r.strategy }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <!-- TDnet Highlights -->
  {% if tdnet_highlights %}
  <div class="section-card">
    <h2>📣 TDnet 適時開示ハイライト</h2>
    {% for item in tdnet_highlights %}
    <div class="tdnet-item {{ 'tdnet-positive' if item.category_score > 60 else '' }}">
      <strong>{{ item.code }} {{ item.company }}</strong>
      <span style="margin-left:10px; font-size:12px; color:#888;">{{ item.time }}</span>
      <br>
      <a href="{{ item.pdf_url }}" target="_blank">{{ item.title }}</a>
      <span style="margin-left:10px; background:#e8f4fd; padding:2px 8px; border-radius:10px;
                   font-size:11px;">{{ item.category }}</span>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- EDINET Highlights -->
  {% if edinet_highlights %}
  <div class="section-card">
    <h2>📋 EDINET 大量保有・臨時報告</h2>
    {% for item in edinet_highlights %}
    <div class="tdnet-item {{ 'tdnet-positive' if item.direction in ('increase', 'new') else '' }}">
      <strong>{{ item.code }} {{ item.filer_name }}</strong>
      <span style="margin-left:8px; background:#f0f0f0; padding:2px 8px; border-radius:10px;
                   font-size:11px;">{{ item.doc_type }}</span>
      <br>
      <span style="font-size:13px;">{{ item.event_category }}</span>
      {% if item.direction %} → <strong>{{ item.direction }}</strong>{% endif %}
      {% if item.ownership_pct %} ({{ item.ownership_pct }}%){% endif %}
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- Macro Context -->
  <div class="section-card">
    <h2>🌐 マクロ環境</h2>
    <div class="macro-grid">
      {% for key, val in macro_indicators.items() %}
      <div class="macro-item">
        <div class="kpi-label">{{ key }}</div>
        <div style="font-size:16px; font-weight:bold;">
          {% if val is not none %}{{ val }}{% else %}—{% endif %}
        </div>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="footer">
    生成日時: {{ generated_at }} | Nemesis Japan Stock Recommendation System
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script>
$(document).ready(function() {
  $('#recommendations-table').DataTable({
    order: [[4, 'desc']],
    pageLength: 25,
    language: { url: '//cdn.datatables.net/plug-ins/1.13.6/i18n/ja.json' }
  });
});
</script>
</body>
</html>"""


class ReportGenerator:
    """Generates HTML and CSV daily recommendation reports."""

    def __init__(self, output_dir: str = REPORTS_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        target_date: date,
        recommendations: pd.DataFrame,
        us_data: dict,
        boj_data: dict,
        estats_data: dict,
        tdnet_df: Optional[pd.DataFrame] = None,
        edinet_df: Optional[pd.DataFrame] = None,
        top_n: int = 30,
    ) -> str:
        """
        Generate HTML report for target_date.

        Returns path to the generated HTML file.
        """
        date_str = target_date.strftime("%Y年%m月%d日")
        weekdays_jp = ["月", "火", "水", "木", "金", "土", "日"]
        weekday = weekdays_jp[target_date.weekday()]

        # Top N recommendations
        top_recs = recommendations.head(top_n).fillna("—").to_dict("records")

        # TDnet highlights (top 20 by score, exclude boring ones)
        tdnet_highlights = []
        if tdnet_df is not None and not tdnet_df.empty:
            tdnet_top = tdnet_df[tdnet_df["category_score"] >= 45].head(20)
            tdnet_highlights = tdnet_top.fillna("").to_dict("records")

        # EDINET highlights
        edinet_highlights = []
        if edinet_df is not None and not edinet_df.empty:
            edinet_highlights = edinet_df.head(20).fillna("").to_dict("records")

        # Macro indicators for display
        macro_indicators = {
            "BOJ政策金利":      f"{boj_data.get('policy_rate', '—')}%",
            "10年JGB":          f"{boj_data.get('jgb_10y', '—')}%",
            "USD/JPY":          f"{boj_data.get('usd_jpy', '—')}",
            "CPI(コア)":        f"{boj_data.get('cpi_core', '—')}",
            "完全失業率":       f"{estats_data.get('unemployment_latest', '—')}%",
            "小売販売(前年比)": f"{estats_data.get('retail_sales_yoy', '—')}%",
        }

        # Strategy used
        strategy = recommendations["strategy"].iloc[0] if not recommendations.empty and "strategy" in recommendations.columns else "multi_factor"

        from datetime import datetime
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

        template = Template(HTML_TEMPLATE)
        html = template.render(
            date_str=date_str,
            weekday=weekday,
            strategy=strategy,
            recommendations=top_recs,
            us_data=us_data,
            boj_data=boj_data,
            estats_data=estats_data,
            tdnet_highlights=tdnet_highlights,
            edinet_highlights=edinet_highlights,
            macro_indicators=macro_indicators,
            generated_at=generated_at,
        )

        # Write HTML
        html_path = os.path.join(self.output_dir, f"{target_date.strftime('%Y-%m-%d')}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML report written: %s", html_path)

        # Write CSV
        csv_path = os.path.join(self.output_dir, f"{target_date.strftime('%Y-%m-%d')}.csv")
        recommendations.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info("CSV report written: %s", csv_path)

        return html_path
