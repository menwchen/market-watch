"""Bank of Korea (한국은행) ECOS API data source."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta

from config import Config


# 한국은행 ECOS API 주요 통계코드 (검증 완료)
BOK_SERIES = {
    # 통화정책 & 물가
    "base_rate": {
        "table": "722Y001", "item": "0101000", "cycle": "M",
        "label": "기준금리", "unit": "%", "group": "monetary",
    },
    "cpi": {
        "table": "901Y009", "item": "0", "cycle": "M",
        "label": "소비자물가지수", "unit": "", "group": "monetary",
    },
    # 고용
    "unemployment": {
        "table": "901Y027", "item": "I61BC", "cycle": "M",
        "label": "실업률", "unit": "%", "group": "economy",
    },
    # 채권시장
    "gov_bond_3y": {
        "table": "817Y002", "item": "010200000", "cycle": "D",
        "label": "국고채 3년", "unit": "%", "group": "bond",
    },
    "gov_bond_10y": {
        "table": "817Y002", "item": "010210000", "cycle": "D",
        "label": "국고채 10년", "unit": "%", "group": "bond",
    },
    # 무역 (국제수지 - 301Y013)
    "current_account": {
        "table": "301Y013", "item": "000000", "cycle": "M",
        "label": "경상수지", "unit": "백만$", "group": "trade",
    },
    "trade_balance": {
        "table": "301Y013", "item": "100000", "cycle": "M",
        "label": "상품수지", "unit": "백만$", "group": "trade",
    },
    "exports": {
        "table": "301Y013", "item": "110000", "cycle": "M",
        "label": "수출", "unit": "백만$", "group": "trade",
    },
    "imports": {
        "table": "301Y013", "item": "120000", "cycle": "M",
        "label": "수입", "unit": "백만$", "group": "trade",
    },
}

BASE_URL = "https://ecos.bok.or.kr/api/StatisticSearch"


class BOKSource:
    """Fetch Korean macroeconomic data from Bank of Korea ECOS API."""

    def __init__(self):
        self.api_key = Config.BOK_API_KEY
        if not self.api_key:
            print("[BOK] Warning: BOK_API_KEY not set. Korean macro data will be unavailable.")

    def _request(self, table_code: str, item_code: str, cycle: str,
                 start_date: str, end_date: str, count: int = 30) -> list[dict]:
        if not self.api_key:
            return []

        url = (
            f"{BASE_URL}/{self.api_key}/json/kr/1/{count}/"
            f"{table_code}/{cycle}/{start_date}/{end_date}/{item_code}"
        )

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "StatisticSearch" not in data:
                return []

            return data["StatisticSearch"].get("row", [])
        except Exception as e:
            print(f"[BOK] Error fetching {table_code}/{item_code}: {e}")
            return []

    def _date_range(self, cycle: str, months_back: int = 24):
        now = datetime.now()
        past = now - timedelta(days=months_back * 30)
        if cycle == "Q":
            return past.strftime("%YQ1"), now.strftime("%YQ4")
        elif cycle == "D":
            return (now - timedelta(days=30)).strftime("%Y%m%d"), now.strftime("%Y%m%d")
        else:
            return past.strftime("%Y%m"), now.strftime("%Y%m")

    def fetch_latest(self, series_key: str) -> dict:
        series = BOK_SERIES.get(series_key)
        if not series:
            return {"error": f"Unknown series: {series_key}"}

        start, end = self._date_range(series["cycle"])
        rows = self._request(series["table"], series["item"], series["cycle"], start, end)

        if not rows:
            return {
                "series_key": series_key, "label": series["label"],
                "unit": series["unit"], "group": series["group"],
                "value": None, "date": None,
            }

        latest = rows[-1]
        try:
            value = float(latest.get("DATA_VALUE", 0))
        except (ValueError, TypeError):
            value = None

        return {
            "series_key": series_key, "label": series["label"],
            "unit": series["unit"], "group": series["group"],
            "value": value, "date": latest.get("TIME", ""),
        }

    def fetch_series(self, series_key: str, months_back: int = 12) -> list[dict]:
        """Fetch time series data for charting."""
        series = BOK_SERIES.get(series_key)
        if not series:
            return []

        start, end = self._date_range(series["cycle"], months_back)
        rows = self._request(series["table"], series["item"], series["cycle"], start, end, count=100)

        result = []
        for row in rows:
            try:
                val = float(row.get("DATA_VALUE", 0))
                result.append({"date": row.get("TIME", ""), "value": val})
            except (ValueError, TypeError):
                continue
        return result

    def fetch_trade_summary(self) -> dict:
        """Fetch comprehensive trade data with derived metrics."""
        if not self.api_key:
            return {}

        exports_data = self.fetch_latest("exports")
        imports_data = self.fetch_latest("imports")
        balance_data = self.fetch_latest("trade_balance")
        ca_data = self.fetch_latest("current_account")

        exports_val = exports_data.get("value")
        imports_val = imports_data.get("value")

        # 실질무역손익 (상품수지)
        real_trade_balance = balance_data.get("value")

        # 수출입 비율
        export_import_ratio = None
        if exports_val and imports_val and imports_val > 0:
            export_import_ratio = round(exports_val / imports_val * 100, 1)

        # 수출 12개월 시계열 (YoY 계산용)
        exports_series = self.fetch_series("exports", months_back=24)
        exports_yoy = None
        if len(exports_series) >= 13:
            current = exports_series[-1]["value"]
            year_ago = exports_series[-13]["value"]
            if year_ago > 0:
                exports_yoy = round((current / year_ago - 1) * 100, 1)

        return {
            "exports": exports_data,
            "imports": imports_data,
            "trade_balance": balance_data,
            "current_account": ca_data,
            "real_trade_balance": real_trade_balance,
            "export_import_ratio": export_import_ratio,
            "exports_yoy": exports_yoy,
            "exports_series": exports_series[-12:] if exports_series else [],
        }

    def fetch_macro_snapshot(self) -> dict:
        if not self.api_key:
            return {}

        result = {}
        for key in BOK_SERIES:
            data = self.fetch_latest(key)
            result[key] = data

        return result
