"""Bank of Korea (한국은행) ECOS API data source."""

from __future__ import annotations

import requests
from datetime import datetime, timedelta

from config import Config


# 한국은행 ECOS API 주요 통계코드 (검증 완료)
BOK_SERIES = {
    "base_rate": {
        "table": "722Y001",
        "item": "0101000",
        "cycle": "M",
        "label": "기준금리",
        "unit": "%",
    },
    "cpi": {
        "table": "901Y009",  # 소비자물가지수 총지수
        "item": "0",
        "cycle": "M",
        "label": "소비자물가",
        "unit": "",
    },
    "unemployment": {
        "table": "901Y027",
        "item": "I61BC",
        "cycle": "M",
        "label": "실업률",
        "unit": "%",
    },
    "gov_bond_3y": {
        "table": "817Y002",
        "item": "010200000",
        "cycle": "D",  # 일별 데이터
        "label": "국고채 3년",
        "unit": "%",
    },
    "gov_bond_10y": {
        "table": "817Y002",
        "item": "010210000",
        "cycle": "D",  # 일별 데이터
        "label": "국고채 10년",
        "unit": "%",
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
                 start_date: str, end_date: str) -> list[dict]:
        """Make a request to the ECOS API."""
        if not self.api_key:
            return []

        # URL format: /api/StatisticSearch/{key}/{format}/{lang}/{start}/{end}/{table}/{cycle}/{start_date}/{end_date}/{item}
        url = (
            f"{BASE_URL}/{self.api_key}/json/kr/1/10/"
            f"{table_code}/{cycle}/{start_date}/{end_date}/{item_code}"
        )

        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if "StatisticSearch" not in data:
                error = data.get("RESULT", {})
                if error:
                    print(f"[BOK] API error: {error.get('MESSAGE', 'Unknown')}")
                return []

            return data["StatisticSearch"].get("row", [])
        except Exception as e:
            print(f"[BOK] Error fetching {table_code}: {e}")
            return []

    def fetch_latest(self, series_key: str) -> dict:
        """Fetch the latest value for a given series."""
        series = BOK_SERIES.get(series_key)
        if not series:
            return {"error": f"Unknown series: {series_key}"}

        now = datetime.now()
        # Calculate date range based on cycle
        if series["cycle"] == "Q":
            start = (now - timedelta(days=365)).strftime("%YQ1")
            end = now.strftime("%YQ4")
        elif series["cycle"] == "D":
            start = (now - timedelta(days=30)).strftime("%Y%m%d")
            end = now.strftime("%Y%m%d")
        else:  # M
            start = (now - timedelta(days=180)).strftime("%Y%m")
            end = now.strftime("%Y%m")

        rows = self._request(
            series["table"], series["item"], series["cycle"], start, end
        )

        if not rows:
            return {
                "series_key": series_key,
                "label": series["label"],
                "unit": series["unit"],
                "value": None,
                "date": None,
            }

        # Get the latest row
        latest = rows[-1]
        try:
            value = float(latest.get("DATA_VALUE", 0))
        except (ValueError, TypeError):
            value = None

        return {
            "series_key": series_key,
            "label": series["label"],
            "unit": series["unit"],
            "value": value,
            "date": latest.get("TIME", ""),
        }

    def fetch_macro_snapshot(self) -> dict:
        """Fetch all key Korean macro indicators."""
        if not self.api_key:
            return {}

        result = {}
        for key in BOK_SERIES:
            data = self.fetch_latest(key)
            result[key] = data

        return result
