"""U.S. Energy Information Administration (EIA) data source.

Uses the EIA API v2 (https://api.eia.gov/v2/) with requests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from config import Config

EIA_BASE_URL = "https://api.eia.gov/v2/"

# Period string -> approximate number of days
_PERIOD_DAYS: dict[str, int] = {
    "1mo": 31,
    "3mo": 93,
    "6mo": 183,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}


class EIASource:
    """Fetch energy market data from the EIA API v2.

    Requires a valid ``EIA_API_KEY`` in the environment / Config.
    All public methods degrade gracefully when the key is missing.
    """

    def __init__(self) -> None:
        self._api_key: str = Config.EIA_API_KEY
        if not self._api_key:
            print("[EIA] Warning: EIA_API_KEY not set. EIA data will be unavailable.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_available(self) -> bool:
        return bool(self._api_key)

    def _request(
        self,
        endpoint: str,
        params: dict | None = None,
    ) -> dict | None:
        """Send a GET request to the EIA API v2.

        Parameters
        ----------
        endpoint : str
            Relative path under the base URL (e.g. ``"petroleum/pri/spt/data"``).
        params : dict, optional
            Additional query parameters.

        Returns
        -------
        dict or None
            Parsed JSON response, or None on failure.
        """
        if not self._is_available():
            return None

        url = f"{EIA_BASE_URL}{endpoint}"
        query: dict = {"api_key": self._api_key}
        if params:
            query.update(params)

        try:
            resp = requests.get(url, params=query, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"[EIA] Request error ({endpoint}): {exc}")
            return None
        except ValueError as exc:
            print(f"[EIA] JSON decode error ({endpoint}): {exc}")
            return None

    @staticmethod
    def _start_date_for_period(period: str) -> str:
        """Return a YYYY-MM-DD string for the start of the given period."""
        days = _PERIOD_DAYS.get(period, 365)
        dt = datetime.now(timezone.utc) - timedelta(days=days)
        return dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_oil_prices(self, period: str = "1y") -> pd.DataFrame:
        """Fetch crude oil spot prices.

        Uses the ``petroleum/pri/spt/data`` endpoint for WTI and Brent
        spot prices.

        Parameters
        ----------
        period : str
            Look-back window (e.g. ``"1y"``).

        Returns
        -------
        pd.DataFrame
            Columns: period, product-name, value.  Empty on error.
        """
        start = self._start_date_for_period(period)
        params = {
            "frequency": "daily",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "start": start,
            "length": "5000",
        }

        data = self._request("petroleum/pri/spt/data", params)
        if data is None:
            return pd.DataFrame()

        try:
            records = data.get("response", {}).get("data", [])
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            if "period" in df.columns:
                df["period"] = pd.to_datetime(df["period"], errors="coerce")
                df = df.sort_values("period").reset_index(drop=True)
            return df
        except Exception as exc:
            print(f"[EIA] Error parsing oil prices: {exc}")
            return pd.DataFrame()

    def fetch_oil_inventory(self) -> dict:
        """Fetch the latest weekly petroleum stock levels.

        Uses the ``petroleum/stoc/wstk/data`` endpoint.

        Returns
        -------
        dict
            Latest inventory data or an empty dict on error.
        """
        params = {
            "frequency": "weekly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": "10",
        }

        data = self._request("petroleum/stoc/wstk/data", params)
        if data is None:
            return {}

        try:
            records = data.get("response", {}).get("data", [])
            if not records:
                return {}

            # Return a summary of the most recent entries
            result: dict = {
                "records": records,
                "count": len(records),
                "latest_period": records[0].get("period") if records else None,
            }
            return result
        except Exception as exc:
            print(f"[EIA] Error parsing oil inventory: {exc}")
            return {}

    def fetch_crude_production(self, period: str = "1y") -> pd.DataFrame:
        """Fetch U.S. crude oil production data.

        Uses the ``petroleum/crd/crpdn/data`` endpoint.

        Parameters
        ----------
        period : str
            Look-back window.

        Returns
        -------
        pd.DataFrame
            Production data.  Empty on error.
        """
        start = self._start_date_for_period(period)
        params = {
            "frequency": "monthly",
            "data[0]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "start": start,
            "length": "5000",
        }

        data = self._request("petroleum/crd/crpdn/data", params)
        if data is None:
            return pd.DataFrame()

        try:
            records = data.get("response", {}).get("data", [])
            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            if "period" in df.columns:
                df["period"] = pd.to_datetime(df["period"], errors="coerce")
                df = df.sort_values("period").reset_index(drop=True)
            return df
        except Exception as exc:
            print(f"[EIA] Error parsing crude production: {exc}")
            return pd.DataFrame()
