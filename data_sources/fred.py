"""FRED (Federal Reserve Economic Data) source using the fredapi package."""

from __future__ import annotations

import os
import ssl
import certifi
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import Config

# Fix SSL certificate verification for macOS
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# Mapping of human-readable names to FRED series IDs
FRED_KEY_SERIES: dict[str, str] = {
    "fed_funds_rate": "FEDFUNDS",
    "cpi": "CPIAUCSL",
    "gdp": "GDP",
    "unemployment": "UNRATE",
    "treasury_10y": "DGS10",
    "treasury_2y": "DGS2",
    "wti_crude": "DCOILWTICO",
    "yield_curve_10y2y": "T10Y2Y",
}

# Period string -> approximate timedelta
_PERIOD_MAP: dict[str, timedelta] = {
    "1mo": timedelta(days=31),
    "3mo": timedelta(days=93),
    "6mo": timedelta(days=183),
    "1y": timedelta(days=365),
    "2y": timedelta(days=730),
    "5y": timedelta(days=1825),
    "10y": timedelta(days=3650),
}


class FREDSource:
    """Fetch macroeconomic data from the FRED API.

    Requires a valid ``FRED_API_KEY`` in the environment / Config.
    All public methods degrade gracefully when the key is missing.
    """

    def __init__(self) -> None:
        self._client = None
        api_key = Config.FRED_API_KEY

        if not api_key:
            print("[FRED] Warning: FRED_API_KEY not set. FRED data will be unavailable.")
            return

        try:
            from fredapi import Fred

            self._client = Fred(api_key=api_key)
        except ImportError:
            print("[FRED] Warning: fredapi package not installed. Run: pip install fredapi")
        except Exception as exc:
            print(f"[FRED] Error initialising client: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_available(self) -> bool:
        return self._client is not None

    @staticmethod
    def _start_date_for_period(period: str) -> datetime:
        delta = _PERIOD_MAP.get(period, timedelta(days=365))
        return datetime.now(timezone.utc) - delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_series(
        self,
        series_id: str,
        period: str = "1y",
    ) -> pd.Series:
        """Download a single FRED time series.

        Parameters
        ----------
        series_id : str
            FRED series identifier (e.g. ``"FEDFUNDS"``).
        period : str
            Look-back window (e.g. ``"1y"``, ``"5y"``).

        Returns
        -------
        pd.Series
            Values indexed by date. Empty Series on error.
        """
        if not self._is_available():
            return pd.Series(dtype=float, name=series_id)

        start = self._start_date_for_period(period)

        try:
            data: pd.Series = self._client.get_series(
                series_id,
                observation_start=start,
            )
            data.name = series_id
            return data.dropna()
        except Exception as exc:
            print(f"[FRED] Error fetching series {series_id}: {exc}")
            return pd.Series(dtype=float, name=series_id)

    def fetch_latest(self, series_id: str) -> dict:
        """Return the most recent observation for a FRED series.

        Returns
        -------
        dict
            Keys: value, date, series_id.
        """
        if not self._is_available():
            return {"series_id": series_id, "value": None, "date": None}

        try:
            data = self._client.get_series(series_id)
            data = data.dropna()

            if data.empty:
                return {"series_id": series_id, "value": None, "date": None}

            latest_date = data.index[-1]
            latest_value = float(data.iloc[-1])

            return {
                "series_id": series_id,
                "value": latest_value,
                "date": latest_date.strftime("%Y-%m-%d"),
            }
        except Exception as exc:
            print(f"[FRED] Error fetching latest for {series_id}: {exc}")
            return {"series_id": series_id, "value": None, "date": None, "error": str(exc)}

    def fetch_macro_snapshot(self) -> dict:
        """Return the latest values for all key macroeconomic series.

        Returns
        -------
        dict
            Mapping of friendly name -> dict with value, date, series_id.
        """
        snapshot: dict[str, dict] = {}

        for name, series_id in FRED_KEY_SERIES.items():
            snapshot[name] = self.fetch_latest(series_id)

        return snapshot
