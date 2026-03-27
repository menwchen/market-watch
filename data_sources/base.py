"""Abstract base class for all data sources."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Base interface that every market data source must implement."""

    @abstractmethod
    def fetch_current_price(self, symbol: str) -> dict:
        """Return the latest quote for *symbol*.

        Returns
        -------
        dict
            Keys: price, change, change_pct, volume, timestamp.
        """

    @abstractmethod
    def fetch_history(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV history for *symbol*.

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume.
            Index: DatetimeIndex.
        """

    @abstractmethod
    def fetch_multiple(
        self,
        symbols: list[str],
        period: str = "3mo",
    ) -> dict[str, pd.DataFrame]:
        """Batch-fetch history for several symbols.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of symbol -> OHLCV DataFrame.
        """
