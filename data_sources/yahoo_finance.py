"""Yahoo Finance data source using the yfinance package."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from config import Config
from data_sources.base import DataSource


class YahooFinanceSource(DataSource):
    """Fetch market data via Yahoo Finance (yfinance)."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(symbol: str) -> str:
        """Map a friendly asset name (e.g. 'WTI') to a Yahoo ticker."""
        return Config.resolve_symbol(symbol)

    @staticmethod
    def _safe_float(value: object) -> float | None:
        """Convert a value to float, returning None on failure."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_current_price(self, symbol: str) -> dict:
        """Return the latest quote for *symbol*.

        Returns
        -------
        dict
            Keys: price, change, change_pct, volume, timestamp.
        """
        ticker_str = self._resolve(symbol)

        try:
            ticker = yf.Ticker(ticker_str)
            info = ticker.fast_info

            price = self._safe_float(getattr(info, "last_price", None))
            prev_close = self._safe_float(getattr(info, "previous_close", None))

            change: float | None = None
            change_pct: float | None = None
            if price is not None and prev_close is not None and prev_close != 0:
                change = round(price - prev_close, 4)
                change_pct = round((change / prev_close) * 100, 4)

            volume = getattr(info, "last_volume", None)
            if volume is not None:
                try:
                    volume = int(volume)
                except (TypeError, ValueError):
                    volume = None

            return {
                "symbol": ticker_str,
                "price": price,
                "change": change,
                "change_pct": change_pct,
                "volume": volume,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            print(f"[YahooFinance] Error fetching price for {ticker_str}: {exc}")
            return {
                "symbol": ticker_str,
                "price": None,
                "change": None,
                "change_pct": None,
                "volume": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }

    def fetch_history(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV history for *symbol*.

        Parameters
        ----------
        symbol : str
            Friendly name or raw Yahoo ticker.
        period : str
            Look-back window (e.g. ``"1mo"``, ``"3mo"``, ``"1y"``).
        interval : str
            Bar size (e.g. ``"1d"``, ``"1h"``, ``"5m"``).

        Returns
        -------
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume. Index is DatetimeIndex.
        """
        ticker_str = self._resolve(symbol)

        try:
            ticker = yf.Ticker(ticker_str)
            df: pd.DataFrame = ticker.history(period=period, interval=interval)

            if df.empty:
                print(f"[YahooFinance] No history returned for {ticker_str}")
                return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

            # Keep only standard OHLCV columns
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            return df[keep]

        except Exception as exc:
            print(f"[YahooFinance] Error fetching history for {ticker_str}: {exc}")
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def fetch_multiple(
        self,
        symbols: list[str],
        period: str = "3mo",
    ) -> dict[str, pd.DataFrame]:
        """Batch-fetch history for a list of symbols.

        Parameters
        ----------
        symbols : list[str]
            Friendly names or raw Yahoo tickers.
        period : str
            Look-back window.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of original symbol -> OHLCV DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}
        ticker_map = {sym: self._resolve(sym) for sym in symbols}
        tickers_str = " ".join(ticker_map.values())

        try:
            data = yf.download(
                tickers=tickers_str,
                period=period,
                interval="1d",
                group_by="ticker",
                threads=True,
            )

            for original, resolved in ticker_map.items():
                try:
                    if len(ticker_map) == 1:
                        df = data
                    else:
                        df = data[resolved] if resolved in data.columns.get_level_values(0) else pd.DataFrame()

                    if isinstance(df, pd.DataFrame) and not df.empty:
                        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                        results[original] = df[keep].dropna(how="all")
                    else:
                        results[original] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
                except Exception as inner_exc:
                    print(f"[YahooFinance] Error processing {original} ({resolved}): {inner_exc}")
                    results[original] = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        except Exception as exc:
            print(f"[YahooFinance] Batch download failed: {exc}")
            for sym in symbols:
                results.setdefault(sym, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))

        return results
