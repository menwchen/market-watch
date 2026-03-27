"""Technical analysis indicators for OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalAnalyzer:
    """Calculate technical indicators from OHLCV price data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume.
        Index should be a DatetimeIndex.
    """

    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

    def __init__(self, df: pd.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        self.df = df.copy()

    # ------------------------------------------------------------------
    # Moving Averages
    # ------------------------------------------------------------------

    def sma(self, period: int = 20) -> pd.Series:
        """Simple Moving Average of Close prices."""
        return self.df["Close"].rolling(window=period).mean()

    def ema(self, period: int = 20) -> pd.Series:
        """Exponential Moving Average of Close prices."""
        return self.df["Close"].ewm(span=period, adjust=False).mean()

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def rsi(self, period: int = 14) -> pd.Series:
        """Relative Strength Index (0-100).

        Uses the smoothed (Wilder) method: first value is SMA of
        gains/losses, subsequent values use exponential smoothing.
        """
        delta = self.df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_series.name = "RSI"
        return rsi_series

    def macd(
        self,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> dict[str, pd.Series]:
        """MACD indicator.

        Returns
        -------
        dict with keys 'macd', 'signal', 'histogram'.
        """
        ema_fast = self.df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["Close"].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def bollinger_bands(self, period: int = 20, std: float = 2.0) -> dict[str, pd.Series]:
        """Bollinger Bands.

        Returns
        -------
        dict with keys 'upper', 'middle', 'lower'.
        """
        middle = self.df["Close"].rolling(window=period).mean()
        rolling_std = self.df["Close"].rolling(window=period).std()
        return {
            "upper": middle + std * rolling_std,
            "middle": middle,
            "lower": middle - std * rolling_std,
        }

    def atr(self, period: int = 14) -> pd.Series:
        """Average True Range."""
        high = self.df["High"]
        low = self.df["Low"]
        prev_close = self.df["Close"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr_series = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        atr_series.name = "ATR"
        return atr_series

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def volume_profile(self) -> dict[str, float]:
        """Basic volume statistics.

        Returns
        -------
        dict with 'average_volume' and 'volume_trend' (percent change of
        recent 5-day avg vs 20-day avg).
        """
        vol = self.df["Volume"]
        avg_volume = float(vol.mean())
        recent_avg = float(vol.tail(5).mean())
        longer_avg = float(vol.tail(20).mean()) if len(vol) >= 20 else avg_volume
        trend_pct = ((recent_avg / longer_avg) - 1.0) * 100.0 if longer_avg != 0 else 0.0
        return {
            "average_volume": avg_volume,
            "volume_trend": round(trend_pct, 2),
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Comprehensive summary of all indicators plus trend/signal info.

        Returns a flat dict suitable for JSON serialisation.
        """
        close = self.df["Close"]
        latest_close = float(close.iloc[-1])

        # Indicators (latest values)
        sma_20 = self.sma(20)
        sma_50 = self.sma(50)
        ema_20 = self.ema(20)
        rsi_val = self.rsi()
        macd_data = self.macd()
        bb = self.bollinger_bands()
        atr_val = self.atr()
        vol = self.volume_profile()

        latest_sma20 = float(sma_20.iloc[-1]) if not np.isnan(sma_20.iloc[-1]) else None
        latest_sma50 = float(sma_50.iloc[-1]) if not np.isnan(sma_50.iloc[-1]) else None
        latest_ema20 = float(ema_20.iloc[-1]) if not np.isnan(ema_20.iloc[-1]) else None
        latest_rsi = float(rsi_val.iloc[-1]) if not np.isnan(rsi_val.iloc[-1]) else None
        latest_macd = float(macd_data["macd"].iloc[-1])
        latest_signal = float(macd_data["signal"].iloc[-1])
        latest_histogram = float(macd_data["histogram"].iloc[-1])
        latest_bb_upper = float(bb["upper"].iloc[-1]) if not np.isnan(bb["upper"].iloc[-1]) else None
        latest_bb_lower = float(bb["lower"].iloc[-1]) if not np.isnan(bb["lower"].iloc[-1]) else None
        latest_atr = float(atr_val.iloc[-1]) if not np.isnan(atr_val.iloc[-1]) else None

        # Trend direction heuristic
        trend = "neutral"
        if latest_sma20 is not None and latest_sma50 is not None:
            if latest_close > latest_sma20 > latest_sma50:
                trend = "bullish"
            elif latest_close < latest_sma20 < latest_sma50:
                trend = "bearish"

        # Signals
        signals: list[str] = []
        if latest_rsi is not None:
            if latest_rsi > 70:
                signals.append("RSI overbought")
            elif latest_rsi < 30:
                signals.append("RSI oversold")
        if latest_histogram > 0 and macd_data["histogram"].iloc[-2] <= 0:
            signals.append("MACD bullish crossover")
        elif latest_histogram < 0 and macd_data["histogram"].iloc[-2] >= 0:
            signals.append("MACD bearish crossover")
        if latest_bb_upper is not None:
            if latest_close > latest_bb_upper:
                signals.append("Price above upper Bollinger Band")
            elif latest_close < latest_bb_lower:  # type: ignore[operator]
                signals.append("Price below lower Bollinger Band")

        return {
            "close": latest_close,
            "sma_20": latest_sma20,
            "sma_50": latest_sma50,
            "ema_20": latest_ema20,
            "rsi": latest_rsi,
            "macd": latest_macd,
            "macd_signal": latest_signal,
            "macd_histogram": latest_histogram,
            "bb_upper": latest_bb_upper,
            "bb_lower": latest_bb_lower,
            "atr": latest_atr,
            "volume_avg": vol["average_volume"],
            "volume_trend_pct": vol["volume_trend"],
            "trend": trend,
            "signals": signals,
        }
