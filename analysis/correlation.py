"""Cross-asset correlation analysis."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


class CorrelationAnalyzer:
    """Compute pairwise correlations across multiple assets.

    Parameters
    ----------
    assets : dict[str, pd.DataFrame]
        Mapping of asset name to OHLCV DataFrame.  Each DataFrame must
        contain a 'Close' column with a DatetimeIndex.
    """

    def __init__(self, assets: dict[str, pd.DataFrame]) -> None:
        if len(assets) < 2:
            raise ValueError("At least two assets are required for correlation analysis.")
        self.assets = assets
        self._returns: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_returns(self) -> pd.DataFrame:
        """Build a DataFrame of daily log-returns aligned on a common index."""
        if self._returns is not None:
            return self._returns

        series_map: dict[str, pd.Series] = {}
        for name, df in self.assets.items():
            close = df["Close"].dropna()
            series_map[name] = np.log(close / close.shift(1))

        self._returns = pd.DataFrame(series_map).dropna()
        return self._returns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correlation_matrix(self) -> pd.DataFrame:
        """Pairwise Pearson correlation of log-returns."""
        returns = self._build_returns()
        return returns.corr()

    def rolling_correlation(
        self,
        asset1: str,
        asset2: str,
        window: int = 30,
    ) -> pd.Series:
        """Rolling Pearson correlation between two assets.

        Parameters
        ----------
        asset1, asset2 : str
            Names matching keys in the ``assets`` dict.
        window : int
            Rolling window size in trading days.
        """
        returns = self._build_returns()
        if asset1 not in returns.columns:
            raise KeyError(f"Asset '{asset1}' not found. Available: {list(returns.columns)}")
        if asset2 not in returns.columns:
            raise KeyError(f"Asset '{asset2}' not found. Available: {list(returns.columns)}")

        rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
        rolling_corr.name = f"{asset1}_vs_{asset2}_corr_{window}d"
        return rolling_corr

    def summary(self) -> dict:
        """Identify strongest positive/negative correlations and notable pairs.

        Returns
        -------
        dict with keys:
            strongest_positive : tuple (asset1, asset2, corr)
            strongest_negative : tuple (asset1, asset2, corr)
            high_correlation_pairs : list of (asset1, asset2, corr) with |corr| > 0.7
            low_correlation_pairs : list of (asset1, asset2, corr) with |corr| < 0.2
        """
        corr_matrix = self.correlation_matrix()
        names = list(corr_matrix.columns)

        best_pos: tuple[str, str, float] = ("", "", -2.0)
        best_neg: tuple[str, str, float] = ("", "", 2.0)
        high_pairs: list[tuple[str, str, float]] = []
        low_pairs: list[tuple[str, str, float]] = []

        for a, b in combinations(names, 2):
            c = float(corr_matrix.loc[a, b])
            if np.isnan(c):
                continue
            if c > best_pos[2]:
                best_pos = (a, b, round(c, 4))
            if c < best_neg[2]:
                best_neg = (a, b, round(c, 4))
            if abs(c) > 0.7:
                high_pairs.append((a, b, round(c, 4)))
            if abs(c) < 0.2:
                low_pairs.append((a, b, round(c, 4)))

        # Sort by absolute correlation descending / ascending
        high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        low_pairs.sort(key=lambda x: abs(x[2]))

        return {
            "strongest_positive": best_pos if best_pos[0] else None,
            "strongest_negative": best_neg if best_neg[0] else None,
            "high_correlation_pairs": high_pairs,
            "low_correlation_pairs": low_pairs,
        }
