"""Market regime detection based on rolling volatility and returns.

Classifies each trading day into one of three regimes -- bull, bear,
or sideways -- and provides transition analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class RegimeDetector:
    """Detect market regimes from daily close prices.

    Classification uses rolling statistics of log-returns and realised
    volatility:

    * **Bull**: positive rolling return *and* volatility <= median.
    * **Bear**: negative rolling return *and* volatility > median.
    * **Sideways**: everything else (small absolute return or mixed signals).

    Parameters
    ----------
    prices : pd.Series
        Daily close prices with a datetime-like index.
        At least 60 observations are recommended.
    """

    MIN_OBSERVATIONS = 20
    REGIMES = ("bull", "bear", "sideways")

    def __init__(self, prices: pd.Series) -> None:
        if prices is None or len(prices) < self.MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {self.MIN_OBSERVATIONS} observations, "
                f"got {0 if prices is None else len(prices)}."
            )

        self.prices: pd.Series = prices.dropna().sort_index()
        self._regimes: pd.Series | None = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_regimes(
        self,
        n_regimes: int = 3,
        window: int = 20,
    ) -> pd.Series:
        """Classify each day into a market regime.

        Parameters
        ----------
        n_regimes : int
            Kept for interface consistency (always 3: bull/bear/sideways).
        window : int
            Rolling window size for return and volatility calculation.

        Returns
        -------
        pd.Series
            Same index as *prices*, values in ``{'bull', 'bear', 'sideways'}``.
            The first ``window`` entries are ``NaN``.
        """
        log_ret = np.log(self.prices / self.prices.shift(1))

        rolling_return = log_ret.rolling(window=window).mean()
        rolling_vol = log_ret.rolling(window=window).std()

        vol_median = rolling_vol.median()
        ret_threshold = 0.0  # sign of the rolling return

        regime = pd.Series(index=self.prices.index, dtype=object)

        for idx in self.prices.index:
            r = rolling_return.get(idx)
            v = rolling_vol.get(idx)

            if r is None or v is None or pd.isna(r) or pd.isna(v):
                regime[idx] = np.nan
                continue

            if r > ret_threshold and v <= vol_median:
                regime[idx] = "bull"
            elif r < ret_threshold and v > vol_median:
                regime[idx] = "bear"
            else:
                regime[idx] = "sideways"

        self._regimes = regime
        return regime

    # ------------------------------------------------------------------
    # Current state
    # ------------------------------------------------------------------

    def current_regime(self) -> str:
        """Return the regime of the most recent observation.

        Automatically runs :meth:`detect_regimes` if not yet called.
        """
        if self._regimes is None:
            self.detect_regimes()
        assert self._regimes is not None

        last = self._regimes.dropna().iloc[-1] if len(self._regimes.dropna()) > 0 else "unknown"
        return str(last)

    # ------------------------------------------------------------------
    # Statistics per regime
    # ------------------------------------------------------------------

    def regime_statistics(self) -> dict:
        """Per-regime descriptive statistics.

        Returns
        -------
        dict
            ``{regime: {avg_return, avg_volatility, total_days, frequency}}``
        """
        if self._regimes is None:
            self.detect_regimes()
        assert self._regimes is not None

        log_ret = np.log(self.prices / self.prices.shift(1))
        rolling_vol = log_ret.rolling(window=20).std()

        stats: dict[str, dict] = {}

        valid = self._regimes.dropna()
        total_valid = len(valid)

        for regime in self.REGIMES:
            mask = valid == regime
            days = int(mask.sum())
            if days == 0:
                stats[regime] = {
                    "avg_return": 0.0,
                    "avg_volatility": 0.0,
                    "total_days": 0,
                    "frequency": 0.0,
                }
                continue

            regime_dates = mask[mask].index
            avg_ret = float(log_ret.reindex(regime_dates).mean())
            avg_vol = float(rolling_vol.reindex(regime_dates).mean())

            stats[regime] = {
                "avg_return": avg_ret,
                "avg_volatility": avg_vol,
                "total_days": days,
                "frequency": days / total_valid if total_valid else 0.0,
            }

        return stats

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------

    def transition_matrix(self) -> pd.DataFrame:
        """Regime-to-regime transition probability matrix.

        Returns
        -------
        pd.DataFrame
            Rows = *from* regime, columns = *to* regime.
            Values are probabilities that sum to 1 across each row.
        """
        if self._regimes is None:
            self.detect_regimes()
        assert self._regimes is not None

        valid = self._regimes.dropna()
        regimes_list = list(self.REGIMES)
        matrix = pd.DataFrame(
            0.0, index=regimes_list, columns=regimes_list
        )

        prev = None
        for val in valid:
            if prev is not None and prev in regimes_list and val in regimes_list:
                matrix.loc[prev, val] += 1
            prev = val

        # Normalise rows to probabilities
        row_sums = matrix.sum(axis=1)
        for regime in regimes_list:
            if row_sums[regime] > 0:
                matrix.loc[regime] = matrix.loc[regime] / row_sums[regime]

        return matrix

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Comprehensive regime analysis.

        Returns
        -------
        dict
            ``current_regime``, ``regime_statistics``, ``transition_matrix``
            (as nested dict), and total observations analysed.
        """
        return {
            "current_regime": self.current_regime(),
            "regime_statistics": self.regime_statistics(),
            "transition_matrix": self.transition_matrix().to_dict(),
            "total_observations": len(self.prices),
        }
