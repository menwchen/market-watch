"""Monte Carlo price simulation using Geometric Brownian Motion (GBM).

Generates thousands of possible future price paths based on historical
drift and volatility, then derives probabilistic forecasts and risk metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """Simulate future asset prices via Geometric Brownian Motion.

    Parameters
    ----------
    prices : pd.Series
        Historical daily Close prices. Index should be datetime-like.
        At least 30 observations are recommended for meaningful statistics.
    """

    MIN_OBSERVATIONS = 5

    def __init__(self, prices: pd.Series) -> None:
        if prices is None or len(prices) < self.MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {self.MIN_OBSERVATIONS} price observations, "
                f"got {0 if prices is None else len(prices)}."
            )

        self.prices: pd.Series = prices.dropna().sort_index()
        self.current_price: float = float(self.prices.iloc[-1])
        self._mu: float | None = None
        self._sigma: float | None = None
        self._simulated: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_params(self) -> tuple[float, float]:
        """Compute daily drift (mu) and volatility (sigma) from log-returns.

        Returns
        -------
        tuple[float, float]
            (mu, sigma) -- annualised values are *not* used here; these are
            daily parameters suitable for direct GBM stepping.
        """
        log_returns = np.log(self.prices / self.prices.shift(1)).dropna()

        if len(log_returns) == 0:
            raise ValueError("Cannot compute parameters: no valid returns.")

        sigma = float(log_returns.std())
        # drift adjusted for the continuous-compounding correction
        mu = float(log_returns.mean()) - 0.5 * sigma ** 2

        self._mu = mu
        self._sigma = sigma
        return mu, sigma

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self, days: int = 30, num_simulations: int = 10_000
    ) -> np.ndarray:
        """Run GBM Monte Carlo simulation.

        Parameters
        ----------
        days : int
            Forecast horizon in trading days.
        num_simulations : int
            Number of independent price paths.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_simulations, days)`` with simulated daily
            closing prices.  Column 0 is the first day *after* the last
            historical observation.
        """
        if days < 1:
            raise ValueError("days must be >= 1")
        if num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")

        mu, sigma = self._calculate_params()

        rng = np.random.default_rng()
        # Random shocks: shape (num_simulations, days)
        Z = rng.standard_normal((num_simulations, days))

        # Daily GBM increments
        daily_returns = np.exp(mu + sigma * Z)

        # Build price paths
        paths = np.zeros((num_simulations, days))
        paths[:, 0] = self.current_price * daily_returns[:, 0]
        for t in range(1, days):
            paths[:, t] = paths[:, t - 1] * daily_returns[:, t]

        self._simulated = paths
        return paths

    # ------------------------------------------------------------------
    # Analytics on simulated paths
    # ------------------------------------------------------------------

    def _ensure_simulated(self) -> np.ndarray:
        if self._simulated is None:
            self.simulate()
        assert self._simulated is not None
        return self._simulated

    @property
    def final_prices(self) -> np.ndarray:
        """1-D array of terminal prices across all simulations."""
        paths = self._ensure_simulated()
        return paths[:, -1]

    def statistics(self) -> dict:
        """Descriptive statistics of the simulated terminal prices.

        Returns
        -------
        dict
            Keys: mean, median, std, min, max, percentile_5, percentile_25,
            percentile_50, percentile_75, percentile_95.
        """
        fp = self.final_prices
        pcts = np.percentile(fp, [5, 25, 50, 75, 95])
        return {
            "mean": float(np.mean(fp)),
            "median": float(np.median(fp)),
            "std": float(np.std(fp)),
            "min": float(np.min(fp)),
            "max": float(np.max(fp)),
            "percentile_5": float(pcts[0]),
            "percentile_25": float(pcts[1]),
            "percentile_50": float(pcts[2]),
            "percentile_75": float(pcts[3]),
            "percentile_95": float(pcts[4]),
        }

    def probability_above(self, target_price: float) -> float:
        """Fraction of simulations ending above *target_price*."""
        fp = self.final_prices
        return float(np.mean(fp > target_price))

    def probability_below(self, target_price: float) -> float:
        """Fraction of simulations ending below *target_price*."""
        fp = self.final_prices
        return float(np.mean(fp < target_price))

    def value_at_risk(self, confidence: float = 0.95) -> float:
        """Value-at-Risk expressed as a price level.

        Parameters
        ----------
        confidence : float
            Confidence level (e.g. 0.95 for 95 %).

        Returns
        -------
        float
            The price level such that only ``(1 - confidence)`` of
            simulations fall below it.
        """
        if not 0 < confidence < 1:
            raise ValueError("confidence must be between 0 and 1 (exclusive).")
        fp = self.final_prices
        return float(np.percentile(fp, (1 - confidence) * 100))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Comprehensive simulation results.

        Includes statistics, VaR at 95 % and 99 %, current price, number
        of simulations and forecast horizon.
        """
        paths = self._ensure_simulated()
        stats = self.statistics()

        return {
            "current_price": self.current_price,
            "num_simulations": paths.shape[0],
            "forecast_days": paths.shape[1],
            "drift_daily": self._mu,
            "volatility_daily": self._sigma,
            "statistics": stats,
            "var_95": self.value_at_risk(0.95),
            "var_99": self.value_at_risk(0.99),
            "forecast_range": {
                "low": stats["percentile_5"],
                "mid": stats["percentile_50"],
                "high": stats["percentile_95"],
            },
            "probability_above_current": self.probability_above(
                self.current_price
            ),
            "probability_below_current": self.probability_below(
                self.current_price
            ),
        }
