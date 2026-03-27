"""Vector Autoregression (VAR) forecaster for multivariate time series.

Fits a VAR model on percentage returns of multiple assets and converts
forecasts back to price levels.  Includes impulse-response analysis and
Granger-causality testing.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


class VARForecaster:
    """Multivariate time-series forecaster using Vector Autoregression.

    Parameters
    ----------
    data : dict[str, pd.Series]
        Mapping of ``asset_name -> Close price Series``.  Each series
        should share a comparable datetime index (daily frequency).
        At least two assets are required.
    """

    MIN_OBSERVATIONS = 30

    def __init__(self, data: dict[str, pd.Series]) -> None:
        if not data or len(data) < 2:
            raise ValueError("At least two asset price series are required.")

        # Align all series on a common date index
        combined = pd.DataFrame(data).sort_index().dropna()

        if len(combined) < self.MIN_OBSERVATIONS:
            raise ValueError(
                f"Need at least {self.MIN_OBSERVATIONS} overlapping "
                f"observations after alignment, got {len(combined)}."
            )

        self._prices: pd.DataFrame = combined
        self._assets: list[str] = list(combined.columns)

        # Compute returns for stationarity
        self._returns: pd.DataFrame = combined.pct_change().dropna()
        self._differenced: bool = False
        self._ensure_stationarity()

        self._model: Any | None = None
        self._results: Any | None = None
        self._fitted_lag: int = 0

    # ------------------------------------------------------------------
    # Stationarity helpers
    # ------------------------------------------------------------------

    def _is_stationary(self, series: pd.Series, significance: float = 0.05) -> bool:
        """ADF test for stationarity."""
        try:
            result = adfuller(series.dropna(), autolag="AIC")
            return result[1] < significance  # p-value < threshold
        except Exception:
            return False

    def _ensure_stationarity(self) -> None:
        """If any return series is non-stationary, difference once more."""
        non_stationary = [
            col
            for col in self._returns.columns
            if not self._is_stationary(self._returns[col])
        ]
        if non_stationary:
            self._returns = self._returns.diff().dropna()
            self._differenced = True

            if len(self._returns) < self.MIN_OBSERVATIONS:
                raise ValueError(
                    "Insufficient data after differencing for stationarity."
                )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, maxlags: int | None = None) -> "VARForecaster":
        """Fit the VAR model.

        Parameters
        ----------
        maxlags : int or None
            Maximum lag order to consider.  If *None*, the optimal lag is
            selected automatically via AIC (capped at ``min(12, n/5)``).

        Returns
        -------
        self
        """
        model = VAR(self._returns)

        if maxlags is None:
            n = len(self._returns)
            cap = min(12, max(1, n // 5))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lag_order = model.select_order(maxlags=cap)
                selected = lag_order.aic
                if selected < 1:
                    selected = 1
            except Exception:
                selected = 1
        else:
            selected = max(1, maxlags)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._results = model.fit(maxlags=selected)
        except np.linalg.LinAlgError:
            # Singular matrix -- fall back to lag=1
            self._results = model.fit(maxlags=1)

        self._model = model
        self._fitted_lag = self._results.k_ar
        return self

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def _ensure_fitted(self) -> None:
        if self._results is None:
            self.fit()

    def forecast(self, steps: int = 30) -> pd.DataFrame:
        """Forecast price levels for all assets.

        Parameters
        ----------
        steps : int
            Number of periods (days) to forecast.

        Returns
        -------
        pd.DataFrame
            Columns are asset names; index is integer step (1..steps).
            Values are *price levels* reconstructed from the return
            forecasts.
        """
        self._ensure_fitted()

        lag_data = self._returns.values[-self._fitted_lag:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_returns = self._results.forecast(lag_data, steps=steps)

        fc_df = pd.DataFrame(
            fc_returns, columns=self._assets, index=range(1, steps + 1)
        )

        # Convert returns back to prices
        if self._differenced:
            # fc_df contains *differenced returns*; integrate once to get
            # returns, then cumulate to prices.
            last_returns = self._returns.iloc[-1].values
            returns = fc_df.cumsum() + last_returns
        else:
            returns = fc_df

        last_prices = self._prices.iloc[-1]
        price_forecast = pd.DataFrame(index=fc_df.index, columns=self._assets, dtype=float)

        for col in self._assets:
            cumulative = (1 + returns[col]).cumprod()
            price_forecast[col] = float(last_prices[col]) * cumulative

        return price_forecast

    # ------------------------------------------------------------------
    # Impulse-response analysis
    # ------------------------------------------------------------------

    def impulse_response(self, periods: int = 10) -> dict:
        """Compute orthogonalised impulse-response functions.

        Returns
        -------
        dict
            ``{impulse_asset: {response_asset: [values...]}}`` for each
            combination over *periods* steps.
        """
        self._ensure_fitted()

        try:
            irf = self._results.irf(periods)
        except Exception as exc:
            return {"error": str(exc)}

        result: dict[str, dict[str, list[float]]] = {}
        for i, impulse in enumerate(self._assets):
            result[impulse] = {}
            for j, response in enumerate(self._assets):
                result[impulse][response] = [
                    float(irf.irfs[t][j, i]) for t in range(periods + 1)
                ]
        return result

    # ------------------------------------------------------------------
    # Granger causality
    # ------------------------------------------------------------------

    def granger_causality(
        self, target: str, cause: str, maxlag: int | None = None
    ) -> dict:
        """Run pairwise Granger-causality test.

        Parameters
        ----------
        target : str
            Name of the asset whose series is the *dependent* variable.
        cause : str
            Name of the asset tested as the *causal* variable.
        maxlag : int or None
            Maximum lag for the test.  Defaults to the fitted lag order.

        Returns
        -------
        dict
            Keys: ``statistic``, ``p_value``, ``significant`` (at 5 %),
            ``lag_used``.
        """
        for name in (target, cause):
            if name not in self._assets:
                raise ValueError(
                    f"'{name}' not found. Available assets: {self._assets}"
                )

        self._ensure_fitted()
        lag = maxlag if maxlag is not None else max(1, self._fitted_lag)

        test_data = self._returns[[target, cause]].dropna()
        if len(test_data) <= lag + 1:
            return {
                "statistic": None,
                "p_value": None,
                "significant": False,
                "lag_used": lag,
                "error": "Insufficient data for Granger test.",
            }

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = grangercausalitytests(test_data, maxlag=lag, verbose=False)

            # Use the largest lag tested
            best_lag = lag
            test_result = results[best_lag][0]
            # Use the ssr_ftest (most common)
            f_stat = test_result["ssr_ftest"][0]
            p_value = test_result["ssr_ftest"][1]

            return {
                "statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "lag_used": best_lag,
            }
        except Exception as exc:
            return {
                "statistic": None,
                "p_value": None,
                "significant": False,
                "lag_used": lag,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Full forecast summary with point estimates and confidence bounds.

        Returns
        -------
        dict
            Per-asset forecast with ``last_price``, ``forecast_price``,
            ``change_pct``, ``confidence_low``, ``confidence_high``.
        """
        self._ensure_fitted()

        fc = self.forecast()
        steps = len(fc)
        last_prices = self._prices.iloc[-1]

        # Approximate confidence intervals from residual std
        resid_std = self._results.resid.std()

        asset_summaries: dict[str, dict] = {}
        for col in self._assets:
            final = float(fc[col].iloc[-1])
            last = float(last_prices[col])
            change = ((final - last) / last) * 100 if last != 0 else 0.0

            # Rough 95 % CI using residual volatility scaled by sqrt(steps)
            std_col = float(resid_std[col]) if col in resid_std.index else 0.0
            margin = 1.96 * std_col * np.sqrt(steps) * last

            asset_summaries[col] = {
                "last_price": last,
                "forecast_price": final,
                "change_pct": change,
                "confidence_low": final - margin,
                "confidence_high": final + margin,
            }

        return {
            "forecast_steps": steps,
            "fitted_lag": self._fitted_lag,
            "differenced": self._differenced,
            "assets": asset_summaries,
        }
