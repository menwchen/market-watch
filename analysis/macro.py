"""Macroeconomic indicator analysis using simple heuristics."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MacroAnalyzer:
    """Analyse macroeconomic data series and produce environment assessments.

    Parameters
    ----------
    data : dict[str, pd.Series | list | float]
        Mapping of macro indicator names to their data.  Values can be a
        ``pd.Series`` (time-series), a list of observations, or a single
        scalar for the latest reading.
    """

    def __init__(self, data: dict[str, pd.Series | list | float]) -> None:
        self.data = data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_series(value: pd.Series | list | float) -> pd.Series:
        """Normalise input to a pd.Series."""
        if isinstance(value, pd.Series):
            return value.dropna()
        if isinstance(value, list):
            return pd.Series(value).dropna()
        return pd.Series([value])

    @staticmethod
    def _latest(series: pd.Series) -> float:
        return float(series.iloc[-1])

    @staticmethod
    def _trend(series: pd.Series, lookback: int = 6) -> str:
        """Simple trend label based on the last *lookback* observations."""
        if len(series) < 2:
            return "insufficient data"
        segment = series.tail(lookback)
        diff = float(segment.iloc[-1] - segment.iloc[0])
        if diff > 0.05 * abs(segment.iloc[0]) if segment.iloc[0] != 0 else diff > 0:
            return "rising"
        if diff < -0.05 * abs(segment.iloc[0]) if segment.iloc[0] != 0 else diff < 0:
            return "falling"
        return "stable"

    # ------------------------------------------------------------------
    # Individual analyses
    # ------------------------------------------------------------------

    def yield_curve_analysis(
        self,
        dgs2: pd.Series | list | float,
        dgs10: pd.Series | list | float,
    ) -> dict:
        """Analyse the 2s10s yield spread.

        Parameters
        ----------
        dgs2 : 2-year Treasury yield series or latest value.
        dgs10 : 10-year Treasury yield series or latest value.

        Returns
        -------
        dict with spread, inversion status, and recession signal.
        """
        s2 = self._to_series(dgs2)
        s10 = self._to_series(dgs10)

        # Align lengths
        min_len = min(len(s2), len(s10))
        s2 = s2.tail(min_len).reset_index(drop=True)
        s10 = s10.tail(min_len).reset_index(drop=True)

        spread = s10 - s2
        current_spread = float(spread.iloc[-1])
        is_inverted = current_spread < 0

        # Simple recession heuristic: curve inverted for extended period
        # or has recently un-inverted (historically recessions follow
        # shortly after un-inversion).
        recession_signal = "none"
        if is_inverted:
            recession_signal = "elevated - curve inverted"
        elif len(spread) >= 6:
            recent = spread.tail(6)
            if (recent < 0).any() and current_spread >= 0:
                recession_signal = "watch - recent un-inversion"

        return {
            "spread_2s10s": round(current_spread, 4),
            "yield_2y": round(float(s2.iloc[-1]), 4),
            "yield_10y": round(float(s10.iloc[-1]), 4),
            "inverted": is_inverted,
            "recession_signal": recession_signal,
            "spread_trend": self._trend(spread),
        }

    def inflation_analysis(self, cpi_series: pd.Series | list | float) -> dict:
        """Analyse CPI / inflation data.

        Parameters
        ----------
        cpi_series : CPI index values (not percent change).

        Returns
        -------
        dict with current rate, trend, and YoY change.
        """
        cpi = self._to_series(cpi_series)
        current = self._latest(cpi)

        # Year-over-year percent change (approximate with 12 observations)
        yoy_change: float | None = None
        if len(cpi) >= 12:
            year_ago = float(cpi.iloc[-12])
            if year_ago != 0:
                yoy_change = round(((current - year_ago) / year_ago) * 100.0, 2)

        # Month-over-month annualised (approximate)
        mom_annualised: float | None = None
        if len(cpi) >= 2:
            prev = float(cpi.iloc[-2])
            if prev != 0:
                mom_annualised = round(((current / prev) - 1.0) * 12.0 * 100.0, 2)

        trend = self._trend(cpi)

        return {
            "current_cpi": round(current, 2),
            "yoy_change_pct": yoy_change,
            "mom_annualised_pct": mom_annualised,
            "trend": trend,
        }

    def fed_rate_analysis(self, fedfunds: pd.Series | list | float) -> dict:
        """Analyse the Federal Funds rate.

        Parameters
        ----------
        fedfunds : Fed Funds effective rate series or latest value.

        Returns
        -------
        dict with current rate, trajectory, and stance.
        """
        ff = self._to_series(fedfunds)
        current = self._latest(ff)
        trend = self._trend(ff)

        # Stance heuristic
        if current >= 5.0:
            stance = "restrictive"
        elif current >= 3.0:
            stance = "moderately restrictive"
        elif current >= 1.5:
            stance = "neutral"
        elif current >= 0.5:
            stance = "accommodative"
        else:
            stance = "highly accommodative"

        return {
            "current_rate": round(current, 4),
            "trajectory": trend,
            "stance": stance,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Overall macro environment assessment.

        Pulls from whichever series are available in ``self.data``.

        Returns
        -------
        dict with:
            policy_stance : str  ('hawkish' / 'dovish' / 'neutral')
            recession_risk : str ('low' / 'moderate' / 'elevated' / 'high')
            key_concerns : list[str]
            details : dict   (individual analysis dicts when data available)
        """
        details: dict[str, dict] = {}
        key_concerns: list[str] = []
        hawkish_score = 0  # positive = hawkish, negative = dovish

        # --- Yield curve ---
        dgs2_raw = self.data.get("DGS2") or self.data.get("dgs2")
        dgs10_raw = self.data.get("DGS10") or self.data.get("dgs10")
        if dgs2_raw is not None and dgs10_raw is not None:
            yc = self.yield_curve_analysis(dgs2_raw, dgs10_raw)
            details["yield_curve"] = yc
            if yc["inverted"]:
                key_concerns.append("Yield curve inverted (recession risk)")
                hawkish_score += 1
            if "un-inversion" in yc.get("recession_signal", ""):
                key_concerns.append("Recent yield curve un-inversion (watch for recession)")

        # --- Inflation ---
        cpi_raw = self.data.get("CPIAUCSL") or self.data.get("cpi")
        if cpi_raw is not None:
            inf = self.inflation_analysis(cpi_raw)
            details["inflation"] = inf
            if inf["yoy_change_pct"] is not None:
                if inf["yoy_change_pct"] > 4.0:
                    key_concerns.append(f"High inflation ({inf['yoy_change_pct']}% YoY)")
                    hawkish_score += 2
                elif inf["yoy_change_pct"] > 3.0:
                    key_concerns.append(f"Above-target inflation ({inf['yoy_change_pct']}% YoY)")
                    hawkish_score += 1
                elif inf["yoy_change_pct"] < 1.5:
                    key_concerns.append(f"Below-target inflation ({inf['yoy_change_pct']}% YoY)")
                    hawkish_score -= 1

        # --- Fed Funds ---
        ff_raw = self.data.get("FEDFUNDS") or self.data.get("fedfunds")
        if ff_raw is not None:
            fed = self.fed_rate_analysis(ff_raw)
            details["fed_rate"] = fed
            if fed["trajectory"] == "rising":
                hawkish_score += 1
            elif fed["trajectory"] == "falling":
                hawkish_score -= 1
            if fed["stance"] == "restrictive":
                key_concerns.append("Restrictive monetary policy")

        # --- Policy stance ---
        if hawkish_score >= 2:
            policy_stance = "hawkish"
        elif hawkish_score <= -2:
            policy_stance = "dovish"
        else:
            policy_stance = "neutral"

        # --- Recession risk ---
        recession_risk = "low"
        yc_detail = details.get("yield_curve", {})
        if yc_detail.get("inverted"):
            recession_risk = "elevated"
        if "un-inversion" in yc_detail.get("recession_signal", ""):
            recession_risk = "high"
        if recession_risk == "low" and hawkish_score >= 2:
            recession_risk = "moderate"

        if not key_concerns:
            key_concerns.append("No major macro concerns identified")

        return {
            "policy_stance": policy_stance,
            "recession_risk": recession_risk,
            "key_concerns": key_concerns,
            "details": details,
        }
