"""Tools available to the ReACT report agent for real market data retrieval."""

import json
from typing import Any

import pandas as pd

from config import Config
from data_sources.yahoo_finance import YahooFinanceSource
from data_sources.fred import FREDSource
from data_sources.eia import EIASource
from data_sources.news import NewsSource
from analysis.technical import TechnicalAnalyzer
from analysis.correlation import CorrelationAnalyzer
from analysis.macro import MacroAnalyzer
from simulation.monte_carlo import MonteCarloSimulator


class ReportTools:
    """Provides tool functions for the ReACT report agent."""

    TOOL_DEFINITIONS = [
        {
            "name": "get_price_data",
            "description": "Get current price and recent history for an asset. "
                           "Assets: WTI, BRENT, GOLD, SPY, SPX, NASDAQ, KOSPI, "
                           "EURUSD, USDJPY, USDKRW, DXY, BTC, ETH, US10Y, etc.",
            "parameters": {"asset": "str - asset name", "period": "str - 1mo/3mo/6mo/1y"},
        },
        {
            "name": "get_technical_analysis",
            "description": "Get technical analysis (RSI, MACD, Bollinger Bands, moving averages, "
                           "trend signals) for an asset.",
            "parameters": {"asset": "str", "period": "str - 3mo/6mo/1y"},
        },
        {
            "name": "get_macro_snapshot",
            "description": "Get current macroeconomic indicators: Fed rate, CPI, GDP, "
                           "unemployment, yield curve, and macro environment assessment.",
            "parameters": {},
        },
        {
            "name": "get_correlation",
            "description": "Get correlation matrix between multiple assets.",
            "parameters": {"assets": "str - comma-separated asset names", "period": "str"},
        },
        {
            "name": "run_monte_carlo",
            "description": "Run Monte Carlo price simulation for an asset. Returns forecast "
                           "range, probabilities, and Value at Risk.",
            "parameters": {"asset": "str", "days": "int - forecast days (default 30)"},
        },
        {
            "name": "get_oil_fundamentals",
            "description": "Get oil market fundamentals: EIA inventory, production data.",
            "parameters": {},
        },
        {
            "name": "get_market_news",
            "description": "Get recent financial news headlines.",
            "parameters": {"query": "str - optional search query"},
        },
    ]

    def __init__(self):
        self.yahoo = YahooFinanceSource()
        self.fred = FREDSource()
        self.eia = EIASource()
        self.news = NewsSource()

    def get_tool_descriptions(self) -> str:
        lines = []
        for t in self.TOOL_DEFINITIONS:
            params = ", ".join(f"{k}: {v}" for k, v in t.get("parameters", {}).items())
            lines.append(f"- **{t['name']}**({params}): {t['description']}")
        return "\n".join(lines)

    def execute(self, tool_name: str, arguments: dict) -> str:
        try:
            method = getattr(self, tool_name, None)
            if method is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            result = method(**arguments)
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_price_data(self, asset: str, period: str = "3mo") -> dict:
        current = self.yahoo.fetch_current_price(asset)
        history = self.yahoo.fetch_history(asset, period=period)
        if history.empty:
            return {"error": f"No data for {asset}", **current}

        close = history["Close"]
        return {
            **current,
            "period_high": float(close.max()),
            "period_low": float(close.min()),
            "period_return": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
            "avg_volume": int(history["Volume"].mean()) if "Volume" in history else 0,
            "data_points": len(history),
            "start_date": str(history.index[0].date()),
            "end_date": str(history.index[-1].date()),
        }

    def get_technical_analysis(self, asset: str, period: str = "3mo") -> dict:
        history = self.yahoo.fetch_history(asset, period=period)
        if history.empty:
            return {"error": f"No data for {asset}"}
        analyzer = TechnicalAnalyzer(history)
        return {"asset": asset, **analyzer.summary()}

    def get_macro_snapshot(self) -> dict:
        snapshot = self.fred.fetch_macro_snapshot()
        if not snapshot:
            return {"error": "FRED API key not configured", "note": "Set FRED_API_KEY in .env"}

        analyzer = MacroAnalyzer(snapshot)
        return {
            "indicators": snapshot,
            "analysis": analyzer.summary(),
        }

    def get_correlation(self, assets: str, period: str = "3mo") -> dict:
        asset_list = [a.strip() for a in assets.split(",")]
        data = self.yahoo.fetch_multiple(asset_list, period=period)
        if not data:
            return {"error": "No data available for correlation"}
        analyzer = CorrelationAnalyzer(data)
        return analyzer.summary()

    def run_monte_carlo(self, asset: str, days: int = 30) -> dict:
        history = self.yahoo.fetch_history(asset, period="1y")
        if history.empty:
            return {"error": f"No data for {asset}"}
        sim = MonteCarloSimulator(history["Close"])
        sim.simulate(days=days, num_simulations=10000)
        return {"asset": asset, "forecast_days": days, **sim.summary()}

    def get_oil_fundamentals(self) -> dict:
        inventory = self.eia.fetch_oil_inventory()
        production = self.eia.fetch_crude_production()
        result = {}
        if inventory:
            result["inventory"] = inventory
        if production is not None and not production.empty:
            result["recent_production"] = {
                "latest": float(production.iloc[-1]),
                "date": str(production.index[-1]),
                "unit": "thousand barrels per day",
            }
        if not result:
            result["note"] = "EIA API key not configured. Set EIA_API_KEY in .env"
        return result

    def get_market_news(self, query: str = None) -> dict:
        if query:
            articles = self.news.fetch_news(query, max_results=5)
        else:
            articles = self.news.fetch_market_news()
        return {"articles": articles[:10]}
