"""
MarketPulse data sources layer.

Provides unified interfaces for fetching market data from
Yahoo Finance, FRED, EIA, and GNews.
"""

from data_sources.base import DataSource
from data_sources.yahoo_finance import YahooFinanceSource
from data_sources.fred import FREDSource
from data_sources.eia import EIASource
from data_sources.news import NewsSource
from data_sources.bok import BOKSource

__all__ = [
    "DataSource",
    "YahooFinanceSource",
    "FREDSource",
    "EIASource",
    "NewsSource",
    "BOKSource",
]
