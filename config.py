import os
from dotenv import load_dotenv

_project_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_project_dir, ".env"), override=True)


class Config:
    # LLM
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    # Data APIs
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")
    EIA_API_KEY = os.getenv("EIA_API_KEY", "")
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

    # Bank of Korea ECOS
    BOK_API_KEY = os.getenv("BOK_API_KEY", "")

    # Cache
    CACHE_DB_PATH = os.path.join(os.path.dirname(__file__), "storage", "cache.db")
    CACHE_TTL_HOURS = 1  # API response cache duration

    # Report
    REPORT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "reports")
    REPORT_MAX_TOOL_CALLS = 5
    REPORT_MAX_SECTIONS = 3

    # Asset symbols mapping
    ASSET_SYMBOLS = {
        # Commodities
        "WTI": "CL=F",
        "BRENT": "BZ=F",
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "NATGAS": "NG=F",
        # Indices
        "SPY": "SPY",
        "SPX": "^GSPC",
        "NASDAQ": "^IXIC",
        "DOW": "^DJI",
        "KOSPI": "^KS11",
        # Forex
        "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X",
        "USDKRW": "USDKRW=X",
        "DXY": "DX-Y.NYB",
        # Crypto
        "BTC": "BTC-USD",
        "ETH": "ETH-USD",
        # Bonds
        "US10Y": "^TNX",
        "US2Y": "^IRX",
    }

    @classmethod
    def resolve_symbol(cls, asset: str) -> str:
        return cls.ASSET_SYMBOLS.get(asset.upper(), asset)
