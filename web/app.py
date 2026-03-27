#!/usr/bin/env python3
"""MarketPulse Web API Server."""

import sys
import os
import json
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from config import Config
from data_sources.yahoo_finance import YahooFinanceSource
from data_sources.fred import FREDSource
from data_sources.eia import EIASource
from data_sources.news import NewsSource
from analysis.technical import TechnicalAnalyzer
from analysis.correlation import CorrelationAnalyzer
from analysis.macro import MacroAnalyzer
from simulation.monte_carlo import MonteCarloSimulator

app = Flask(__name__)
CORS(app)

yahoo = YahooFinanceSource()
fred = FREDSource()
eia = EIASource()
news = NewsSource()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/snapshot")
def api_snapshot():
    assets_str = request.args.get("assets", "WTI,SPY,NASDAQ,GOLD,BTC,EURUSD,DXY,US10Y")
    assets = [a.strip() for a in assets_str.split(",")]

    prices = {}
    for asset in assets:
        try:
            prices[asset] = yahoo.fetch_current_price(asset)
        except Exception as e:
            prices[asset] = {"error": str(e)}

    macro = {}
    if Config.FRED_API_KEY:
        try:
            snapshot = fred.fetch_macro_snapshot()
            for k, v in snapshot.items():
                if isinstance(v, dict) and v.get("value") is not None:
                    macro[k] = v
        except Exception:
            pass

    return jsonify({"prices": prices, "macro": macro})


@app.route("/api/history")
def api_history():
    asset = request.args.get("asset", "SPY")
    period = request.args.get("period", "3mo")
    interval = request.args.get("interval", "1d")

    history = yahoo.fetch_history(asset, period=period, interval=interval)
    if history.empty:
        return jsonify({"error": f"No data for {asset}"}), 404

    data = {
        "asset": asset,
        "dates": [str(d.date()) for d in history.index],
        "open": history["Open"].tolist(),
        "high": history["High"].tolist(),
        "low": history["Low"].tolist(),
        "close": history["Close"].tolist(),
        "volume": history["Volume"].tolist() if "Volume" in history else [],
    }
    return jsonify(data)


@app.route("/api/technical")
def api_technical():
    asset = request.args.get("asset", "SPY")
    period = request.args.get("period", "3mo")

    history = yahoo.fetch_history(asset, period=period)
    if history.empty:
        return jsonify({"error": f"No data for {asset}"}), 404

    analyzer = TechnicalAnalyzer(history)
    return jsonify({"asset": asset, **analyzer.summary()})


@app.route("/api/simulate")
def api_simulate():
    asset = request.args.get("asset", "WTI")
    days = int(request.args.get("days", 30))
    num_sims = int(request.args.get("simulations", 10000))

    history = yahoo.fetch_history(asset, period="1y")
    if history.empty:
        return jsonify({"error": f"No data for {asset}"}), 404

    sim = MonteCarloSimulator(history["Close"])
    paths = sim.simulate(days=days, num_simulations=num_sims)
    summary = sim.summary()

    # Sample 100 paths for visualization
    import numpy as np
    indices = np.linspace(0, num_sims - 1, min(100, num_sims), dtype=int)
    sample_paths = paths[indices].tolist()

    # Histogram of final prices
    final_prices = paths[:, -1]
    hist_counts, hist_edges = np.histogram(final_prices, bins=50)

    return jsonify({
        "asset": asset,
        "days": days,
        **summary,
        "sample_paths": sample_paths,
        "histogram": {
            "counts": hist_counts.tolist(),
            "edges": hist_edges.tolist(),
        },
    })


@app.route("/api/correlation")
def api_correlation():
    assets_str = request.args.get("assets", "WTI,SPY,GOLD,BTC,DXY")
    period = request.args.get("period", "3mo")
    assets = [a.strip() for a in assets_str.split(",")]

    data = yahoo.fetch_multiple(assets, period=period)
    if not data:
        return jsonify({"error": "No data available"}), 404

    analyzer = CorrelationAnalyzer(data)
    matrix = analyzer.correlation_matrix()
    summary = analyzer.summary()

    return jsonify({
        "assets": list(matrix.columns),
        "matrix": matrix.values.tolist(),
        **summary,
    })


@app.route("/api/news")
def api_news():
    query = request.args.get("query", None)
    if query:
        articles = news.fetch_news(query, max_results=10)
    else:
        articles = news.fetch_market_news()
    return jsonify({"articles": articles[:10]})


@app.route("/api/report", methods=["POST"])
def api_report():
    if not Config.ANTHROPIC_API_KEY:
        return jsonify({"error": "ANTHROPIC_API_KEY not configured"}), 500

    body = request.get_json()
    assets = body.get("assets", ["WTI", "SPY", "BTC"])
    period = body.get("period", "3mo")
    lang = body.get("lang", "ko")

    from report.agent import ReportAgent
    language = "English" if lang == "en" else "Korean"
    agent = ReportAgent(language=language)
    report = agent.generate_report(assets=assets, period=period)

    return jsonify({"report": report})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    debug = os.environ.get("RENDER") is None
    print(f"\n  Song Jongun's Market Watch")
    print(f"  http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
