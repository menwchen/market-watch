#!/usr/bin/env python3
"""MarketPulse - Real Market Data Simulation & Analysis Engine.

Usage:
    python main.py snapshot
    python main.py snapshot --assets WTI,SPY,BTC,EURUSD
    python main.py simulate --asset WTI --days 30
    python main.py correlation --assets WTI,SPY,GLD,BTC,DXY
    python main.py report --assets WTI,SPY,BTC --period 3mo
    python main.py report --assets WTI,SPY,BTC --period 3mo --lang en
"""

import argparse
import sys
import json

from config import Config


def cmd_snapshot(args):
    """Show current market snapshot."""
    from data_sources.yahoo_finance import YahooFinanceSource

    yahoo = YahooFinanceSource()
    assets = [a.strip() for a in args.assets.split(",")]

    print("\n" + "=" * 65)
    print("  MarketPulse - Market Snapshot")
    print("=" * 65)
    print(f"{'Asset':<10} {'Price':>12} {'Change':>10} {'Change%':>10}")
    print("-" * 65)

    for asset in assets:
        data = yahoo.fetch_current_price(asset)
        if "error" in data:
            print(f"{asset:<10} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
        else:
            price = data.get("price", 0)
            change = data.get("change", 0)
            pct = data.get("change_pct", 0)
            symbol = "+" if change >= 0 else ""
            print(f"{asset:<10} {price:>12.2f} {symbol}{change:>9.2f} {symbol}{pct:>8.2f}%")

    print("=" * 65)

    # Macro snapshot if FRED key available
    if Config.FRED_API_KEY:
        from data_sources.fred import FREDSource
        from analysis.macro import MacroAnalyzer
        print("\n  Macro Indicators")
        print("-" * 65)
        fred = FREDSource()
        snapshot = fred.fetch_macro_snapshot()
        if snapshot:
            for key, val in snapshot.items():
                if isinstance(val, dict):
                    print(f"  {key}: {val.get('value', 'N/A')} ({val.get('date', '')})")
                else:
                    print(f"  {key}: {val}")
            # Convert FRED snapshot dicts to scalar values for MacroAnalyzer
            macro_data = {}
            for k, v in snapshot.items():
                if isinstance(v, dict) and v.get("value") is not None:
                    macro_data[k] = v["value"]
            analyzer = MacroAnalyzer(macro_data)
            summary = analyzer.summary()
            print(f"\n  Environment: {summary.get('environment', 'N/A')}")
            print(f"  Recession risk: {summary.get('recession_risk', 'N/A')}")
    print()


def cmd_simulate(args):
    """Run Monte Carlo simulation for an asset."""
    from data_sources.yahoo_finance import YahooFinanceSource
    from simulation.monte_carlo import MonteCarloSimulator

    yahoo = YahooFinanceSource()
    history = yahoo.fetch_history(args.asset, period="1y")

    if history.empty:
        print(f"Error: No data available for {args.asset}")
        sys.exit(1)

    print(f"\nRunning Monte Carlo simulation for {args.asset}...")
    print(f"  Historical data: {len(history)} days")
    print(f"  Forecast: {args.days} days ahead")
    print(f"  Simulations: {args.simulations:,}")

    sim = MonteCarloSimulator(history["Close"])
    sim.simulate(days=args.days, num_simulations=args.simulations)
    result = sim.summary()

    stats = result.get("statistics", {})
    forecast = result.get("forecast_range", {})

    print(f"\n{'=' * 50}")
    print(f"  {args.asset} Monte Carlo Simulation Results")
    print(f"{'=' * 50}")
    print(f"  Current price:   ${result['current_price']:.2f}")
    print(f"  Forecast mean:   ${stats.get('mean', 0):.2f}")
    print(f"  Forecast median: ${stats.get('median', 0):.2f}")
    print(f"\n  Forecast range (90% CI):")
    print(f"    Low  (5th pct): ${stats.get('percentile_5', 0):.2f}")
    print(f"    Mid (50th pct): ${stats.get('percentile_50', 0):.2f}")
    print(f"    High(95th pct): ${stats.get('percentile_95', 0):.2f}")
    print(f"\n  VaR (95%):       ${result.get('var_95', 0):.2f}")
    print(f"  VaR (99%):       ${result.get('var_99', 0):.2f}")
    print(f"  Upside prob:     {result.get('probability_above_current', 0)*100:.1f}%")
    print(f"  Downside prob:   {result.get('probability_below_current', 0)*100:.1f}%")
    print(f"  Daily volatility: {result.get('volatility_daily', 0)*100:.2f}%")
    print(f"{'=' * 50}\n")


def cmd_correlation(args):
    """Show correlation matrix between assets."""
    from data_sources.yahoo_finance import YahooFinanceSource
    from analysis.correlation import CorrelationAnalyzer

    yahoo = YahooFinanceSource()
    assets = [a.strip() for a in args.assets.split(",")]

    print(f"\nFetching data for {len(assets)} assets...")
    data = yahoo.fetch_multiple(assets, period=args.period)

    if not data:
        print("Error: No data available")
        sys.exit(1)

    analyzer = CorrelationAnalyzer(data)
    matrix_df = analyzer.correlation_matrix()
    summary = analyzer.summary()

    print(f"\n{'=' * 60}")
    print("  Correlation Matrix (daily returns)")
    print(f"{'=' * 60}")

    asset_names = list(matrix_df.columns)
    # Header
    print(f"{'':>10}", end="")
    for name in asset_names:
        print(f"{name:>10}", end="")
    print()

    # Rows
    for row_name in asset_names:
        print(f"{row_name:>10}", end="")
        for col_name in asset_names:
            val = matrix_df.loc[row_name, col_name]
            print(f"{val:>10.3f}", end="")
        print()

    # Notable pairs
    sp = summary.get("strongest_positive")
    sn = summary.get("strongest_negative")
    if sp:
        print(f"\n  Strongest positive: {sp[0]} / {sp[1]} = {sp[2]:.3f}")
    if sn:
        print(f"  Strongest negative: {sn[0]} / {sn[1]} = {sn[2]:.3f}")

    print(f"{'=' * 60}\n")


def cmd_report(args):
    """Generate comprehensive market analysis report."""
    if not Config.ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set in .env")
        print("Report generation requires Claude API access.")
        sys.exit(1)

    from report.agent import ReportAgent

    assets = [a.strip() for a in args.assets.split(",")]
    language = "English" if args.lang == "en" else "Korean"

    print(f"\nMarketPulse Report Generator")
    print(f"  Assets: {', '.join(assets)}")
    print(f"  Period: {args.period}")
    print(f"  Language: {language}")

    agent = ReportAgent(language=language)
    report = agent.generate_report(assets=assets, period=args.period)

    print(f"\n{'=' * 60}")
    print("  REPORT PREVIEW (first 500 chars)")
    print(f"{'=' * 60}")
    print(report[:500])
    print(f"\n... ({len(report)} total characters)")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="MarketPulse - Real Market Data Simulation & Analysis"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # snapshot
    snap = subparsers.add_parser("snapshot", help="Current market snapshot")
    snap.add_argument(
        "--assets", default="WTI,SPY,NASDAQ,GOLD,BTC,EURUSD,DXY,US10Y",
        help="Comma-separated asset names",
    )

    # simulate
    sim = subparsers.add_parser("simulate", help="Monte Carlo simulation")
    sim.add_argument("--asset", required=True, help="Asset to simulate")
    sim.add_argument("--days", type=int, default=30, help="Forecast days")
    sim.add_argument("--simulations", type=int, default=10000, help="Number of simulations")

    # correlation
    corr = subparsers.add_parser("correlation", help="Correlation analysis")
    corr.add_argument("--assets", required=True, help="Comma-separated assets")
    corr.add_argument("--period", default="3mo", help="Period (1mo/3mo/6mo/1y)")

    # report
    rep = subparsers.add_parser("report", help="Generate analysis report")
    rep.add_argument("--assets", required=True, help="Comma-separated assets")
    rep.add_argument("--period", default="3mo", help="Period (1mo/3mo/6mo/1y)")
    rep.add_argument("--lang", default="ko", choices=["ko", "en"], help="Report language")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "snapshot": cmd_snapshot,
        "simulate": cmd_simulate,
        "correlation": cmd_correlation,
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
