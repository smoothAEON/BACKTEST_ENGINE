"""CLI for the broker runtime and standalone trade-ledger reporting."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from backtest import (  # noqa: E402
    BacktestRunConfig,
    CandleFeeder,
    ReportConfig,
    load_trades_csv,
    run_backtest,
    write_backtest_artifacts,
)
from core.logging_setup import setup_logging  # noqa: E402
from core.market_metadata import normalize_instrument, normalize_timeframe  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broker runtime and standalone ledger reporting CLI")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a broker/trader backtest from a JSON config")
    run_parser.add_argument("--config", required=True, help="Path to the runtime JSON config")

    report_parser = subparsers.add_parser("report-ledger", help="Run standalone reporting on a trades CSV")
    report_parser.add_argument("--data-root", required=True, help="Root directory containing candle CSV folders")
    report_parser.add_argument("--trades-csv", required=True, help="Input trades ledger CSV path")
    report_parser.add_argument("--report-dir", default="reports/backtest_run", help="Output directory")
    report_parser.add_argument("--start", help="Optional UTC start filter (YYYY-MM-DD)")
    report_parser.add_argument("--end", help="Optional UTC end filter (YYYY-MM-DD)")
    report_parser.add_argument("--instruments", nargs="+", metavar="INST", help="Optional instrument filter")
    report_parser.add_argument("--timeframes", nargs="+", metavar="TF", help="Optional timeframe filter")
    report_parser.add_argument(
        "--manifest-scope",
        default="selected",
        choices=["selected", "full", "none"],
        help="Controls whether the manifest uses selected datasets, all datasets, or is skipped",
    )

    return parser.parse_args()


def _parse_date(date_str: str | None):
    if not date_str:
        return None
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _derive_manifest_targets_from_trades(trades_df: pd.DataFrame | None) -> tuple[list[str], list[str]]:
    if trades_df is None or trades_df.empty:
        return [], []

    instruments: set[str] = set()
    if "instrument" in trades_df.columns:
        for item in trades_df["instrument"].tolist():
            value = str(item or "").strip()
            if not value or value.lower() == "nan":
                continue
            instruments.add(normalize_instrument(value))

    timeframes: set[str] = set()
    if "timeframe" in trades_df.columns:
        for item in trades_df["timeframe"].tolist():
            value = str(item or "").strip()
            if not value or value.lower() == "nan":
                continue
            timeframes.add(normalize_timeframe(value))

    return sorted(instruments), sorted(timeframes)


def _build_data_manifest(
    feeder: CandleFeeder,
    args: argparse.Namespace,
    trades_df: pd.DataFrame,
) -> dict:
    if args.manifest_scope == "none":
        return {}
    if args.manifest_scope == "full":
        return feeder.build_manifest()

    explicit_instruments = [normalize_instrument(item) for item in (args.instruments or [])]
    explicit_timeframes = [normalize_timeframe(item) for item in (args.timeframes or [])]

    selected_instruments = list(explicit_instruments)
    selected_timeframes = list(explicit_timeframes)
    if not selected_instruments:
        derived_instruments, derived_timeframes = _derive_manifest_targets_from_trades(trades_df)
        selected_instruments = derived_instruments
        if not selected_timeframes:
            selected_timeframes = derived_timeframes

    if not selected_instruments:
        return {}

    return feeder.build_manifest(
        instruments=selected_instruments,
        timeframes=selected_timeframes or [],
    )


def _run_broker(args: argparse.Namespace) -> int:
    logger = logging.getLogger(__name__)
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("config does not exist: %s", config_path)
        return 2

    try:
        config = BacktestRunConfig.from_path(config_path)
        artifacts = run_backtest(config)
    except Exception as exc:  # surface actionable runtime/config errors
        logger.error(str(exc))
        return 3

    logger.info("Report dir: %s", artifacts.paths["report_dir"])
    logger.info("Ending equity: %s", artifacts.aggregate_summary.get("ending_equity_usd"))
    logger.info("Closed trades: %s", artifacts.aggregate_summary.get("total_closed_trades"))
    return 0


def _run_report_ledger(args: argparse.Namespace) -> int:
    logger = logging.getLogger(__name__)
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error("data-root does not exist: %s", data_root)
        return 4

    trades_path = Path(args.trades_csv)
    if not trades_path.exists():
        logger.error("trades-csv does not exist: %s", trades_path)
        return 5

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        trades_df = load_trades_csv(trades_path)
        feeder = CandleFeeder(data_root=data_root)
        data_manifest = _build_data_manifest(feeder, args, trades_df)
        config = ReportConfig(
            data_root=data_root,
            report_dir=report_dir,
            source_trades_csv=trades_path,
            start_utc=_parse_date(args.start),
            end_utc=_parse_date(args.end),
            instruments=list(args.instruments or []),
            timeframes=list(args.timeframes or []),
        )
        artifacts = write_backtest_artifacts(
            trades_df,
            report_dir=report_dir,
            config=config,
            data_manifest=data_manifest,
        )
    except ValueError as exc:
        logger.error(str(exc))
        return 6

    summary = artifacts["summary"].get("trade_summary", {})
    logger.info("Report dir: %s", artifacts["paths"]["report_dir"])
    logger.info("Total trades: %s", summary.get("total_trades"))
    logger.info("Resolved trades: %s", summary.get("resolved_trades"))
    logger.info("Open trades: %s", summary.get("open_trades"))
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.command == "run":
        return _run_broker(args)
    if args.command == "report-ledger":
        return _run_report_ledger(args)
    raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    args = _parse_args()
    setup_logging(log_level=args.log_level)
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
