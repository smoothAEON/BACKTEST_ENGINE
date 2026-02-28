"""Trade-ledger reporting utilities for standalone backtesting workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.market_metadata import get_instrument_class, get_pip_value, normalize_instrument

from .models import OPTIONAL_TRADE_COLUMNS, REQUIRED_TRADE_COLUMNS, ReportConfig, iso_utc


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _profit_factor(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    wins = float(series[series > 0].sum())
    losses = float(series[series < 0].sum())
    if losses == 0:
        return None if wins == 0 else float("inf")
    return float(wins / abs(losses))


def _max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    equity = series.fillna(0.0).astype(float).cumsum()
    return float((equity - equity.cummax()).min())


def _normalize_side(value: Any) -> str:
    side = str(value or "").strip().upper()
    if side in {"LONG", "BUY"}:
        return "LONG"
    if side in {"SHORT", "SELL"}:
        return "SHORT"
    raise ValueError(f"Unsupported side value: {value}")


def _validate_trade_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_TRADE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required trade columns: {missing}")


def load_trades_csv(path: str | Path) -> pd.DataFrame:
    """Read raw trades CSV for reporting."""
    return pd.read_csv(path)


def enrich_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and enrich raw trades with derived performance fields."""
    df = trades_df.copy()
    _validate_trade_columns(df)

    for col in OPTIONAL_TRADE_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["instrument"] = df["instrument"].map(normalize_instrument)
    df["side"] = df["side"].map(_normalize_side)

    df["entry_time_utc"] = pd.to_datetime(df["entry_time_utc"], utc=True, errors="coerce")
    df["exit_time_utc"] = pd.to_datetime(df["exit_time_utc"], utc=True, errors="coerce")
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["fees"] = pd.to_numeric(df["fees"], errors="coerce").fillna(0.0)

    if df["entry_time_utc"].isna().any():
        raise ValueError("entry_time_utc contains invalid timestamps")
    if df["entry_price"].isna().any():
        raise ValueError("entry_price contains invalid numbers")

    if "trade_id" not in df.columns:
        df["trade_id"] = None
    df["trade_id"] = df["trade_id"].astype(str).replace({"": None, "nan": None})
    missing_trade_id = df["trade_id"].isna()
    if missing_trade_id.any():
        df.loc[missing_trade_id, "trade_id"] = [f"T{idx+1:06d}" for idx in df.index[missing_trade_id]]

    df["timeframe"] = df["timeframe"].astype(str).replace({"nan": None, "None": None})
    df["source_tag"] = df["source_tag"].astype(str).replace({"nan": None, "None": None})
    df["notes"] = df["notes"].astype(str).replace({"nan": None, "None": None})

    resolved = df["exit_time_utc"].notna() & df["exit_price"].notna()
    pip_values = df["instrument"].map(get_pip_value).astype(float)

    long_mask = resolved & (df["side"] == "LONG")
    short_mask = resolved & (df["side"] == "SHORT")

    df["realized_pips"] = None
    df.loc[long_mask, "realized_pips"] = (
        (df.loc[long_mask, "exit_price"].astype(float) - df.loc[long_mask, "entry_price"].astype(float))
        / pip_values.loc[long_mask]
    )
    df.loc[short_mask, "realized_pips"] = (
        (df.loc[short_mask, "entry_price"].astype(float) - df.loc[short_mask, "exit_price"].astype(float))
        / pip_values.loc[short_mask]
    )
    df["realized_pips"] = pd.to_numeric(df["realized_pips"], errors="coerce")

    df["holding_minutes"] = (
        (df["exit_time_utc"] - df["entry_time_utc"]).dt.total_seconds() / 60.0
    )
    df.loc[~resolved, "holding_minutes"] = None

    df["gross_pnl_quote"] = None
    qty_mask = resolved & df["quantity"].notna()
    long_qty_mask = qty_mask & (df["side"] == "LONG")
    short_qty_mask = qty_mask & (df["side"] == "SHORT")
    df.loc[long_qty_mask, "gross_pnl_quote"] = (
        (df.loc[long_qty_mask, "exit_price"].astype(float) - df.loc[long_qty_mask, "entry_price"].astype(float))
        * df.loc[long_qty_mask, "quantity"].astype(float)
    )
    df.loc[short_qty_mask, "gross_pnl_quote"] = (
        (df.loc[short_qty_mask, "entry_price"].astype(float) - df.loc[short_qty_mask, "exit_price"].astype(float))
        * df.loc[short_qty_mask, "quantity"].astype(float)
    )
    df["gross_pnl_quote"] = pd.to_numeric(df["gross_pnl_quote"], errors="coerce")

    df["net_pnl_quote"] = df["gross_pnl_quote"] - df["fees"]
    df.loc[df["gross_pnl_quote"].isna(), "net_pnl_quote"] = None

    df["outcome"] = "OPEN"
    df.loc[resolved & (df["realized_pips"] > 0), "outcome"] = "WIN"
    df.loc[resolved & (df["realized_pips"] < 0), "outcome"] = "LOSS"
    df.loc[resolved & (df["realized_pips"] == 0), "outcome"] = "FLAT"

    df["month"] = df["entry_time_utc"].dt.strftime("%Y-%m")
    return df


def _filter_trades(df: pd.DataFrame, config: ReportConfig) -> pd.DataFrame:
    filtered = df.copy()

    if config.start_utc is not None:
        start_ts = pd.Timestamp(pd.to_datetime(config.start_utc, utc=True))
        filtered = filtered[filtered["entry_time_utc"] >= start_ts]
    if config.end_utc is not None:
        end_ts = pd.Timestamp(pd.to_datetime(config.end_utc, utc=True))
        filtered = filtered[filtered["entry_time_utc"] <= end_ts]

    if config.instruments:
        allowed = {normalize_instrument(item) for item in config.instruments}
        filtered = filtered[filtered["instrument"].isin(allowed)]

    if config.timeframes:
        allowed_tfs = {str(item).upper() for item in config.timeframes}
        filtered = filtered[filtered["timeframe"].astype(str).str.upper().isin(allowed_tfs)]

    return filtered.reset_index(drop=True)


def _group_trade_metrics(df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    if df.empty or group_col not in df.columns:
        return []

    rows: list[dict[str, Any]] = []
    for key, grp in df.groupby(group_col, dropna=False):
        resolved = grp[grp["outcome"].isin(["WIN", "LOSS", "FLAT"])]
        wins = int((resolved["outcome"] == "WIN").sum())
        losses = int((resolved["outcome"] == "LOSS").sum())
        flats = int((resolved["outcome"] == "FLAT").sum())
        resolved_count = int(len(resolved))

        rows.append(
            {
                group_col: str(key),
                "total_trades": int(len(grp)),
                "resolved_trades": resolved_count,
                "open_trades": int((grp["outcome"] == "OPEN").sum()),
                "wins": wins,
                "losses": losses,
                "flats": flats,
                "win_rate_resolved": (wins / resolved_count) if resolved_count > 0 else None,
                "expectancy_pips_per_trade_open_zero": _safe_float(grp["realized_pips"].fillna(0.0).mean()),
                "expectancy_pips_per_resolved": _safe_float(resolved["realized_pips"].mean()) if resolved_count > 0 else None,
                "profit_factor_resolved": _profit_factor(resolved["realized_pips"]) if resolved_count > 0 else None,
            }
        )

    rows.sort(key=lambda item: (item.get("total_trades", 0), str(item.get(group_col))), reverse=True)
    return rows


def _monthly_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    for month, grp in df.groupby("month"):
        resolved = grp[grp["outcome"].isin(["WIN", "LOSS", "FLAT"])]
        rows.append(
            {
                "month": month,
                "total_trades": int(len(grp)),
                "resolved_trades": int(len(resolved)),
                "open_trades": int((grp["outcome"] == "OPEN").sum()),
                "wins": int((resolved["outcome"] == "WIN").sum()),
                "losses": int((resolved["outcome"] == "LOSS").sum()),
                "flats": int((resolved["outcome"] == "FLAT").sum()),
                "realized_pips_total_open_zero": _safe_float(grp["realized_pips"].fillna(0.0).sum()),
                "expectancy_pips_per_trade_open_zero": _safe_float(grp["realized_pips"].fillna(0.0).mean()),
            }
        )

    rows.sort(key=lambda row: row["month"])
    return rows


def _default_stress_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "S1": {
            "description": "Moderate extra slippage per side (class-based pips)",
            "per_side_pips": {"FX": 0.2, "JPY": 0.2, "METAL": 0.5, "ENERGY": 0.5, "CRYPTO": 2.0},
        },
        "S2": {
            "description": "Heavy extra slippage per side (class-based pips)",
            "per_side_pips": {"FX": 0.5, "JPY": 0.5, "METAL": 1.0, "ENERGY": 1.0, "CRYPTO": 5.0},
        },
    }


def _compute_stress_metrics(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []

    base = df.copy()
    base["instrument_class"] = base["instrument"].map(lambda item: get_instrument_class(str(item)))
    resolved = base["outcome"].isin(["WIN", "LOSS", "FLAT"]) & base["realized_pips"].notna()

    rows: list[dict[str, Any]] = []
    for scenario, payload in _default_stress_scenarios().items():
        tmp = base.copy()
        per_side = payload["per_side_pips"]
        tmp["slippage_roundtrip_pips"] = tmp["instrument_class"].map(
            lambda cls: 2.0 * float(per_side.get(str(cls), per_side.get("FX", 0.0)))
        )
        tmp["stressed_realized_pips"] = tmp["realized_pips"]
        tmp.loc[resolved, "stressed_realized_pips"] = (
            tmp.loc[resolved, "stressed_realized_pips"].astype(float)
            - tmp.loc[resolved, "slippage_roundtrip_pips"].astype(float)
        )

        resolved_df = tmp[resolved]
        rows.append(
            {
                "scenario": scenario,
                "description": payload["description"],
                "per_side_pips": per_side,
                "expectancy_pips_per_resolved": _safe_float(resolved_df["stressed_realized_pips"].mean())
                if not resolved_df.empty
                else None,
                "expectancy_pips_per_trade_open_zero": _safe_float(tmp["stressed_realized_pips"].fillna(0.0).mean()),
                "profit_factor_resolved": _profit_factor(resolved_df["stressed_realized_pips"]) if not resolved_df.empty else None,
            }
        )
    return rows


def _build_summary_from_enriched(
    trades_df: pd.DataFrame,
    config: ReportConfig,
    data_manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    total_count = int(len(trades_df))
    resolved_df = trades_df[trades_df["outcome"].isin(["WIN", "LOSS", "FLAT"])].copy()
    open_count = int((trades_df["outcome"] == "OPEN").sum())

    wins = int((resolved_df["outcome"] == "WIN").sum())
    losses = int((resolved_df["outcome"] == "LOSS").sum())
    flats = int((resolved_df["outcome"] == "FLAT").sum())
    resolved_count = int(len(resolved_df))

    pips_open_zero = trades_df["realized_pips"].fillna(0.0)
    net_open_zero = trades_df["net_pnl_quote"].fillna(0.0)
    has_quote_pnl = trades_df["net_pnl_quote"].notna().any()

    summary = {
        "run_config": config.to_dict(),
        "schema": {
            "required_trade_columns": list(REQUIRED_TRADE_COLUMNS),
            "optional_trade_columns": list(OPTIONAL_TRADE_COLUMNS),
            "side_values": ["LONG", "SHORT"],
        },
        "window": {
            "start_utc": iso_utc(config.start_utc),
            "end_utc": iso_utc(config.end_utc),
        },
        "trade_summary": {
            "total_trades": total_count,
            "resolved_trades": resolved_count,
            "open_trades": open_count,
            "wins": wins,
            "losses": losses,
            "flats": flats,
            "win_rate_resolved": (wins / resolved_count) if resolved_count > 0 else None,
            "loss_rate_resolved": (losses / resolved_count) if resolved_count > 0 else None,
            "flat_rate_resolved": (flats / resolved_count) if resolved_count > 0 else None,
            "expectancy_pips_per_trade_open_zero": _safe_float(pips_open_zero.mean()) if total_count > 0 else None,
            "expectancy_pips_per_resolved": _safe_float(resolved_df["realized_pips"].mean()) if resolved_count > 0 else None,
            "profit_factor_resolved": _profit_factor(resolved_df["realized_pips"]) if resolved_count > 0 else None,
            "max_drawdown_pips_open_zero": _max_drawdown(pips_open_zero) if total_count > 0 else 0.0,
            "max_drawdown_quote_open_zero": _max_drawdown(net_open_zero) if has_quote_pnl else None,
            "avg_holding_minutes_resolved": _safe_float(resolved_df["holding_minutes"].mean()) if resolved_count > 0 else None,
        },
        "breakdowns": {
            "by_instrument": _group_trade_metrics(trades_df, "instrument"),
            "by_side": _group_trade_metrics(trades_df, "side"),
            "by_timeframe": _group_trade_metrics(trades_df, "timeframe"),
            "by_month": _monthly_metrics(trades_df),
        },
        "stress_scenarios": _compute_stress_metrics(trades_df),
        "data_manifest": data_manifest or {},
    }
    return summary


def build_summary(
    trades_df: pd.DataFrame,
    config: ReportConfig,
    data_manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build summary metrics from raw trade ledger dataframe."""
    enriched = enrich_trades(trades_df)
    filtered = _filter_trades(enriched, config)
    return _build_summary_from_enriched(filtered, config=config, data_manifest=data_manifest)


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for row in rows:
        values: list[str] = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def _write_markdown_report(report_path: Path, summary: dict[str, Any]) -> None:
    run_cfg = summary.get("run_config", {})
    trade_summary = summary.get("trade_summary", {})

    sections: list[str] = []
    sections.append("# Standalone Backtesting Report")
    sections.append("")
    sections.append("## Scope")
    sections.append("")
    sections.append(
        "Trade-ledger analysis generated from CSV candles and an external trades ledger."
    )
    sections.append("")
    sections.append(f"- Data root: `{run_cfg.get('data_root')}`")
    sections.append(f"- Price type: `{run_cfg.get('price_type')}`")
    sections.append(f"- Start: `{summary.get('window', {}).get('start_utc')}`")
    sections.append(f"- End: `{summary.get('window', {}).get('end_utc')}`")
    sections.append("")
    sections.append("## Summary Metrics")
    sections.append("")
    sections.append(_md_table([
        {"metric": "Total trades", "value": trade_summary.get("total_trades")},
        {"metric": "Resolved trades", "value": trade_summary.get("resolved_trades")},
        {"metric": "Open trades", "value": trade_summary.get("open_trades")},
        {"metric": "Wins", "value": trade_summary.get("wins")},
        {"metric": "Losses", "value": trade_summary.get("losses")},
        {"metric": "Flats", "value": trade_summary.get("flats")},
        {"metric": "Win rate (resolved)", "value": trade_summary.get("win_rate_resolved")},
        {"metric": "Expectancy pips/trade (open=0)", "value": trade_summary.get("expectancy_pips_per_trade_open_zero")},
        {"metric": "Expectancy pips/resolved", "value": trade_summary.get("expectancy_pips_per_resolved")},
        {"metric": "Profit factor (resolved)", "value": trade_summary.get("profit_factor_resolved")},
        {"metric": "Max drawdown pips (open=0)", "value": trade_summary.get("max_drawdown_pips_open_zero")},
        {"metric": "Max drawdown quote (open=0)", "value": trade_summary.get("max_drawdown_quote_open_zero")},
    ], ["metric", "value"]).rstrip())
    sections.append("")

    sections.append("## Breakdowns")
    sections.append("")
    for title, key, cols in [
        ("By Instrument", "by_instrument", ["instrument", "total_trades", "resolved_trades", "open_trades", "wins", "losses", "win_rate_resolved", "expectancy_pips_per_trade_open_zero"]),
        ("By Side", "by_side", ["side", "total_trades", "resolved_trades", "open_trades", "wins", "losses", "win_rate_resolved", "expectancy_pips_per_trade_open_zero"]),
        ("By Timeframe", "by_timeframe", ["timeframe", "total_trades", "resolved_trades", "open_trades", "wins", "losses", "win_rate_resolved", "expectancy_pips_per_trade_open_zero"]),
        ("By Month", "by_month", ["month", "total_trades", "resolved_trades", "open_trades", "wins", "losses", "realized_pips_total_open_zero"]),
    ]:
        sections.append(f"### {title}")
        sections.append("")
        sections.append(_md_table(summary.get("breakdowns", {}).get(key, []), cols).rstrip())
        sections.append("")

    sections.append("## Stress Scenarios")
    sections.append("")
    sections.append(_md_table(summary.get("stress_scenarios", []), [
        "scenario",
        "expectancy_pips_per_resolved",
        "expectancy_pips_per_trade_open_zero",
        "profit_factor_resolved",
    ]).rstrip())
    sections.append("")

    report_path.write_text("\n".join(sections), encoding="utf-8")


def write_backtest_artifacts(
    trades_df: pd.DataFrame,
    report_dir: str | Path,
    config: ReportConfig,
    data_manifest: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Write enriched ledger and summary/report artifacts."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched = enrich_trades(trades_df)
    filtered = _filter_trades(enriched, config)
    summary = _build_summary_from_enriched(filtered, config=config, data_manifest=data_manifest)

    trades_out = filtered.copy()
    for col in ("entry_time_utc", "exit_time_utc"):
        if col in trades_out.columns:
            trades_out[col] = pd.to_datetime(trades_out[col], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    trades_path = out_dir / "trades_enriched.csv"
    summary_path = out_dir / "summary.json"
    report_path = out_dir / "report.md"
    manifest_path = out_dir / "data_manifest.json"
    run_cfg_path = out_dir / "run_config.json"

    trades_out.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    manifest_path.write_text(json.dumps(data_manifest or {}, indent=2, default=_json_default), encoding="utf-8")
    run_cfg_path.write_text(json.dumps(config.to_dict(), indent=2, default=_json_default), encoding="utf-8")
    _write_markdown_report(report_path, summary)

    return {
        "summary": summary,
        "paths": {
            "report_dir": str(out_dir),
            "trades_enriched_csv": str(trades_path),
            "summary_json": str(summary_path),
            "report_md": str(report_path),
            "data_manifest_json": str(manifest_path),
            "run_config_json": str(run_cfg_path),
        },
    }
