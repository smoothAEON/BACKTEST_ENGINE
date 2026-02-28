"""MIT entry execution simulation and batch artifact helpers."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from core.market_metadata import get_pip_value, normalize_instrument, normalize_timeframe

logger = logging.getLogger(__name__)

_EPOCH_NS = datetime(1970, 1, 1, tzinfo=timezone.utc)

# Nanosecond durations for standard OANDA timeframes used to compute bar-close fill times.
_TIMEFRAME_NS: dict[str, int] = {
    "S5": 5_000_000_000,
    "S10": 10_000_000_000,
    "S15": 15_000_000_000,
    "S30": 30_000_000_000,
    "M1": 60_000_000_000,
    "M2": 120_000_000_000,
    "M4": 240_000_000_000,
    "M5": 300_000_000_000,
    "M10": 600_000_000_000,
    "M15": 900_000_000_000,
    "M30": 1_800_000_000_000,
    "H1": 3_600_000_000_000,
    "H2": 7_200_000_000_000,
    "H3": 10_800_000_000_000,
    "H4": 14_400_000_000_000,
    "H6": 21_600_000_000_000,
    "H8": 28_800_000_000_000,
    "H12": 43_200_000_000_000,
    "D": 86_400_000_000_000,
    "W": 604_800_000_000_000,
}


def _timeframe_to_ns(tf: str | None) -> int | None:
    """Return nanosecond bar duration for a normalised timeframe, or None if unknown."""
    if not tf:
        return None
    return _TIMEFRAME_NS.get(str(tf).upper())


def _datetime_to_ns(dt: datetime) -> int:
    """Convert a timezone-aware datetime to nanoseconds since Unix epoch."""
    dt_utc = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    delta = dt_utc - _EPOCH_NS
    return int((delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000)

from .models import ReportConfig
from .offline_oanda_provider import CandleFeeder, CandleSlice
from .reporting import write_backtest_artifacts

REQUIRED_ORDER_COLUMNS: tuple[str, ...] = (
    "order_id",
    "instrument",
    "side",
    "submit_time_utc",
    "trigger_price",
)

OPTIONAL_ORDER_COLUMNS: tuple[str, ...] = (
    "timeframe",
    "quantity",
    "expiry_time_utc",
    "source_tag",
    "notes",
    "order_type",
)


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


def _datetime_to_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    dt_value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return dt_value.isoformat().replace("+00:00", "Z")


def _timestamp_from_ns(value: int) -> datetime:
    return pd.Timestamp(int(value), tz="UTC").to_pydatetime()


def _normalize_side(raw_value: Any) -> str:
    side = str(raw_value or "").strip().upper()
    if side in {"LONG", "BUY"}:
        return "LONG"
    if side in {"SHORT", "SELL"}:
        return "SHORT"
    raise ValueError(f"Unsupported side value: {raw_value}")


def _require_text(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"{field} is required")
    return value


def _optional_text(row: dict[str, Any], field: str) -> str | None:
    value = str(row.get(field) or "").strip()
    return value or None


def _parse_required_datetime(row: dict[str, Any], field: str) -> datetime:
    value = _require_text(row, field).replace("Z", "+00:00")
    try:
        dt_value = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} contains invalid timestamps") from exc
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


def _parse_optional_datetime(row: dict[str, Any], field: str) -> datetime | None:
    value = _optional_text(row, field)
    if value is None:
        return None
    try:
        dt_value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} contains invalid timestamps") from exc
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


def _parse_required_float(row: dict[str, Any], field: str) -> float:
    value = _require_text(row, field)
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} contains invalid numbers") from exc


def _parse_optional_float(row: dict[str, Any], field: str) -> float | None:
    value = _optional_text(row, field)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} contains invalid numbers") from exc


@dataclass(frozen=True)
class MitOrderIntent:
    """Validated MIT order intent for entry-only simulation."""

    order_id: str
    instrument: str
    side: str
    submit_time_utc: datetime
    trigger_price: float
    timeframe: str | None = None
    quantity: float | None = None
    expiry_time_utc: datetime | None = None
    source_tag: str | None = None
    notes: str | None = None
    order_type: str = "MIT"

    def to_record(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "side": self.side,
            "submit_time_utc": _datetime_to_iso(self.submit_time_utc),
            "trigger_price": self.trigger_price,
            "quantity": self.quantity,
            "expiry_time_utc": _datetime_to_iso(self.expiry_time_utc),
            "source_tag": self.source_tag,
            "notes": self.notes,
            "order_type": self.order_type,
        }


@dataclass(frozen=True)
class FillResult:
    """Entry fill outcome for one MIT order."""

    order_id: str
    status: str
    filled_time_utc: datetime | None
    filled_price: float | None
    trigger_time_utc: datetime | None
    reason: str | None
    simulator_name: str
    slippage_pips: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "status": self.status,
            "filled_time_utc": _datetime_to_iso(self.filled_time_utc),
            "filled_price": self.filled_price,
            "trigger_time_utc": _datetime_to_iso(self.trigger_time_utc),
            "reason": self.reason,
            "simulator_name": self.simulator_name,
            "slippage_pips": self.slippage_pips,
        }


@dataclass
class ExecutionConfig:
    """Configuration for entry simulation batches."""

    max_workers: int = 4
    max_ram_bytes: int = 16 * 1024**3
    manifest_scope: str = "selected"
    emit_open_trades: bool = True
    default_fill_simulator: str = "first_touch_mit"
    cache_policy: str = "lru"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": max(1, int(self.max_workers)),
            "max_ram_bytes": max(0, int(self.max_ram_bytes)),
            "manifest_scope": str(self.manifest_scope),
            "emit_open_trades": bool(self.emit_open_trades),
            "default_fill_simulator": str(self.default_fill_simulator),
            "cache_policy": str(self.cache_policy),
        }


class ExecutionSimulator(Protocol):
    """Pluggable MIT entry simulator interface."""

    def simulate_mit_entry(self, order: MitOrderIntent, candles: CandleSlice) -> FillResult:
        ...


@dataclass
class SimulationArtifacts:
    """In-memory results for one simulation batch."""

    orders: list[MitOrderIntent]
    fills: list[FillResult]
    fill_summary: dict[str, Any]
    data_manifest: dict[str, Any] = field(default_factory=dict)
    generated_trades: pd.DataFrame | None = None


class FirstTouchMITSimulator:
    """Minimal deterministic fallback simulator for first-touch MIT entries."""

    simulator_name = "first_touch_mit"

    def simulate_mit_entry(self, order: MitOrderIntent, candles: CandleSlice) -> FillResult:
        if candles.rows == 0:
            return FillResult(
                order_id=order.order_id,
                status="UNFILLED",
                filled_time_utc=None,
                filled_price=None,
                trigger_time_utc=None,
                reason="no_candles_after_submit",
                simulator_name=self.simulator_name,
                slippage_pips=None,
            )

        if order.side == "LONG":
            hit_mask = candles.ask_low <= order.trigger_price
        else:
            hit_mask = candles.bid_high >= order.trigger_price

        hit_indexes = np.flatnonzero(hit_mask)

        # Issue 2 fix: exclude any hit on a bar that opens at or after expiry.
        # An order expires at expiry_time_utc; the bar opening at that exact time is
        # outside the order's valid window.
        if order.expiry_time_utc is not None and hit_indexes.size > 0:
            expiry_ns = _datetime_to_ns(order.expiry_time_utc)
            hit_indexes = hit_indexes[candles.time_ns[hit_indexes] < expiry_ns]

        if hit_indexes.size > 0:
            hit_index = int(hit_indexes[0])
            bar_open_ns = int(candles.time_ns[hit_index])

            # Issue 1 fix: only record fill at bar open when the bar itself opened
            # at or through the trigger (i.e. ask_open <= trigger for LONG).  If the
            # bar opened above the trigger and the price drifted down to it later,
            # we cannot know the exact fill time — use bar close (conservative).
            if order.side == "LONG":
                opened_at_trigger = bool(candles.ask_open[hit_index] <= order.trigger_price)
            else:
                opened_at_trigger = bool(candles.bid_open[hit_index] >= order.trigger_price)

            if opened_at_trigger:
                fill_ns = bar_open_ns
            else:
                tf_ns = _timeframe_to_ns(order.timeframe)
                if tf_ns is not None:
                    fill_ns = bar_open_ns + tf_ns
                else:
                    logger.warning(
                        "Cannot determine intrabar fill time for order %s: unknown timeframe %r. "
                        "Using bar open time (optimistic).",
                        order.order_id,
                        order.timeframe,
                    )
                    fill_ns = bar_open_ns

            fill_time = _timestamp_from_ns(fill_ns)
            trigger_time = _timestamp_from_ns(bar_open_ns)
            return FillResult(
                order_id=order.order_id,
                status="FILLED",
                filled_time_utc=fill_time,
                filled_price=order.trigger_price,
                trigger_time_utc=trigger_time,
                reason="first_touch",
                simulator_name=self.simulator_name,
                slippage_pips=0.0,
            )

        if order.expiry_time_utc is not None:
            last_time = _timestamp_from_ns(int(candles.time_ns[-1]))
            if last_time >= order.expiry_time_utc:
                return FillResult(
                    order_id=order.order_id,
                    status="EXPIRED",
                    filled_time_utc=None,
                    filled_price=None,
                    trigger_time_utc=None,
                    reason="expired_without_touch",
                    simulator_name=self.simulator_name,
                    slippage_pips=None,
                )

        return FillResult(
            order_id=order.order_id,
            status="UNFILLED",
            filled_time_utc=None,
            filled_price=None,
            trigger_time_utc=None,
            reason="not_touched",
            simulator_name=self.simulator_name,
            slippage_pips=None,
        )


def load_orders_csv(path: str | Path) -> list[MitOrderIntent]:
    """Read and validate raw MIT order intents from CSV."""
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_ORDER_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing required order columns: {missing}")

        seen_ids: set[str] = set()
        orders: list[MitOrderIntent] = []
        for row in reader:
            order_id = _require_text(row, "order_id")
            if order_id in seen_ids:
                raise ValueError(f"Duplicate order_id: {order_id}")
            seen_ids.add(order_id)

            order_type = (_optional_text(row, "order_type") or "MIT").upper()
            if order_type != "MIT":
                raise ValueError(f"Unsupported order_type for {order_id}: {order_type}")

            submit_time = _parse_required_datetime(row, "submit_time_utc")
            expiry_time = _parse_optional_datetime(row, "expiry_time_utc")
            if expiry_time is not None and expiry_time < submit_time:
                raise ValueError(f"expiry_time_utc is before submit_time_utc for {order_id}")

            timeframe = _optional_text(row, "timeframe")
            orders.append(
                MitOrderIntent(
                    order_id=order_id,
                    instrument=normalize_instrument(_require_text(row, "instrument")),
                    timeframe=normalize_timeframe(timeframe) if timeframe else None,
                    side=_normalize_side(row.get("side")),
                    submit_time_utc=submit_time,
                    trigger_price=_parse_required_float(row, "trigger_price"),
                    quantity=_parse_optional_float(row, "quantity"),
                    expiry_time_utc=expiry_time,
                    source_tag=_optional_text(row, "source_tag"),
                    notes=_optional_text(row, "notes"),
                    order_type=order_type,
                )
            )
        return orders


def _resolve_timeframe(order: MitOrderIntent, feeder_or_store: CandleFeeder) -> str:
    if order.timeframe:
        return normalize_timeframe(order.timeframe)

    available = feeder_or_store.list_timeframes(order.instrument)
    if len(available) == 1:
        return available[0]
    if not available:
        raise FileNotFoundError(f"No candle data available for {order.instrument}")
    raise ValueError(f"timeframe is required for {order.order_id} because multiple datasets exist for {order.instrument}")


def _build_fill_summary(orders: list[MitOrderIntent], fills: list[FillResult]) -> dict[str, Any]:
    status_counts = {
        "FILLED": 0,
        "EXPIRED": 0,
        "UNFILLED": 0,
        "REJECTED": 0,
    }
    order_map = {order.order_id: order for order in orders}

    latency_minutes: list[float] = []
    slippage_pips: list[float] = []
    for fill in fills:
        status_key = str(fill.status).upper()
        if status_key in status_counts:
            status_counts[status_key] += 1

        order = order_map.get(fill.order_id)
        if order is None:
            continue

        if fill.filled_time_utc is not None:
            latency_minutes.append((fill.filled_time_utc - order.submit_time_utc).total_seconds() / 60.0)

        if fill.slippage_pips is not None:
            slippage_pips.append(float(fill.slippage_pips))
        elif fill.filled_price is not None:
            pip_value = float(get_pip_value(order.instrument))
            slippage_pips.append(abs(float(fill.filled_price) - float(order.trigger_price)) / pip_value)

    total_orders = len(orders)
    filled_orders = status_counts["FILLED"]
    return {
        "total_orders": total_orders,
        "filled_orders": filled_orders,
        "expired_orders": status_counts["EXPIRED"],
        "unfilled_orders": status_counts["UNFILLED"],
        "rejected_orders": status_counts["REJECTED"],
        "fill_rate": (filled_orders / total_orders) if total_orders > 0 else None,
        "avg_latency_minutes": (sum(latency_minutes) / len(latency_minutes)) if latency_minutes else None,
        "avg_trigger_to_fill_slippage_pips": (sum(slippage_pips) / len(slippage_pips)) if slippage_pips else None,
    }


def fills_to_trades(fill_results: list[FillResult], orders: list[MitOrderIntent]) -> pd.DataFrame:
    """Materialize FILLED entry results into open trades for reporting compatibility."""
    order_map = {order.order_id: order for order in orders}
    rows: list[dict[str, Any]] = []
    for fill in fill_results:
        if fill.status != "FILLED":
            continue
        order = order_map.get(fill.order_id)
        if order is None:
            continue
        rows.append(
            {
                "trade_id": order.order_id,
                "instrument": order.instrument,
                "side": order.side,
                "entry_time_utc": _datetime_to_iso(fill.filled_time_utc),
                "entry_price": fill.filled_price,
                "quantity": order.quantity,
                "timeframe": order.timeframe,
                "source_tag": order.source_tag,
                "notes": order.notes,
            }
        )

    columns = [
        "trade_id",
        "instrument",
        "side",
        "entry_time_utc",
        "entry_price",
        "quantity",
        "timeframe",
        "source_tag",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def _simulate_group(
    grouped_orders: list[tuple[int, MitOrderIntent]],
    feeder_or_store: CandleFeeder,
    simulator: ExecutionSimulator,
) -> list[tuple[int, FillResult]]:
    results: list[tuple[int, FillResult]] = []
    for order_index, order in grouped_orders:
        candles = feeder_or_store.load_slice(
            order.instrument,
            order.timeframe or "",
            start_utc=order.submit_time_utc,
            end_utc=order.expiry_time_utc,
        )
        result = simulator.simulate_mit_entry(order, candles)
        results.append((order_index, result))
    return results


def simulate_mit_entries(
    orders: list[MitOrderIntent],
    feeder_or_store: CandleFeeder,
    simulator: ExecutionSimulator | None = None,
    config: ExecutionConfig | None = None,
    data_manifest: Optional[dict[str, Any]] = None,
) -> SimulationArtifacts:
    """Simulate entry-only MIT fills with shared candle cache and thread batches."""
    execution_config = config or ExecutionConfig()
    if hasattr(feeder_or_store, "max_ram_bytes"):
        feeder_or_store.max_ram_bytes = max(0, int(execution_config.max_ram_bytes))

    if not orders:
        return SimulationArtifacts(
            orders=[],
            fills=[],
            fill_summary=_build_fill_summary([], []),
            data_manifest=data_manifest or {},
            generated_trades=fills_to_trades([], []),
        )

    active_simulator = simulator or FirstTouchMITSimulator()
    resolved_orders: list[MitOrderIntent] = []
    grouped: dict[tuple[str, str], list[tuple[int, MitOrderIntent]]] = {}
    for order in orders:
        timeframe = _resolve_timeframe(order, feeder_or_store)
        resolved_order = MitOrderIntent(
            order_id=order.order_id,
            instrument=order.instrument,
            timeframe=timeframe,
            side=order.side,
            submit_time_utc=order.submit_time_utc,
            trigger_price=order.trigger_price,
            quantity=order.quantity,
            expiry_time_utc=order.expiry_time_utc,
            source_tag=order.source_tag,
            notes=order.notes,
            order_type=order.order_type,
        )
        order_index = len(resolved_orders)
        resolved_orders.append(resolved_order)
        grouped.setdefault((resolved_order.instrument, timeframe), []).append((order_index, resolved_order))

    max_workers = max(1, int(execution_config.max_workers))
    max_workers = min(max_workers, max(1, len(grouped)))

    ordered_results: list[FillResult | None] = [None] * len(resolved_orders)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_simulate_group, batch, feeder_or_store, active_simulator): key
            for key, batch in grouped.items()
        }
        for future in as_completed(future_map):
            for order_index, fill in future.result():
                ordered_results[order_index] = fill

    fills = [fill for fill in ordered_results if fill is not None]
    generated_trades = fills_to_trades(fills, resolved_orders) if execution_config.emit_open_trades else None
    fill_summary = _build_fill_summary(resolved_orders, fills)
    return SimulationArtifacts(
        orders=resolved_orders,
        fills=fills,
        fill_summary=fill_summary,
        data_manifest=data_manifest or {},
        generated_trades=generated_trades,
    )


def write_execution_artifacts(
    artifacts: SimulationArtifacts,
    report_dir: str | Path,
    report_config: ReportConfig,
    execution_config: ExecutionConfig,
) -> dict[str, Any]:
    """Write fill artifacts and optionally generate open-trade report artifacts."""
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    order_map = {order.order_id: order for order in artifacts.orders}
    fill_rows: list[dict[str, Any]] = []
    for fill in artifacts.fills:
        order = order_map.get(fill.order_id)
        record = order.to_record() if order is not None else {"order_id": fill.order_id}
        record.update(fill.to_record())
        fill_rows.append(record)

    fill_columns = [
        "order_id",
        "instrument",
        "timeframe",
        "side",
        "submit_time_utc",
        "trigger_price",
        "quantity",
        "expiry_time_utc",
        "source_tag",
        "notes",
        "order_type",
        "status",
        "filled_time_utc",
        "filled_price",
        "trigger_time_utc",
        "reason",
        "simulator_name",
        "slippage_pips",
    ]
    fills_df = pd.DataFrame(fill_rows, columns=fill_columns)

    fills_path = out_dir / "fills.csv"
    fill_summary_path = out_dir / "fill_summary.json"
    execution_cfg_path = out_dir / "execution_config.json"
    manifest_path = out_dir / "data_manifest.json"
    run_cfg_path = out_dir / "run_config.json"

    fills_df.to_csv(fills_path, index=False)
    fill_summary_path.write_text(json.dumps(artifacts.fill_summary, indent=2, default=_json_default), encoding="utf-8")
    execution_cfg_path.write_text(json.dumps(execution_config.to_dict(), indent=2, default=_json_default), encoding="utf-8")
    manifest_path.write_text(json.dumps(artifacts.data_manifest or {}, indent=2, default=_json_default), encoding="utf-8")
    run_cfg_path.write_text(json.dumps(report_config.to_dict(), indent=2, default=_json_default), encoding="utf-8")

    report_artifacts: dict[str, Any] = {"summary": None, "paths": {}}
    if execution_config.emit_open_trades and artifacts.generated_trades is not None:
        report_artifacts = write_backtest_artifacts(
            artifacts.generated_trades,
            report_dir=out_dir,
            config=report_config,
            data_manifest=artifacts.data_manifest,
        )

    return {
        "fill_summary": artifacts.fill_summary,
        "summary": report_artifacts.get("summary"),
        "paths": {
            "report_dir": str(out_dir),
            "fills_csv": str(fills_path),
            "fill_summary_json": str(fill_summary_path),
            "execution_config_json": str(execution_cfg_path),
            "data_manifest_json": str(manifest_path),
            "run_config_json": str(run_cfg_path),
            **report_artifacts.get("paths", {}),
        },
    }
