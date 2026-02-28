"""Broker/trader runtime for candle-driven backtests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib
import importlib.util
import json
import logging
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd

from core.market_metadata import (
    get_instrument_class,
    get_pip_value,
    normalize_instrument,
    normalize_timeframe,
    round_price,
)

from .models import iso_utc
from .offline_oanda_provider import CandleFeeder, CandleSlice

logger = logging.getLogger(__name__)

EXECUTION_TIMEFRAME = "M1"
_EPOCH_NS = datetime(1970, 1, 1, tzinfo=timezone.utc)
_TIMEFRAME_NS: dict[str, int] = {
    "M1": 60_000_000_000,
    "M5": 300_000_000_000,
    "M15": 900_000_000_000,
    "M30": 1_800_000_000_000,
    "H1": 3_600_000_000_000,
    "H4": 14_400_000_000_000,
    "D": 86_400_000_000_000,
    "W": 604_800_000_000_000,
}


def _parse_datetime(value: Any | None) -> datetime | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    normalized = str(value).replace("Z", "+00:00")
    dt_value = datetime.fromisoformat(normalized)
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=timezone.utc)
    return dt_value.astimezone(timezone.utc)


def _datetime_to_ns(value: datetime) -> int:
    dt_value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    delta = dt_value - _EPOCH_NS
    return int((delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000)


def _ns_to_datetime(value: int) -> datetime:
    return pd.Timestamp(int(value), tz="UTC").to_pydatetime()


def _normalize_side(raw_value: Any) -> str:
    side = str(raw_value or "").strip().upper()
    if side in {"LONG", "BUY"}:
        return "LONG"
    if side in {"SHORT", "SELL"}:
        return "SHORT"
    raise ValueError(f"Unsupported side value: {raw_value}")


def _timeframe_duration_ns(timeframe: str) -> int:
    normalized = normalize_timeframe(timeframe)
    if normalized not in _TIMEFRAME_NS:
        raise ValueError(f"Unsupported timeframe for runtime scheduling: {normalized}")
    return _TIMEFRAME_NS[normalized]


def _normalize_visible_timeframes(primary_timeframe: str, raw_value: Any | None) -> list[str]:
    items = list(raw_value or [])
    normalized: list[str] = []
    for candidate in [EXECUTION_TIMEFRAME, primary_timeframe, *items]:
        tf = normalize_timeframe(candidate)
        if tf not in normalized:
            normalized.append(tf)
    return normalized


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat().replace("+00:00", "Z")
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _readonly_slice(frame: CandleSlice) -> CandleSlice:
    for name in (
        "time_ns",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ):
        getattr(frame, name).setflags(write=False)
    return frame


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    MIT = "MIT"

    @classmethod
    def from_value(cls, value: Any) -> "OrderType":
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError(f"Unsupported order_type: {value}") from exc


@dataclass(frozen=True)
class BracketSpec:
    stop_loss: float
    take_profit: float
    trailing_stop_distance: float | None = None

    @classmethod
    def from_raw(cls, value: Any | None) -> "BracketSpec | None":
        if value in (None, "", "None"):
            return None
        if isinstance(value, BracketSpec):
            return value
        if not isinstance(value, dict):
            raise ValueError(
                "bracket must be a mapping with stop_loss, take_profit, and optional trailing_stop_distance"
            )
        trailing_raw = value.get("trailing_stop_distance")
        trailing_distance: float | None
        if trailing_raw in (None, "", "None"):
            trailing_distance = None
        else:
            trailing_distance = float(trailing_raw)
            if trailing_distance <= 0:
                raise ValueError("trailing_stop_distance must be positive")
        return cls(
            stop_loss=float(value["stop_loss"]),
            take_profit=float(value["take_profit"]),
            trailing_stop_distance=trailing_distance,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "stop_loss": float(self.stop_loss),
            "take_profit": float(self.take_profit),
        }
        if self.trailing_stop_distance is not None:
            data["trailing_stop_distance"] = float(self.trailing_stop_distance)
        return data


@dataclass(frozen=True)
class CandleEvent:
    strategy_id: str
    instrument: str
    timeframe: str
    time_utc: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid_open: float
    bid_high: float
    bid_low: float
    bid_close: float
    ask_open: float
    ask_high: float
    ask_low: float
    ask_close: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "time_utc": iso_utc(self.time_utc),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "bid_open": float(self.bid_open),
            "bid_high": float(self.bid_high),
            "bid_low": float(self.bid_low),
            "bid_close": float(self.bid_close),
            "ask_open": float(self.ask_open),
            "ask_high": float(self.ask_high),
            "ask_low": float(self.ask_low),
            "ask_close": float(self.ask_close),
        }


@dataclass(frozen=True)
class OrderRequest:
    side: str
    order_type: OrderType
    quantity: float
    price: float | None = None
    expiry_time_utc: datetime | None = None
    reduce_only: bool = False
    bracket: BracketSpec | None = None
    client_tag: str | None = None
    notes: str | None = None

    @classmethod
    def from_raw(cls, value: Any) -> "OrderRequest":
        if isinstance(value, OrderRequest):
            return value
        if not isinstance(value, dict):
            raise ValueError("Trader output must contain OrderRequest objects or dict-like payloads")
        order_type = OrderType.from_value(value.get("order_type"))
        price = value.get("price")
        if price in ("", "None"):
            price = None
        return cls(
            side=_normalize_side(value.get("side")),
            order_type=order_type,
            quantity=float(value.get("quantity")),
            price=None if price is None else float(price),
            expiry_time_utc=_parse_datetime(value.get("expiry_time_utc")),
            reduce_only=bool(value.get("reduce_only", False)),
            bracket=BracketSpec.from_raw(value.get("bracket")),
            client_tag=None if value.get("client_tag") in (None, "") else str(value.get("client_tag")),
            notes=None if value.get("notes") in (None, "") else str(value.get("notes")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": float(self.quantity),
            "price": None if self.price is None else float(self.price),
            "expiry_time_utc": iso_utc(self.expiry_time_utc),
            "reduce_only": bool(self.reduce_only),
            "bracket": None if self.bracket is None else self.bracket.to_dict(),
            "client_tag": self.client_tag,
            "notes": self.notes,
        }


@dataclass
class StrategySessionConfig:
    strategy_id: str
    trader_class: str
    instrument: str
    timeframe: str
    visible_timeframes: list[str] = field(default_factory=list)
    starting_balance_usd: float = 100_000.0
    leverage: float = 30.0
    commission_usd_per_fill: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.instrument = normalize_instrument(self.instrument)
        self.timeframe = normalize_timeframe(self.timeframe)
        self.visible_timeframes = _normalize_visible_timeframes(self.timeframe, self.visible_timeframes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "trader_class": self.trader_class,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "visible_timeframes": list(self.visible_timeframes),
            "starting_balance_usd": float(self.starting_balance_usd),
            "leverage": float(self.leverage),
            "commission_usd_per_fill": float(self.commission_usd_per_fill),
            "params": dict(self.params or {}),
        }


@dataclass
class SlippageConfig:
    pips_by_order_type: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, value: Any | None) -> "SlippageConfig":
        if value in (None, "", "None"):
            return cls()
        if isinstance(value, SlippageConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError("slippage must be a mapping")
        normalized: dict[str, dict[str, float]] = {}
        for raw_order_type, raw_bucket in value.items():
            order_type = OrderType.from_value(raw_order_type).value
            if not isinstance(raw_bucket, dict):
                raise ValueError(f"slippage bucket for {raw_order_type} must be a mapping")
            normalized[order_type] = {str(key).strip().upper(): float(item) for key, item in raw_bucket.items()}
        return cls(pips_by_order_type=normalized)

    def pips_for(self, order_type: OrderType, instrument: str) -> float:
        bucket = self.pips_by_order_type.get(order_type.value, {})
        instrument_class = get_instrument_class(instrument).upper()
        return float(bucket.get(instrument_class, bucket.get("DEFAULT", 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            order_type: {key: float(value) for key, value in bucket.items()}
            for order_type, bucket in self.pips_by_order_type.items()
        }


@dataclass
class BacktestRunConfig:
    data_root: Path
    report_dir: Path
    strategies: list[StrategySessionConfig]
    start_utc: datetime | None = None
    end_utc: datetime | None = None
    account_currency: str = "USD"
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    _config_dir: Path | None = None

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        config_dir: Path | None = None,
    ) -> "BacktestRunConfig":
        if not isinstance(payload, dict):
            raise ValueError("Backtest config must be a JSON object")
        strategies_raw = payload.get("strategies")
        if not isinstance(strategies_raw, list) or not strategies_raw:
            raise ValueError("Backtest config requires a non-empty strategies list")

        sessions: list[StrategySessionConfig] = []
        seen_ids: set[str] = set()
        for item in strategies_raw:
            if not isinstance(item, dict):
                raise ValueError("Each strategy entry must be a JSON object")
            strategy_id = str(item.get("strategy_id") or "").strip()
            if not strategy_id:
                raise ValueError("strategy_id is required")
            if strategy_id in seen_ids:
                raise ValueError(f"Duplicate strategy_id: {strategy_id}")
            seen_ids.add(strategy_id)
            trader_class = str(item.get("trader_class") or "").strip()
            if not trader_class:
                raise ValueError(f"trader_class is required for {strategy_id}")
            sessions.append(
                StrategySessionConfig(
                    strategy_id=strategy_id,
                    trader_class=trader_class,
                    instrument=str(item.get("instrument") or ""),
                    timeframe=str(item.get("timeframe") or ""),
                    visible_timeframes=list(item.get("visible_timeframes") or []),
                    starting_balance_usd=float(item.get("starting_balance_usd", 100_000.0)),
                    leverage=float(item.get("leverage", 30.0)),
                    commission_usd_per_fill=float(item.get("commission_usd_per_fill", 0.0)),
                    params=dict(item.get("params") or {}),
                )
            )

        account_currency = str(payload.get("account_currency", "USD")).strip().upper()
        if account_currency != "USD":
            raise ValueError("Only USD account_currency is supported in v1")

        data_root = Path(payload.get("data_root") or "")
        report_dir = Path(payload.get("report_dir") or "")
        if not str(data_root):
            raise ValueError("data_root is required")
        if not str(report_dir):
            raise ValueError("report_dir is required")

        return cls(
            data_root=data_root,
            report_dir=report_dir,
            strategies=sessions,
            start_utc=_parse_datetime(payload.get("start_utc")),
            end_utc=_parse_datetime(payload.get("end_utc")),
            account_currency=account_currency,
            slippage=SlippageConfig.from_raw(payload.get("slippage")),
            _config_dir=config_dir,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "BacktestRunConfig":
        config_path = Path(path)
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = cls.from_dict(payload, config_dir=config_path.parent)
        if not config.data_root.is_absolute():
            config.data_root = (config_path.parent / config.data_root).resolve()
        if not config.report_dir.is_absolute():
            config.report_dir = (config_path.parent / config.report_dir).resolve()
        return config

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "report_dir": str(self.report_dir),
            "start_utc": iso_utc(self.start_utc),
            "end_utc": iso_utc(self.end_utc),
            "account_currency": self.account_currency,
            "slippage": self.slippage.to_dict(),
            "strategies": [item.to_dict() for item in self.strategies],
        }


@dataclass(frozen=True)
class AccountSnapshot:
    time_utc: datetime
    strategy_id: str
    balance_usd: float
    equity_usd: float
    used_margin_usd: float
    free_margin_usd: float
    unrealized_pnl_usd: float
    realized_pnl_usd: float
    margin_blocked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_utc": iso_utc(self.time_utc),
            "strategy_id": self.strategy_id,
            "balance_usd": float(self.balance_usd),
            "equity_usd": float(self.equity_usd),
            "used_margin_usd": float(self.used_margin_usd),
            "free_margin_usd": float(self.free_margin_usd),
            "unrealized_pnl_usd": float(self.unrealized_pnl_usd),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "margin_blocked": bool(self.margin_blocked),
        }


@dataclass(frozen=True)
class PositionRecord:
    strategy_id: str
    instrument: str
    side: str
    quantity: float
    entry_time_utc: datetime
    entry_price: float
    exit_time_utc: datetime
    exit_price: float
    realized_pnl_usd: float
    reason: str
    timeframe: str
    execution_timeframe: str = EXECUTION_TIMEFRAME
    position_id: str = ""
    realized_pips: float = 0.0
    holding_minutes: float = 0.0
    fees_usd: float = 0.0
    mae_pips: float = 0.0
    mfe_pips: float = 0.0
    mae_pnl_usd: float = 0.0
    mfe_pnl_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_time_utc": iso_utc(self.entry_time_utc),
            "entry_price": float(self.entry_price),
            "exit_time_utc": iso_utc(self.exit_time_utc),
            "exit_price": float(self.exit_price),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "reason": self.reason,
            "timeframe": self.timeframe,
            "execution_timeframe": self.execution_timeframe,
            "position_id": self.position_id,
            "realized_pips": float(self.realized_pips),
            "holding_minutes": float(self.holding_minutes),
            "fees_usd": float(self.fees_usd),
            "mae_pips": float(self.mae_pips),
            "mfe_pips": float(self.mfe_pips),
            "mae_pnl_usd": float(self.mae_pnl_usd),
            "mfe_pnl_usd": float(self.mfe_pnl_usd),
        }


@dataclass
class StrategyArtifacts:
    orders: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: dict[str, Any]
    paths: dict[str, str]


@dataclass
class BacktestArtifacts:
    strategies: dict[str, StrategyArtifacts]
    aggregate_equity_curve: pd.DataFrame
    aggregate_summary: dict[str, Any]
    paths: dict[str, str]


@dataclass(frozen=True)
class PriceQuote:
    time_utc: datetime
    bid: float
    ask: float
    mid: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_utc": iso_utc(self.time_utc),
            "bid": float(self.bid),
            "ask": float(self.ask),
            "mid": float(self.mid),
        }


@dataclass(frozen=True)
class AccountView:
    time_utc: datetime
    balance_usd: float
    equity_usd: float
    used_margin_usd: float
    free_margin_usd: float
    unrealized_pnl_usd: float
    realized_pnl_usd: float
    margin_blocked: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_utc": iso_utc(self.time_utc),
            "balance_usd": float(self.balance_usd),
            "equity_usd": float(self.equity_usd),
            "used_margin_usd": float(self.used_margin_usd),
            "free_margin_usd": float(self.free_margin_usd),
            "unrealized_pnl_usd": float(self.unrealized_pnl_usd),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "margin_blocked": bool(self.margin_blocked),
        }


@dataclass(frozen=True)
class PositionView:
    position_id: str
    strategy_id: str
    instrument: str
    timeframe: str
    execution_timeframe: str
    side: str
    quantity: float
    avg_entry_price: float
    entry_time_utc: datetime
    bracket_order_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "execution_timeframe": self.execution_timeframe,
            "side": self.side,
            "quantity": float(self.quantity),
            "avg_entry_price": float(self.avg_entry_price),
            "entry_time_utc": iso_utc(self.entry_time_utc),
            "bracket_order_ids": list(self.bracket_order_ids),
        }


@dataclass(frozen=True)
class OrderView:
    order_id: str
    strategy_id: str
    instrument: str
    timeframe: str
    execution_timeframe: str
    side: str
    order_type: str
    quantity: float
    requested_price: float | None
    submit_time_utc: datetime
    activation_time_utc: datetime | None
    expiry_time_utc: datetime | None
    reduce_only: bool
    status: str
    source: str
    client_tag: str | None
    notes: str | None
    bracket_role: str | None
    parent_position_id: str | None
    rejection_reason: str | None
    fill_time_utc: datetime | None
    fill_price: float | None
    fill_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "execution_timeframe": self.execution_timeframe,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": float(self.quantity),
            "requested_price": None if self.requested_price is None else float(self.requested_price),
            "submit_time_utc": iso_utc(self.submit_time_utc),
            "activation_time_utc": iso_utc(self.activation_time_utc),
            "expiry_time_utc": iso_utc(self.expiry_time_utc),
            "reduce_only": bool(self.reduce_only),
            "status": self.status,
            "source": self.source,
            "client_tag": self.client_tag,
            "notes": self.notes,
            "bracket_role": self.bracket_role,
            "parent_position_id": self.parent_position_id,
            "rejection_reason": self.rejection_reason,
            "fill_time_utc": iso_utc(self.fill_time_utc),
            "fill_price": None if self.fill_price is None else float(self.fill_price),
            "fill_reason": self.fill_reason,
        }


@dataclass(frozen=True)
class TradeView:
    position_id: str
    strategy_id: str
    instrument: str
    timeframe: str
    execution_timeframe: str
    side: str
    quantity: float
    entry_time_utc: datetime
    entry_price: float
    exit_time_utc: datetime
    exit_price: float
    realized_pnl_usd: float
    realized_pips: float
    holding_minutes: float
    fees_usd: float
    reason: str
    mae_pips: float = 0.0
    mfe_pips: float = 0.0
    mae_pnl_usd: float = 0.0
    mfe_pnl_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "execution_timeframe": self.execution_timeframe,
            "side": self.side,
            "quantity": float(self.quantity),
            "entry_time_utc": iso_utc(self.entry_time_utc),
            "entry_price": float(self.entry_price),
            "exit_time_utc": iso_utc(self.exit_time_utc),
            "exit_price": float(self.exit_price),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "realized_pips": float(self.realized_pips),
            "holding_minutes": float(self.holding_minutes),
            "fees_usd": float(self.fees_usd),
            "reason": self.reason,
            "mae_pips": float(self.mae_pips),
            "mfe_pips": float(self.mfe_pips),
            "mae_pnl_usd": float(self.mae_pnl_usd),
            "mfe_pnl_usd": float(self.mfe_pnl_usd),
        }


@dataclass(frozen=True)
class BrokerEvent:
    event_type: str
    time_utc: datetime
    order_id: str | None = None
    position_id: str | None = None
    reason: str | None = None
    fill_price: float | None = None
    quantity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "time_utc": iso_utc(self.time_utc),
            "order_id": self.order_id,
            "position_id": self.position_id,
            "reason": self.reason,
            "fill_price": None if self.fill_price is None else float(self.fill_price),
            "quantity": None if self.quantity is None else float(self.quantity),
        }


class Trader(Protocol):
    def on_clock(self, broker: "InstrumentBrokerApi") -> Any:
        ...


@dataclass
class _OrderState:
    order_id: str
    strategy_id: str
    instrument: str
    timeframe: str
    side: str
    order_type: OrderType
    quantity: float
    requested_price: float | None
    submit_time_utc: datetime
    activation_time_utc: datetime | None
    expiry_time_utc: datetime | None
    reduce_only: bool
    client_tag: str | None
    notes: str | None
    source: str
    bracket: BracketSpec | None = None
    bracket_role: str | None = None
    parent_position_id: str | None = None
    status: str = "WORKING"
    rejection_reason: str | None = None
    fill_time_utc: datetime | None = None
    fill_price: float | None = None
    slippage_pips: float | None = None
    commission_usd: float | None = None
    fill_reason: str | None = None
    execution_timeframe: str = EXECUTION_TIMEFRAME

    def to_record(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "execution_timeframe": self.execution_timeframe,
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": float(self.quantity),
            "requested_price": None if self.requested_price is None else float(self.requested_price),
            "submit_time_utc": iso_utc(self.submit_time_utc),
            "activation_time_utc": iso_utc(self.activation_time_utc),
            "expiry_time_utc": iso_utc(self.expiry_time_utc),
            "reduce_only": bool(self.reduce_only),
            "client_tag": self.client_tag,
            "notes": self.notes,
            "source": self.source,
            "bracket_role": self.bracket_role,
            "parent_position_id": self.parent_position_id,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "fill_time_utc": iso_utc(self.fill_time_utc),
            "fill_price": None if self.fill_price is None else float(self.fill_price),
            "slippage_pips": None if self.slippage_pips is None else float(self.slippage_pips),
            "commission_usd": None if self.commission_usd is None else float(self.commission_usd),
            "fill_reason": self.fill_reason,
        }

    def to_view(self) -> OrderView:
        return OrderView(
            order_id=self.order_id,
            strategy_id=self.strategy_id,
            instrument=self.instrument,
            timeframe=self.timeframe,
            execution_timeframe=self.execution_timeframe,
            side=self.side,
            order_type=self.order_type.value,
            quantity=float(self.quantity),
            requested_price=None if self.requested_price is None else float(self.requested_price),
            submit_time_utc=self.submit_time_utc,
            activation_time_utc=self.activation_time_utc,
            expiry_time_utc=self.expiry_time_utc,
            reduce_only=bool(self.reduce_only),
            status=self.status,
            source=self.source,
            client_tag=self.client_tag,
            notes=self.notes,
            bracket_role=self.bracket_role,
            parent_position_id=self.parent_position_id,
            rejection_reason=self.rejection_reason,
            fill_time_utc=self.fill_time_utc,
            fill_price=self.fill_price,
            fill_reason=self.fill_reason,
        )


@dataclass
class _FillState:
    fill_id: str
    order_id: str
    strategy_id: str
    instrument: str
    timeframe: str
    side: str
    order_type: OrderType
    quantity: float
    fill_time_utc: datetime
    fill_price: float
    slippage_pips: float
    commission_usd: float
    fill_reason: str
    position_id: str | None
    execution_timeframe: str = EXECUTION_TIMEFRAME

    def to_record(self) -> dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "execution_timeframe": self.execution_timeframe,
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": float(self.quantity),
            "fill_time_utc": iso_utc(self.fill_time_utc),
            "fill_price": float(self.fill_price),
            "slippage_pips": float(self.slippage_pips),
            "commission_usd": float(self.commission_usd),
            "fill_reason": self.fill_reason,
            "position_id": self.position_id,
        }


@dataclass
class _OpenPosition:
    position_id: str
    strategy_id: str
    instrument: str
    timeframe: str
    side: str
    quantity: float
    avg_entry_price: float
    entry_time_utc: datetime
    trailing_stop_distance: float | None = None
    entry_fees_usd: float = 0.0
    bracket_order_ids: list[str] = field(default_factory=list)
    mae_pips: float = 0.0
    mfe_pips: float = 0.0
    mae_pnl_usd: float = 0.0
    mfe_pnl_usd: float = 0.0


@dataclass
class _SessionState:
    config: StrategySessionConfig
    frames: dict[str, CandleSlice]
    frame_close_ns: dict[str, np.ndarray]
    trader: Trader
    visible_counts: dict[str, int]
    exec_cursor: int = 0
    broker_api: InstrumentBrokerApi | None = None
    recent_events: list[BrokerEvent] = field(default_factory=list)
    cycle_closed_timeframes: tuple[str, ...] = field(default_factory=tuple)
    manual_order_ids: list[str] = field(default_factory=list)
    bracket_order_ids: list[str] = field(default_factory=list)
    order_states: list[_OrderState] = field(default_factory=list)
    fill_states: list[_FillState] = field(default_factory=list)
    closed_positions: list[PositionRecord] = field(default_factory=list)
    account_snapshots: list[AccountSnapshot] = field(default_factory=list)
    open_position: _OpenPosition | None = None
    balance_usd: float = 0.0
    margin_blocked: bool = False
    order_seq: int = 0
    fill_seq: int = 0
    position_seq: int = 0

    def __post_init__(self) -> None:
        self.balance_usd = float(self.config.starting_balance_usd)


def _max_drawdown_from_series(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    numeric = values.astype(float)
    return float((numeric - numeric.cummax()).min())


def _sanitize_module_name(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return f"_backtest_trader_{digest}"


def _load_trader_class(spec: str, base_dir: Path | None = None) -> type:
    raw_spec = str(spec or "").strip()
    if not raw_spec:
        raise ValueError("trader_class is required")

    if ":" in raw_spec:
        target, class_name = raw_spec.rsplit(":", 1)
    else:
        target, class_name = raw_spec.rsplit(".", 1)

    target = target.strip()
    class_name = class_name.strip()
    if not class_name:
        raise ValueError(f"Invalid trader_class spec: {spec}")

    is_file_ref = target.endswith(".py") or "\\" in target or "/" in target
    if is_file_ref:
        file_path = Path(target)
        if not file_path.is_absolute():
            file_path = ((base_dir or Path.cwd()) / file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Trader module file not found: {file_path}")
        module_name = _sanitize_module_name(file_path)
        module_spec = importlib.util.spec_from_file_location(module_name, file_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Unable to import trader module from {file_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Trader class {class_name} not found in {target}") from exc


class _ConversionStore:
    def __init__(self, feeder: CandleFeeder):
        self._feeder = feeder
        self._cache: dict[tuple[str, str], CandleSlice] = {}

    def require_pair(self, instrument: str, timeframe: str) -> None:
        quote = normalize_instrument(instrument).split("_", 1)[1]
        if quote == "USD":
            return
        direct = f"{quote}_USD"
        inverse = f"USD_{quote}"
        normalized_tf = normalize_timeframe(timeframe)
        if normalized_tf in self._feeder.list_timeframes(direct):
            self._get_frame(direct, timeframe)
            return
        if normalized_tf in self._feeder.list_timeframes(inverse):
            self._get_frame(inverse, timeframe)
            return
        raise FileNotFoundError(
            f"Missing USD conversion dataset for {instrument} at {timeframe}. Expected {direct} or {inverse}."
        )

    def to_usd(self, instrument: str, timeframe: str, value_in_quote: float, time_utc: datetime) -> float:
        normalized = normalize_instrument(instrument)
        quote = normalized.split("_", 1)[1]
        if quote == "USD":
            return float(value_in_quote)

        direct = f"{quote}_USD"
        inverse = f"USD_{quote}"
        time_ns = _datetime_to_ns(time_utc)
        normalized_tf = normalize_timeframe(timeframe)
        if normalized_tf in self._feeder.list_timeframes(direct):
            rate = self._lookup_close(self._get_frame(direct, timeframe), time_ns)
            return float(value_in_quote) * rate
        rate = self._lookup_close(self._get_frame(inverse, timeframe), time_ns)
        if rate == 0:
            raise ZeroDivisionError(f"USD conversion rate is zero for {inverse}")
        return float(value_in_quote) / rate

    def _get_frame(self, instrument: str, timeframe: str) -> CandleSlice:
        key = (normalize_instrument(instrument), normalize_timeframe(timeframe))
        frame = self._cache.get(key)
        if frame is None:
            frame = self._feeder.load_slice(key[0], key[1])
            self._cache[key] = frame
        return frame

    @staticmethod
    def _lookup_close(frame: CandleSlice, time_ns: int) -> float:
        idx = int(np.searchsorted(frame.time_ns, time_ns, side="right")) - 1
        if idx < 0:
            raise ValueError(
                f"No conversion candle at or before {_ns_to_datetime(time_ns)} for {frame.instrument}/{frame.timeframe}"
            )
        return float(frame.close[idx])


def _extract_orders(raw_value: Any, callback_name: str) -> list[OrderRequest]:
    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise ValueError(f"Trader {callback_name}() must return a list or None")
    return [OrderRequest.from_raw(item) for item in raw_value]


class InstrumentBrokerApi:
    """Bounded broker view for a single strategy and instrument."""

    def __init__(self, engine: "BrokerEngine", session: _SessionState):
        self._engine = engine
        self._session = session
        self._event_time_utc: datetime | None = None
        self._closed_timeframes: tuple[str, ...] = ()

    def _set_cycle_context(self, event_time_utc: datetime, closed_timeframes: tuple[str, ...]) -> None:
        self._event_time_utc = event_time_utc
        self._closed_timeframes = tuple(closed_timeframes)

    def now_utc(self) -> datetime:
        if self._event_time_utc is None:
            raise RuntimeError("Broker clock is not active")
        return self._event_time_utc

    def instrument(self) -> str:
        return self._session.config.instrument

    def primary_timeframe(self) -> str:
        return self._session.config.timeframe

    def execution_timeframe(self) -> str:
        return EXECUTION_TIMEFRAME

    def current_bar(self, timeframe: str = EXECUTION_TIMEFRAME) -> CandleEvent | None:
        return self._engine._current_visible_candle(self._session, timeframe)

    def history(self, timeframe: str, n: int | None = None) -> CandleSlice:
        return self._engine._visible_history(self._session, timeframe, n)

    def available_timeframes(self) -> tuple[str, ...]:
        return tuple(self._session.config.visible_timeframes)

    def closed_timeframes(self) -> tuple[str, ...]:
        return tuple(self._closed_timeframes)

    def get_price(self) -> float:
        candle = self._engine._execution_candle(self._session)
        return float(candle.close)

    def get_quote(self) -> PriceQuote:
        candle = self._engine._execution_candle(self._session)
        return PriceQuote(
            time_utc=candle.time_utc,
            bid=float(candle.bid_close),
            ask=float(candle.ask_close),
            mid=float(candle.close),
        )

    def get_cash(self) -> float:
        return float(self._session.balance_usd)

    def get_account(self) -> AccountView:
        return self._engine._account_view(self._session)

    def get_position(self) -> PositionView | None:
        return self._engine._position_view(self._session)

    def get_working_orders(self) -> tuple[OrderView, ...]:
        return tuple(order.to_view() for order in self._session.order_states if order.status == "WORKING")

    def get_trade_log(self) -> tuple[TradeView, ...]:
        return tuple(self._engine._trade_view(item) for item in self._session.closed_positions)

    def get_recent_events(self) -> tuple[BrokerEvent, ...]:
        return tuple(self._session.recent_events)

    def submit_order(self, order: OrderRequest | dict[str, Any]) -> str:
        request = OrderRequest.from_raw(order)
        return self._engine._accept_order(self._session, request, self.now_utc())

    def cancel_order(self, order_id: str) -> None:
        self._engine._cancel_order_via_api(self._session, str(order_id))


class BrokerEngine:
    def __init__(self, config: BacktestRunConfig):
        self.config = config
        self.feeder = CandleFeeder(config.data_root)
        self.conversions = _ConversionStore(self.feeder)

    def run(self) -> BacktestArtifacts:
        sessions = self._build_sessions()
        self._run_event_loop(sessions)
        return self._write_artifacts(sessions)

    def _build_sessions(self) -> list[_SessionState]:
        if not self.config.data_root.exists():
            raise FileNotFoundError(f"data_root does not exist: {self.config.data_root}")

        sessions: list[_SessionState] = []
        base_dir = self.config._config_dir
        for item in self.config.strategies:
            if item.starting_balance_usd <= 0:
                raise ValueError(f"starting_balance_usd must be positive for {item.strategy_id}")
            if item.leverage <= 0:
                raise ValueError(f"leverage must be positive for {item.strategy_id}")

            trader_cls = _load_trader_class(item.trader_class, base_dir=base_dir)
            trader = trader_cls(
                strategy_id=item.strategy_id,
                instrument=item.instrument,
                timeframe=item.timeframe,
                params=dict(item.params or {}),
            )
            if not hasattr(trader, "on_clock"):
                raise TypeError(f"Trader {item.strategy_id} must implement on_clock(broker)")

            frames: dict[str, CandleSlice] = {}
            frame_close_ns: dict[str, np.ndarray] = {}
            visible_counts: dict[str, int] = {}
            for timeframe in item.visible_timeframes:
                frame = self.feeder.load_slice(
                    item.instrument,
                    timeframe,
                    start_utc=self.config.start_utc,
                    end_utc=self.config.end_utc,
                )
                if frame.rows == 0:
                    raise ValueError(f"No candles available for {item.strategy_id} ({item.instrument}/{timeframe})")
                frames[timeframe] = frame
                frame_close_ns[timeframe] = frame.time_ns + _timeframe_duration_ns(timeframe)
                visible_counts[timeframe] = 0

            if EXECUTION_TIMEFRAME not in frames:
                raise ValueError(f"M1 data is required for {item.strategy_id} ({item.instrument})")

            self.conversions.require_pair(item.instrument, EXECUTION_TIMEFRAME)
            session = _SessionState(
                config=item,
                frames=frames,
                frame_close_ns=frame_close_ns,
                trader=trader,
                visible_counts=visible_counts,
            )
            session.broker_api = InstrumentBrokerApi(self, session)
            sessions.append(session)
        return sessions

    def _run_event_loop(self, sessions: list[_SessionState]) -> None:
        schedule: list[tuple[int, int]] = []
        for idx, session in enumerate(sessions):
            frame = session.frames[EXECUTION_TIMEFRAME]
            if frame.rows > 0:
                schedule.append((int(frame.time_ns[0]), idx))
        schedule.sort()

        while schedule:
            current_bar_open_ns = schedule[0][0]
            due_indexes: list[int] = []
            while schedule and schedule[0][0] == current_bar_open_ns:
                _, session_index = schedule.pop(0)
                due_indexes.append(session_index)
            due_indexes.sort(key=lambda item: sessions[item].config.strategy_id)

            for session_index in due_indexes:
                session = sessions[session_index]
                session.recent_events.clear()
                self._process_existing_orders_for_bar(session)
                self._advance_cycle(session)
                self._record_account_snapshot(session)
                self._invoke_trader(session)
                self._record_account_snapshot(session)
                session.exec_cursor += 1
                exec_frame = session.frames[EXECUTION_TIMEFRAME]
                if session.exec_cursor < exec_frame.rows:
                    schedule.append((int(exec_frame.time_ns[session.exec_cursor]), session_index))
            schedule.sort()

        for session in sessions:
            self._finalize_open_orders(session)
            self._record_account_snapshot(session)

    def _advance_cycle(self, session: _SessionState) -> None:
        event_time = self._current_exec_close_time(session)
        event_ns = _datetime_to_ns(event_time)
        previous = dict(session.visible_counts)
        for timeframe in session.config.visible_timeframes:
            visible = int(np.searchsorted(session.frame_close_ns[timeframe], event_ns, side="right"))
            session.visible_counts[timeframe] = max(session.visible_counts[timeframe], visible)
        closed_timeframes = tuple(
            timeframe
            for timeframe in session.config.visible_timeframes
            if session.visible_counts[timeframe] > previous.get(timeframe, 0)
        )
        session.cycle_closed_timeframes = closed_timeframes
        if session.broker_api is None:
            raise RuntimeError("Broker API not initialized")
        session.broker_api._set_cycle_context(event_time, closed_timeframes)
        self._trail_stop_if_needed(session, event_time)
        self._update_position_excursions(session)

    def _invoke_trader(self, session: _SessionState) -> None:
        raw_orders = session.trader.on_clock(session.broker_api)
        for request in _extract_orders(raw_orders, "on_clock"):
            self._accept_order(session, request, self._current_exec_close_time(session))

    def _process_existing_orders_for_bar(self, session: _SessionState) -> None:
        if session.exec_cursor >= session.frames[EXECUTION_TIMEFRAME].rows:
            return

        bar_open_time = self._current_exec_open_time(session)
        active_brackets = self._active_bracket_orders(session, bar_open_time)
        if session.open_position is not None and active_brackets:
            chosen = self._choose_bracket_fill(session, active_brackets)
            if chosen is not None:
                self._fill_order(session, chosen, bar_open_time)
                self._cancel_other_brackets(session, keep_order_id=None, reason="position_closed", event_time_utc=bar_open_time)
                return

        for order in self._iter_active_manual_orders(session, bar_open_time):
            if order.expiry_time_utc is not None and bar_open_time >= order.expiry_time_utc:
                order.status = "EXPIRED"
                order.rejection_reason = "expired_before_bar"
                self._emit_event(session, "ORDER_CANCELED", bar_open_time, order=order, reason="expired_before_bar")
                continue
            if self._order_can_fill(session, order):
                self._fill_order(session, order, bar_open_time)
                return

    def _record_account_snapshot(self, session: _SessionState) -> None:
        candle = self._current_mark_candle(session)
        unrealized = self._compute_unrealized_pnl_usd(session, candle)
        used_margin = self._compute_used_margin_usd(session, candle)
        equity = session.balance_usd + unrealized
        free_margin = equity - used_margin
        margin_blocked = free_margin < 0
        if margin_blocked:
            session.margin_blocked = True

        snapshot = AccountSnapshot(
            time_utc=candle.time_utc,
            strategy_id=session.config.strategy_id,
            balance_usd=round(float(session.balance_usd), 10),
            equity_usd=round(float(equity), 10),
            used_margin_usd=round(float(used_margin), 10),
            free_margin_usd=round(float(free_margin), 10),
            unrealized_pnl_usd=round(float(unrealized), 10),
            realized_pnl_usd=round(float(session.balance_usd - session.config.starting_balance_usd), 10),
            margin_blocked=margin_blocked,
        )
        if not session.account_snapshots or session.account_snapshots[-1].to_dict() != snapshot.to_dict():
            session.account_snapshots.append(snapshot)

    def _accept_order(self, session: _SessionState, request: OrderRequest, submit_time_utc: datetime) -> str:
        current_candle = self._execution_candle(session)
        if request.quantity <= 0:
            return self._record_rejected_order(session, request, submit_time_utc, "invalid_quantity")
        if request.order_type in {OrderType.LIMIT, OrderType.STOP, OrderType.MIT} and request.price is None:
            return self._record_rejected_order(session, request, submit_time_utc, "price_required")

        next_activation = self._next_activation_time(session)
        open_position = session.open_position
        open_quantity = 0.0 if open_position is None else float(open_position.quantity)

        if request.reduce_only:
            if open_position is None:
                return self._record_rejected_order(session, request, submit_time_utc, "reduce_only_without_position")
            if request.side == open_position.side:
                return self._record_rejected_order(session, request, submit_time_utc, "reduce_only_same_direction")
            if request.quantity > open_quantity + 1e-12:
                return self._record_rejected_order(session, request, submit_time_utc, "reduce_only_size_exceeds_position")
        elif open_position is not None and request.side == open_position.side:
            return self._record_rejected_order(session, request, submit_time_utc, "same_direction_position_exists")

        if self._has_working_manual_order(session):
            return self._record_rejected_order(session, request, submit_time_utc, "existing_working_order")

        projected_price = request.price
        if projected_price is None:
            projected_price = current_candle.ask_close if request.side == "LONG" else current_candle.bid_close
        projected_quantity = self._projected_position_quantity(session, request)
        if projected_quantity > 0:
            projected_margin = self._compute_margin_for_price(session, projected_quantity, projected_price, submit_time_utc)
            latest = session.account_snapshots[-1] if session.account_snapshots else None
            free_margin = latest.free_margin_usd if latest is not None else session.config.starting_balance_usd
            if session.margin_blocked:
                return self._record_rejected_order(session, request, submit_time_utc, "margin_blocked")
            if projected_margin > free_margin + 1e-9:
                return self._record_rejected_order(session, request, submit_time_utc, "insufficient_margin")

        if open_position is not None:
            self._cancel_other_brackets(
                session,
                keep_order_id=None,
                reason="manual_order_submitted",
                event_time_utc=submit_time_utc,
            )

        order = _OrderState(
            order_id=self._next_order_id(session),
            strategy_id=session.config.strategy_id,
            instrument=session.config.instrument,
            timeframe=session.config.timeframe,
            side=request.side,
            order_type=request.order_type,
            quantity=float(request.quantity),
            requested_price=None if request.price is None else round_price(session.config.instrument, request.price),
            submit_time_utc=submit_time_utc,
            activation_time_utc=next_activation,
            expiry_time_utc=request.expiry_time_utc,
            reduce_only=bool(request.reduce_only),
            client_tag=request.client_tag,
            notes=request.notes,
            source="TRADER",
            bracket=request.bracket,
        )
        session.order_states.append(order)
        session.manual_order_ids.append(order.order_id)
        self._emit_event(session, "ORDER_ACCEPTED", submit_time_utc, order=order, reason="accepted")
        return order.order_id

    def _record_rejected_order(
        self,
        session: _SessionState,
        request: OrderRequest,
        submit_time_utc: datetime,
        reason: str,
    ) -> str:
        order = _OrderState(
            order_id=self._next_order_id(session),
            strategy_id=session.config.strategy_id,
            instrument=session.config.instrument,
            timeframe=session.config.timeframe,
            side=request.side,
            order_type=request.order_type,
            quantity=float(request.quantity),
            requested_price=None if request.price is None else round_price(session.config.instrument, request.price),
            submit_time_utc=submit_time_utc,
            activation_time_utc=None,
            expiry_time_utc=request.expiry_time_utc,
            reduce_only=bool(request.reduce_only),
            client_tag=request.client_tag,
            notes=request.notes,
            source="TRADER",
            bracket=request.bracket,
            status="REJECTED",
            rejection_reason=reason,
        )
        session.order_states.append(order)
        self._emit_event(session, "ORDER_REJECTED", submit_time_utc, order=order, reason=reason)
        return order.order_id

    def _cancel_order_via_api(self, session: _SessionState, order_id: str) -> None:
        order = self._order_by_id(session, order_id)
        if order is None:
            raise ValueError(f"Unknown order_id: {order_id}")
        if order.source != "TRADER":
            raise ValueError("Trader may only cancel its own manual orders")
        if order.status != "WORKING":
            raise ValueError(f"Order is not working: {order_id}")
        order.status = "CANCELED"
        order.rejection_reason = "canceled_by_trader"
        self._emit_event(
            session,
            "ORDER_CANCELED",
            self._current_exec_close_time(session),
            order=order,
            reason="canceled_by_trader",
        )

    def _has_working_manual_order(self, session: _SessionState) -> bool:
        for order_id in session.manual_order_ids:
            order = self._order_by_id(session, order_id)
            if order is not None and order.status == "WORKING":
                return True
        return False

    def _active_bracket_orders(self, session: _SessionState, bar_open_time: datetime) -> list[_OrderState]:
        result: list[_OrderState] = []
        for order_id in list(session.bracket_order_ids):
            order = self._order_by_id(session, order_id)
            if order is None or order.status != "WORKING":
                continue
            if order.activation_time_utc is None or order.activation_time_utc > bar_open_time:
                continue
            if order.expiry_time_utc is not None and bar_open_time >= order.expiry_time_utc:
                order.status = "EXPIRED"
                order.rejection_reason = "expired_before_bar"
                self._emit_event(session, "ORDER_CANCELED", bar_open_time, order=order, reason="expired_before_bar")
                continue
            result.append(order)
        return result

    def _choose_bracket_fill(self, session: _SessionState, orders: list[_OrderState]) -> _OrderState | None:
        stop_order: _OrderState | None = None
        target_order: _OrderState | None = None
        for order in orders:
            if order.bracket_role == "STOP_LOSS":
                stop_order = order
            elif order.bracket_role == "TAKE_PROFIT":
                target_order = order
        stop_hit = stop_order is not None and self._order_can_fill(session, stop_order)
        target_hit = target_order is not None and self._order_can_fill(session, target_order)
        if stop_hit and target_hit:
            return stop_order
        if stop_hit:
            return stop_order
        if target_hit:
            return target_order
        return None

    def _iter_active_manual_orders(self, session: _SessionState, bar_open_time: datetime) -> list[_OrderState]:
        active: list[_OrderState] = []
        for order_id in list(session.manual_order_ids):
            order = self._order_by_id(session, order_id)
            if order is None or order.status != "WORKING":
                continue
            if order.activation_time_utc is None or order.activation_time_utc > bar_open_time:
                continue
            active.append(order)
        active.sort(key=lambda item: (item.activation_time_utc or item.submit_time_utc, item.order_id))
        return active

    def _order_can_fill(self, session: _SessionState, order: _OrderState) -> bool:
        candle = self._execution_candle(session)
        if order.order_type == OrderType.MARKET:
            return True
        trigger = float(order.requested_price or 0.0)
        if order.order_type == OrderType.LIMIT:
            return candle.ask_low <= trigger if order.side == "LONG" else candle.bid_high >= trigger
        if order.order_type == OrderType.STOP:
            return candle.ask_high >= trigger if order.side == "LONG" else candle.bid_low <= trigger
        if order.order_type == OrderType.MIT:
            return candle.ask_low <= trigger if order.side == "LONG" else candle.bid_high >= trigger
        return False

    def _fill_order(self, session: _SessionState, order: _OrderState, fill_time_utc: datetime) -> None:
        fill_price, slippage_pips, fill_reason = self._determine_fill(session, order)
        projected_quantity = self._projected_position_quantity_for_fill(session, order)
        if projected_quantity > 0:
            projected_margin = self._compute_margin_for_price(session, projected_quantity, fill_price, fill_time_utc)
            equity = session.balance_usd + self._compute_unrealized_pnl_usd(session, self._execution_candle(session))
            if projected_margin > equity + 1e-9:
                order.status = "REJECTED_MARGIN"
                order.rejection_reason = "insufficient_margin_at_fill"
                self._emit_event(session, "ORDER_REJECTED", fill_time_utc, order=order, reason="insufficient_margin_at_fill")
                return

        commission = float(session.config.commission_usd_per_fill)
        session.balance_usd -= commission

        order.status = "FILLED"
        order.fill_time_utc = fill_time_utc
        order.fill_price = float(fill_price)
        order.slippage_pips = float(slippage_pips)
        order.commission_usd = commission
        order.fill_reason = fill_reason

        fill = _FillState(
            fill_id=self._next_fill_id(session),
            order_id=order.order_id,
            strategy_id=session.config.strategy_id,
            instrument=session.config.instrument,
            timeframe=session.config.timeframe,
            side=order.side,
            order_type=order.order_type,
            quantity=float(order.quantity),
            fill_time_utc=fill_time_utc,
            fill_price=float(fill_price),
            slippage_pips=float(slippage_pips),
            commission_usd=commission,
            fill_reason=fill_reason,
            position_id=session.open_position.position_id if session.open_position is not None else None,
        )
        self._emit_event(session, "ORDER_FILLED", fill_time_utc, order=order, fill=fill, reason=fill_reason)
        if order.bracket_role == "STOP_LOSS":
            self._emit_event(session, "STOP_LOSS_FILLED", fill_time_utc, order=order, fill=fill, reason=fill_reason)
        elif order.bracket_role == "TAKE_PROFIT":
            self._emit_event(session, "TAKE_PROFIT_FILLED", fill_time_utc, order=order, fill=fill, reason=fill_reason)

        self._apply_fill_to_position(session, order, fill)
        fill.position_id = session.open_position.position_id if session.open_position is not None else fill.position_id
        session.fill_states.append(fill)

    def _determine_fill(self, session: _SessionState, order: _OrderState) -> tuple[float, float, str]:
        candle = self._execution_candle(session)
        trigger = float(order.requested_price or 0.0)
        if order.order_type == OrderType.MARKET:
            base_price = candle.ask_open if order.side == "LONG" else candle.bid_open
            reason = "next_bar_open"
        elif order.order_type == OrderType.LIMIT:
            if order.side == "LONG":
                base_price = min(candle.ask_open, trigger) if candle.ask_open <= trigger else trigger
            else:
                base_price = max(candle.bid_open, trigger) if candle.bid_open >= trigger else trigger
            reason = "limit_fill"
        elif order.order_type == OrderType.STOP:
            base_price = max(trigger, candle.ask_open) if order.side == "LONG" else min(trigger, candle.bid_open)
            reason = "stop_fill"
        else:
            if order.side == "LONG" and candle.ask_open <= trigger:
                base_price = candle.ask_open
                reason = "gap_favorable"
            elif order.side == "SHORT" and candle.bid_open >= trigger:
                base_price = candle.bid_open
                reason = "gap_favorable"
            else:
                base_price = trigger
                reason = "intrabar_touch"

        adjusted = self._apply_slippage(order, base_price)
        pip_value = float(get_pip_value(order.instrument))
        slippage_pips = 0.0 if pip_value == 0 else abs(adjusted - base_price) / pip_value
        return round_price(order.instrument, adjusted), slippage_pips, reason

    def _apply_slippage(self, order: _OrderState, base_price: float) -> float:
        pips = self.config.slippage.pips_for(order.order_type, order.instrument)
        if pips == 0:
            return float(base_price)
        delta = float(get_pip_value(order.instrument)) * pips
        return float(base_price) + delta if order.side == "LONG" else float(base_price) - delta

    def _apply_fill_to_position(self, session: _SessionState, order: _OrderState, fill: _FillState) -> None:
        position = session.open_position
        if position is None:
            new_position = _OpenPosition(
                position_id=self._next_position_id(session),
                strategy_id=session.config.strategy_id,
                instrument=session.config.instrument,
                timeframe=session.config.timeframe,
                side=order.side,
                quantity=float(order.quantity),
                avg_entry_price=float(fill.fill_price),
                entry_time_utc=fill.fill_time_utc,
                trailing_stop_distance=None if order.bracket is None else order.bracket.trailing_stop_distance,
                entry_fees_usd=float(fill.commission_usd),
            )
            session.open_position = new_position
            fill.position_id = new_position.position_id
            self._emit_event(
                session,
                "POSITION_OPENED",
                fill.fill_time_utc,
                order=order,
                fill=fill,
                position_id=new_position.position_id,
                reason="entry_filled",
            )
            self._create_brackets_if_needed(session, order, new_position, fill.fill_time_utc)
            return

        if order.side == position.side:
            old_qty = position.quantity
            position.quantity += float(order.quantity)
            total_notional = (position.avg_entry_price * old_qty) + (fill.fill_price * float(order.quantity))
            position.avg_entry_price = float(total_notional / position.quantity)
            position.entry_fees_usd += float(fill.commission_usd)
            fill.position_id = position.position_id
            self._sync_bracket_quantities(session)
            return

        closing_quantity = min(float(order.quantity), float(position.quantity))
        realized_quote = (
            (fill.fill_price - position.avg_entry_price) * closing_quantity
            if position.side == "LONG"
            else (position.avg_entry_price - fill.fill_price) * closing_quantity
        )
        realized_usd = self.conversions.to_usd(
            position.instrument,
            EXECUTION_TIMEFRAME,
            realized_quote,
            fill.fill_time_utc,
        )
        session.balance_usd += realized_usd
        fill.position_id = position.position_id

        if float(order.quantity) < float(position.quantity) - 1e-12:
            position.quantity -= float(order.quantity)
            # Partial reductions keep the same position lifecycle; a trade row is emitted only on final close.
            self._sync_bracket_quantities(session)
            return

        reason = self._close_reason_for_order(order)
        self._close_position(
            session,
            position,
            fill,
            order=order,
            realized_usd=realized_usd,
            closed_quantity=closing_quantity,
            reason=reason,
        )
        remainder = float(order.quantity) - closing_quantity
        session.open_position = None
        if remainder > 1e-12 and not order.reduce_only:
            new_position = _OpenPosition(
                position_id=self._next_position_id(session),
                strategy_id=session.config.strategy_id,
                instrument=session.config.instrument,
                timeframe=session.config.timeframe,
                side=order.side,
                quantity=remainder,
                avg_entry_price=float(fill.fill_price),
                entry_time_utc=fill.fill_time_utc,
                trailing_stop_distance=None if order.bracket is None else order.bracket.trailing_stop_distance,
                entry_fees_usd=float(fill.commission_usd),
            )
            session.open_position = new_position
            fill.position_id = new_position.position_id
            self._emit_event(
                session,
                "POSITION_OPENED",
                fill.fill_time_utc,
                order=order,
                fill=fill,
                position_id=new_position.position_id,
                reason="reversal_opened",
            )
            self._create_brackets_if_needed(session, order, new_position, fill.fill_time_utc)

    def _create_brackets_if_needed(
        self,
        session: _SessionState,
        order: _OrderState,
        position: _OpenPosition,
        fill_time_utc: datetime,
    ) -> None:
        if order.bracket is None:
            return
        activation_time = self._next_activation_time(session)
        exit_side = "SHORT" if position.side == "LONG" else "LONG"
        stop_order = _OrderState(
            order_id=self._next_order_id(session),
            strategy_id=session.config.strategy_id,
            instrument=session.config.instrument,
            timeframe=session.config.timeframe,
            side=exit_side,
            order_type=OrderType.STOP,
            quantity=float(position.quantity),
            requested_price=round_price(session.config.instrument, order.bracket.stop_loss),
            submit_time_utc=fill_time_utc,
            activation_time_utc=activation_time,
            expiry_time_utc=None,
            reduce_only=True,
            client_tag=None,
            notes="broker_bracket_stop",
            source="BROKER_BRACKET",
            bracket_role="STOP_LOSS",
            parent_position_id=position.position_id,
        )
        target_order = _OrderState(
            order_id=self._next_order_id(session),
            strategy_id=session.config.strategy_id,
            instrument=session.config.instrument,
            timeframe=session.config.timeframe,
            side=exit_side,
            order_type=OrderType.LIMIT,
            quantity=float(position.quantity),
            requested_price=round_price(session.config.instrument, order.bracket.take_profit),
            submit_time_utc=fill_time_utc,
            activation_time_utc=activation_time,
            expiry_time_utc=None,
            reduce_only=True,
            client_tag=None,
            notes="broker_bracket_target",
            source="BROKER_BRACKET",
            bracket_role="TAKE_PROFIT",
            parent_position_id=position.position_id,
        )
        session.order_states.extend([stop_order, target_order])
        session.bracket_order_ids.extend([stop_order.order_id, target_order.order_id])
        position.bracket_order_ids = [stop_order.order_id, target_order.order_id]

    def _sync_bracket_quantities(self, session: _SessionState) -> None:
        if session.open_position is None:
            return
        for order_id in session.open_position.bracket_order_ids:
            order = self._order_by_id(session, order_id)
            if order is not None and order.status == "WORKING":
                order.quantity = float(session.open_position.quantity)

    def _trail_stop_if_needed(self, session: _SessionState, event_time_utc: datetime) -> None:
        position = session.open_position
        if position is None or position.trailing_stop_distance is None:
            return

        stop_order = self._position_stop_order(session, position)
        if stop_order is None or stop_order.requested_price is None:
            return

        candle = self._current_mark_candle(session)
        distance = float(position.trailing_stop_distance)
        if position.side == "LONG":
            candidate = round_price(position.instrument, float(candle.bid_close) - distance)
            if candidate <= float(stop_order.requested_price) + 1e-12:
                return
        else:
            candidate = round_price(position.instrument, float(candle.ask_close) + distance)
            if candidate >= float(stop_order.requested_price) - 1e-12:
                return

        stop_order.requested_price = float(candidate)
        self._emit_event(
            session,
            "ORDER_MODIFIED",
            event_time_utc,
            order=stop_order,
            position_id=position.position_id,
            reason="trailing_stop_updated",
        )

    def _update_position_excursions(self, session: _SessionState) -> None:
        position = session.open_position
        if position is None:
            return

        candle = self._execution_candle(session)
        pip_value = float(get_pip_value(position.instrument))
        if pip_value <= 0:
            return

        if position.side == "LONG":
            favorable_price = float(candle.bid_high)
            adverse_price = float(candle.bid_low)
            favorable_pips = (favorable_price - position.avg_entry_price) / pip_value
            adverse_pips = (adverse_price - position.avg_entry_price) / pip_value
            favorable_quote = (favorable_price - position.avg_entry_price) * position.quantity
            adverse_quote = (adverse_price - position.avg_entry_price) * position.quantity
        else:
            favorable_price = float(candle.ask_low)
            adverse_price = float(candle.ask_high)
            favorable_pips = (position.avg_entry_price - favorable_price) / pip_value
            adverse_pips = (position.avg_entry_price - adverse_price) / pip_value
            favorable_quote = (position.avg_entry_price - favorable_price) * position.quantity
            adverse_quote = (position.avg_entry_price - adverse_price) * position.quantity

        favorable_usd = self.conversions.to_usd(
            position.instrument,
            EXECUTION_TIMEFRAME,
            favorable_quote,
            candle.time_utc,
        )
        adverse_usd = self.conversions.to_usd(
            position.instrument,
            EXECUTION_TIMEFRAME,
            adverse_quote,
            candle.time_utc,
        )
        position.mfe_pips = max(float(position.mfe_pips), float(favorable_pips))
        position.mae_pips = min(float(position.mae_pips), float(adverse_pips))
        position.mfe_pnl_usd = max(float(position.mfe_pnl_usd), float(favorable_usd))
        position.mae_pnl_usd = min(float(position.mae_pnl_usd), float(adverse_usd))

    def _position_stop_order(self, session: _SessionState, position: _OpenPosition) -> _OrderState | None:
        for order_id in position.bracket_order_ids:
            order = self._order_by_id(session, order_id)
            if order is None or order.status != "WORKING":
                continue
            if order.bracket_role == "STOP_LOSS":
                return order
        return None

    def _cancel_other_brackets(
        self,
        session: _SessionState,
        keep_order_id: str | None,
        reason: str,
        event_time_utc: datetime | None,
    ) -> None:
        cancel_time = event_time_utc or self._current_exec_close_time(session)
        for order_id in list(session.bracket_order_ids):
            if keep_order_id is not None and order_id == keep_order_id:
                continue
            order = self._order_by_id(session, order_id)
            if order is None or order.status != "WORKING":
                continue
            order.status = "CANCELED"
            order.rejection_reason = reason
            self._emit_event(session, "ORDER_CANCELED", cancel_time, order=order, reason=reason)

    def _close_position(
        self,
        session: _SessionState,
        position: _OpenPosition,
        fill: _FillState,
        *,
        order: _OrderState,
        realized_usd: float,
        closed_quantity: float,
        reason: str,
    ) -> None:
        pip_value = float(get_pip_value(position.instrument))
        realized_pips = (
            (fill.fill_price - position.avg_entry_price) / pip_value
            if position.side == "LONG"
            else (position.avg_entry_price - fill.fill_price) / pip_value
        )
        hold_minutes = (fill.fill_time_utc - position.entry_time_utc).total_seconds() / 60.0
        session.closed_positions.append(
            PositionRecord(
                strategy_id=position.strategy_id,
                instrument=position.instrument,
                side=position.side,
                quantity=float(closed_quantity),
                entry_time_utc=position.entry_time_utc,
                entry_price=float(position.avg_entry_price),
                exit_time_utc=fill.fill_time_utc,
                exit_price=float(fill.fill_price),
                realized_pnl_usd=float(realized_usd - position.entry_fees_usd - fill.commission_usd),
                reason=reason,
                timeframe=position.timeframe,
                position_id=position.position_id,
                realized_pips=float(realized_pips),
                holding_minutes=float(hold_minutes),
                fees_usd=float(position.entry_fees_usd + fill.commission_usd),
                mae_pips=float(position.mae_pips),
                mfe_pips=float(position.mfe_pips),
                mae_pnl_usd=float(position.mae_pnl_usd),
                mfe_pnl_usd=float(position.mfe_pnl_usd),
            )
        )
        self._emit_event(
            session,
            "POSITION_CLOSED",
            fill.fill_time_utc,
            order=order,
            fill=fill,
            position_id=position.position_id,
            reason=reason,
        )
        self._cancel_other_brackets(session, keep_order_id=None, reason=reason.lower(), event_time_utc=fill.fill_time_utc)

    def _close_reason_for_order(self, order: _OrderState) -> str:
        if order.bracket_role == "STOP_LOSS":
            return "STOP_LOSS"
        if order.bracket_role == "TAKE_PROFIT":
            return "TAKE_PROFIT"
        if order.reduce_only:
            return "MANUAL_EXIT"
        return "REVERSAL"

    def _compute_unrealized_pnl_usd(self, session: _SessionState, candle: CandleEvent) -> float:
        position = session.open_position
        if position is None:
            return 0.0
        delta = (
            (candle.bid_close - position.avg_entry_price) * position.quantity
            if position.side == "LONG"
            else (position.avg_entry_price - candle.ask_close) * position.quantity
        )
        return self.conversions.to_usd(position.instrument, EXECUTION_TIMEFRAME, delta, candle.time_utc)

    def _compute_margin_for_price(self, session: _SessionState, quantity: float, price: float, time_utc: datetime) -> float:
        notional_quote = abs(float(quantity) * float(price))
        notional_usd = self.conversions.to_usd(session.config.instrument, EXECUTION_TIMEFRAME, notional_quote, time_utc)
        return abs(notional_usd) / float(session.config.leverage)

    def _compute_used_margin_usd(self, session: _SessionState, candle: CandleEvent) -> float:
        position = session.open_position
        if position is None:
            return 0.0
        mark_price = candle.bid_close if position.side == "LONG" else candle.ask_close
        return self._compute_margin_for_price(session, position.quantity, mark_price, candle.time_utc)

    def _projected_position_quantity(self, session: _SessionState, request: OrderRequest) -> float:
        position = session.open_position
        if position is None:
            return 0.0 if request.reduce_only else float(request.quantity)
        if request.side == position.side:
            return float(position.quantity + request.quantity)
        if request.reduce_only:
            return float(position.quantity - request.quantity)
        if request.quantity <= position.quantity:
            return float(position.quantity - request.quantity)
        return float(request.quantity - position.quantity)

    def _projected_position_quantity_for_fill(self, session: _SessionState, order: _OrderState) -> float:
        request = OrderRequest(
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.requested_price,
            expiry_time_utc=order.expiry_time_utc,
            reduce_only=order.reduce_only,
            bracket=order.bracket,
            client_tag=order.client_tag,
            notes=order.notes,
        )
        return self._projected_position_quantity(session, request)

    def _next_activation_time(self, session: _SessionState) -> datetime | None:
        next_index = session.exec_cursor + 1
        exec_frame = session.frames[EXECUTION_TIMEFRAME]
        if next_index >= exec_frame.rows:
            return None
        return _ns_to_datetime(int(exec_frame.time_ns[next_index]))

    def _next_order_id(self, session: _SessionState) -> str:
        session.order_seq += 1
        return f"{session.config.strategy_id}-O{session.order_seq:06d}"

    def _next_fill_id(self, session: _SessionState) -> str:
        session.fill_seq += 1
        return f"{session.config.strategy_id}-F{session.fill_seq:06d}"

    def _next_position_id(self, session: _SessionState) -> str:
        session.position_seq += 1
        return f"{session.config.strategy_id}-P{session.position_seq:06d}"

    def _order_by_id(self, session: _SessionState, order_id: str) -> _OrderState | None:
        for order in session.order_states:
            if order.order_id == order_id:
                return order
        return None

    def _finalize_open_orders(self, session: _SessionState) -> None:
        cancel_time = self._current_mark_candle(session).time_utc
        for order in session.order_states:
            if order.status == "WORKING":
                order.status = "CANCELED"
                order.rejection_reason = "end_of_data"
                self._emit_event(session, "ORDER_CANCELED", cancel_time, order=order, reason="end_of_data")

    def _validate_requested_timeframe(self, session: _SessionState, timeframe: str) -> str:
        normalized = normalize_timeframe(timeframe)
        if normalized not in session.config.visible_timeframes:
            raise ValueError(f"Timeframe {normalized} is not configured for {session.config.strategy_id}")
        return normalized

    def _current_visible_candle(self, session: _SessionState, timeframe: str) -> CandleEvent | None:
        normalized = self._validate_requested_timeframe(session, timeframe)
        visible = session.visible_counts[normalized]
        if visible <= 0:
            return None
        return self._build_candle_event(session, normalized, visible - 1)

    def _visible_history(self, session: _SessionState, timeframe: str, n: int | None = None) -> CandleSlice:
        normalized = self._validate_requested_timeframe(session, timeframe)
        visible = session.visible_counts[normalized]
        if n is None:
            start = 0
        else:
            if n <= 0:
                raise ValueError("history length must be positive")
            start = max(0, visible - int(n))
        return _readonly_slice(session.frames[normalized].slice_by_index(start, visible))

    def _build_candle_event(self, session: _SessionState, timeframe: str, row_index: int) -> CandleEvent:
        frame = session.frames[timeframe]
        close_ns = int(session.frame_close_ns[timeframe][row_index])
        return CandleEvent(
            strategy_id=session.config.strategy_id,
            instrument=session.config.instrument,
            timeframe=timeframe,
            time_utc=_ns_to_datetime(close_ns),
            open=float(frame.open[row_index]),
            high=float(frame.high[row_index]),
            low=float(frame.low[row_index]),
            close=float(frame.close[row_index]),
            volume=float(frame.volume[row_index]),
            bid_open=float(frame.bid_open[row_index]),
            bid_high=float(frame.bid_high[row_index]),
            bid_low=float(frame.bid_low[row_index]),
            bid_close=float(frame.bid_close[row_index]),
            ask_open=float(frame.ask_open[row_index]),
            ask_high=float(frame.ask_high[row_index]),
            ask_low=float(frame.ask_low[row_index]),
            ask_close=float(frame.ask_close[row_index]),
        )

    def _execution_candle(self, session: _SessionState) -> CandleEvent:
        return self._build_candle_event(session, EXECUTION_TIMEFRAME, session.exec_cursor)

    def _current_mark_candle(self, session: _SessionState) -> CandleEvent:
        exec_frame = session.frames[EXECUTION_TIMEFRAME]
        if exec_frame.rows == 0:
            raise ValueError(f"No candles available for {session.config.strategy_id}")
        row_index = min(max(session.exec_cursor, 0), exec_frame.rows - 1)
        visible = session.visible_counts[EXECUTION_TIMEFRAME]
        if visible > 0:
            row_index = min(row_index, visible - 1)
        return self._build_candle_event(session, EXECUTION_TIMEFRAME, row_index)

    def _current_exec_open_time(self, session: _SessionState) -> datetime:
        exec_frame = session.frames[EXECUTION_TIMEFRAME]
        return _ns_to_datetime(int(exec_frame.time_ns[session.exec_cursor]))

    def _current_exec_close_time(self, session: _SessionState) -> datetime:
        open_time = self._current_exec_open_time(session)
        return _ns_to_datetime(_datetime_to_ns(open_time) + _timeframe_duration_ns(EXECUTION_TIMEFRAME))

    def _account_view(self, session: _SessionState) -> AccountView:
        if session.account_snapshots:
            latest = session.account_snapshots[-1]
            return AccountView(
                time_utc=latest.time_utc,
                balance_usd=float(latest.balance_usd),
                equity_usd=float(latest.equity_usd),
                used_margin_usd=float(latest.used_margin_usd),
                free_margin_usd=float(latest.free_margin_usd),
                unrealized_pnl_usd=float(latest.unrealized_pnl_usd),
                realized_pnl_usd=float(latest.realized_pnl_usd),
                margin_blocked=bool(latest.margin_blocked),
            )
        mark = self._current_mark_candle(session)
        unrealized = self._compute_unrealized_pnl_usd(session, mark)
        used_margin = self._compute_used_margin_usd(session, mark)
        equity = session.balance_usd + unrealized
        return AccountView(
            time_utc=mark.time_utc,
            balance_usd=float(session.balance_usd),
            equity_usd=float(equity),
            used_margin_usd=float(used_margin),
            free_margin_usd=float(equity - used_margin),
            unrealized_pnl_usd=float(unrealized),
            realized_pnl_usd=float(session.balance_usd - session.config.starting_balance_usd),
            margin_blocked=bool((equity - used_margin) < 0),
        )

    def _position_view(self, session: _SessionState) -> PositionView | None:
        position = session.open_position
        if position is None:
            return None
        return PositionView(
            position_id=position.position_id,
            strategy_id=position.strategy_id,
            instrument=position.instrument,
            timeframe=position.timeframe,
            execution_timeframe=EXECUTION_TIMEFRAME,
            side=position.side,
            quantity=float(position.quantity),
            avg_entry_price=float(position.avg_entry_price),
            entry_time_utc=position.entry_time_utc,
            bracket_order_ids=tuple(position.bracket_order_ids),
        )

    def _trade_view(self, record: PositionRecord) -> TradeView:
        return TradeView(
            position_id=record.position_id,
            strategy_id=record.strategy_id,
            instrument=record.instrument,
            timeframe=record.timeframe,
            execution_timeframe=record.execution_timeframe,
            side=record.side,
            quantity=float(record.quantity),
            entry_time_utc=record.entry_time_utc,
            entry_price=float(record.entry_price),
            exit_time_utc=record.exit_time_utc,
            exit_price=float(record.exit_price),
            realized_pnl_usd=float(record.realized_pnl_usd),
            realized_pips=float(record.realized_pips),
            holding_minutes=float(record.holding_minutes),
            fees_usd=float(record.fees_usd),
            reason=record.reason,
            mae_pips=float(record.mae_pips),
            mfe_pips=float(record.mfe_pips),
            mae_pnl_usd=float(record.mae_pnl_usd),
            mfe_pnl_usd=float(record.mfe_pnl_usd),
        )

    def _emit_event(
        self,
        session: _SessionState,
        event_type: str,
        time_utc: datetime,
        *,
        order: _OrderState | None = None,
        fill: _FillState | None = None,
        position_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        session.recent_events.append(
            BrokerEvent(
                event_type=event_type,
                time_utc=time_utc,
                order_id=None if order is None else order.order_id,
                position_id=position_id if position_id is not None else (None if fill is None else fill.position_id),
                reason=reason,
                fill_price=None if fill is None else float(fill.fill_price),
                quantity=None if fill is None else float(fill.quantity),
            )
        )

    def _write_artifacts(self, sessions: list[_SessionState]) -> BacktestArtifacts:
        self.config.report_dir.mkdir(parents=True, exist_ok=True)
        manifest_timeframes = sorted({tf for session in sessions for tf in session.config.visible_timeframes})
        data_manifest = self.feeder.build_manifest(
            instruments=sorted({item.config.instrument for item in sessions}),
            timeframes=manifest_timeframes,
        )
        (self.config.report_dir / "run_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, default=_json_default),
            encoding="utf-8",
        )
        (self.config.report_dir / "data_manifest.json").write_text(
            json.dumps(data_manifest, indent=2, default=_json_default),
            encoding="utf-8",
        )

        strategies_root = self.config.report_dir / "strategies"
        aggregate_root = self.config.report_dir / "aggregate"
        strategies_root.mkdir(parents=True, exist_ok=True)
        aggregate_root.mkdir(parents=True, exist_ok=True)

        strategy_artifacts: dict[str, StrategyArtifacts] = {}
        equity_frames: dict[str, pd.DataFrame] = {}
        for session in sessions:
            strategy_dir = strategies_root / session.config.strategy_id
            strategy_dir.mkdir(parents=True, exist_ok=True)

            orders_df = pd.DataFrame([item.to_record() for item in session.order_states])
            fills_df = pd.DataFrame([item.to_record() for item in session.fill_states])
            trades_df = pd.DataFrame([item.to_dict() for item in session.closed_positions])
            equity_df = pd.DataFrame([item.to_dict() for item in session.account_snapshots])
            summary = self._build_strategy_summary(session, orders_df, fills_df, trades_df, equity_df)

            orders_path = strategy_dir / "orders.csv"
            fills_path = strategy_dir / "fills.csv"
            trades_path = strategy_dir / "trades.csv"
            equity_path = strategy_dir / "equity_curve.csv"
            summary_path = strategy_dir / "summary.json"
            report_path = strategy_dir / "report.md"

            orders_df.to_csv(orders_path, index=False)
            fills_df.to_csv(fills_path, index=False)
            trades_df.to_csv(trades_path, index=False)
            equity_df.to_csv(equity_path, index=False)
            summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
            report_path.write_text(self._render_strategy_report(summary), encoding="utf-8")

            strategy_artifacts[session.config.strategy_id] = StrategyArtifacts(
                orders=orders_df,
                fills=fills_df,
                trades=trades_df,
                equity_curve=equity_df,
                summary=summary,
                paths={
                    "strategy_dir": str(strategy_dir),
                    "orders_csv": str(orders_path),
                    "fills_csv": str(fills_path),
                    "trades_csv": str(trades_path),
                    "equity_curve_csv": str(equity_path),
                    "summary_json": str(summary_path),
                    "report_md": str(report_path),
                },
            )
            equity_frames[session.config.strategy_id] = equity_df

        aggregate_df = self._build_aggregate_equity_curve(equity_frames)
        aggregate_summary = self._build_aggregate_summary(strategy_artifacts, aggregate_df)
        aggregate_equity_path = aggregate_root / "portfolio_equity_curve.csv"
        aggregate_summary_path = aggregate_root / "summary.json"
        aggregate_report_path = aggregate_root / "report.md"
        aggregate_df.to_csv(aggregate_equity_path, index=False)
        aggregate_summary_path.write_text(json.dumps(aggregate_summary, indent=2, default=_json_default), encoding="utf-8")
        aggregate_report_path.write_text(self._render_aggregate_report(aggregate_summary), encoding="utf-8")

        return BacktestArtifacts(
            strategies=strategy_artifacts,
            aggregate_equity_curve=aggregate_df,
            aggregate_summary=aggregate_summary,
            paths={
                "report_dir": str(self.config.report_dir),
                "run_config_json": str(self.config.report_dir / "run_config.json"),
                "data_manifest_json": str(self.config.report_dir / "data_manifest.json"),
                "aggregate_equity_curve_csv": str(aggregate_equity_path),
                "aggregate_summary_json": str(aggregate_summary_path),
                "aggregate_report_md": str(aggregate_report_path),
            },
        )

    def _build_strategy_summary(
        self,
        session: _SessionState,
        orders_df: pd.DataFrame,
        fills_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        equity_df: pd.DataFrame,
    ) -> dict[str, Any]:
        pnl_series = trades_df.get("realized_pnl_usd", pd.Series(dtype=float))
        wins = int((pnl_series > 0).sum()) if not trades_df.empty else 0
        losses = int((pnl_series < 0).sum()) if not trades_df.empty else 0
        flats = int((pnl_series == 0).sum()) if not trades_df.empty else 0
        total_trades = int(len(trades_df))
        final_balance = float(equity_df["balance_usd"].iloc[-1]) if not equity_df.empty else float(session.config.starting_balance_usd)
        final_equity = float(equity_df["equity_usd"].iloc[-1]) if not equity_df.empty else final_balance

        margin_blocked_events = 0
        if not equity_df.empty and "margin_blocked" in equity_df.columns:
            flags = equity_df["margin_blocked"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
            flags = flags.astype(bool)
            margin_blocked_events = int((flags & ~flags.shift(fill_value=False)).sum())

        return {
            "strategy_id": session.config.strategy_id,
            "instrument": session.config.instrument,
            "timeframe": session.config.timeframe,
            "execution_timeframe": EXECUTION_TIMEFRAME,
            "visible_timeframes": list(session.config.visible_timeframes),
            "starting_balance_usd": float(session.config.starting_balance_usd),
            "ending_balance_usd": final_balance,
            "ending_equity_usd": final_equity,
            "return_pct": ((final_equity / session.config.starting_balance_usd) - 1.0) if session.config.starting_balance_usd else None,
            "realized_pnl_usd": float(pnl_series.sum()) if not trades_df.empty else 0.0,
            "realized_pips": float(trades_df.get("realized_pips", pd.Series(dtype=float)).sum()) if not trades_df.empty else 0.0,
            "open_position": None
            if session.open_position is None
            else {
                "position_id": session.open_position.position_id,
                "side": session.open_position.side,
                "quantity": float(session.open_position.quantity),
                "avg_entry_price": float(session.open_position.avg_entry_price),
                "entry_time_utc": iso_utc(session.open_position.entry_time_utc),
            },
            "order_summary": {
                "total_orders": int(len(orders_df)),
                "filled_orders": int((orders_df.get("status") == "FILLED").sum()) if not orders_df.empty else 0,
                "rejected_orders": int((orders_df.get("status") == "REJECTED").sum()) if not orders_df.empty else 0,
                "rejected_margin_orders": int((orders_df.get("status") == "REJECTED_MARGIN").sum()) if not orders_df.empty else 0,
                "expired_orders": int((orders_df.get("status") == "EXPIRED").sum()) if not orders_df.empty else 0,
                "canceled_orders": int((orders_df.get("status") == "CANCELED").sum()) if not orders_df.empty else 0,
            },
            "fill_summary": {
                "total_fills": int(len(fills_df)),
                "total_commission_usd": float(fills_df.get("commission_usd", pd.Series(dtype=float)).sum()) if not fills_df.empty else 0.0,
                "avg_slippage_pips": float(fills_df.get("slippage_pips", pd.Series(dtype=float)).mean()) if not fills_df.empty else 0.0,
            },
            "trade_summary": {
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "flats": flats,
                "win_rate": (wins / total_trades) if total_trades else None,
                "avg_realized_pnl_usd": float(pnl_series.mean()) if total_trades else None,
                "avg_realized_pips": float(trades_df.get("realized_pips", pd.Series(dtype=float)).mean()) if total_trades else None,
                "avg_holding_minutes": float(trades_df.get("holding_minutes", pd.Series(dtype=float)).mean()) if total_trades else None,
            },
            "risk": {
                "max_drawdown_usd": _max_drawdown_from_series(equity_df["equity_usd"]) if not equity_df.empty else 0.0,
                "margin_blocked_events": margin_blocked_events,
            },
        }

    def _build_aggregate_equity_curve(self, equity_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not equity_frames:
            return pd.DataFrame(
                columns=[
                    "time_utc",
                    "balance_usd",
                    "equity_usd",
                    "used_margin_usd",
                    "free_margin_usd",
                    "unrealized_pnl_usd",
                    "realized_pnl_usd",
                ]
            )

        all_times: set[str] = set()
        for frame in equity_frames.values():
            if not frame.empty:
                all_times.update(frame["time_utc"].astype(str).tolist())
        ordered_times = sorted(all_times)
        base = pd.DataFrame({"time_utc": ordered_times})
        numeric_columns = [
            "balance_usd",
            "equity_usd",
            "used_margin_usd",
            "free_margin_usd",
            "unrealized_pnl_usd",
            "realized_pnl_usd",
        ]
        for strategy_id, frame in equity_frames.items():
            if frame.empty:
                continue
            subset = frame[["time_utc", *numeric_columns]].copy()
            subset["time_utc"] = subset["time_utc"].astype(str)
            merged = base.merge(subset, on="time_utc", how="left")
            merged[numeric_columns] = merged[numeric_columns].ffill().fillna(0.0)
            for column in numeric_columns:
                base[f"{strategy_id}_{column}"] = merged[column]
        for column in numeric_columns:
            matching = [name for name in base.columns if name.endswith(f"_{column}")]
            base[column] = base[matching].sum(axis=1) if matching else 0.0
        return base[["time_utc", *numeric_columns]]

    def _build_aggregate_summary(self, strategy_artifacts: dict[str, StrategyArtifacts], aggregate_df: pd.DataFrame) -> dict[str, Any]:
        total_start = sum(item.summary.get("starting_balance_usd", 0.0) for item in strategy_artifacts.values())
        final_equity = float(aggregate_df["equity_usd"].iloc[-1]) if not aggregate_df.empty else float(total_start)
        return {
            "starting_balance_usd": float(total_start),
            "ending_equity_usd": float(final_equity),
            "execution_timeframe": EXECUTION_TIMEFRAME,
            "return_pct": ((final_equity / total_start) - 1.0) if total_start else None,
            "max_drawdown_usd": _max_drawdown_from_series(aggregate_df["equity_usd"]) if not aggregate_df.empty else 0.0,
            "total_closed_trades": int(sum(item.summary.get("trade_summary", {}).get("total_trades", 0) for item in strategy_artifacts.values())),
            "rejected_orders": int(sum(item.summary.get("order_summary", {}).get("rejected_orders", 0) for item in strategy_artifacts.values())),
            "rejected_margin_orders": int(sum(item.summary.get("order_summary", {}).get("rejected_margin_orders", 0) for item in strategy_artifacts.values())),
            "margin_blocked_events": int(sum(item.summary.get("risk", {}).get("margin_blocked_events", 0) for item in strategy_artifacts.values())),
            "strategies": {
                strategy_id: {
                    "ending_equity_usd": artifacts.summary.get("ending_equity_usd"),
                    "realized_pnl_usd": artifacts.summary.get("realized_pnl_usd"),
                    "return_pct": artifacts.summary.get("return_pct"),
                }
                for strategy_id, artifacts in strategy_artifacts.items()
            },
        }

    @staticmethod
    def _render_strategy_report(summary: dict[str, Any]) -> str:
        return "\n".join(
            [
                f"# Broker Report: {summary.get('strategy_id')}",
                "",
                f"- Instrument: `{summary.get('instrument')}`",
                f"- Timeframe: `{summary.get('timeframe')}`",
                f"- Execution timeframe: `{summary.get('execution_timeframe')}`",
                f"- Starting balance (USD): `{summary.get('starting_balance_usd')}`",
                f"- Ending equity (USD): `{summary.get('ending_equity_usd')}`",
                f"- Realized PnL (USD): `{summary.get('realized_pnl_usd')}`",
                f"- Closed trades: `{summary.get('trade_summary', {}).get('total_trades')}`",
                f"- Win rate: `{summary.get('trade_summary', {}).get('win_rate')}`",
                f"- Max drawdown (USD): `{summary.get('risk', {}).get('max_drawdown_usd')}`",
                "",
            ]
        )

    @staticmethod
    def _render_aggregate_report(summary: dict[str, Any]) -> str:
        lines = [
            "# Aggregate Portfolio Report",
            "",
            f"- Starting balance (USD): `{summary.get('starting_balance_usd')}`",
            f"- Ending equity (USD): `{summary.get('ending_equity_usd')}`",
            f"- Execution timeframe: `{summary.get('execution_timeframe')}`",
            f"- Return: `{summary.get('return_pct')}`",
            f"- Max drawdown (USD): `{summary.get('max_drawdown_usd')}`",
            f"- Closed trades: `{summary.get('total_closed_trades')}`",
            f"- Rejected orders: `{summary.get('rejected_orders')}`",
            "",
            "## Per Trader",
            "",
        ]
        for strategy_id, payload in summary.get("strategies", {}).items():
            lines.append(f"- `{strategy_id}` ending equity: `{payload.get('ending_equity_usd')}` return: `{payload.get('return_pct')}`")
        lines.append("")
        return "\n".join(lines)


def run_backtest(config: BacktestRunConfig) -> BacktestArtifacts:
    """Run a broker/trader backtest and write artifacts to disk."""
    return BrokerEngine(config).run()
