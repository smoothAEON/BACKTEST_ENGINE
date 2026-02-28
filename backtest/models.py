"""Data models for candle-feeder and trade-ledger reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


REQUIRED_TRADE_COLUMNS: tuple[str, ...] = (
    "instrument",
    "side",
    "entry_time_utc",
    "entry_price",
)

OPTIONAL_TRADE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "exit_time_utc",
    "exit_price",
    "quantity",
    "fees",
    "timeframe",
    "source_tag",
    "notes",
)


def iso_utc(value: Any) -> Optional[str]:
    """Serialize datetime-like values to ISO8601 UTC string when possible."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat().replace("+00:00", "Z")
        except TypeError:
            return str(value)
    return str(value)


@dataclass
class ReportConfig:
    """Configuration for trade-ledger reporting runs."""

    data_root: Optional[Path] = None
    report_dir: Optional[Path] = None
    source_trades_csv: Optional[Path] = None
    start_utc: Optional[Any] = None
    end_utc: Optional[Any] = None
    instruments: list[str] = field(default_factory=list)
    timeframes: list[str] = field(default_factory=list)
    price_type: str = "BA"
    slippage_profile: str = "default"
    notes: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("data_root", "report_dir", "source_trades_csv"):
            value = payload.get(key)
            if value is not None:
                payload[key] = str(value)
        payload["start_utc"] = iso_utc(payload.get("start_utc"))
        payload["end_utc"] = iso_utc(payload.get("end_utc"))
        payload["instruments"] = [str(item).upper() for item in (payload.get("instruments") or [])]
        payload["timeframes"] = [str(item).upper() for item in (payload.get("timeframes") or [])]
        return payload
