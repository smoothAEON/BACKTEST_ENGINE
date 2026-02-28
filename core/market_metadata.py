"""Shared market metadata and normalization helpers."""

from __future__ import annotations

import re

# User-facing aliases for common symbols.
INSTRUMENT_ALIASES: dict[str, str] = {
    "GOLD": "XAU_USD",
    "SILVER": "XAG_USD",
    "OIL": "WTICO_USD",
    "BTC": "BTC_USD",
    "ETH": "ETH_USD",
}

# Canonical timeframe aliases used across commands/providers.
TIMEFRAME_ALIASES: dict[str, str] = {
    "1m": "M1",
    "m1": "M1",
    "5m": "M5",
    "m5": "M5",
    "15m": "M15",
    "m15": "M15",
    "30m": "M30",
    "m30": "M30",
    "1h": "H1",
    "h1": "H1",
    "4h": "H4",
    "h4": "H4",
    "1d": "D",
    "d": "D",
    "d1": "D",
    "daily": "D",
    "1w": "W",
    "w": "W",
    "w1": "W",
    "weekly": "W",
}

SUPPORTED_TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D", "W"}

_PAIR_RE = re.compile(r"^[A-Z0-9]{3,}_[A-Z0-9]{3,}$")


def resolve_instrument_alias(raw: str) -> str:
    """Resolve user alias to canonical OANDA instrument if available."""
    key = raw.strip().upper()
    return INSTRUMENT_ALIASES.get(key, key)


def normalize_instrument(raw: str, *, allow_aliases: bool = True) -> str:
    """
    Normalize user input to canonical OANDA instrument format.

    Examples:
    - eurusd -> EUR_USD
    - eur/usd -> EUR_USD
    - gold -> XAU_USD
    """
    if not raw or not raw.strip():
        raise ValueError("Instrument is required.")

    normalized = raw.strip().upper().replace("/", "_").replace("-", "_")
    normalized = normalized.replace(" ", "")

    if allow_aliases:
        normalized = resolve_instrument_alias(normalized)

    if "_" not in normalized and len(normalized) == 6 and normalized.isalnum():
        normalized = f"{normalized[:3]}_{normalized[3:]}"

    if not _PAIR_RE.match(normalized):
        raise ValueError(f"Invalid instrument format: {raw}")

    return normalized


def normalize_timeframe(raw: str) -> str:
    """Normalize timeframe aliases to canonical form (M1/M5/M15/M30/H1/H4/D/W)."""
    if not raw or not raw.strip():
        raise ValueError("Timeframe is required.")

    key = raw.strip().lower()
    if key in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[key]

    normalized = raw.strip().upper()
    if normalized in SUPPORTED_TIMEFRAMES:
        return normalized

    raise ValueError(
        f"Unsupported timeframe: {raw}. "
        "Supported aliases: 1m/m1, 5m/m5, 15m/m15, 30m/m30, 1h/h1, 4h/h4, 1d/d1, 1w/w1."
    )


def get_price_precision(instrument: str) -> int:
    """Get display precision by instrument class."""
    inst = normalize_instrument(instrument, allow_aliases=True)

    if inst.endswith("_JPY"):
        return 3
    if inst.startswith("XAU_") or inst.startswith("XAG_") or inst.startswith("WTICO_"):
        return 2
    if inst.startswith("BTC_") or inst.startswith("ETH_"):
        return 2
    return 5


def get_pip_value(instrument: str) -> float:
    """Get pip value for instrument."""
    inst = normalize_instrument(instrument, allow_aliases=True)
    if inst.endswith("_JPY") or inst.startswith("XAU_") or inst.startswith("XAG_") or inst.startswith("WTICO_"):
        return 0.01
    return 0.0001


def get_instrument_class(instrument: str) -> str:
    """Classify instrument for threshold/precision policies."""
    inst = normalize_instrument(instrument, allow_aliases=True)
    if inst.endswith("_JPY"):
        return "JPY"
    if inst.startswith("XAU_") or inst.startswith("XAG_"):
        return "METAL"
    if inst.startswith("WTICO_") or inst.startswith("BRENT_"):
        return "ENERGY"
    if inst.startswith("BTC_") or inst.startswith("ETH_"):
        return "CRYPTO"
    return "FX"


def round_price(instrument: str, value: float) -> float:
    """Round price using instrument-aware precision."""
    return round(float(value), get_price_precision(instrument))


def format_price(instrument: str, value: float) -> str:
    """Format price string using instrument-aware precision."""
    precision = get_price_precision(instrument)
    return f"{float(value):,.{precision}f}"
