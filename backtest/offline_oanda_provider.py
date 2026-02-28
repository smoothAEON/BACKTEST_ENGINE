"""CSV candle feeder with strict BA schema validation."""

from __future__ import annotations

import csv
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.market_metadata import normalize_instrument, normalize_timeframe

logger = logging.getLogger(__name__)

_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CACHE_SOFT_LIMIT_BYTES = 16 * 1024**3
_ARRAY_COLUMNS: tuple[str, ...] = (
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
)
_REQUIRED_BA_COLUMNS = {"time", *_ARRAY_COLUMNS}


def _parse_time_ns(raw_value: Any) -> int | None:
    raw = str(raw_value or "").strip()
    if not raw:
        return None

    normalized = raw.replace("Z", "+00:00")
    try:
        dt_value = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=timezone.utc)
    else:
        dt_value = dt_value.astimezone(timezone.utc)

    delta = dt_value - _EPOCH_UTC
    return ((delta.days * 86400) + delta.seconds) * 1_000_000_000 + (delta.microseconds * 1_000)


def _coerce_time_ns(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, datetime):
        dt_value = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        delta = dt_value - _EPOCH_UTC
        return ((delta.days * 86400) + delta.seconds) * 1_000_000_000 + (delta.microseconds * 1_000)
    return _parse_time_ns(value)


def _ns_to_iso_utc(value: int | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(int(value), tz="UTC").isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class CandleSlice:
    """Immutable columnar candle container used by the loader and simulator."""

    instrument: str
    timeframe: str
    time_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    bid_open: np.ndarray
    bid_high: np.ndarray
    bid_low: np.ndarray
    bid_close: np.ndarray
    ask_open: np.ndarray
    ask_high: np.ndarray
    ask_low: np.ndarray
    ask_close: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.time_ns.size)

    @property
    def estimated_size_bytes(self) -> int:
        return int(
            self.time_ns.nbytes
            + self.open.nbytes
            + self.high.nbytes
            + self.low.nbytes
            + self.close.nbytes
            + self.volume.nbytes
            + self.bid_open.nbytes
            + self.bid_high.nbytes
            + self.bid_low.nbytes
            + self.bid_close.nbytes
            + self.ask_open.nbytes
            + self.ask_high.nbytes
            + self.ask_low.nbytes
            + self.ask_close.nbytes
        )

    @property
    def start_time_utc(self) -> str | None:
        if self.rows == 0:
            return None
        return _ns_to_iso_utc(int(self.time_ns[0]))

    @property
    def end_time_utc(self) -> str | None:
        if self.rows == 0:
            return None
        return _ns_to_iso_utc(int(self.time_ns[-1]))

    def slice_by_index(self, start_idx: int, end_idx: int) -> CandleSlice:
        start = max(0, int(start_idx))
        end = min(self.rows, int(end_idx))
        if end < start:
            end = start
        return CandleSlice(
            instrument=self.instrument,
            timeframe=self.timeframe,
            time_ns=self.time_ns[start:end],
            open=self.open[start:end],
            high=self.high[start:end],
            low=self.low[start:end],
            close=self.close[start:end],
            volume=self.volume[start:end],
            bid_open=self.bid_open[start:end],
            bid_high=self.bid_high[start:end],
            bid_low=self.bid_low[start:end],
            bid_close=self.bid_close[start:end],
            ask_open=self.ask_open[start:end],
            ask_high=self.ask_high[start:end],
            ask_low=self.ask_low[start:end],
            ask_close=self.ask_close[start:end],
        )

    def slice_by_time(
        self,
        start_utc: Any | None = None,
        end_utc: Any | None = None,
    ) -> CandleSlice:
        start_ns = _coerce_time_ns(start_utc)
        end_ns = _coerce_time_ns(end_utc)

        left = 0 if start_ns is None else int(np.searchsorted(self.time_ns, start_ns, side="left"))
        right = self.rows if end_ns is None else int(np.searchsorted(self.time_ns, end_ns, side="right"))
        return self.slice_by_index(left, right)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": pd.to_datetime(self.time_ns, utc=True),
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
                "bid_open": self.bid_open,
                "bid_high": self.bid_high,
                "bid_low": self.bid_low,
                "bid_close": self.bid_close,
                "ask_open": self.ask_open,
                "ask_high": self.ask_high,
                "ask_low": self.ask_low,
                "ask_close": self.ask_close,
            }
        )


def _empty_slice(instrument: str, timeframe: str) -> CandleSlice:
    empty_float = np.asarray([], dtype=np.float64)
    return CandleSlice(
        instrument=instrument,
        timeframe=timeframe,
        time_ns=np.asarray([], dtype=np.int64),
        open=empty_float,
        high=empty_float,
        low=empty_float,
        close=empty_float,
        volume=empty_float,
        bid_open=empty_float,
        bid_high=empty_float,
        bid_low=empty_float,
        bid_close=empty_float,
        ask_open=empty_float,
        ask_high=empty_float,
        ask_low=empty_float,
        ask_close=empty_float,
    )


def _stable_sort_and_dedupe(frame: CandleSlice) -> CandleSlice:
    if frame.rows <= 1:
        return frame

    order = np.argsort(frame.time_ns, kind="mergesort")
    ordered = CandleSlice(
        instrument=frame.instrument,
        timeframe=frame.timeframe,
        time_ns=frame.time_ns[order],
        open=frame.open[order],
        high=frame.high[order],
        low=frame.low[order],
        close=frame.close[order],
        volume=frame.volume[order],
        bid_open=frame.bid_open[order],
        bid_high=frame.bid_high[order],
        bid_low=frame.bid_low[order],
        bid_close=frame.bid_close[order],
        ask_open=frame.ask_open[order],
        ask_high=frame.ask_high[order],
        ask_low=frame.ask_low[order],
        ask_close=frame.ask_close[order],
    )

    if ordered.rows <= 1:
        return ordered

    keep_mask = np.ones(ordered.rows, dtype=bool)
    keep_mask[:-1] = ordered.time_ns[:-1] != ordered.time_ns[1:]
    if keep_mask.all():
        return ordered

    return CandleSlice(
        instrument=ordered.instrument,
        timeframe=ordered.timeframe,
        time_ns=ordered.time_ns[keep_mask],
        open=ordered.open[keep_mask],
        high=ordered.high[keep_mask],
        low=ordered.low[keep_mask],
        close=ordered.close[keep_mask],
        volume=ordered.volume[keep_mask],
        bid_open=ordered.bid_open[keep_mask],
        bid_high=ordered.bid_high[keep_mask],
        bid_low=ordered.bid_low[keep_mask],
        bid_close=ordered.bid_close[keep_mask],
        ask_open=ordered.ask_open[keep_mask],
        ask_high=ordered.ask_high[keep_mask],
        ask_low=ordered.ask_low[keep_mask],
        ask_close=ordered.ask_close[keep_mask],
    )


def _read_frame_csv(path: Path, instrument: str, timeframe: str) -> CandleSlice:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = sorted(_REQUIRED_BA_COLUMNS.difference(fieldnames))
        if missing:
            raise ValueError(f"BA schema validation failed for {path}: missing columns {missing}")

        time_values: list[int] = []
        numeric_values: dict[str, list[float]] = {name: [] for name in _ARRAY_COLUMNS}
        for row in reader:
            time_ns = _parse_time_ns(row.get("time"))
            if time_ns is None:
                continue

            parsed_row: dict[str, float] = {}
            try:
                for column in _ARRAY_COLUMNS:
                    parsed_row[column] = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value in {path}: {exc}") from exc

            time_values.append(time_ns)
            for column, value in parsed_row.items():
                numeric_values[column].append(value)

    if not time_values:
        raise ValueError(f"Candle file has no valid rows: {path}")

    frame = CandleSlice(
        instrument=instrument,
        timeframe=timeframe,
        time_ns=np.asarray(time_values, dtype=np.int64),
        open=np.asarray(numeric_values["open"], dtype=np.float64),
        high=np.asarray(numeric_values["high"], dtype=np.float64),
        low=np.asarray(numeric_values["low"], dtype=np.float64),
        close=np.asarray(numeric_values["close"], dtype=np.float64),
        volume=np.asarray(numeric_values["volume"], dtype=np.float64),
        bid_open=np.asarray(numeric_values["bid_open"], dtype=np.float64),
        bid_high=np.asarray(numeric_values["bid_high"], dtype=np.float64),
        bid_low=np.asarray(numeric_values["bid_low"], dtype=np.float64),
        bid_close=np.asarray(numeric_values["bid_close"], dtype=np.float64),
        ask_open=np.asarray(numeric_values["ask_open"], dtype=np.float64),
        ask_high=np.asarray(numeric_values["ask_high"], dtype=np.float64),
        ask_low=np.asarray(numeric_values["ask_low"], dtype=np.float64),
        ask_close=np.asarray(numeric_values["ask_close"], dtype=np.float64),
    )
    return _stable_sort_and_dedupe(frame)


class CandleFeeder:
    """Load and validate candle datasets stored as OANDA-style BA CSV files."""

    def __init__(self, data_root: str | Path, max_ram_bytes: int = _CACHE_SOFT_LIMIT_BYTES):
        self.data_root = Path(data_root)
        self.max_ram_bytes = max(0, int(max_ram_bytes))
        self._frames: OrderedDict[tuple[str, str], CandleSlice] = OrderedDict()
        self._frame_sizes: dict[tuple[str, str], int] = {}
        self._cache_bytes = 0
        self._pending_bytes = 0
        self._loading_keys: set[tuple[str, str]] = set()
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)

    def _csv_path(self, instrument: str, timeframe: str) -> Path:
        inst = normalize_instrument(instrument)
        tf = normalize_timeframe(timeframe)
        return self.data_root / inst / f"candles_{inst}_{tf}.csv"

    def _estimate_load_bytes(self, path: Path) -> int:
        try:
            return max(1, int(path.stat().st_size))
        except OSError:
            return 1

    def _evict_oldest_locked(self) -> bool:
        if not self._frames:
            return False
        old_key, _ = self._frames.popitem(last=False)
        released = self._frame_sizes.pop(old_key, 0)
        self._cache_bytes = max(0, self._cache_bytes - released)
        self._cv.notify_all()
        return True

    def _reserve_pending_locked(self, required_bytes: int) -> None:
        if self.max_ram_bytes <= 0:
            return

        required = max(0, int(required_bytes))
        while (self._cache_bytes + self._pending_bytes + required) > self.max_ram_bytes:
            if self._evict_oldest_locked():
                continue
            if (self._cache_bytes + self._pending_bytes) == 0:
                break
            self._cv.wait(timeout=0.05)
        self._pending_bytes += required

    def _promote_cached_frame_locked(self, key: tuple[str, str]) -> CandleSlice | None:
        cached = self._frames.get(key)
        if cached is None:
            return None
        self._frames.move_to_end(key)
        return cached

    def _load_frame(self, instrument: str, timeframe: str) -> CandleSlice:
        inst = normalize_instrument(instrument)
        tf = normalize_timeframe(timeframe)
        key = (inst, tf)

        with self._cv:
            cached = self._promote_cached_frame_locked(key)
            if cached is not None:
                return cached

            while key in self._loading_keys:
                self._cv.wait()
                cached = self._promote_cached_frame_locked(key)
                if cached is not None:
                    return cached

            self._loading_keys.add(key)
            csv_path = self._csv_path(inst, tf)
            pending_bytes = self._estimate_load_bytes(csv_path)
            self._reserve_pending_locked(pending_bytes)

        try:
            if not csv_path.exists():
                raise FileNotFoundError(f"Candle file not found: {csv_path}")
            frame = _read_frame_csv(csv_path, inst, tf)
            logger.debug("Loaded candles %s/%s rows=%s", inst, tf, frame.rows)
        except Exception:
            with self._cv:
                self._pending_bytes = max(0, self._pending_bytes - pending_bytes)
                self._loading_keys.discard(key)
                self._cv.notify_all()
            raise

        actual_size = frame.estimated_size_bytes
        with self._cv:
            self._pending_bytes = max(0, self._pending_bytes - pending_bytes)
            if self.max_ram_bytes > 0:
                while (self._cache_bytes + actual_size) > self.max_ram_bytes:
                    if self._evict_oldest_locked():
                        continue
                    if self._cache_bytes == 0:
                        break
                    self._cv.wait(timeout=0.05)

            self._frames[key] = frame
            self._frame_sizes[key] = actual_size
            self._cache_bytes += actual_size
            self._loading_keys.discard(key)
            self._cv.notify_all()
            return frame

    def list_instruments(self) -> list[str]:
        """Return normalized instruments that have at least one candle CSV."""
        if not self.data_root.exists():
            return []

        instruments: list[str] = []
        for child in sorted(self.data_root.iterdir()):
            if not child.is_dir():
                continue
            has_csv = any(child.glob("candles_*_*.csv"))
            if has_csv:
                instruments.append(child.name.upper())
        return instruments

    def list_timeframes(self, instrument: str) -> list[str]:
        """Return available timeframes for one instrument."""
        inst = normalize_instrument(instrument)
        folder = self.data_root / inst
        if not folder.exists() or not folder.is_dir():
            return []

        result: set[str] = set()
        prefix = f"candles_{inst}_"
        for file_path in folder.glob(f"{prefix}*.csv"):
            stem = file_path.stem
            tf = stem.replace(prefix, "", 1).upper()
            if tf:
                result.add(tf)
        return sorted(result)

    def load_slice(
        self,
        instrument: str,
        timeframe: str,
        start_utc: Any | None = None,
        end_utc: Any | None = None,
        limit: int | None = None,
    ) -> CandleSlice:
        """Load columnar candles for one instrument/timeframe with optional filters."""
        frame = self._load_frame(instrument, timeframe)
        sliced = frame.slice_by_time(start_utc=start_utc, end_utc=end_utc)

        if limit is not None and sliced.rows > 0:
            count = max(1, int(limit))
            left = max(0, sliced.rows - count)
            sliced = sliced.slice_by_index(left, sliced.rows)
        return sliced

    def load_candles(
        self,
        instrument: str,
        timeframe: str,
        start_utc: Any | None = None,
        end_utc: Any | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Load candles for one instrument/timeframe with optional range and limit filters."""
        return self.load_slice(
            instrument,
            timeframe,
            start_utc=start_utc,
            end_utc=end_utc,
            limit=limit,
        ).to_dataframe()

    def get_frame(self, instrument: str, timeframe: str) -> pd.DataFrame:
        """Return full cached frame for compatibility with previous interface naming."""
        return self.load_candles(instrument, timeframe)

    def evict_instrument(self, instrument: str) -> None:
        """Evict cached frames for one instrument."""
        inst = normalize_instrument(instrument)
        with self._cv:
            remove_keys = [key for key in self._frames if key[0] == inst]
            for key in remove_keys:
                self._frames.pop(key, None)
                released = self._frame_sizes.pop(key, 0)
                self._cache_bytes = max(0, self._cache_bytes - released)
            self._cv.notify_all()

    def build_manifest(
        self,
        instruments: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build dataset coverage manifest for selected instruments/timeframes."""
        selected_instruments = [normalize_instrument(item) for item in (instruments or self.list_instruments())]
        selected_instruments = sorted(set(selected_instruments))

        selected_timeframes = [normalize_timeframe(item) for item in (timeframes or [])]
        selected_timeframes = sorted(set(selected_timeframes))

        files: dict[str, dict[str, Any]] = {}
        rows_total = 0
        valid_file_count = 0

        for instrument in selected_instruments:
            tf_list = selected_timeframes or self.list_timeframes(instrument)
            for timeframe in tf_list:
                key = f"{instrument}/{timeframe}"
                csv_path = self._csv_path(instrument, timeframe)
                entry: dict[str, Any] = {
                    "instrument": instrument,
                    "timeframe": timeframe,
                    "file_path": str(csv_path),
                    "exists": bool(csv_path.exists()),
                    "rows": 0,
                    "first_time_utc": None,
                    "last_time_utc": None,
                    "ba_schema_ok": False,
                    "error": None,
                }
                if not csv_path.exists():
                    entry["error"] = "file_missing"
                    files[key] = entry
                    continue

                try:
                    frame = self._load_frame(instrument, timeframe)
                    entry["rows"] = frame.rows
                    entry["first_time_utc"] = frame.start_time_utc
                    entry["last_time_utc"] = frame.end_time_utc
                    entry["ba_schema_ok"] = True
                    rows_total += frame.rows
                    valid_file_count += 1
                except Exception as exc:  # defensive manifest generation
                    entry["error"] = f"{type(exc).__name__}: {exc}"

                files[key] = entry

        return {
            "data_root": str(self.data_root),
            "instrument_count": len(selected_instruments),
            "timeframes_filter": selected_timeframes,
            "file_count": len(files),
            "valid_file_count": valid_file_count,
            "rows_total": rows_total,
            "files": files,
        }


# Compatibility alias requested by migration plan.
OfflineOANDAProvider = CandleFeeder
