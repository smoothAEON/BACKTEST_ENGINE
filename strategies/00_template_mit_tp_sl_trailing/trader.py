"""Minimal broker-runtime trader template for the offline engine."""

from __future__ import annotations

from typing import Any

from core.market_metadata import get_pip_value, round_price


class TemplateMITTrader:
    """Safe no-op scaffold for engine-compatible trader implementations."""

    def __init__(self, strategy_id: str, instrument: str, timeframe: str, params: dict[str, Any]):
        self.strategy_id = strategy_id
        self.instrument = instrument
        self.timeframe = timeframe
        self.params = dict(params or {})
        self.last_signal_time: str | None = None

    def on_clock(self, broker) -> None:
        """Read broker-owned data and submit orders through the broker API."""
        for event in broker.get_recent_events():
            if event.event_type in {"STOP_LOSS_FILLED", "TAKE_PROFIT_FILLED"}:
                self.last_signal_time = None

        if "M30" not in broker.available_timeframes():
            return
        if "M30" not in broker.closed_timeframes():
            return

        signal_bar = broker.current_bar("M30")
        if signal_bar is None:
            return
        signal_key = signal_bar.time_utc.isoformat()
        if self.last_signal_time == signal_key:
            return

        m30_history = broker.history("M30", 3)
        if m30_history.rows < 3:
            return
        if broker.get_position() is not None or broker.get_working_orders():
            return

        if float(m30_history.close[-1]) <= float(m30_history.open[-1]):
            return

        entry = broker.current_bar("M1")
        if entry is None:
            return

        broker.submit_order(self.build_example_mit_order("LONG", float(entry.ask_close)))
        self.last_signal_time = signal_key

    def build_example_mit_order(self, side: str, entry_price: float) -> dict[str, Any]:
        """Build an example MIT order payload with a broker-managed bracket."""
        normalized_side = str(side).strip().upper()
        if normalized_side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")

        quantity = float(self.params.get("quantity", 1000))
        stop_pips = float(self.params.get("stop_pips", 20))
        target_pips = float(self.params.get("target_pips", 40))
        trailing_stop_pips = self.params.get("trailing_stop_pips")
        pip_value = float(get_pip_value(self.instrument))
        rounded_entry_price = float(round_price(self.instrument, float(entry_price)))

        if normalized_side == "LONG":
            stop_loss = rounded_entry_price - (stop_pips * pip_value)
            take_profit = rounded_entry_price + (target_pips * pip_value)
        else:
            stop_loss = rounded_entry_price + (stop_pips * pip_value)
            take_profit = rounded_entry_price - (target_pips * pip_value)

        bracket = {
            "stop_loss": float(round_price(self.instrument, stop_loss)),
            "take_profit": float(round_price(self.instrument, take_profit)),
        }
        if trailing_stop_pips not in (None, "", "None"):
            trailing_distance = float(trailing_stop_pips) * pip_value
            if trailing_distance <= 0:
                raise ValueError("trailing_stop_pips must be positive")
            bracket["trailing_stop_distance"] = float(round_price(self.instrument, trailing_distance))

        return {
            "side": normalized_side,
            "order_type": "MIT",
            "quantity": quantity,
            "price": rounded_entry_price,
            "bracket": bracket,
            "notes": "template_example_order",
        }
