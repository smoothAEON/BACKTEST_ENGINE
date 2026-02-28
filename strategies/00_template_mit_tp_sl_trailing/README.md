# 00 - Template Trader

## Purpose

This folder is the starter template for engine-compatible traders.
It is intentionally a no-op until you add your own signal logic.

## Files

- `trader.py`: safe scaffold with `on_clock` and an example MIT order helper
- `config.json`: a complete runnable backtest config for this folder

## Engine Contract

Your trader must implement:

```python
class TemplateMITTrader:
    def __init__(self, strategy_id: str, instrument: str, timeframe: str, params: dict):
        ...

    def on_clock(self, broker) -> None:
        ...
```

`on_clock(broker)` is called once per completed `M1` candle.
The broker runtime owns fills, attached brackets, PnL, margin, and report generation.

## How To Customize

Replace the no-op block in `on_clock` with your entry and exit logic.
When you want a sample MIT order payload with a stop-loss and take-profit bracket, call `build_example_mit_order(...)`.
That helper is only an illustrative utility in this template, not a special engine hook.

## Direct Run Command

Run the template directly with:

```bash
python run_backtest.py run --config strategies/00_template_mit_tp_sl_trailing/config.json
```

The default run should succeed and produce zero trades until you add logic that submits orders.

## Output Expectation

Reports are written to `reports/00_template_mit_tp_sl_trailing`.
Zero orders, fills, and trades is the expected result for the unmodified template.
