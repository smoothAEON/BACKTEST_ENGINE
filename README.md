# Offline Backtesting Engine And FX Calendar Utility

This repository contains two main tools:

1. `run_backtest.py`: a local, deterministic broker simulator for strategy backtesting against OANDA-style bid/ask candle CSV files.
2. `fx_cal.py`: a standalone FXEmpire economic-calendar fetcher that can pull paginated event data, scrape supporting metadata, and optionally stream live updates over websocket.

The backtesting side of the project is fully offline once your candle CSVs are on disk. The calendar utility is the only part that makes network requests, and only when you run it explicitly.

## What The Backtesting Engine Does

The engine simulates a small broker for one or more strategies. Each configured strategy gets its own USD-denominated account, its own order book, and its own position lifecycle. The runtime loads local candle data, advances a shared timeline, exposes a restricted broker API to each strategy, and writes both per-strategy and portfolio-level reports.

At a high level, a backtest run looks like this:

1. Load a JSON config file.
2. Resolve paths relative to the config file location.
3. Load the requested candle datasets from `data/{INSTRUMENT}/candles_{INSTRUMENT}_{TIMEFRAME}.csv`.
4. Import each trader class from a Python file path or module path.
5. Replay the `M1` candles in chronological order.
6. Process working orders at the start of each executable bar.
7. Advance visible candle history for all configured timeframes.
8. Call each trader's `on_clock(broker)` callback.
9. Accept any newly submitted orders.
10. Update account snapshots, positions, and broker events.
11. Write CSV, JSON, and Markdown artifacts when the replay completes.

## Backtesting Architecture

The core runtime lives under `backtest/`:

| File | Purpose |
| --- | --- |
| `backtest/runtime.py` | Broker engine, event loop, order handling, account and position state, artifact writing |
| `backtest/offline_oanda_provider.py` | CSV loader with strict bid/ask schema validation and cached candle slices |
| `backtest/reporting.py` | Standalone trade-ledger enrichment, summaries, stress scenarios, and Markdown reports |
| `backtest/execution.py` | MIT-oriented execution helpers for batch simulation workflows outside the main runtime |
| `backtest/models.py` | Shared reporting models and trade-ledger schema constants |

The execution flow is built around three concepts:

- `CandleFeeder` loads immutable `CandleSlice` objects from disk and validates the expected OANDA-style BA schema.
- `BrokerEngine` runs the replay loop and owns all canonical order, fill, position, and account state.
- `InstrumentBrokerApi` is the bounded interface strategies use inside `on_clock(broker)`.

## Data Requirements

The runtime expects local candle files with this naming convention:

```text
data/{INSTRUMENT}/candles_{INSTRUMENT}_{TIMEFRAME}.csv
```

Example:

```text
data/EUR_USD/candles_EUR_USD_M1.csv
data/EUR_USD/candles_EUR_USD_M30.csv
data/EUR_USD/candles_EUR_USD_H1.csv
```

Every CSV must include these 14 columns:

- `time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `bid_open`
- `bid_high`
- `bid_low`
- `bid_close`
- `ask_open`
- `ask_high`
- `ask_low`
- `ask_close`

Important runtime constraints:

- `M1` data is mandatory for every strategy. The engine clock and execution layer are driven by completed `M1` candles.
- Additional timeframes are optional, but they must exist on disk if a strategy lists them in `visible_timeframes`.
- The loader sorts candles by timestamp and removes duplicate timestamps deterministically.
- `start_utc` and `end_utc` filters are applied when slices are loaded.
- If an instrument is not quoted in USD, the runtime needs a conversion dataset at `M1` for either `{QUOTE}_USD` or `USD_{QUOTE}` so it can convert margin and PnL into USD.

## Strategy Interface

Strategies are plain Python classes. The runtime constructs them with:

```python
class MyTrader:
    def __init__(self, strategy_id: str, instrument: str, timeframe: str, params: dict):
        ...
```

A strategy must implement:

```python
def on_clock(self, broker) -> Any:
    ...
```

This is the current runtime contract. The callback is `on_clock(broker)`, not `on_candle(candle)`.

During each replay cycle, `on_clock(broker)` is called once per completed `M1` bar for that strategy. Inside the callback, the strategy can inspect market state and account state, then either:

- call `broker.submit_order(...)` directly, or
- return a list of order payloads / `OrderRequest` objects

The broker API intentionally exposes only a narrow, strategy-safe surface:

- `broker.now_utc()`: current engine time for the callback
- `broker.instrument()`: normalized instrument, such as `EUR_USD`
- `broker.primary_timeframe()`: the strategy's configured signal timeframe
- `broker.execution_timeframe()`: the execution timeframe, currently always `M1`
- `broker.current_bar(timeframe="M1")`: latest fully visible candle for a configured timeframe
- `broker.history(timeframe, n=None)`: visible candle history as a `CandleSlice`
- `broker.available_timeframes()`: all normalized timeframes visible to the strategy
- `broker.closed_timeframes()`: timeframes that just completed on the current cycle
- `broker.get_price()`: current mid close from the active `M1` bar
- `broker.get_quote()`: bid/ask/mid close quote from the active `M1` bar
- `broker.get_cash()`: current account cash balance
- `broker.get_account()`: balance, equity, used margin, free margin, unrealized PnL, realized PnL, margin-blocked flag
- `broker.get_position()`: current open position, if any
- `broker.get_working_orders()`: currently active working orders
- `broker.get_trade_log()`: closed trades for that strategy
- `broker.get_recent_events()`: broker events emitted since the previous callback
- `broker.submit_order(order)`: accept a new order
- `broker.cancel_order(order_id)`: cancel a working order

The bundled example strategy is in `strategies/00_template_mit_tp_sl_trailing/trader.py`.

## Runtime Timing Model

The timing model is important because it controls when strategies can see data and when orders can fill.

- The scheduler iterates chronologically over each strategy's `M1` dataset.
- At the start of a cycle, the engine first processes previously accepted working orders against the next executable bar.
- After that, the engine advances the visible candle counts for all configured timeframes.
- A timeframe appears in `broker.closed_timeframes()` only when a new bar of that timeframe has just completed.
- The engine then calls `on_clock(broker)`.
- Orders submitted during that callback are accepted immediately, but they do not activate until the next executable `M1` bar.

Practical consequence:

- A `MARKET` order submitted during cycle `N` is filled on the next bar open (cycle `N+1`), not on the same callback tick.
- Pending orders (`LIMIT`, `STOP`, `MIT`) are also activated from the next executable bar onward.

This makes the runtime deterministic and prevents look-ahead bias from same-bar execution after a strategy has already observed a closed candle.

## Supported Order Types And Fill Semantics

The runtime accepts four order types:

- `MARKET`
- `LIMIT`
- `STOP`
- `MIT`

All orders also carry:

- `side`: `LONG`/`BUY` or `SHORT`/`SELL`
- `quantity`
- optional `price` for non-market orders
- optional `expiry_time_utc`
- optional `reduce_only`
- optional broker-managed `bracket`
- optional `client_tag`
- optional `notes`

### `MARKET`

- Must have a positive quantity.
- Does not need a price.
- Fills at the next bar open.
- Long market entries use the bar's `ask_open`.
- Short market entries use the bar's `bid_open`.
- Slippage, if configured, is applied after the base fill price is determined.

### `LIMIT`

- Requires `price`.
- Fills when the bar trades through the limit.
- If the next bar opens at a more favorable price than the limit, the runtime uses the favorable open instead of the requested trigger.
- Long limits fill from ask-side logic.
- Short limits fill from bid-side logic.

### `STOP`

- Requires `price`.
- Fills when the bar trades through the stop level.
- If the next bar gaps beyond the stop, the runtime fills at the worse opening price implied by the gap.
- Long stops use ask-side logic.
- Short stops use bid-side logic.

### `MIT` (Market If Touched)

- Requires `price`.
- Fills on the first favorable touch of the trigger.
- If price gaps through the trigger in a favorable direction, the runtime fills at the favorable open.
- Otherwise it fills at the trigger price on intrabar touch.
- This is useful for pullback entries where you want a market-style fill after a favorable retracement.

## Position, Margin, And Bracket Behavior

Each strategy session has one account and, operationally, one live position slot for its configured instrument.

The runtime enforces the following behavior:

- Only `USD` is supported as `account_currency`.
- A trader-submitted order is rejected if `quantity <= 0`.
- `LIMIT`, `STOP`, and `MIT` orders are rejected if `price` is missing.
- Only one trader-submitted working order can exist at a time. Broker-generated bracket orders are separate.
- Same-direction trader-submitted entries are rejected while a position is already open.
- Opposite-direction trader-submitted orders can reduce, fully close, or reverse an open position depending on size.
- `reduce_only` orders must oppose the current position and cannot exceed the current position size.
- Free margin is checked before order acceptance and again at fill time.
- If free margin goes negative, the session is marked as margin-blocked and new entries are rejected until the run ends.

Attached brackets are broker-managed:

- A bracket can include `stop_loss`, `take_profit`, and optional `trailing_stop_distance`.
- After an entry fills, the engine automatically creates a reduce-only stop order and a reduce-only target order.
- Those bracket orders activate on the next executable bar.
- If one bracket fills, the sibling bracket is canceled automatically.
- If `trailing_stop_distance` is set, the stop-loss order ratchets only in the favorable direction using the latest visible `M1` mark.

## Backtest Configuration

The main entry point is:

```bash
python run_backtest.py run --config path/to/config.json
```

Example runtime config:

```json
{
  "data_root": "data",
  "report_dir": "reports/demo_run",
  "start_utc": "2025-01-01T00:00:00Z",
  "end_utc": "2025-06-01T00:00:00Z",
  "account_currency": "USD",
  "slippage": {
    "MARKET": {"FX": 0.1, "JPY": 0.1, "METAL": 0.2},
    "MIT": {"DEFAULT": 0.0}
  },
  "strategies": [
    {
      "strategy_id": "demo",
      "trader_class": "strategies/00_template_mit_tp_sl_trailing/trader.py:TemplateMITTrader",
      "instrument": "EUR_USD",
      "timeframe": "M30",
      "visible_timeframes": ["M1", "M30", "H1"],
      "starting_balance_usd": 100000,
      "leverage": 30,
      "commission_usd_per_fill": 0.0,
      "params": {
        "quantity": 1000,
        "stop_pips": 20,
        "target_pips": 40,
        "trailing_stop_pips": 10
      }
    }
  ]
}
```

Key config rules:

- `data_root` and `report_dir` are required.
- If they are relative paths, they are resolved relative to the config file, not the current shell directory.
- `strategies` must be a non-empty list.
- `strategy_id` values must be unique.
- `trader_class` can be either:
  - a file reference: `path/to/trader.py:ClassName`
  - a module reference: `package.module:ClassName` or `package.module.ClassName`
- `visible_timeframes` is normalized automatically to include both `M1` and the primary `timeframe`, even if you omit them.
- `account_currency` must be `USD` in the current version.

## Backtest Output Artifacts

A `run` invocation writes a full report directory:

```text
report_dir/
├── run_config.json
├── data_manifest.json
├── aggregate/
│   ├── portfolio_equity_curve.csv
│   ├── summary.json
│   └── report.md
└── strategies/
    └── {strategy_id}/
        ├── orders.csv
        ├── fills.csv
        ├── trades.csv
        ├── equity_curve.csv
        ├── summary.json
        └── report.md
```

What each artifact is for:

- `run_config.json`: normalized copy of the resolved runtime config
- `data_manifest.json`: which datasets were available and used
- `orders.csv`: every accepted, rejected, canceled, and filled order, including rejection reasons
- `fills.csv`: fill-by-fill execution records
- `trades.csv`: closed-position records with realized USD PnL, realized pips, holding time, fees, and per-trade `mae_*`/`mfe_*` in both pips and USD; excursion is measured from execution-bar bid/ask extremes, `mae_*` is signed adverse excursion (negative or zero), `mfe_*` is signed favorable excursion (positive or zero), and partial reductions do not emit separate trade rows
- `equity_curve.csv`: account snapshots over time
- per-strategy `summary.json`: strategy-level performance, trade counts, return, and risk metrics
- per-strategy `report.md`: human-readable strategy report
- `aggregate/portfolio_equity_curve.csv`: summed portfolio equity across strategies
- `aggregate/summary.json`: portfolio-level summary including ending equity, total closed trades, and max drawdown
- `aggregate/report.md`: human-readable portfolio report

## Standalone Ledger Reporting (`report-ledger`)

The repo also supports analysis of an external trade ledger without running the broker runtime. This is useful when you already have a `trades.csv` from another system and want the enrichment and report layer only.

Command:

```bash
python run_backtest.py report-ledger \
  --data-root data \
  --trades-csv path/to/trades.csv \
  --report-dir reports/ledger_only
```

Supported filters and options:

- `--start YYYY-MM-DD`: inclusive UTC entry-time filter
- `--end YYYY-MM-DD`: inclusive UTC entry-time filter
- `--instruments INST [INST ...]`: instrument filter
- `--timeframes TF [TF ...]`: timeframe filter
- `--manifest-scope selected|full|none`: control how `data_manifest.json` is built

The trade ledger must contain at least:

- `instrument`
- `side`
- `entry_time_utc`
- `entry_price`

Optional fields such as `exit_time_utc`, `exit_price`, `quantity`, `fees`, `timeframe`, `source_tag`, and `notes` are supported and enriched automatically.

The reporting layer computes:

- normalized side and instrument values
- realized pips
- holding time
- gross and net quote-currency PnL
- win/loss/flat/open outcomes
- expectancy
- profit factor
- drawdown
- breakdowns by instrument, side, timeframe, and month
- simple stress scenarios that apply extra class-based slippage

The `report-ledger` output directory contains:

- `trades_enriched.csv`
- `summary.json`
- `report.md`
- `data_manifest.json`
- `run_config.json`

## Economic Calendar Utility (`fx_cal.py`)

`fx_cal.py` is a separate utility for fetching economic calendar data from FXEmpire's public web surface through an unofficial API workflow.

It supports three related jobs:

1. Fetch paginated calendar events over HTTP.
2. Scrape supporting metadata (countries, timezones, regions, categories) from the calendar page's embedded `__NEXT_DATA__` JSON.
3. Connect to FXEmpire's websocket endpoint and stream incremental updates for events that were already fetched.

This script is independent from the backtesting runtime. You can use it to build or refresh macro-event datasets, inspect the current calendar, or pipe live updates into another process.

## What `fx_cal.py` Fetches

The script works with:

- HTTP API base: `https://www.fxempire.com/api/v1`
- Websocket endpoint: `wss://ec-ws-prod.fxempire.com`
- Locale-specific calendar pages for metadata scraping

For normal event fetches, it requests the paginated `economic-calendar` API, validates that the response contains the expected top-level keys, merges day groups across pages, and flattens the response into one event record per row.

For metadata fetches, it downloads the public calendar page for the selected locale, extracts the `__NEXT_DATA__` script tag, decodes the JSON payload, and returns:

- `countries`
- `timezones`
- `regions`
- `categories`

## `fx_cal.py` CLI Options

Basic usage:

```bash
python fx_cal.py --date-range current-week
```

Supported locales:

- `en`
- `it`
- `es`
- `pt`
- `de`
- `ar`
- `fr`

Supported named date ranges:

- `previous-day`
- `current-day`
- `next-day`
- `current-week`
- `next-week`

Supported filters:

- `--impact`
- `--country`
- `--region`
- `--category`

These filter arguments accept comma-separated values. The script normalizes whitespace and removes empty fragments before building the query string.

Date selection rules:

- You can use `--date-range ...`, or
- you can use `--date-from YYYY-MM-DD --date-to YYYY-MM-DD`

You cannot mix the two approaches. If one of `--date-from` or `--date-to` is provided, the other is required.

Additional behavior flags:

- `--timezone`: passed through to FXEmpire as the calendar timezone parameter, default `UTC`
- `--max-pages`: stop after a fixed number of pages even if the API has more
- `--format json|jsonl`: choose between pretty JSON or JSON Lines
- `--output path`: write to a file instead of stdout
- `--metadata`: return metadata instead of events
- `--stream`: keep running after the initial fetch and listen for websocket updates
- `--stream-jsonl`: append only streamed updates as JSONL after the initial output
- `--timeout`: HTTP and websocket timeout, default 10 seconds

Validation rule:

- `--stream-jsonl` requires `--stream`

## `fx_cal.py` Output Shapes

### Standard event fetch (`--format json`)

The JSON output includes:

- `request`: normalized request parameters
- `range`: API-reported start and end dates
- `pages_fetched`
- `event_count`
- `events`: flat event list
- `days`: grouped day buckets as returned by the source

### Standard event fetch (`--format jsonl`)

The output contains one flattened event object per line. This is useful for downstream processing with command-line tools or append-only pipelines.

### Metadata fetch (`--metadata`)

The output contains:

- `countries`
- `timezones`
- `regions`
- `categories`

### Streaming mode (`--stream`)

When streaming is enabled:

1. The script first performs the normal HTTP fetch.
2. It seeds an in-memory event cache keyed by event ID.
3. It connects to the websocket endpoint.
4. It listens for `update...` message types.
5. It merges partial updates into the cached event state.
6. If `--stream-jsonl` is enabled, it appends tracked updates as JSONL records.

The stream logic retries on transient connection failures with exponential backoff and stops after 5 failed reconnect attempts.

## Practical `fx_cal.py` Examples

Fetch the current week in JSON:

```bash
python fx_cal.py --date-range current-week --output calendar.json
```

Fetch a custom range in JSONL:

```bash
python fx_cal.py \
  --date-from 2026-03-01 \
  --date-to 2026-03-07 \
  --format jsonl \
  --output calendar.jsonl
```

Fetch only metadata:

```bash
python fx_cal.py --locale en --metadata --output fxempire_metadata.json
```

Fetch and then stream live updates:

```bash
python fx_cal.py \
  --date-range next-week \
  --stream \
  --stream-jsonl \
  --output calendar_stream.jsonl
```

## Installation

Use Python 3.10+.

Install the repository dependencies:

```bash
python -m pip install -r requirements.txt
```

Notes:

- The core backtesting runtime itself is centered on `numpy` and `pandas`, plus the standard library.
- `fx_cal.py` uses only the standard library for HTTP fetches unless you enable `--stream`.
- If you want websocket streaming, install `websockets` separately because the script imports it lazily only when `--stream` is used.

## Running Tests

Run the current test suite with:

```bash
python -m unittest discover -s tests -v
```

Run a focused test module:

```bash
python -m unittest tests.test_execution -v
python -m unittest tests.test_cli -v
```

## Repository Layout

```text
.
├── run_backtest.py
├── fx_cal.py
├── backtest/
├── core/
├── strategies/
├── data/
├── tests/
└── reports/
```

## Summary

If you are here for the trading simulator, start with `run_backtest.py run --config ...` and the template strategy under `strategies/00_template_mit_tp_sl_trailing/`.

If you are here for macro-event data, use `fx_cal.py` directly. It can fetch a snapshot, expose source metadata, or stay connected for live incremental updates.
