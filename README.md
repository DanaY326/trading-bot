# Meme-stock market maker

Quotes both sides on a fixed set of meme stocks (`mm/config.py` → `MEME_UNIVERSE`), earning the
spread while managing inventory and standing aside when these names get violent.

```
python run.py sim                     # simulated day, no account needed
python run.py sim --days 20 --seed 1  # P&L distribution over many days
python -m unittest tests.test_mm tests.test_alpaca   # Alpaca tests need alpaca-py
pip install -r requirements.txt && python run.py paper   # Alpaca paper trading
```

## How it quotes
- **Spread**: `max(min_half_spread_bps, vol_mult × σ × √horizon)`, from an EWMA of realized volatility.
- **Inventory skew**: when long, both quotes shift down so we sell more readily than we buy.
- **Long-only by default**: meme names are often hard to borrow, so we never go short.
- **Passive only**: quotes are clamped so they never cross the market.

## Calm markets: touch mode
When the expected 30s move is under 2.5 ticks (`touch_max_move_ticks`), the bot joins the
best bid/offer instead of quoting a vol-scaled distance away, steps a tick ahead when the
market is 3+ ticks wide, and leans by whole ticks for inventory. It pays on cheap,
tick-constrained names (AMC, BB) and not on GME at $25, where the tick is small relative to
how far price wanders; the gate makes that call per stock and switches off in a squeeze.
Tuned on Sep 23 - Oct 7 2026, checked on Oct 8-21 (held out).

## Meme-stock guards (any one pulls quotes)
Jump/squeeze detector with cooldown · realized vol cap · stale quote (likely an LULD halt) ·
market spread too wide · price floor · first 5 min after the open · flatten 10 min before close.

## Risk
Per-symbol share and notional caps · gross notional cap · daily loss kill switch (cancels and
flattens with collared limit orders, never market orders) · order rate limit.

## Live pilot
`python run.py paper --profile pilot --feed sip` runs AMC only, with 10-share orders and a
$25 daily stop. Every run logs to `logs/`, and `python report.py logs/*.jsonl` measures the real
fill rate and adverse selection and prints the replay settings that match them. See
[PILOT.md](PILOT.md) for the full runbook.

## Stress testing (GME, Jan 2021)
```
python stress.py synthetic --seeds 3                 # approximate squeeze days, no data needed
python stress.py yahoo --start 2021-01-22 --end 2021-02-05   # real daily OHLC from Yahoo
python scripts/fetch_yahoo.py GME data/gme_recent.csv        # real 1-min bars, last 30 days only
python scripts/fetch_bars.py GME 2021-01-19 2021-02-05 data/gme_2021_01.csv
python stress.py bars data/gme_2021_01.csv --spread-bps 60
```
Runs each day with guards on/off, under `probabilistic` and `toxic` (informed-flow) fill
models, with order/cancel latency. Use it to check mechanics: guards, kill switch, halts and
trapped inventory. Fills are queue-aware: random "friendly" fills only happen while our order
is at or better than the market's best price; behind it we only fill when the price runs
through us. P&L is still only indicative, because the replay doesn't know how many shares are
queued ahead of us. With minute bars, friendly fills are a share of the real volume that traded
(`--participation`, default 1% while joining the touch); sells pay SEC/FINRA fees.
Override any setting with `--set strategy.touch_max_move_ticks=0` and pick days with
`--day-from/--day-to` to keep a held-out test period.
