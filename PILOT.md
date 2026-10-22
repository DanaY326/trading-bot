# Live pilot: measuring real fills

Every backtest result in this repo depends on two numbers the replay has to assume:

- **participation**: when our order sits at the best bid/offer, what share of the volume
  trading on that side do we get? (Default assumption: 1%.)
- **toxicity**: how much more often a fill is followed by the price moving against us than
  chance would predict. (Default assumption: 0.5.)

At 0.1% participation, AMC's held-out profit drops from about +$2,200 to about +$200 over
10 days. Only real fills can settle these numbers. The pilot measures them as cheaply as
possible. Its goal is information, not profit.

## What it trades

`python run.py paper --profile pilot` uses `pilot_config()` in `mm/config.py`:

| | |
|---|---|
| Symbol | AMC only (tick-constrained, where touch mode tested best) |
| Order size | 10 shares |
| Max position | 50 shares, about $150 at $3 |
| Daily loss limit | $25: cancel everything and flatten |
| Order rate | 40/minute |

## Before the first run

1. **Data feed.** Touch mode means joining the national best bid/offer. The free IEX feed
   shows only IEX's quotes, which are often worse, so the bot would quietly sit *behind*
   the real touch and the measurement would be meaningless. You need Alpaca's SIP data
   (the paid plan). Then pass `--feed sip`. The bot warns at startup if you don't.
2. **Day-trading rules.** Every round trip is a day trade. Accounts under $25k have been
   limited to 3 day trades per 5 days under FINRA's pattern-day-trader rule, and Alpaca
   enforces this by rejecting orders. Check the rule's current status for your account.
   The bot warns at startup and refuses to trade live while warnings are unresolved
   (`--accept-warnings` overrides).
3. **Environment.**
   ```
   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
   export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
   .venv/bin/python -m unittest tests.test_mm tests.test_alpaca   # 41 tests
   ```
4. **Flat account.** The bot refuses to start if the account already holds AMC.

## Step 1: one paper session (plumbing only)

```
.venv/bin/python run.py paper --profile pilot --feed sip
```

Let it run for an hour during market hours, then stop it with Ctrl-C (that cancels all
orders). Then check that `logs/` has a file and that `report.py` reads it.

**Don't calibrate from paper.** Alpaca's paper engine fills a limit order as soon as the
market touches its price, ignoring the queue of orders ahead of it. That overstates
participation even more than the backtest does.

## Step 2: live pilot

```
.venv/bin/python run.py paper --profile pilot --feed sip --live --i-understand-this-can-lose-money
```

Start it before 9:30 ET; it waits for the open, sits out the first 5 minutes, and flattens
10 minutes before the close. Run **at least 5 full days**. A day gives roughly 1,000–4,000
fills, but toxicity varies from day to day, so you want several days of them.

Watch for these in the console (they're also written to the log as `note` events):

- `KILL SWITCH`: the day's loss hit $25. It flattens and stops quoting for the day.
- `position mismatch`: the websocket dropped a fill. It adopts the broker's position and
  flattens. If this happens more than once, stop and investigate.
- `order rejected`: usually day-trading limits or a bad price. Repeated rejections mean stop.

## Step 3: grade it

Run the report within 30 days of the run (it needs Yahoo's minute volume, which only goes
back 30 days):

```
.venv/bin/python report.py logs/*.jsonl
```

It prints P&L, fills split by type (`join`: ordinary flow at the touch; `cleared`: the price
ran through our level; `taker`: our order crossed on arrival), markouts at 5/30/60s, and the
calibrated replay command:

```
calibrated replay: python stress.py bars data/amc_recent.csv --symbol AMC --fill-models toxic --participation 0.0040 --toxicity 0.62
```

The calibration inverts the replay's exact fill model. Tested on replays with known
settings, it recovered participation within about 10% and toxicity within about 0.03
(`CalibrationLoopTest`).

## Step 4: decide

Refresh the bars (`python scripts/fetch_yahoo.py AMC data/amc_recent.csv`) and run the
calibrated replay command. Then:

- **Calibrated replay positive, and live P&L in line with it:** the edge looks real at
  pilot size. Scale order size slowly. Participation won't scale linearly with size, so
  re-measure at each step.
- **Calibrated replay negative:** the edge doesn't survive real fills. Stop. The backtests
  were optimistic about participation or toxicity, which is exactly what the pilot was
  for.
- **Live P&L far from the calibrated replay:** the replay is missing something real, such
  as latency spikes, partial fills or queue dynamics. Compare markouts by fill type to find
  it before trusting either number.

Sensible stop rules regardless: a cumulative pilot loss over $100, measured toxicity above
0.75, or participation below 0.001.

## Known limits of the measurement

- The bot polls quotes about once a second. A level swept between two snapshots looks like
  an ordinary `join` fill, so live toxicity reads slightly high (the conservative
  direction).
- "Improving" participation (alone at a better price) is rarely measurable on AMC, because
  it trades a penny wide almost all day. The report says so rather than guessing.
- Regulatory fees are estimated at fill time. Alpaca books the actual fees separately.
