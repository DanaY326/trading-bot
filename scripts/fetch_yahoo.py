"""Download the last ~30 days of 1-minute bars from Yahoo into the CSV format stress.py reads.

    python scripts/fetch_yahoo.py GME data/gme_recent.csv
    python stress.py bars data/gme_recent.csv --spread-bps 20

Yahoo only serves minute data for the last 30 days. For the 2021 squeeze at minute
resolution use scripts/fetch_bars.py (Alpaca, free account); for daily-level 2021 data
use `python stress.py yahoo`.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mm.yahoo import minute_bars  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    symbol, out = sys.argv[1:]
    bars = minute_bars(symbol)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "ts", "open", "high", "low", "close", "volume"])
        w.writerows((symbol, *bar) for bar in bars)
    print(f"wrote {len(bars)} bars to {out}")


if __name__ == "__main__":
    main()
