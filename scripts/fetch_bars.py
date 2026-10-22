"""Download raw (unadjusted) 1-minute bars from Alpaca into the CSV format stress.py reads.

    APCA_API_KEY_ID=... APCA_API_SECRET_KEY=... \
        python scripts/fetch_bars.py GME 2021-01-19 2021-02-05 data/gme_2021_01.csv

Raw prices matter: GME split 4:1 in 2022, and split-adjusted prices would put the
squeeze at ~$120 instead of $483, which changes tick sizes and position caps.
"""
import csv
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


def main() -> None:
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    symbol, start, end, out = sys.argv[1:]
    ny = ZoneInfo("America/New_York")
    client = StockHistoricalDataClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"])
    bars = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=datetime.fromisoformat(start).replace(tzinfo=ny),
        end=datetime.fromisoformat(end).replace(tzinfo=ny) + timedelta(days=1),
        adjustment=Adjustment.RAW,
        feed=DataFeed.SIP,
    )).data.get(symbol, [])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            w.writerow([symbol, int(b.timestamp.timestamp()), b.open, b.high, b.low, b.close, b.volume])
    print(f"wrote {len(bars)} bars to {out}")


if __name__ == "__main__":
    main()
