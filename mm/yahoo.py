"""Yahoo Finance chart API: unofficial, no key, may change or rate-limit without notice.

Daily bars go back decades. 1-minute bars only cover the last 30 days, at most 8 days
per request. Prices come back split-adjusted (GME's 2022 4:1 split turns the $483 high
into $120.75), so we undo splits to get the prices that actually traded.
"""
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

from .scenarios import NY, DaySpec

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"


def _chart(symbol: str, start_ts: int, end_ts: int, interval: str) -> dict:
    query = urllib.parse.urlencode({"period1": start_ts, "period2": end_ts, "interval": interval,
                                    "events": "split", "includePrePost": "false"})
    req = urllib.request.Request(URL.format(symbol=symbol, query=query),
                                 headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            chart = json.load(resp)["chart"]
    except urllib.error.HTTPError as e:
        chart = json.load(e)["chart"]  # Yahoo explains bad requests in the body
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e.reason):
            raise RuntimeError("SSL certificates missing: run 'Install Certificates.command' in your "
                               "Python app folder, or set SSL_CERT_FILE=/etc/ssl/cert.pem") from e
        raise
    if chart["error"]:
        raise RuntimeError(f"yahoo: {chart['error'].get('description', chart['error'])}")
    return chart["result"][0]


def _splits_since(symbol: str, start_ts: int) -> list[tuple[int, float]]:
    # Splits after the requested window still adjust its prices, so ask up to today.
    result = _chart(symbol, start_ts, int(time.time()), "3mo")
    return [(s["date"], s["numerator"] / s["denominator"])
            for s in result.get("events", {}).get("splits", {}).values()]


def _bars(result: dict, splits: list[tuple[int, float]]) -> list[tuple[int, float, float, float, float, float]]:
    q = result["indicators"]["quote"][0]
    out = []
    for i, ts in enumerate(result.get("timestamp", [])):
        o, h, l, c, v = (q[k][i] for k in ("open", "high", "low", "close", "volume"))
        if None in (o, h, l, c):
            continue
        f = math.prod(ratio for split_ts, ratio in splits if split_ts > ts)
        out.append((ts, round(o * f, 4), round(h * f, 4), round(l * f, 4), round(c * f, 4), (v or 0) / f))
    return out


def daily_specs(symbol: str, start: date, end: date) -> list[DaySpec]:
    """Real daily OHLC turned into synthetic-day specs.

    Daily bars don't say how many halts there were, how wide the spread was, or whether
    the high came before the low, so those are estimated from the day's range.
    """
    start_ts = int(datetime.combine(start, datetime.min.time(), NY).timestamp())
    end_ts = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), NY).timestamp())
    splits = _splits_since(symbol, start_ts)
    specs = []
    for ts, o, h, l, c, v in _bars(_chart(symbol, start_ts, end_ts, "1d"), splits):
        log_range = math.log(h / l)
        specs.append(DaySpec(
            day=datetime.fromtimestamp(ts, NY).date().isoformat(),
            open=o, high=h, low=l, close=c,
            halts=min(12, int(log_range / 0.15)),  # rough: one LULD halt per ~15% of range
            spread_bps=min(150.0, max(10.0, 10 + 80 * log_range)),
            first="random",
            volume=v,
        ))
    return specs


def minute_bars(symbol: str, days: int = 30) -> list[tuple[int, float, float, float, float, float]]:
    """Regular-hours 1-minute bars for the last `days` days (Yahoo's limit is 30)."""
    now = int(time.time())
    start = now - min(days, 29) * 86400  # stay just inside the 30-day window
    splits = _splits_since(symbol, start)
    bars = []
    for chunk_start in range(start, now, 7 * 86400):
        result = _chart(symbol, chunk_start, min(chunk_start + 7 * 86400, now), "1m")
        bars += _bars(result, splits)
    return sorted(set(bars))
