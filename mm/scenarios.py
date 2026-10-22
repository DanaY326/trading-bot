"""Stress scenarios: synthetic GME squeeze days, or replay of real minute bars.

Both produce a 1-second quote path. Between known prices (a day's open/high/low/close,
or each bar's OHLC) we draw a Brownian bridge, so the path hits the real extremes but
still has realistic second-to-second noise for the vol estimator and fill model.
Missing data (LULD halts) is left as a gap: no quotes, then a jump on reopen.
"""
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from .models import Book, round_to_tick, tick_size

NY = ZoneInfo("America/New_York")
SESSION_S = int(6.5 * 3600)
HALT_S = 300  # LULD halts are 5 minutes


@dataclass(frozen=True)
class DaySpec:
    day: str
    open: float
    high: float
    low: float
    close: float
    halts: int
    spread_bps: float
    first: str  # which extreme printed first: "high", "low", or "random" (chosen per seed)
    volume: float = 0.0  # shares traded that day, if known; spread evenly across minutes


# Pre-split (raw) prices, approximate and from memory. Treat these as a plausible
# shape of the squeeze, not ground truth; replay real bars for that.
GME_SQUEEZE_2021 = (
    DaySpec("2021-01-22", 42.59, 76.76, 42.32, 65.01, 4, 25, "high"),
    DaySpec("2021-01-25", 96.73, 159.18, 61.13, 76.79, 8, 40, "high"),
    DaySpec("2021-01-26", 88.56, 150.00, 80.20, 147.98, 4, 40, "low"),
    DaySpec("2021-01-27", 354.83, 380.00, 249.00, 347.51, 6, 80, "low"),
    DaySpec("2021-01-28", 265.00, 483.00, 112.25, 193.60, 10, 150, "high"),
    DaySpec("2021-01-29", 379.71, 413.98, 250.00, 325.00, 4, 80, "high"),
    DaySpec("2021-02-01", 316.56, 322.00, 212.00, 225.00, 2, 60, "high"),
    DaySpec("2021-02-02", 140.76, 158.00, 74.22, 90.00, 3, 60, "high"),
)


@dataclass
class Day:
    label: str
    symbol: str
    open_ts: float
    close_ts: float
    books: list[Book]
    volume: list[tuple[float, float]] | None = None  # (minute start ts, shares traded)


def _bridge(t0: int, p0: float, t1: int, p1: float, sigma: float, rng: random.Random,
            lo: float = 0.0, hi: float = math.inf) -> list[tuple[int, float]]:
    """1-second log-price Brownian bridge from (t0, p0) to (t1, p1), excluding t0."""
    n = t1 - t0
    if n <= 0:
        return []
    x0, x1 = math.log(p0), math.log(p1)
    xlo, xhi = math.log(lo) if lo > 0 else -math.inf, math.log(hi)
    w = [0.0]
    for _ in range(n):
        w.append(w[-1] + rng.gauss(0, sigma))
    out = []
    for i in range(1, n + 1):
        x = x0 + (x1 - x0) * i / n + w[i] - w[n] * i / n
        # Reflect off the bounds rather than clamping: clamping pins the price flat at
        # the high/low for long stretches, which looks calm and lures the bot in.
        if x > xhi:
            x = 2 * xhi - x
        elif x < xlo:
            x = 2 * xlo - x
        out.append((t0 + i, math.exp(min(xhi, max(xlo, x)))))
    return out


def _books(symbol: str, path: list[tuple[int, float]], spread_bps: float,
           rng: random.Random) -> list[Book]:
    books, prev_t, reopen_t = [], None, -math.inf
    for t, px in path:
        if prev_t is not None and t - prev_t > 30:
            reopen_t = t  # spreads blow out right after a halt
        prev_t = t
        widen = 3.0 if t - reopen_t < 60 else 1.0
        tick = tick_size(px)
        # Spread in whole ticks, at least one: cheap stocks like AMC really do trade a
        # penny wide, and rounding each side separately would make that three.
        ticks = max(1, round(px * spread_bps / 1e4 * math.exp(rng.gauss(0, 0.3)) * widen / tick))
        bid = round_to_tick(px - ticks * tick / 2, tick, "down")
        if bid > 0:
            books.append(Book(symbol, bid, round(bid + ticks * tick, 4), 100, 100, float(t)))
    return books


def _drop_halts(path: list[tuple[int, float]], n: int) -> list[tuple[int, float]]:
    """Halt where the path moved most over 5 minutes, which is what triggers LULD."""
    moves = sorted(
        ((abs(math.log(path[i][1] / path[i - HALT_S][1])), path[i][0]) for i in range(HALT_S, len(path))),
        reverse=True,
    )
    starts: list[int] = []
    for _, t in moves:
        if len(starts) == n:
            break
        if all(abs(t - s) > 2 * HALT_S for s in starts):
            starts.append(t)
    return [(t, p) for t, p in path if not any(s <= t < s + HALT_S for s in starts)]


def synthetic_day(spec: DaySpec, seed: int, symbol: str = "GME") -> Day:
    rng = random.Random(seed)
    t1 = int(rng.uniform(0.05, 0.45) * SESSION_S)
    t2 = int(rng.uniform(t1 / SESSION_S + 0.1, 0.95) * SESSION_S)
    first = rng.choice(("high", "low")) if spec.first == "random" else spec.first
    e1, e2 = (spec.high, spec.low) if first == "high" else (spec.low, spec.high)
    # Noise scaled to the day's range: wild days are wild second-to-second too.
    sigma = 0.25 * math.log(spec.high / spec.low) / math.sqrt(SESSION_S / 4)
    path = [(0, spec.open)]
    for (ta, pa), (tb, pb) in zip(
        [(0, spec.open), (t1, e1), (t2, e2)], [(t1, e1), (t2, e2), (SESSION_S, spec.close)]
    ):
        path += _bridge(ta, pa, tb, pb, sigma, rng, spec.low, spec.high)
    path = _drop_halts(path, spec.halts)
    volume = ([(float(t), spec.volume / (SESSION_S / 60)) for t in range(0, SESSION_S, 60)]
              if spec.volume else None)
    return Day(spec.day, symbol, 0.0, float(SESSION_S), _books(symbol, path, spec.spread_bps, rng),
               volume)


def load_bars_csv(path: str, symbol: str, spread_bps: float, seed: int = 0) -> list[Day]:
    """Minute bars (symbol,ts,open,high,low,close,volume; ts = bar start, epoch s) -> Days.

    Intrabar order follows the usual heuristic: up bars go O-L-H-C, down bars O-H-L-C.
    """
    rng = random.Random(seed)
    by_date: dict = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["symbol"] != symbol:
                continue
            ts = int(float(row["ts"]))
            local = datetime.fromtimestamp(ts, NY)
            if dtime(9, 30) <= local.time() < dtime(16, 0):
                by_date[local.date()].append(
                    (ts, *(float(row[k]) for k in ("open", "high", "low", "close", "volume")))
                )

    days = []
    for d, bars in sorted(by_date.items()):
        bars.sort()
        path: list[tuple[int, float]] = []
        prev = None  # (close_t, close_px, sigma) of the previous bar, if contiguous
        for ts, o, h, l, c, _ in bars:
            sigma = max(math.log(h / l), 1e-4) / math.sqrt(4 * math.log(2) * 60)  # Parkinson
            if prev and ts - prev[0] <= 15:
                path += _bridge(prev[0], prev[1], ts, o, prev[2], rng)
            else:
                path.append((ts, o))
            e1, e2 = (l, h) if c >= o else (h, l)
            for (ta, pa), (tb, pb) in (((ts, o), (ts + 15, e1)), ((ts + 15, e1), (ts + 30, e2)),
                                       ((ts + 30, e2), (ts + 45, c))):
                path += _bridge(ta, pa, tb, pb, sigma, rng, l, h)
            prev = (ts + 45, c, sigma)
        open_ts = datetime.combine(d, dtime(9, 30), NY).timestamp()
        close_ts = datetime.combine(d, dtime(16, 0), NY).timestamp()
        days.append(Day(d.isoformat(), symbol, open_ts, close_ts, _books(symbol, path, spread_bps, rng),
                        [(float(b[0]), b[5]) for b in bars]))
    return days
