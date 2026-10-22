"""Databento MBO (market-by-order) data: loading, session times, and a synthetic generator.

Real files come from scripts/fetch_mbo.py, one symbol-day per file. The synthetic
generator produces the same event format with price-time matching, so the order-book
replay can be exercised end to end before buying any data.
"""
import random
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from typing import Iterator

from .broker.orderbook import ADD, ASK, BID, CANCEL, FILL, PX, TRADE, Event
from .scenarios import NY


@dataclass
class MboDay:
    symbol: str
    day: date
    open_ts: float
    close_ts: float
    events: Iterator[Event]


def session_bounds(day: date) -> tuple[float, float]:
    # Regular hours; half days (e.g. day after Thanksgiving) close at 13:00 and aren't handled.
    return (datetime.combine(day, dtime(9, 30), NY).timestamp(),
            datetime.combine(day, dtime(16, 0), NY).timestamp())


def load_dbn(path: str, chunk: int = 500_000) -> MboDay:
    """Open a one-symbol, one-day MBO file (.dbn or .dbn.zst) as a lazy event stream."""
    import databento as db

    store = db.DBNStore.from_file(path)
    if str(store.schema) != "mbo":
        raise ValueError(f"{path}: expected schema mbo, got {store.schema}")
    symbols = store.symbols or ["?"]
    if len(symbols) != 1:
        raise ValueError(f"{path}: expected one symbol, got {symbols}")
    day = datetime.fromtimestamp(store.start / 1e9, NY).date()

    def events() -> Iterator[Event]:
        for arr in store.to_ndarray(count=chunk):
            yield from zip(*(arr[k].tolist() for k in ("ts_recv", "action", "side", "price", "size", "order_id")))

    return MboDay(symbols[0], day, *session_bounds(day), events())


def synthetic_day(day: date, symbol: str = "SYN", mid: float = 3.00, seconds: float = 6.5 * 3600,
                  seed: int = 0, tick: float = 0.01, start_before_open_s: float = 600) -> MboDay:
    """A plausible penny-wide book: limit adds and cancels near the touch, and market
    orders that consume the queue first-in-first-out, emitting T + F + C like a real feed.
    The price only moves when a level is exhausted, as on a real tick-constrained stock."""
    open_ts, close_ts = session_bounds(day)
    rng = random.Random(seed)
    t0 = open_ts - start_before_open_s
    end = min(close_ts, open_ts + seconds)

    def gen() -> Iterator[Event]:
        ids = iter(range(1, 10**12))
        levels: dict[bytes, dict[int, dict[int, int]]] = {BID: {}, ASK: {}}
        best_bid = round(mid / tick) - 0  # in ticks
        t = t0

        def ns(x: float) -> int:
            return int(x * 1e9)

        def px(ticks: int) -> int:
            return round(ticks * tick * PX)

        def add(side: bytes, ticks: int, size: int):
            oid = next(ids)
            levels[side].setdefault(ticks, {})[oid] = size
            return (ns(t), ADD, side, px(ticks), size, oid)

        # seed both sides a few levels deep
        for k in range(5):
            for _ in range(rng.randint(3, 8)):
                yield add(BID, best_bid - k, rng.choice((100, 100, 200, 300, 500)))
                yield add(ASK, best_bid + 1 + k, rng.choice((100, 100, 200, 300, 500)))
        bb, ba = best_bid, best_bid + 1
        while t < end:
            t += rng.expovariate(40.0)  # ~40 book events a second
            u = rng.random()
            # replenish an emptied side so the book never goes one-sided
            for side in (BID, ASK):
                if not levels[side]:
                    other = levels[ASK if side == BID else BID]
                    if side == BID:
                        ref = min(other) - 1 if other else bb
                    else:
                        ref = max(other) + 1 if other else ba
                    yield add(side, ref, 500)
            bb, ba = max(levels[BID]), min(levels[ASK])
            if u < 0.48:  # add near the touch
                side = BID if rng.random() < 0.5 else ASK
                ticks = (bb - rng.choice((0, 0, 0, 1, 2))) if side == BID else (ba + rng.choice((0, 0, 0, 1, 2)))
                if side == BID and ticks >= ba:
                    ticks = ba - 1
                if side == ASK and ticks <= bb:
                    ticks = bb + 1
                yield add(side, ticks, rng.choice((100, 100, 100, 200, 300, 500, 1000)))
            elif u < 0.93:  # cancel a random resting order near the touch
                side = BID if rng.random() < 0.5 else ASK
                lvl_ticks = bb - rng.choice((0, 0, 1, 2)) if side == BID else ba + rng.choice((0, 0, 1, 2))
                q = levels[side].get(lvl_ticks)
                if q:
                    oid = rng.choice(list(q))
                    size = q.pop(oid)
                    if not q:
                        del levels[side][lvl_ticks]
                    yield (ns(t), CANCEL, side, px(lvl_ticks), size, oid)
            else:  # marketable order sweeping the opposite side FIFO
                aggressor = BID if rng.random() < 0.5 else ASK
                resting = ASK if aggressor == BID else BID
                want = rng.choice((100, 100, 200, 300, 500, 1000, 2000))
                while want > 0 and levels[resting]:
                    lvl = min(levels[resting]) if resting == ASK else max(levels[resting])
                    q = levels[resting][lvl]
                    oid, size = next(iter(q.items()))
                    take = min(size, want)
                    yield (ns(t), TRADE, aggressor, px(lvl), take, 0)
                    yield (ns(t), FILL, resting, px(lvl), take, oid)
                    yield (ns(t), CANCEL, resting, px(lvl), take, oid)
                    want -= take
                    if take == size:
                        del q[oid]
                        if not q:
                            del levels[resting][lvl]
                    else:
                        q[oid] = size - take

    return MboDay(symbol, day, open_ts, close_ts, gen())


def minute_bars(trades: list[tuple[float, float, int]], symbol: str) -> list[tuple]:
    """(symbol, minute_start_ts, open, high, low, close, volume) rows from executions."""
    bars: dict[int, list] = {}
    for ts, px, size in trades:
        m = int(ts // 60 * 60)
        b = bars.get(m)
        if b is None:
            bars[m] = [px, px, px, px, size]
        else:
            b[1] = max(b[1], px)
            b[2] = min(b[2], px)
            b[3] = px
            b[4] += size
    return [(symbol, m, *b) for m, b in sorted(bars.items())]
