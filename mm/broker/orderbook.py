"""Replay of one exchange's full order book (Databento MBO), with queue position.

Unlike ReplayBroker, nothing about fills is assumed. Our order joins the back of the
queue at its price, behind every displayed order already there, and fills only when:
  * an execution (F record) hits a resting order *behind* us at our price, after every
    order that was ahead of us has traded or cancelled (price-time priority); or
  * an execution happens at a price worse than ours: the market traded through us; or
  * it arrives marketable, in which case it takes the best opposing price.
Adverse selection therefore comes from what actually traded next, not a parameter.

MBO conventions (Databento): A/C/M/R change the book; T (aggressor) and F (each resting
order filled) don't, and a C follows every F to remove the filled shares. Events are
(ts_ns, action, side, price_1e9, size, order_id) with action/side as bytes, e.g. b"A".

Simplifications, all on the conservative side: we ignore hidden/midpoint liquidity
(which a displayed order would actually have priority over at the same price), assume
our order doesn't change anyone else's behaviour, and fill the full order on arrival
when it's marketable.
"""
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from ..models import Book, Fill, Session, Side, regulatory_fee
from .base import Broker

ADD, CANCEL, MODIFY, CLEAR, TRADE, FILL = b"A", b"C", b"M", b"R", b"T", b"F"
BID, ASK = b"B", b"A"
PX = 1_000_000_000  # Databento fixed-point price scale

Event = tuple[int, bytes, bytes, int, int, int]


@dataclass
class _Ours:
    id: str
    side: Side
    px: int            # fixed-point
    qty: int
    remaining: int
    ahead: dict[int, int] = field(default_factory=dict)  # order_id -> shares still ahead of us
    executed: dict[int, int] = field(default_factory=dict)  # F'd shares awaiting their C

    @property
    def book_side(self) -> bytes:
        return BID if self.side is Side.BUY else ASK

    def ahead_total(self) -> int:
        return sum(self.ahead.values())


class OrderBookBroker(Broker):
    def __init__(self, symbol: str, events: Iterable[Event], open_ts: float, close_ts: float,
                 latency_s: float = 1.0):
        self.symbol = symbol
        self.open_ts, self.close_ts = open_ts, close_ts
        self.latency_s = latency_s
        self._events: Iterator[Event] = iter(events)
        self._next: Event | None = next(self._events, None)
        self.orders: dict[int, list] = {}  # order_id -> [side, price, size]
        self.levels: dict[bytes, dict[int, dict[int, int]]] = {BID: defaultdict(dict), ASK: defaultdict(dict)}
        self.last_event_ts = 0.0
        self.t = open_ts
        self._pending: deque[tuple[float, str, object]] = deque()
        self._live: dict[str, _Ours] = {}
        self._fills: list[Fill] = []
        self._acks: list[tuple[str, float, str]] = []
        self._ids = itertools.count(1)
        self.order_count = 0
        self.queue_at_join: list[int] = []  # shares ahead of us when each order went live
        self.trades: list[tuple[float, float, int]] = []  # (ts, px, size) of every execution
        self._run_until(open_ts)  # build the book through the pre-market

    # ---- Broker interface ----------------------------------------------------

    def now(self) -> float:
        return self.t

    def session(self) -> Session:
        is_open = self.open_ts <= self.t < self.close_ts
        return Session(is_open, self.t - self.open_ts, self.close_ts - self.t)

    def get_books(self, symbols) -> dict[str, Book]:
        bid, ask = self.best(BID), self.best(ASK)
        if self.symbol not in symbols or bid is None or ask is None:
            return {}
        return {self.symbol: Book(self.symbol, bid / PX, ask / PX, self.level_size(BID, bid),
                                  self.level_size(ASK, ask), self.last_event_ts)}

    def place(self, symbol, side, qty, px) -> str:
        oid = f"{int(self.open_ts)}-{next(self._ids)}"
        self.order_count += 1
        self._pending.append((self.t + self.latency_s, "place",
                              _Ours(oid, side, round(px * PX), qty, qty)))
        return oid

    def cancel(self, order_id) -> None:
        self._pending.append((self.t + self.latency_s, "cancel", order_id))

    def poll(self):
        fills, self._fills = self._fills, []
        pending = {p.id for _, kind, p in self._pending if kind == "place"}
        return fills, set(self._live) | pending

    def poll_acks(self):
        acks, self._acks = self._acks, []
        return acks

    def advance(self, dt: float = 1.0) -> None:
        self._run_until(self.t + dt)
        self.t += dt

    # ---- book ------------------------------------------------------------------

    def best(self, side: bytes) -> int | None:
        prices = self.levels[side]
        if not prices:
            return None
        return max(prices) if side == BID else min(prices)

    def level_size(self, side: bytes, price: int) -> int:
        return sum(self.levels[side].get(price, {}).values())

    def _run_until(self, until: float) -> None:
        while self._next is not None and self._next[0] / 1e9 <= until:
            ts_ns, action, side, price, size, oid = self._next
            ts = ts_ns / 1e9
            self._apply_pending(ts, inclusive=False)  # ties go to the market
            self._on_event(ts, action, side, price, size, oid)
            self.last_event_ts = ts
            self._next = next(self._events, None)
        self._apply_pending(until, inclusive=True)

    def _on_event(self, ts: float, action: bytes, side: bytes, price: int, size: int, oid: int) -> None:
        if action == ADD:
            self.orders[oid] = [side, price, size]
            self.levels[side][price][oid] = size
        elif action == CANCEL:
            self._reduce(oid, size)
        elif action == MODIFY:
            self._modify(oid, side, price, size)
        elif action == CLEAR:
            self.orders.clear()
            self.levels = {BID: defaultdict(dict), ASK: defaultdict(dict)}
            for o in self._live.values():
                o.ahead.clear()
                o.executed.clear()
        elif action == FILL:
            self._on_execution(ts, side, price, size, oid)
        elif action == TRADE:
            # Venue volume and bars come from T records (one per aggressor print,
            # including hidden liquidity); F records are only used for our queue.
            self.trades.append((ts, price / PX, size))

    def _reduce(self, oid: int, size: int) -> None:
        o = self.orders.get(oid)
        if o is None:
            return
        side, price, have = o
        left = have - size
        level = self.levels[side][price]
        if left > 0:
            o[2] = left
            level[oid] = left
        else:
            del self.orders[oid]
            level.pop(oid, None)
            if not level:
                del self.levels[side][price]
        for ours in self._live.values():
            if oid in ours.ahead:
                # Shares already counted off by the F that preceded this C aren't removed twice.
                already = min(ours.executed.pop(oid, 0), size)
                ours.ahead[oid] -= size - already
                if ours.ahead[oid] <= 0 or left <= 0:
                    del ours.ahead[oid]

    def _modify(self, oid: int, side: bytes, price: int, size: int) -> None:
        old = self.orders.get(oid)
        if old is None:
            self._on_event(0.0, ADD, side, price, size, oid)
            return
        old_side, old_price, old_size = old
        same_level = side == old_side and price == old_price
        if same_level:
            self.levels[side][price][oid] = size
        else:
            level = self.levels[old_side][old_price]
            level.pop(oid, None)
            if not level:
                del self.levels[old_side][old_price]
            self.levels[side][price][oid] = size
        self.orders[oid] = [side, price, size]
        # Exchanges keep time priority only for a size reduction at the same price.
        keeps_priority = same_level and size <= old_size
        for ours in self._live.values():
            if oid in ours.ahead:
                if keeps_priority:
                    ours.ahead[oid] = min(ours.ahead[oid], size)
                else:
                    del ours.ahead[oid]
                    ours.executed.pop(oid, None)

    # ---- our fills -------------------------------------------------------------

    def _on_execution(self, ts: float, side: bytes, price: int, size: int, oid: int) -> None:
        for ours in list(self._live.values()):
            if ours.book_side != side:
                continue
            through = price < ours.px if ours.side is Side.BUY else price > ours.px
            if through:
                self._fill_ours(ours, min(ours.remaining, size), ts)
            elif price == ours.px:
                if oid in ours.ahead:
                    ours.ahead[oid] -= size
                    ours.executed[oid] = ours.executed.get(oid, 0) + size
                    if ours.ahead[oid] <= 0:
                        del ours.ahead[oid]
                        ours.executed.pop(oid, None)  # nothing left for its C to remove
                elif ours.ahead_total() <= 0:
                    # Everything ahead of us is gone, so this execution reaches us first.
                    self._fill_ours(ours, min(ours.remaining, size), ts)

    def _fill_ours(self, ours: _Ours, qty: int, ts: float, px: float | None = None) -> None:
        if qty <= 0:
            return
        fill_px = px if px is not None else ours.px / PX
        self._fills.append(Fill(ours.id, self.symbol, ours.side, qty, fill_px, ts,
                                regulatory_fee(ours.side, qty, fill_px)))
        ours.remaining -= qty
        if ours.remaining <= 0:
            self._live.pop(ours.id, None)

    def _apply_pending(self, until: float, inclusive: bool) -> None:
        while self._pending and (self._pending[0][0] < until or inclusive and self._pending[0][0] == until):
            t, kind, payload = self._pending.popleft()
            if kind == "cancel":
                if self._live.pop(payload, None):
                    self._acks.append((payload, t, "done"))
                continue
            ours: _Ours = payload
            opposite = self.best(ASK if ours.side is Side.BUY else BID)
            self._acks.append((ours.id, t, "live"))
            if opposite is not None and (ours.px >= opposite if ours.side is Side.BUY else ours.px <= opposite):
                self._live[ours.id] = ours
                self._fill_ours(ours, ours.remaining, t, opposite / PX)  # marketable: take it
                continue
            # Join the back of the queue: every displayed order at our price is ahead.
            ours.ahead = dict(self.levels[ours.book_side].get(ours.px, {}))
            self.queue_at_join.append(ours.ahead_total())
            self._live[ours.id] = ours

