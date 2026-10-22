"""Toy meme-stock market for offline testing.

Price follows GBM plus Poisson jumps (squeezes/dumps). Resting orders fill two ways:
  * uninformed flow: probability A * exp(-k * distance_bps) per second
  * adverse selection: the market moves through our price and we get run over
The second is what kills market makers in meme names, so the sim keeps it.
"""
import itertools
import math
import random
from dataclasses import dataclass

from ..models import Book, Fill, Session, Side, regulatory_fee, round_to_tick, tick_size
from .base import Broker


@dataclass
class SimSymbol:
    mid: float
    sigma_per_sec: float = 0.0006   # ~38% intraday vol per sqrt(6.5h); meme-ish
    jump_rate_per_sec: float = 1 / 1800
    jump_sigma: float = 0.03
    half_spread_bps: float = 8.0


@dataclass
class _Order:
    id: str
    symbol: str
    side: Side
    qty: int
    px: float


class SimBroker(Broker):
    def __init__(self, symbols: dict[str, SimSymbol], session_len_s: float = 6.5 * 3600,
                 fill_a: float = 0.3, fill_k: float = 0.08, seed: int | None = None):
        self.symbols = symbols
        self.session_len_s = session_len_s
        self.fill_a = fill_a
        self.fill_k = fill_k
        self.rng = random.Random(seed)
        self.t = 0.0
        self._ids = itertools.count(1)
        self._orders: dict[str, _Order] = {}
        self._fills: list[Fill] = []
        self.order_count = 0

    def now(self) -> float:
        return self.t

    def session(self) -> Session:
        return Session(self.t < self.session_len_s, self.t, self.session_len_s - self.t)

    def _book(self, sym: str) -> Book:
        s = self.symbols[sym]
        tick = tick_size(s.mid)
        half = max(tick, s.mid * s.half_spread_bps / 1e4)
        bid = round_to_tick(s.mid - half, tick, "down")
        ask = round_to_tick(s.mid + half, tick, "up")
        return Book(sym, bid, ask, 100, 100, self.t)

    def get_books(self, symbols) -> dict[str, Book]:
        return {s: self._book(s) for s in symbols}

    def place(self, symbol, side, qty, px) -> str:
        oid = str(next(self._ids))
        self.order_count += 1
        self._orders[oid] = _Order(oid, symbol, side, qty, px)
        self._match_marketable(self._orders[oid])
        return oid

    def cancel(self, order_id) -> None:
        self._orders.pop(order_id, None)

    def poll(self):
        fills, self._fills = self._fills, []
        return fills, set(self._orders)

    def _fill(self, o: _Order, px: float) -> None:
        self._fills.append(Fill(o.id, o.symbol, o.side, o.qty, px, self.t,
                                regulatory_fee(o.side, o.qty, px)))
        del self._orders[o.id]

    def _match_marketable(self, o: _Order) -> None:
        book = self._book(o.symbol)
        if o.side is Side.BUY and o.px >= book.ask:
            self._fill(o, book.ask)
        elif o.side is Side.SELL and o.px <= book.bid:
            self._fill(o, book.bid)

    def advance(self, dt: float = 1.0) -> None:
        self.t += dt
        for s in self.symbols.values():
            z = self.rng.gauss(0, 1)
            r = s.sigma_per_sec * math.sqrt(dt) * z - 0.5 * s.sigma_per_sec ** 2 * dt
            if self.rng.random() < s.jump_rate_per_sec * dt:
                r += self.rng.gauss(0, s.jump_sigma)
            s.mid *= math.exp(r)

        for o in list(self._orders.values()):
            book = self._book(o.symbol)
            mid = book.mid
            # Adverse: market traded through us.
            if (o.side is Side.BUY and book.ask <= o.px) or (o.side is Side.SELL and book.bid >= o.px):
                self._fill(o, o.px)
                continue
            dist_bps = abs(mid - o.px) / mid * 1e4
            if self.rng.random() < self.fill_a * math.exp(-self.fill_k * dist_bps) * dt:
                self._fill(o, o.px)
