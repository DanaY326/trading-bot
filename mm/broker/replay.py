"""Replays a recorded/synthetic quote path, with order latency.

Fill models:
  * pessimistic: we only fill when the market trades through our price (every fill is
    adverse). A lower bound on how a strategy does.
  * probabilistic: also fills uninformed flow, A * exp(-k * distance_bps) per second,
    but only while our order is at or better than the market's best price. Behind the
    touch, incoming orders hit everyone ahead of us first, so we only fill when the
    price runs through our level, which is always an adverse fill.
  * With `volume` (real shares traded per minute), the friendly fill rate comes from
    that instead: we get `participation` of the shares that trade on our side while
    we're joining the touch (there's a queue ahead of us), or `improve_participation`
    while we're alone at a better price.
  * toxic: same average fill rate, but a fraction `toxicity` of the flow is informed:
    it only trades with you when the price is about to move against you over the next
    `toxic_horizon_s`. Replay knows the future path, which is what makes this possible.

Latency applies to both new orders and cancels, so a cancel sent as the price rips
through you can still get filled, which is exactly the risk in a squeeze.
"""
import bisect
import itertools
import math
import random
from collections import deque

from ..models import Book, Fill, Session, Side, regulatory_fee
from .base import Broker
from .sim import _Order


class ReplayBroker(Broker):
    def __init__(self, quotes: dict[str, list[Book]], open_ts: float, close_ts: float,
                 latency_s: float = 1.0, fill_model: str = "probabilistic",
                 fill_a: float = 0.3, fill_k: float = 0.08, toxicity: float = 0.5,
                 toxic_horizon_s: float = 30.0, volume: dict[str, list[tuple[float, float]]] | None = None,
                 participation: float = 0.01, improve_participation: float = 0.25,
                 seed: int | None = None):
        self.feeds = {s: sorted(b, key=lambda x: x.ts) for s, b in quotes.items()}
        self._feed_ts = {s: [b.ts for b in books] for s, books in self.feeds.items()}
        self.toxicity = toxicity if fill_model == "toxic" else 0.0
        self.toxic_horizon_s = toxic_horizon_s
        self.volume = volume or {}
        self._volume_ts = {s: [t for t, _ in v] for s, v in self.volume.items()}
        self.participation = participation
        self.improve_participation = improve_participation
        self.open_ts, self.close_ts = open_ts, close_ts
        self.latency_s = latency_s
        self.fill_model = fill_model
        self.fill_a, self.fill_k = fill_a, fill_k
        self.rng = random.Random(seed)
        self.t = open_ts
        self._idx = {s: 0 for s in self.feeds}
        self._current: dict[str, Book] = {}
        self._pending: deque[tuple[float, str, object]] = deque()  # (effective_t, kind, payload)
        self._live: dict[str, _Order] = {}
        self._fills: list[Fill] = []
        self._acks: list[tuple[str, float]] = []
        self._ids = itertools.count(1)
        self.order_count = 0
        for book in self._advance_feeds(open_ts):
            self._current[book.symbol] = book

    def now(self) -> float:
        return self.t

    def session(self) -> Session:
        is_open = self.open_ts <= self.t < self.close_ts
        return Session(is_open, self.t - self.open_ts, self.close_ts - self.t)

    def get_books(self, symbols) -> dict[str, Book]:
        return {s: self._current[s] for s in symbols if s in self._current}

    def place(self, symbol, side, qty, px) -> str:
        oid = f"{int(self.open_ts)}-{next(self._ids)}"  # unique across days in one log
        self.order_count += 1
        self._pending.append((self.t + self.latency_s, "place", _Order(oid, symbol, side, qty, px)))
        return oid

    def cancel(self, order_id) -> None:
        self._pending.append((self.t + self.latency_s, "cancel", order_id))

    def poll(self):
        fills, self._fills = self._fills, []
        pending = {p.id for _, kind, p in self._pending if kind == "place"}
        return fills, set(self._live) | pending

    def poll_acks(self) -> list[tuple[str, float, str]]:
        acks, self._acks = self._acks, []
        return acks

    def advance(self, dt: float = 1.0) -> None:
        end = self.t + dt
        if self.fill_model in ("probabilistic", "toxic"):
            self._friendly_fills(dt)
        for book in self._advance_feeds(end):
            # Ties go to the market: an order or cancel landing in the same instant as
            # a price move is too late to dodge it.
            self._apply_pending(book.ts, inclusive=False)
            prev = self._current.get(book.symbol)
            self._current[book.symbol] = book
            for o in [o for o in self._live.values() if o.symbol == book.symbol]:
                self._match_through(o, prev, book)
        self._apply_pending(end)
        self.t = end

    def _friendly_fills(self, dt: float) -> None:
        """Uninformed flow over the coming step, judged against the book in force now.

        Only orders already resting count; ones that arrive during the step start
        collecting flow on the next one. (Judging against the end-of-step book instead
        would credit brief states the order never actually sat in.)
        """
        now = self.t
        fill_t = now + dt - 1e-3  # the flow arrives during the step, before the next quote lands
        for o in list(self._live.values()):
            book = self._current[o.symbol]
            if now - book.ts > 5:  # halted: nobody is trading
                continue
            if (o.px < book.bid) if o.side is Side.BUY else (o.px > book.ask):
                continue  # behind the touch: flow hits the orders ahead of us
            if o.symbol in self.volume:
                improving = o.px > book.bid if o.side is Side.BUY else o.px < book.ask
                share = self.improve_participation if improving else self.participation
                # Half the minute's volume trades against each side of the book.
                rate = share * self._minute_volume(o.symbol, now) / 120 / o.qty
            else:
                dist_bps = abs(book.mid - o.px) / book.mid * 1e4
                rate = self.fill_a * math.exp(-self.fill_k * dist_bps)
            if self.toxicity:
                # Informed flow at double rate when adverse, none otherwise: same
                # average rate as `probabilistic` if moves are a coin flip.
                rate *= (1 - self.toxicity) + self.toxicity * 2 * self._adverse_ahead(o, fill_t)
            if self.rng.random() < rate * dt:
                self._fill(o, o.px, fill_t)

    def _minute_volume(self, symbol: str, now: float) -> float:
        i = bisect.bisect_right(self._volume_ts[symbol], now) - 1
        if i < 0 or now - self._volume_ts[symbol][i] >= 60:
            return 0.0
        return self.volume[symbol][i][1]

    def _adverse_ahead(self, o: _Order, now: float) -> bool:
        ts = self._feed_ts[o.symbol]
        i = bisect.bisect_right(ts, now + self.toxic_horizon_s) - 1
        future_mid = self.feeds[o.symbol][i].mid
        return future_mid < o.px if o.side is Side.BUY else future_mid > o.px

    def _advance_feeds(self, until: float) -> list[Book]:
        out = []
        for s, books in self.feeds.items():
            i = self._idx[s]
            while i < len(books) and books[i].ts <= until:
                out.append(books[i])
                i += 1
            self._idx[s] = i
        return sorted(out, key=lambda b: b.ts)

    def _apply_pending(self, until: float, inclusive: bool = True) -> None:
        while self._pending and (self._pending[0][0] < until or inclusive and self._pending[0][0] == until):
            t, kind, payload = self._pending.popleft()
            if kind == "cancel":
                if self._live.pop(payload, None):
                    self._acks.append((payload, t, "done"))
                continue
            self._live[payload.id] = payload
            self._acks.append((payload.id, t, "live"))
            book = self._current.get(payload.symbol)
            # Marketable on arrival: take liquidity at the touch.
            if book and payload.side is Side.BUY and payload.px >= book.ask:
                self._fill(payload, book.ask, t)
            elif book and payload.side is Side.SELL and payload.px <= book.bid:
                self._fill(payload, book.bid, t)

    def _match_through(self, o: _Order, prev: Book | None, book: Book) -> None:
        """Fill when the market moves through our price.

        Either the other side reaches us (ask falls to our bid), or the level we were
        resting at, at or behind the touch, gets cleared out (bid drops below it). If
        we were improving the touch we were the best bid, so a falling market bid
        tells us nothing; only the first condition applies.
        """
        if o.side is Side.BUY:
            cleared = prev is not None and prev.bid >= o.px > book.bid
            hit = book.ask <= o.px or cleared
        else:
            cleared = prev is not None and prev.ask <= o.px < book.ask
            hit = book.bid >= o.px or cleared
        if hit:
            self._fill(o, o.px, book.ts)

    def _fill(self, o: _Order, px: float, ts: float) -> None:
        self._fills.append(Fill(o.id, o.symbol, o.side, o.qty, px, ts, regulatory_fee(o.side, o.qty, px)))
        self._live.pop(o.id, None)
