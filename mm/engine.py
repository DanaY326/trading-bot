import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field, replace

from .broker.base import Broker
from .config import Config
from .models import Book, Quote, Side, tick_size
from .risk import Portfolio, RateLimiter
from .signals import EwmaVol, JumpDetector
from .strategy import compute_quote, exit_quote, flatten_quote

log = logging.getLogger(__name__)


@dataclass
class WorkingOrder:
    id: str
    px: float
    remaining: int


@dataclass
class SymbolState:
    vol: EwmaVol
    jumps: JumpDetector
    last_quote_ts: float = -math.inf
    guarded_since: float | None = None
    working: dict[Side, WorkingOrder] = field(default_factory=dict)


class Engine:
    def __init__(self, broker: Broker, cfg: Config, recorder=None):
        self.broker = broker
        self.cfg = cfg
        self.recorder = recorder
        self._last_position_check = -math.inf
        self._suspect_positions: dict[str, int] = {}
        self._books: dict[str, Book] = {}
        self.portfolio = Portfolio(cfg.symbols)
        self.limiter = RateLimiter(cfg.risk.max_orders_per_min)
        g = cfg.guards
        self.state = {
            s: SymbolState(EwmaVol(cfg.strategy.ewma_halflife_s),
                           JumpDetector(g.jump_window_s, g.jump_threshold_bps, g.jump_cooldown_s))
            for s in cfg.symbols
        }
        self.killed = False
        self.marks: dict[str, float] = {}
        self.reasons: Counter[str] = Counter()  # why each symbol was / wasn't quoting, per step

    # ---- main loop -------------------------------------------------------

    def step(self) -> None:
        now = self.broker.now()
        self._sync_orders()

        session = self.broker.session()
        if not session.is_open:
            self.cancel_all()
            self.reasons["closed"] += 1
            return

        books = self.broker.get_books(self.cfg.symbols)
        self._books = books
        for sym, book in books.items():
            if book.valid:
                self.marks[sym] = book.mid
                if self.recorder:
                    self.recorder.book(book)
        self._check_positions(now)

        pnl = self.portfolio.pnl(self.marks)
        if not self.killed and pnl <= -self.cfg.risk.daily_loss_limit:
            log.error("KILL SWITCH: pnl %.2f breached limit -%.2f; flattening",
                      pnl, self.cfg.risk.daily_loss_limit)
            self.killed = True

        for sym in self.cfg.symbols:
            quote = self._target_quote(sym, books.get(sym), session, now)
            self.reasons[quote.reason or "quoting"] += 1
            self._reconcile(sym, quote, now)

    def _check_positions(self, now: float) -> None:
        positions = getattr(self.broker, "positions", None)
        if positions is None or now - self._last_position_check < self.cfg.reconcile_every_s:
            return
        self._last_position_check = now
        actual = positions()
        mismatched = {s: actual.get(s, 0) for s in self.cfg.symbols
                      if actual.get(s, 0) != self.portfolio.positions[s].qty}
        # Fills can be in flight between the REST snapshot and the websocket, so only
        # act when the same mismatch is still there on the next check.
        confirmed = {s: q for s, q in mismatched.items() if self._suspect_positions.get(s) == q}
        self._suspect_positions = mismatched
        if mismatched and not confirmed:
            self._last_position_check = now - self.cfg.reconcile_every_s + 5  # recheck soon
        for sym, qty in confirmed.items():
            msg = (f"position mismatch on {sym}: ours {self.portfolio.positions[sym].qty}, "
                   f"broker {qty}; adopting broker's and flattening")
            log.error(msg)
            if self.recorder:
                self.recorder.note(now, msg)
            self.portfolio.positions[sym].qty = qty
            self.killed = True

    def run(self) -> None:
        log.info("starting on %s", ", ".join(self.cfg.symbols))
        try:
            while True:
                t0 = time.monotonic()
                try:
                    self.step()
                except Exception:
                    log.exception("step failed; pulling quotes")
                    self.cancel_all()
                time.sleep(max(0.0, self.cfg.loop_interval_s - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            log.info("shutting down; cancelling all orders")
            self.cancel_all()

    # ---- decision ---------------------------------------------------------

    def _target_quote(self, sym: str, book: Book | None, session, now: float) -> Quote:
        g, st = self.cfg.guards, self.state[sym]
        if book is None or not book.valid:
            return Quote.none("no_book")

        if book.ts > st.last_quote_ts:
            st.vol.update(book.ts, book.mid)
            st.jumps.update(book.ts, book.mid)
            st.last_quote_ts = book.ts

        position = self.portfolio.positions[sym].qty
        if self.killed or session.secs_to_close <= g.flatten_before_close_s:
            q = flatten_quote(book, position, g.flatten_collar_bps)
            return q if position else Quote.none("killed" if self.killed else "eod")

        reason = self._guard_reason(st, book, session, now)
        if reason is None:
            st.guarded_since = None
            q = compute_quote(book, position, st.vol.sigma_per_sec, self.cfg.strategy)
            return self._cap_gross(q)

        if st.guarded_since is None:
            st.guarded_since = now
        # Can't trade into a halt; once it lifts, the guard clock has usually run out.
        if not position or not g.exit_on_guard or reason == "stale_or_halted":
            return Quote.none(reason)
        if now - st.guarded_since >= g.exit_after_s:
            return replace(flatten_quote(book, position, g.flatten_collar_bps), reason="exit_aggressive")
        return exit_quote(book, position)

    def _guard_reason(self, st: SymbolState, book: Book, session, now: float) -> str | None:
        g = self.cfg.guards
        if now - book.ts > g.stale_quote_s:
            return "stale_or_halted"
        if book.mid < g.min_price:
            return "below_min_price"
        if book.spread_bps > g.max_market_spread_bps:
            return "market_too_wide"
        if session.secs_since_open < g.skip_open_s:
            return "opening"
        if st.vol.samples < self.cfg.strategy.min_vol_samples:
            return "warming_up"
        if st.vol.sigma_per_sec * math.sqrt(self.cfg.strategy.horizon_s) * 1e4 > g.max_horizon_vol_bps:
            return "vol_too_high"
        if st.jumps.active(book.ts):
            return "jump_cooldown"
        return None

    def _cap_gross(self, q: Quote) -> Quote:
        if not q.bid_qty:
            return q
        headroom = self.cfg.risk.max_gross_notional - self.portfolio.gross_notional(self.marks)
        qty = max(0, min(q.bid_qty, int(headroom // q.bid_px)))
        return Quote(q.bid_px if qty else 0.0, qty, q.ask_px, q.ask_qty, q.reason)

    # ---- order management -------------------------------------------------

    def _sync_orders(self) -> None:
        fills, open_ids = self.broker.poll()
        if self.recorder:
            for oid, ts, state in self.broker.poll_acks():
                self.recorder.ack(ts, oid, state)
        for f in fills:
            self.portfolio.on_fill(f)
            if self.recorder:
                self.recorder.fill(f)
            log.info("FILL %s %s %d @ %.4f  pos=%d", f.symbol, f.side.value, f.qty, f.px,
                     self.portfolio.positions[f.symbol].qty)
            w = self.state[f.symbol].working.get(f.side)
            if w and w.id == f.order_id:
                w.remaining -= f.qty
        for st in self.state.values():
            for side, w in list(st.working.items()):
                if w.id not in open_ids or w.remaining <= 0:
                    del st.working[side]

    def _reconcile(self, sym: str, q: Quote, now: float) -> None:
        self._reconcile_side(sym, Side.BUY, q.bid_px, q.bid_qty, now, q.reason)
        self._reconcile_side(sym, Side.SELL, q.ask_px, q.ask_qty, now, q.reason)

    def _cancel(self, oid: str) -> None:
        self.broker.cancel(oid)
        if self.recorder:
            self.recorder.cancel(self.broker.now(), oid)

    def _reconcile_side(self, sym: str, side: Side, px: float, qty: int, now: float,
                        reason: str = "") -> None:
        working = self.state[sym].working
        w = working.get(side)
        if qty <= 0:
            if w:  # cancels always go through, regardless of rate limit
                self._cancel(w.id)
                del working[side]
            return
        threshold = self.cfg.strategy.requote_threshold_ticks * tick_size(px)
        if w and w.remaining == qty and abs(w.px - px) < threshold - 1e-9:
            return
        if not self.limiter.allow(now):
            return
        if w:
            self._cancel(w.id)
            del working[side]
        oid = self.broker.place(sym, side, qty, px)
        if oid:
            working[side] = WorkingOrder(oid, px, qty)
            if self.recorder:
                self.recorder.order(now, sym, oid, side, qty, px, self._books.get(sym), reason or "quote")

    def cancel_all(self) -> None:
        for st in self.state.values():
            for w in st.working.values():
                self._cancel(w.id)
            st.working.clear()
