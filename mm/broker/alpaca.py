"""Alpaca adapter (paper by default). Requires `pip install alpaca-py`.

Quotes are polled over REST; fills arrive on the trade-updates websocket, which runs
in a background thread and hands events to the engine through a queue.
"""
import logging
import queue
import threading
import time
from datetime import date

from alpaca.common.exceptions import APIError
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import GetCalendarRequest, LimitOrderRequest
from alpaca.trading.stream import TradingStream

from ..models import Book, Fill, Session, Side, regulatory_fee
from .base import Broker

log = logging.getLogger(__name__)

_TERMINAL = {"fill", "canceled", "expired", "rejected", "replaced"}


class AlpacaBroker(Broker):
    def __init__(self, key: str, secret: str, paper: bool = True, feed: str = "iex"):
        self.trading = TradingClient(key, secret, paper=paper)
        self.data = StockHistoricalDataClient(key, secret)
        self.feed = DataFeed(feed)
        self._events: queue.Queue = queue.Queue()
        self._acks: list[tuple[str, float, str]] = []
        self._open: set[str] = set()
        self._closed: set[str] = set()
        self._lock = threading.Lock()
        self._session_cache: tuple[float, float, float] | None = None  # (fetched_at, open_ts, close_ts)

        self.stream = TradingStream(key, secret, paper=paper)
        self.stream.subscribe_trade_updates(self._on_trade_update)
        threading.Thread(target=self.stream.run, daemon=True, name="alpaca-trade-stream").start()

        # Start clean: anything left over from a previous run is not ours to manage.
        self.trading.cancel_orders()

    async def _on_trade_update(self, update) -> None:
        self._events.put(update)

    def now(self) -> float:
        return time.time()

    def session(self) -> Session:
        now = time.time()
        if self._session_cache is None or now - self._session_cache[0] > 60:
            clock = self.trading.get_clock()
            cal = self.trading.get_calendar(GetCalendarRequest(start=date.today(), end=date.today()))
            if not clock.is_open or not cal:
                self._session_cache = (now, 0.0, 0.0)
            else:
                # calendar open/close are naive exchange-local; clock.next_close is tz-aware
                close_ts = clock.next_close.timestamp()
                open_ts = close_ts - (cal[0].close - cal[0].open).total_seconds()
                self._session_cache = (now, open_ts, close_ts)
        _, open_ts, close_ts = self._session_cache
        is_open = open_ts <= now < close_ts
        return Session(is_open, now - open_ts if is_open else 0.0, close_ts - now if is_open else 0.0)

    def get_books(self, symbols) -> dict[str, Book]:
        quotes = self.data.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=self.feed)
        )
        return {
            sym: Book(sym, float(q.bid_price), float(q.ask_price), float(q.bid_size),
                      float(q.ask_size), q.timestamp.timestamp())
            for sym, q in quotes.items()
        }

    def place(self, symbol, side, qty, px) -> str | None:
        # alpaca-py doesn't validate price increments; the server rejects sub-penny
        # prices at or above $1, and float noise like 3.0100000000000002 would trip it.
        px = round(px, 2 if px >= 1.0 else 4)
        try:
            order = self.trading.submit_order(LimitOrderRequest(
                symbol=symbol, qty=qty, limit_price=px, time_in_force=TimeInForce.DAY,
                side=OrderSide.BUY if side is Side.BUY else OrderSide.SELL,
            ))
        except APIError as e:
            log.warning("order rejected %s %s %d@%.4f: %s", symbol, side.value, qty, px, e)
            return None
        oid = str(order.id)
        with self._lock:
            if oid not in self._closed:  # stream may have beaten the REST response
                self._open.add(oid)
        return oid

    def cancel(self, order_id) -> None:
        try:
            self.trading.cancel_order_by_id(order_id)
        except APIError as e:
            log.debug("cancel %s failed (likely already done): %s", order_id, e)

    def poll(self):
        fills: list[Fill] = []
        while True:
            try:
                u = self._events.get_nowait()
            except queue.Empty:
                break
            event = getattr(u.event, "value", u.event)
            oid = str(u.order.id)
            if event in ("fill", "partial_fill") and u.qty:
                side = Side.BUY if u.order.side == OrderSide.BUY else Side.SELL
                qty, px = int(float(u.qty)), float(u.price)
                # Alpaca books regulatory fees as separate account activity; estimate them
                # here so the daily loss limit sees them.
                fills.append(Fill(oid, u.order.symbol, side, qty, px, u.timestamp.timestamp(),
                                  regulatory_fee(side, qty, px)))
            if event == "new":
                self._acks.append((oid, u.timestamp.timestamp(), "live"))
            elif event in ("canceled", "expired", "rejected"):
                self._acks.append((oid, u.timestamp.timestamp(), "done"))
            if event in _TERMINAL:
                with self._lock:
                    self._open.discard(oid)
                    self._closed.add(oid)
        with self._lock:
            return fills, set(self._open)

    def poll_acks(self) -> list[tuple[str, float, str]]:
        acks, self._acks = self._acks, []
        return acks

    def positions(self) -> dict[str, int]:
        return {p.symbol: int(float(p.qty)) for p in self.trading.get_all_positions()}

    def account_warnings(self, touch_mode: bool) -> list[str]:
        """Things that will quietly break a market-making run; checked before starting."""
        acct = self.trading.get_account()
        warnings = []
        if acct.trading_blocked or acct.account_blocked:
            warnings.append("account is blocked from trading")
        if float(acct.equity or 0) < 25_000:
            warnings.append(
                f"equity ${float(acct.equity or 0):,.0f} is under $25k: if the pattern-day-trader "
                f"rule still applies to your account, Alpaca will block orders after "
                f"{3 - int(acct.daytrade_count or 0)} more day trade(s) in a rolling 5 days, "
                f"and every round trip here is a day trade")
        if touch_mode and self.feed == DataFeed.IEX:
            warnings.append(
                "touch mode on the IEX feed: IEX's best bid/offer is often worse than the real "
                "national one, so 'joining the touch' would really mean sitting behind it. "
                "Use --feed sip (paid Alpaca data plan) for touch mode")
        return warnings

    def close(self) -> None:
        self.trading.cancel_orders()
        try:
            self.stream.stop()  # hands the shutdown to the stream's own event loop
        except Exception:  # best effort on shutdown
            pass
