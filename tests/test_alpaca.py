"""AlpacaBroker against the real alpaca-py models, with the network clients faked out.

Skipped when alpaca-py isn't installed (`pip install -r requirements.txt`).
"""
import asyncio
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import mock

try:
    from alpaca.trading.models import TradeUpdate
    import mm.broker.alpaca as A
except ImportError:  # pragma: no cover
    A = None

from mm.models import Side


class FakeTrading:
    def __init__(self, *args, **kwargs):
        self.submitted, self.cancelled, self.cancel_all_calls = [], [], 0
        self.equity = "100000"

    def submit_order(self, req):
        self.submitted.append(req)
        return SimpleNamespace(id=uuid.uuid4())

    def cancel_order_by_id(self, oid):
        self.cancelled.append(oid)

    def cancel_orders(self):
        self.cancel_all_calls += 1

    def get_account(self):
        return SimpleNamespace(trading_blocked=False, account_blocked=False, equity=self.equity,
                               daytrade_count=1)

    def get_all_positions(self):
        return [SimpleNamespace(symbol="AMC", qty="10")]

    def get_clock(self):
        return SimpleNamespace(is_open=True,
                               next_close=datetime.now(timezone.utc) + timedelta(hours=1))

    def get_calendar(self, req):
        d = datetime(2026, 10, 22)
        return [SimpleNamespace(open=d.replace(hour=9, minute=30), close=d.replace(hour=16))]


class FakeStream:
    def __init__(self, *args, **kwargs):
        self.stopped = False

    def subscribe_trade_updates(self, handler):
        self.handler = handler

    def run(self):
        pass

    def stop(self):
        self.stopped = True


def update(event: str, oid: str, side: str = "buy", qty: str | None = None, price: str | None = None) -> "TradeUpdate":
    now = "2026-10-22T13:35:01.456789Z"
    order = {"id": oid, "client_order_id": str(uuid.uuid4()), "created_at": now, "updated_at": now,
             "submitted_at": now, "symbol": "AMC", "asset_class": "us_equity", "qty": "10",
             "order_class": "", "order_type": "limit", "type": "limit", "side": side,
             "time_in_force": "day", "limit_price": "3.01", "extended_hours": False,
             "status": {"fill": "filled", "partial_fill": "partially_filled"}.get(event, event)}
    return TradeUpdate.model_validate({"event": event, "order": order, "timestamp": now,
                                       "qty": qty, "price": price})


@unittest.skipIf(A is None, "alpaca-py not installed")
class AlpacaBrokerTest(unittest.TestCase):
    def setUp(self):
        patches = [mock.patch.object(A, "TradingClient", FakeTrading),
                   mock.patch.object(A, "StockHistoricalDataClient", mock.MagicMock()),
                   mock.patch.object(A, "TradingStream", FakeStream)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.broker = A.AlpacaBroker("key", "secret", paper=True)
        self.broker._open.clear()

    def push(self, u):
        asyncio.run(self.broker._on_trade_update(u))

    def test_starts_by_cancelling_leftovers(self):
        self.assertEqual(self.broker.trading.cancel_all_calls, 1)

    def test_prices_rounded_to_valid_increments(self):
        self.broker.place("AMC", Side.BUY, 10, 3.0100000000000002)
        self.broker.place("AMC", Side.BUY, 10, 0.51234)
        self.assertEqual([r.limit_price for r in self.broker.trading.submitted], [3.01, 0.5123])

    def test_trade_updates_become_fills_acks_and_open_orders(self):
        a = self.broker.place("AMC", Side.BUY, 10, 3.01)
        b = self.broker.place("AMC", Side.SELL, 10, 3.02)
        self.push(update("new", a))
        self.push(update("partial_fill", a, qty="4", price="3.01"))
        self.push(update("fill", a, qty="6", price="3.01"))
        self.push(update("new", b, side="sell"))
        fills, open_ids = self.broker.poll()
        self.assertEqual([(f.side, f.qty, f.px) for f in fills], [(Side.BUY, 4, 3.01), (Side.BUY, 6, 3.01)])
        self.assertEqual(open_ids, {b})
        self.push(update("canceled", b, side="sell"))
        fills, open_ids = self.broker.poll()
        self.assertEqual((fills, open_ids), ([], set()))
        self.assertEqual([(oid, state) for oid, _, state in self.broker.poll_acks()],
                         [(a, "live"), (b, "live"), (b, "done")])

    def test_sell_fills_carry_fees(self):
        a = self.broker.place("AMC", Side.SELL, 10, 3.01)
        self.push(update("fill", a, side="sell", qty="10", price="3.01"))
        (f,), _ = self.broker.poll()
        self.assertGreater(f.fee, 0)

    def test_fill_before_rest_response_is_not_left_open(self):
        oid = str(uuid.uuid4())
        self.push(update("fill", oid, qty="10", price="3.01"))
        self.broker.poll()
        with mock.patch.object(self.broker.trading, "submit_order", return_value=SimpleNamespace(id=oid)):
            self.broker.place("AMC", Side.BUY, 10, 3.01)
        self.assertNotIn(oid, self.broker.poll()[1])

    def test_session_from_clock_and_calendar(self):
        s = self.broker.session()
        self.assertTrue(s.is_open)
        self.assertAlmostEqual(s.secs_to_close, 3600, delta=5)
        self.assertAlmostEqual(s.secs_since_open, 5.5 * 3600, delta=5)

    def test_account_warnings(self):
        self.broker.trading.equity = "1000"
        warnings = self.broker.account_warnings(touch_mode=True)
        self.assertTrue(any("25k" in w for w in warnings))
        self.assertTrue(any("IEX" in w for w in warnings))
        self.broker.trading.equity = "100000"
        self.broker.feed = A.DataFeed.SIP
        self.assertEqual(self.broker.account_warnings(touch_mode=True), [])

    def test_close_stops_stream_and_cancels(self):
        self.broker.close()
        self.assertTrue(self.broker.stream.stopped)
        self.assertEqual(self.broker.trading.cancel_all_calls, 2)

    def test_positions(self):
        self.assertEqual(self.broker.positions(), {"AMC": 10})


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(A is None, "alpaca-py not installed")
class PaperSessionSmokeTest(unittest.TestCase):
    """The live code path end to end: run.py's paper command, real AlpacaBroker,
    Engine, Recorder, then report.py on the log. Only the network is faked."""

    def test_pilot_session_quotes_at_touch_and_logs(self):
        import os, sys, tempfile
        from alpaca.data.models import Quote
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import report
        import run

        t0 = time.time()
        clock = {"t": t0}

        def latest_quote(req):
            clock["t"] += 1.0
            bid = 3.00 + 0.01 * (int(clock["t"] - t0) // 40 % 2)  # drift a tick now and then
            ts = datetime.fromtimestamp(clock["t"], timezone.utc)
            return {"AMC": Quote("AMC", {"t": ts.isoformat(), "bp": bid, "bs": 50, "bx": "Q",
                                          "ap": bid + 0.01, "as": 50, "ax": "Q", "c": ["R"], "z": "A"})}

        data = mock.MagicMock()
        data.return_value.get_stock_latest_quote.side_effect = latest_quote
        steps = {"n": 0}

        def stop_after(self):  # Engine.run, but bounded and without sleeping
            while steps["n"] < 120:
                steps["n"] += 1
                self.step()

        class FlatAccount(FakeTrading):
            def get_all_positions(self):
                return []

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(A, "TradingClient", FlatAccount), \
                mock.patch.object(A, "StockHistoricalDataClient", data), \
                mock.patch.object(A, "TradingStream", FakeStream), \
                mock.patch.object(A.time, "time", lambda: clock["t"]), \
                mock.patch("mm.engine.Engine.run", stop_after), \
                mock.patch.dict(os.environ, {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"}):
            args = SimpleNamespace(live=False, i_understand_this_can_lose_money=False, feed="sip",
                                   profile="pilot", log_dir=tmp, accept_warnings=False)
            run.cmd_paper(args)
            (log,) = [os.path.join(tmp, f) for f in os.listdir(tmp)]
            events = report.load_events([log])
            out = report.analyze(events, volume_csv=os.devnull)

        orders = [e for e in events if e["k"] == "order"]
        self.assertTrue(orders, "pilot never quoted")
        self.assertTrue(all(o["px"] == o["bid"] for o in orders if o["side"] == "buy"))  # joined the touch
        self.assertTrue(all(o["qty"] == 10 for o in orders))
        self.assertIn("AMC", out["symbols"])
