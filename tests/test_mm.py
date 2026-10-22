import unittest

from mm.broker.sim import SimBroker, SimSymbol
from mm.config import Config, StrategyConfig
from mm.engine import Engine
from mm.models import Book, Fill, Side, round_to_tick
from mm.risk import Position
from mm.signals import JumpDetector
from mm.strategy import compute_quote, flatten_quote

BOOK = Book("GME", 24.99, 25.01, 100, 100, 0.0)
CFG = StrategyConfig()


class StrategyTest(unittest.TestCase):
    def test_flat_long_only_quotes_bid_only(self):
        q = compute_quote(BOOK, 0, 0.0005, CFG)
        self.assertGreater(q.bid_qty, 0)
        self.assertEqual(q.ask_qty, 0)
        self.assertLess(q.bid_px, BOOK.mid)

    def test_long_inventory_skews_quotes_down(self):
        flat = compute_quote(BOOK, 0, 0.0005, StrategyConfig(long_only=False))
        long = compute_quote(BOOK, 150, 0.0005, StrategyConfig(long_only=False))
        self.assertLess(long.bid_px, flat.bid_px)
        self.assertLess(long.ask_px, flat.ask_px)

    def test_no_bid_at_max_position(self):
        q = compute_quote(BOOK, CFG.max_position_shares, 0.0005, CFG)
        self.assertEqual(q.bid_qty, 0)
        self.assertGreater(q.ask_qty, 0)

    def test_never_crosses_market(self):
        q = compute_quote(BOOK, 199, 0.00001, CFG)  # extreme skew
        self.assertGreater(q.ask_px, BOOK.bid)
        q = compute_quote(BOOK, 0, 0.00001, StrategyConfig(long_only=False, inventory_skew=50))
        self.assertLess(q.bid_px, BOOK.ask)

    def test_wider_when_volatile(self):
        calm = compute_quote(BOOK, 50, 0.0001, CFG)
        wild = compute_quote(BOOK, 50, 0.002, CFG)
        self.assertGreater(wild.ask_px - wild.bid_px, calm.ask_px - calm.bid_px)

    def test_flatten_crosses_with_collar(self):
        q = flatten_quote(BOOK, 30, 50)
        self.assertEqual(q.ask_qty, 30)
        self.assertLess(q.ask_px, BOOK.bid)
        self.assertGreater(q.ask_px, BOOK.bid * 0.99)

    def test_round_to_tick(self):
        self.assertEqual(round_to_tick(12.3, 0.01, "down"), 12.3)
        self.assertEqual(round_to_tick(12.301, 0.01, "up"), 12.31)
        self.assertEqual(round_to_tick(12.309, 0.01, "down"), 12.3)


class PositionTest(unittest.TestCase):
    def test_round_trip_and_flip(self):
        p = Position()
        p.apply(Side.BUY, 10, 10.0)
        p.apply(Side.BUY, 10, 12.0)
        self.assertAlmostEqual(p.avg_px, 11.0)
        p.apply(Side.SELL, 15, 13.0)
        self.assertAlmostEqual(p.realized, 30.0)
        self.assertEqual(p.qty, 5)
        p.apply(Side.SELL, 10, 14.0)  # flips short
        self.assertAlmostEqual(p.realized, 45.0)
        self.assertEqual(p.qty, -5)
        self.assertAlmostEqual(p.avg_px, 14.0)


class GuardTest(unittest.TestCase):
    def test_jump_detector_cooldown(self):
        j = JumpDetector(window_s=10, threshold_bps=150, cooldown_s=60)
        j.update(0, 10.0)
        j.update(5, 10.2)  # +2%
        self.assertTrue(j.active(30))
        self.assertFalse(j.active(70))

    def test_kill_switch_flattens(self):
        broker = SimBroker({"GME": SimSymbol(mid=25.0, sigma_per_sec=0.0002)}, seed=0)
        engine = Engine(broker, Config(symbols=("GME",)))
        engine.portfolio.on_fill(Fill("x", "GME", Side.BUY, 100, 30.0, 0))  # instantly down $500
        engine.step()
        self.assertTrue(engine.killed)
        broker.advance()
        engine.step()
        self.assertEqual(engine.portfolio.positions["GME"].qty, 0)


class SimSmokeTest(unittest.TestCase):
    def test_full_day_ends_flat_within_limits(self):
        cfg = Config()
        broker = SimBroker({s: SimSymbol(mid=10.0) for s in cfg.symbols}, seed=42)
        engine = Engine(broker, cfg)
        max_pos = 0
        while broker.session().is_open:
            engine.step()
            broker.advance()
            max_pos = max(max_pos, *(abs(p.qty) for p in engine.portfolio.positions.values()))
        self.assertLessEqual(max_pos, cfg.strategy.max_position_shares)
        for p in engine.portfolio.positions.values():
            self.assertEqual(p.qty, 0)


class ReplayTest(unittest.TestCase):
    def _books(self, mids):
        return [Book("X", m - 0.01, m + 0.01, 100, 100, float(t)) for t, m in enumerate(mids)]

    def test_late_cancel_still_fills(self):
        # Price drops through our bid in the second our cancel is in flight.
        from mm.broker.replay import ReplayBroker
        b = ReplayBroker({"X": self._books([10.0, 10.0, 10.0, 9.0, 9.0])}, 0, 5,
                         latency_s=1.0, fill_model="pessimistic")
        oid = b.place("X", Side.BUY, 10, 9.95)
        b.advance()            # order live at t=1
        b.advance()            # t=2
        b.cancel(oid)          # takes effect at t=3, same time as the drop
        b.advance()
        fills, open_ids = b.poll()
        self.assertEqual([f.qty for f in fills], [10])
        self.assertNotIn(oid, open_ids)

    def test_toxic_fills_skew_adverse(self):
        from mm.broker.replay import ReplayBroker
        mids = [10.0] * 50 + [9.0] * 50  # buys made early are all adverse
        adverse = benign = 0
        for seed in range(200):
            b = ReplayBroker({"X": self._books(mids)}, 0, 100, latency_s=0, fill_model="toxic",
                             toxicity=1.0, toxic_horizon_s=60, seed=seed)
            b.place("X", Side.BUY, 1, 9.99)
            b.place("X", Side.SELL, 1, 10.01)
            b.advance()
            b.advance()
            for f in b.poll()[0]:
                adverse += f.side is Side.BUY
                benign += f.side is Side.SELL
        self.assertGreater(adverse, 0)
        self.assertEqual(benign, 0)


class ScenarioTest(unittest.TestCase):
    def test_synthetic_day_respects_ohlc(self):
        from mm.scenarios import GME_SQUEEZE_2021, synthetic_day
        spec = GME_SQUEEZE_2021[4]
        day = synthetic_day(spec, seed=0)
        mids = [b.mid for b in day.books]
        self.assertAlmostEqual(day.books[0].mid, spec.open, delta=spec.open * 0.02)
        self.assertLessEqual(max(mids), spec.high * 1.02)
        self.assertGreaterEqual(min(mids), spec.low * 0.98)
        gaps = sum(1 for a, b in zip(day.books, day.books[1:]) if b.ts - a.ts > 60)
        self.assertEqual(gaps, spec.halts)

    def test_bars_csv_roundtrip(self):
        import os, tempfile
        from datetime import datetime
        from mm.scenarios import NY, load_bars_csv
        t0 = int(datetime(2021, 1, 28, 9, 30, tzinfo=NY).timestamp())
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("symbol,ts,open,high,low,close,volume\n")
            f.write(f"GME,{t0},265,270,260,268,1000\nGME,{t0 + 60},268,275,266,274,1000\n")
            f.write(f"GME,{t0 + 600},300,305,295,301,1000\n")  # halt gap
        try:
            (day,) = load_bars_csv(f.name, "GME", spread_bps=40)
        finally:
            os.unlink(f.name)
        self.assertEqual(day.open_ts, t0)
        self.assertTrue(all(258 < b.mid < 307 for b in day.books))
        self.assertTrue(any(b.ts - a.ts > 300 for a, b in zip(day.books, day.books[1:])))


class GuardExitTest(unittest.TestCase):
    def _engine(self, **guards):
        from dataclasses import replace
        from mm.config import GuardConfig
        broker = SimBroker({"GME": SimSymbol(mid=25.0, sigma_per_sec=0.0002)}, seed=0)
        cfg = Config(symbols=("GME",), guards=replace(GuardConfig(), skip_open_s=0, **guards))
        engine = Engine(broker, cfg)
        engine.portfolio.on_fill(Fill("x", "GME", Side.BUY, 10, 25.0, 0))
        return broker, engine

    def _step(self, broker, engine):
        engine.step()
        broker.advance()
        return engine.state["GME"].working

    def test_guard_with_inventory_keeps_exit_then_crosses(self):
        broker, engine = self._engine(max_market_spread_bps=1.0, exit_after_s=5)  # always "too wide"
        working = self._step(broker, engine)
        self.assertNotIn(Side.BUY, working)
        self.assertGreater(working[Side.SELL].px, broker._book("GME").bid)  # passive, inside the touch
        for _ in range(6):
            self._step(broker, engine)
        self.assertEqual(engine.portfolio.positions["GME"].qty, 0)  # out by passive or aggressive exit

    def test_no_exit_variant_goes_dark(self):
        broker, engine = self._engine(max_market_spread_bps=1.0, exit_on_guard=False)
        self.assertEqual(self._step(broker, engine), {})


class YahooTest(unittest.TestCase):
    def test_bars_undo_later_splits(self):
        from mm.yahoo import _bars
        result = {"timestamp": [100, 200, 300],
                  "indicators": {"quote": [{"open": [10.0, None, 20.0], "high": [12.0, 1, 21.0],
                                            "low": [9.0, 1, 19.0], "close": [11.0, 1, 20.5],
                                            "volume": [400, 1, 100]}]}}
        bars = _bars(result, splits=[(250, 4.0)])
        self.assertEqual(bars[0], (100, 40.0, 48.0, 36.0, 44.0, 100.0))  # before split: x4
        self.assertEqual(bars[1][:5], (300, 20.0, 21.0, 19.0, 20.5))     # after split: as-is


class QueueFillTest(unittest.TestCase):
    def _broker(self, books, **kw):
        from mm.broker.replay import ReplayBroker
        return ReplayBroker({"X": books}, 0, len(books), latency_s=0, fill_a=1e9, fill_k=0, **kw)

    def _flat(self):
        return [Book("X", 10.00, 10.01, 100, 100, float(t)) for t in range(10)]

    def test_behind_touch_gets_no_friendly_fills(self):
        b = self._broker(self._flat(), fill_model="probabilistic", seed=0)
        b.place("X", Side.BUY, 1, 9.98)  # two ticks behind the bid
        for _ in range(9):
            b.advance()
        self.assertEqual(b.poll()[0], [])

    def test_at_touch_gets_friendly_fills(self):
        b = self._broker(self._flat(), fill_model="probabilistic", seed=0)
        b.place("X", Side.BUY, 1, 10.00)
        b.advance()  # goes live at the end of this step
        b.advance()  # collects flow during this one
        self.assertEqual(len(b.poll()[0]), 1)

    def test_behind_touch_fills_when_level_cleared(self):
        books = [Book("X", 10.00, 10.01, 100, 100, 0.0), Book("X", 10.00, 10.01, 100, 100, 1.0),
                 Book("X", 9.97, 9.98, 100, 100, 2.0)]  # bid falls through 9.98
        b = self._broker(books, fill_model="pessimistic")
        b.place("X", Side.BUY, 1, 9.98)
        b.advance()
        b.advance()
        self.assertEqual([f.px for f in b.poll()[0]], [9.98])

    def test_cheap_stock_trades_one_tick_wide(self):
        import random
        from mm.scenarios import _books
        books = _books("AMC", [(t, 3.005 + 0.001 * (t % 7)) for t in range(200)], 10, random.Random(0))
        self.assertTrue(all(round(b.ask - b.bid, 4) == 0.01 for b in books))


class TouchModeTest(unittest.TestCase):
    CALM = 0.00001  # per-second vol well under the touch gate

    def test_calm_large_tick_joins_touch(self):
        book = Book("AMC", 3.00, 3.01, 1000, 1000, 0.0)
        q = compute_quote(book, 0, self.CALM, StrategyConfig(long_only=False))
        self.assertEqual((q.bid_px, q.ask_px), (3.00, 3.01))
        q = compute_quote(book, 50, self.CALM, CFG)  # a quarter of max inventory: back off a tick
        self.assertEqual(q.bid_px, 2.99)

    def test_wide_market_steps_ahead(self):
        book = Book("AMC", 3.00, 3.04, 1000, 1000, 0.0)
        q = compute_quote(book, 0, self.CALM, StrategyConfig(long_only=False))
        self.assertEqual((q.bid_px, q.ask_px), (3.01, 3.03))

    def test_long_inventory_leans_in_ticks(self):
        book = Book("AMC", 3.00, 3.06, 1000, 1000, 0.0)
        q = compute_quote(book, 200, self.CALM, CFG)  # max inventory: lean 4 ticks
        self.assertEqual(q.ask_px, 3.01)              # clamped one tick above the bid

    def test_volatile_relative_to_tick_falls_back(self):
        book = Book("GME", 300.00, 300.01, 100, 100, 0.0)
        q = compute_quote(book, 0, 0.0005, CFG)  # ~80c expected move = 80 ticks
        self.assertLess(q.bid_px, 299.5)


class VolumeFillTest(unittest.TestCase):
    def test_fill_rate_follows_real_volume(self):
        from mm.broker.replay import ReplayBroker
        books = [Book("X", 3.00, 3.01, 100, 100, float(t)) for t in range(600)]
        counts = []
        for shares_per_min in (0, 120_000):
            vol = [(float(t), shares_per_min) for t in range(0, 600, 60)]
            b = ReplayBroker({"X": books}, 0, 600, latency_s=0, volume={"X": vol},
                             participation=0.01, seed=0)
            n = 0
            for _ in range(599):
                if not b._live:
                    b.place("X", Side.BUY, 10, 3.00)
                b.advance()
                n += len(b.poll()[0])
            counts.append(n)
        self.assertEqual(counts[0], 0)
        # 0.01 * 1000 shares/s / 10 = a fill every second it rests, but a replacement
        # order needs a step to go live, so about one fill every two steps.
        self.assertGreater(counts[1], 250)

    def test_sells_pay_regulatory_fees(self):
        from mm.models import regulatory_fee
        self.assertEqual(regulatory_fee(Side.BUY, 100, 25.0), 0.0)
        self.assertAlmostEqual(regulatory_fee(Side.SELL, 100, 25.0), 100 * 25 * 27.8e-6 + 0.0166)


class CalibrationLoopTest(unittest.TestCase):
    """Run the bot on a replay with known fill behaviour, record it, and check the
    report recovers that behaviour. This is the loop a live pilot relies on."""

    def test_report_recovers_participation_and_toxicity(self):
        import csv, os, tempfile
        import report
        from mm.broker.replay import ReplayBroker
        from mm.config import pilot_config
        from mm.recorder import Recorder
        from mm.scenarios import DaySpec, synthetic_day

        spec = DaySpec("calm", 3.00, 3.08, 2.95, 3.02, 0, 30, "random", volume=20e6)
        day = synthetic_day(spec, seed=3, symbol="AMC")
        with tempfile.TemporaryDirectory() as tmp:
            vol_csv = os.path.join(tmp, "vol.csv")
            with open(vol_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["symbol", "ts", "open", "high", "low", "close", "volume"])
                w.writerows(("AMC", int(t), 0, 0, 0, 0, v) for t, v in day.volume)
            rec = Recorder(path=os.path.join(tmp, "run.jsonl"))
            broker = ReplayBroker({"AMC": day.books}, day.open_ts, day.close_ts, fill_model="toxic",
                                  toxicity=0.5, volume={"AMC": day.volume}, participation=0.01, seed=1)
            engine = Engine(broker, pilot_config(), rec)
            while broker.session().is_open:
                engine.step()
                broker.advance()
            rec.close()
            r = report.analyze(report.load_events([rec.path]), vol_csv)["symbols"]["AMC"]
        self.assertAlmostEqual(r["participation"]["join"], 0.01, delta=0.003)
        self.assertAlmostEqual(r["toxicity"], 0.5, delta=0.15)


class PositionReconcileTest(unittest.TestCase):
    def test_persistent_mismatch_trips_kill_switch(self):
        from dataclasses import replace
        broker = SimBroker({"GME": SimSymbol(mid=25.0, sigma_per_sec=0.0002)}, seed=0)
        broker.positions = lambda: {"GME": 7}  # broker says we hold shares we never saw fill
        engine = Engine(broker, replace(Config(symbols=("GME",)), reconcile_every_s=10))
        engine.step()
        self.assertFalse(engine.killed)  # first sighting might be a fill still in flight
        for _ in range(6):
            broker.advance()
            engine.step()
        self.assertTrue(engine.killed)
        self.assertGreaterEqual(engine.portfolio.positions["GME"].volume, 7)  # adopted the 7 and sold them

    def test_transient_mismatch_is_ignored(self):
        from dataclasses import replace
        broker = SimBroker({"GME": SimSymbol(mid=25.0, sigma_per_sec=0.0002)}, seed=0)
        answers = iter([{"GME": 7}] + [{}] * 100)
        broker.positions = lambda: next(answers)
        engine = Engine(broker, replace(Config(symbols=("GME",)), reconcile_every_s=10))
        for _ in range(30):
            engine.step()
            broker.advance()
        self.assertFalse(engine.killed)


if __name__ == "__main__":
    unittest.main()
