"""Grade a paper/live run from its event log, and calibrate the replay to match it.

    python report.py logs/2026-10-22_093000.jsonl [more.jsonl ...]
    python report.py logs/*.jsonl --volume-csv data/amc_recent.csv   # offline volume source

Measures the two things the stress tests could only assume:
  * participation: of the shares that traded on our side while our order was at the best
    price, what share did we get? (stress.py --participation / --improve-participation)
  * toxicity: how much more often a fill is followed by an adverse move over 30s than a
    random moment of sitting at the touch would be (stress.py --toxicity)
Market volume comes from Yahoo minute bars (last 30 days only) unless --volume-csv is given.
"""
import argparse
import bisect
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field

from mm.models import Side
from mm.risk import Position

HORIZONS = (5, 30, 60)
TOXIC_HORIZON_S = 30  # must match ReplayBroker.toxic_horizon_s for the calibration to mean anything


@dataclass
class Order:
    sym: str
    side: Side
    px: float
    qty: int
    t0: float
    t1: float | None = None
    live_at: float | None = None  # exchange acks; None if the log has none
    done_at: float | None = None  # left the book without filling (cancel took effect)
    remaining: int = 0
    crossed: bool = False  # marketable when sent
    fills: list = field(default_factory=list)  # (ts, qty)

    def remaining_at(self, t: float) -> int:
        return self.qty - sum(q for ts, q in self.fills if ts <= t)


@dataclass
class Timeline:
    ts: list[float] = field(default_factory=list)
    bid: list[float] = field(default_factory=list)
    ask: list[float] = field(default_factory=list)

    def index_at(self, t: float) -> int:
        return bisect.bisect_right(self.ts, t) - 1

    def mid_at(self, t: float) -> float | None:
        i = self.index_at(t)
        return (self.bid[i] + self.ask[i]) / 2 if i >= 0 else None


def load_events(paths: list[str]) -> list[dict]:
    events = []
    for p in paths:
        with open(p) as f:
            events += [json.loads(line) for line in f if line.strip()]
    return sorted(events, key=lambda e: e["ts"])


def minute_volume(sym: str, volume_csv: str | None) -> dict[int, float]:
    if volume_csv:
        with open(volume_csv) as f:
            return {int(float(r["ts"])): float(r["volume"]) for r in csv.DictReader(f) if r["symbol"] == sym}
    try:
        from mm.yahoo import minute_bars
        return {int(b[0]): b[5] for b in minute_bars(sym)}
    except Exception as e:  # network, certificates, or the run is older than 30 days
        print(f"  (no market volume for {sym}: {e}; participation can't be measured)")
        return {}


def _adverse(side: Side, px: float, future_mid: float | None) -> bool:
    if future_mid is None:
        return False
    return future_mid < px if side is Side.BUY else future_mid > px


MIN_EXPOSURE_FILLS = 20  # below this many fills a participation estimate is mostly noise


def _expected_fills(segments: list, p: float, tox: float, adverse_only: bool = False) -> float:
    """Expected fill count under the replay's model: each step an order fills with
    probability min(1, rate * dt), rate = p * side_shares_per_s / qty * toxic multiplier."""
    return sum(min(1.0, p * v / max(q, 1) * ((1 - tox) + 2 * tox * adv) * dt)
               for dt, v, q, adv in segments if adv or not adverse_only)


def _bisect(f, lo: float, hi: float, target: float, steps: int = 40) -> float:
    """Root of increasing f(x) = target on [lo, hi], clamped to the ends."""
    if f(hi) <= target:
        return hi
    if f(lo) >= target:
        return lo
    for _ in range(steps):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return (lo + hi) / 2


def _calibrate(segments: list, fills: int, adverse_fills: int | None) -> tuple[float | None, float | None]:
    """(participation, toxicity) that reproduce both the fill count and how many of
    those fills preceded an adverse move, under the replay's exact fill model.

    Solving the model directly, rather than taking ratios, accounts for fills
    saturating in busy minutes and for adverse moments clustering in high volume.
    """
    if fills < MIN_EXPOSURE_FILLS or not segments:
        return None, None

    def p_for(tox: float) -> float:
        return _bisect(lambda p: _expected_fills(segments, p, tox), 0.0, 1.0, fills)

    if adverse_fills is None:
        return p_for(0.0), None

    def adverse_share(tox: float) -> float:  # increasing in tox
        p = p_for(tox)
        return _expected_fills(segments, p, tox, adverse_only=True) / max(_expected_fills(segments, p, tox), 1e-12)

    tox = _bisect(adverse_share, 0.0, 1.0, adverse_fills / fills, steps=25)
    return p_for(tox), tox


def analyze(events: list[dict], volume_csv: str | None = None, ack_delay_s: float = 1.0) -> dict:
    books: dict[str, Timeline] = defaultdict(Timeline)
    orders: dict[str, Order] = {}
    fills: list[dict] = []
    notes = []
    for e in events:
        k = e["k"]
        if k == "book":
            tl = books[e["sym"]]
            tl.ts.append(e["ts"]); tl.bid.append(e["bid"]); tl.ask.append(e["ask"])
        elif k == "order":
            side = Side(e["side"])
            crossed = e["ask"] is not None and (e["px"] >= e["ask"] if side is Side.BUY else e["px"] <= e["bid"])
            orders[e["id"]] = Order(e["sym"], side, e["px"], e["qty"], e["ts"], remaining=e["qty"], crossed=crossed)
        elif k == "ack" and e["id"] in orders:
            if e.get("state", "live") == "live":
                orders[e["id"]].live_at = e["ts"]
            else:
                orders[e["id"]].done_at = e["ts"]
        elif k == "cancel" and e["id"] in orders and orders[e["id"]].t1 is None:
            orders[e["id"]].t1 = e["ts"]
        elif k == "fill":
            fills.append(e)
            o = orders.get(e["id"])
            if o:
                o.remaining -= e["qty"]
                o.fills.append((e["ts"], e["qty"]))
                if o.remaining <= 0 and o.t1 is None:
                    o.t1 = e["ts"]
        elif k == "note":
            notes.append(e["msg"])
    end = events[-1]["ts"] if events else 0.0

    results = {}
    for sym, tl in books.items():
        vol = minute_volume(sym, volume_csv)

        def side_shares_per_s(t: float) -> float:
            # Half of each minute's volume trades against each side of the book.
            return vol.get(int(t // 60 * 60), 0.0) / 120

        # Time at the touch, the market volume we could have shared, and how often a
        # random moment there was followed by an adverse move (the baseline for toxicity).
        exposure = {"join": [0.0, 0.0], "improve": [0.0, 0.0]}  # seconds, market side-shares
        segments = {"join": [], "improve": []}  # (seconds, side-shares/s, order qty, adverse later)
        adverse_time = total_time = 0.0
        for o in (o for o in orders.values() if o.sym == sym):
            # Only count time the order was actually live at the exchange.
            t_start = o.live_at if o.live_at is not None else o.t0 + ack_delay_s
            if o.remaining <= 0:  # filled: t1 is the last fill
                t_end = o.t1
            elif o.done_at is not None:
                t_end = o.done_at
            else:  # cancel sent but no ack logged: it rested a little longer
                t_end = o.t1 + ack_delay_s if o.t1 is not None else end
            i = max(tl.index_at(t_start), 0)
            while i < len(tl.ts) and tl.ts[i] < t_end:
                a = max(tl.ts[i], t_start)
                b = min(tl.ts[i + 1] if i + 1 < len(tl.ts) else t_end, t_end)
                dt = b - a
                touch = tl.bid[i] if o.side is Side.BUY else tl.ask[i]
                # An order at or through the other side isn't resting; it's taking.
                crossing = o.px >= tl.ask[i] if o.side is Side.BUY else o.px <= tl.bid[i]
                if dt > 0 and not crossing:
                    better = o.px > touch if o.side is Side.BUY else o.px < touch
                    if abs(o.px - touch) < 1e-9 or better:
                        kind = "improve" if better else "join"
                        # Judge "about to move against us" from the end of the interval,
                        # where the replay places the flow that arrives during it.
                        adverse = _adverse(o.side, o.px, tl.mid_at(b - 1e-3 + TOXIC_HORIZON_S))
                        exposure[kind][0] += dt
                        exposure[kind][1] += dt * side_shares_per_s(a)
                        segments[kind].append((dt, side_shares_per_s(a), o.remaining_at(a), adverse))
                        if not better:
                            total_time += dt
                            adverse_time += dt * adverse
                i += 1

        cats: dict[str, dict] = defaultdict(lambda: {"fills": 0, "shares": 0, **{h: 0.0 for h in HORIZONS}})
        adverse_join_fills = 0
        pos = Position()
        for f in (f for f in fills if f["sym"] == sym):
            side = Side(f["side"])
            pos.apply(side, f["qty"], f["px"], f.get("fee", 0.0))
            o = orders.get(f["id"])
            # Classify by the book in force at the moment of the fill. If our level was
            # already gone, the market ran through us; otherwise it was ordinary flow.
            # Books are logged about once a second, so a sweep that lands between
            # snapshots shows up as "join" here: live toxicity reads a little high.
            at = tl.index_at(f["ts"])
            if o is None:
                cat = "unknown"
            elif (o.crossed or abs(f["px"] - o.px) > 1e-9
                  or (o.live_at is not None and f["ts"] - o.live_at < 0.05)):
                cat = "taker"  # marketable on arrival: it never rested
            elif at >= 0 and (tl.bid[at] < o.px if side is Side.BUY else tl.ask[at] > o.px):
                cat = "cleared"  # our level got run through: adverse by construction
            elif at >= 0 and (o.px > tl.bid[at] if side is Side.BUY else o.px < tl.ask[at]):
                cat = "improve"
            else:
                cat = "join"
            c = cats[cat]
            c["fills"] += 1
            c["shares"] += f["qty"]
            sign = 1 if side is Side.BUY else -1
            for h in HORIZONS:
                m = tl.mid_at(f["ts"] + h)
                if m is not None:
                    c[h] += sign * (m - f["px"]) * f["qty"]
            if cat == "join":
                adverse_join_fills += _adverse(side, f["px"], tl.mid_at(f["ts"] + TOXIC_HORIZON_S))

        mark = (tl.bid[-1] + tl.ask[-1]) / 2 if tl.ts else 0.0
        a = adverse_time / total_time if total_time else None
        join_fills = cats["join"]["fills"] if "join" in cats else 0
        f_adv = adverse_join_fills / join_fills if join_fills else None
        p_join, tox = _calibrate(segments["join"], join_fills, adverse_join_fills)
        p_improve, _ = _calibrate(segments["improve"], cats["improve"]["fills"] if "improve" in cats else 0, None)
        part = {"join": p_join, "improve": p_improve}
        results[sym] = dict(
            pnl=pos.realized + pos.unrealized(mark), fees=sum(f.get("fee", 0.0) for f in fills if f["sym"] == sym),
            end_pos=pos.qty, orders=sum(o.sym == sym for o in orders.values()), cats=dict(cats),
            exposure=exposure, participation=part, base_adverse=a, fill_adverse=f_adv, toxicity=tox,
            hours=sum(b - a for a, b in zip(tl.ts, tl.ts[1:]) if b - a < 300) / 3600,  # skip overnight
        )
    return {"symbols": results, "notes": notes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--volume-csv", help="minute bars CSV to use instead of fetching from Yahoo")
    ap.add_argument("--ack-delay", type=float, default=1.0,
                    help="assumed send-to-live delay for logs without exchange acks")
    args = ap.parse_args()
    out = analyze(load_events(args.logs), args.volume_csv, args.ack_delay)
    for msg in out["notes"]:
        print(f"NOTE: {msg}")
    for sym, r in out["symbols"].items():
        print(f"\n{sym}: {r['hours']:.1f}h of book data, {r['orders']} orders, "
              f"P&L ${r['pnl']:,.2f} (fees ${r['fees']:,.2f}), ending position {r['end_pos']}")
        print(f"  {'fill type':<10}{'fills':>7}{'shares':>9}" + "".join(f"{f'{h}s markout':>14}" for h in HORIZONS))
        for cat in ("join", "improve", "cleared", "taker", "unknown"):
            c = r["cats"].get(cat)
            if c:
                print(f"  {cat:<10}{c['fills']:>7}{c['shares']:>9}" +
                      "".join(f"{100 * c[h] / c['shares']:>+12.2f}c" for h in HORIZONS))
        for k in ("join", "improve"):
            secs, mkt = r["exposure"][k]
            p = r["participation"][k]
            print(f"  at touch ({k}): {secs / 60:,.0f} min, {mkt:,.0f} shares traded on our side"
                  + (f", we got {p:.2%}" if p is not None else ""))
        if r["toxicity"] is not None:
            print(f"  adverse 30s later: {r['fill_adverse']:.0%} of join fills vs {r['base_adverse']:.0%} "
                  f"of random moments at the touch -> toxicity {r['toxicity']:.2f}")
        flags = []
        if r["participation"]["join"] is not None:
            flags.append(f"--participation {r['participation']['join']:.4f}")
        if r["participation"]["improve"] is not None:
            flags.append(f"--improve-participation {r['participation']['improve']:.3f}")
        if r["toxicity"] is not None:
            flags.append(f"--toxicity {r['toxicity']:.2f}")
        if flags:
            print(f"  calibrated replay: python stress.py bars data/{sym.lower()}_recent.csv --symbol {sym} "
                  f"--fill-models toxic {' '.join(flags)}")


if __name__ == "__main__":
    main()
