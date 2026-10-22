"""Stress-test the bot on the January 2021 GME squeeze.

    python stress.py synthetic                      # approximate squeeze days, no data needed
    python stress.py yahoo --start 2021-01-22 --end 2021-02-05   # real daily OHLC from Yahoo
    python stress.py bars data/gme_2021_01.csv      # real minute bars (scripts/fetch_bars.py)

Every day is run for each guard variant x fill model x seed, with a fresh engine per day
(the loss limit is daily). Any config field can be overridden for experiments:

    python stress.py bars data/gme_recent.csv --set strategy.touch_max_vol_bps=40 \
        --variants guarded --fill-models toxic --day-from 2026-10-08
 "no_guards" turns off the meme-stock circuit breakers but
keeps the kill switch and end-of-day flatten, so you can see what the guards buy you.
"guarded_no_exit" goes dark when a guard trips even while holding inventory.
"""
import argparse
import csv
import logging
import math
import statistics
from dataclasses import dataclass, replace
from datetime import date

from mm.broker.replay import ReplayBroker
from mm.config import Config, GuardConfig
from mm.engine import Engine
from mm.scenarios import GME_SQUEEZE_2021, Day, load_bars_csv, synthetic_day
from mm.yahoo import daily_specs

VARIANTS = {
    "guarded": GuardConfig(),
    "guarded_no_exit": GuardConfig(exit_on_guard=False),
    "no_guards": GuardConfig(max_horizon_vol_bps=math.inf, jump_threshold_bps=math.inf,
                             max_market_spread_bps=math.inf, stale_quote_s=math.inf, skip_open_s=0),
}


@dataclass
class DayResult:
    pnl: float
    max_drawdown: float
    max_inventory_usd: float
    volume: int
    quoting_frac: float
    killed: bool
    end_pos: int


def run_day(day: Day, cfg: Config, fill_model: str, seed: int, series: list | None = None,
            **broker_kw) -> DayResult:
    volume = {day.symbol: day.volume} if day.volume else None
    broker = ReplayBroker({day.symbol: day.books}, day.open_ts, day.close_ts,
                          fill_model=fill_model, volume=volume, seed=seed, **broker_kw)
    engine = Engine(broker, cfg)
    pos = engine.portfolio.positions[day.symbol]
    peak = drawdown = max_inv = 0.0
    while broker.session().is_open:
        engine.step()
        broker.advance(cfg.loop_interval_s)
        pnl = engine.portfolio.pnl(engine.marks)
        peak = max(peak, pnl)
        drawdown = max(drawdown, peak - pnl)
        mid = engine.marks.get(day.symbol, 0.0)
        max_inv = max(max_inv, abs(pos.qty) * mid)
        if series is not None and int(broker.now() - day.open_ts) % 10 == 0:
            series.append((broker.now() - day.open_ts, mid, pos.qty, round(pnl, 2)))
    engine.step()
    return DayResult(
        pnl=engine.portfolio.pnl(engine.marks),
        max_drawdown=drawdown,
        max_inventory_usd=max_inv,
        volume=pos.volume,
        quoting_frac=engine.reasons["quoting"] / max(1, sum(engine.reasons.values())),
        killed=engine.killed,
        end_pos=pos.qty,
    )


def _changed(section) -> list[str]:
    """Fields of a config section that differ from their defaults."""
    default = type(section)()
    return [f for f in section.__dataclass_fields__ if getattr(section, f) != getattr(default, f)]


def apply_overrides(cfg: Config, sets: list[str]) -> Config:
    for item in sets:
        key, _, raw = item.partition("=")
        section_name, _, field_name = key.partition(".")
        if not field_name:  # top-level field, e.g. loop_interval_s=0.5
            cfg = replace(cfg, **{key: type(getattr(cfg, key))(raw)})
            continue
        section = getattr(cfg, section_name)
        current = getattr(section, field_name)  # AttributeError on a typo, which is what we want
        value = (raw.lower() in ("1", "true", "yes")) if isinstance(current, bool) else type(current)(raw)
        cfg = replace(cfg, **{section_name: replace(section, **{field_name: value})})
    return cfg


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["synthetic", "yahoo", "bars"])
    ap.add_argument("csv", nargs="?", help="minute-bar CSV for `bars`")
    ap.add_argument("--symbol", default="GME")
    ap.add_argument("--start", default="2021-01-22", help="first day for `yahoo`")
    ap.add_argument("--end", default="2021-02-05", help="last day for `yahoo`")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--latency", type=float, default=1.0, help="order/cancel latency, seconds")
    ap.add_argument("--fill-models", default="probabilistic,toxic",
                    help="comma-separated: probabilistic, toxic, pessimistic")
    ap.add_argument("--toxicity", type=float, default=0.5, help="informed share of flow for `toxic`")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                    help="override a config field, e.g. strategy.vol_mult=0.3 (repeatable)")
    ap.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated guard variants")
    ap.add_argument("--day-from", default="", help="only run days on/after this label (YYYY-MM-DD)")
    ap.add_argument("--day-to", default="9999", help="only run days on/before this label")
    ap.add_argument("--participation", type=float, default=0.01,
                    help="share of real volume we get while joining the touch (bars/yahoo only)")
    ap.add_argument("--improve-participation", type=float, default=0.25,
                    help="share of real volume we get while alone at a better price")
    ap.add_argument("--fill-k", type=float, default=0.08,
                    help="fill-rate decay per bp from mid; lower = flow sweeps deeper into the book")
    ap.add_argument("--spread-bps", type=float, default=40.0, help="assumed spread for bar replay")
    ap.add_argument("--out", help="write a 10s-resolution P&L/inventory series (seed 0) to this CSV")
    args = ap.parse_args()
    if args.source == "bars" and not args.csv:
        ap.error("bars needs a CSV path")
    fill_models = args.fill_models.split(",")
    specs = GME_SQUEEZE_2021
    if args.source == "yahoo":
        specs = daily_specs(args.symbol, date.fromisoformat(args.start), date.fromisoformat(args.end))
        print(f"fetched {len(specs)} days from Yahoo: " +
              ", ".join(f"{s.day} {s.low:.2f}-{s.high:.2f}" for s in specs))

    def days_for(seed: int) -> list[Day]:
        if args.source != "bars":
            days = [synthetic_day(spec, seed, args.symbol) for spec in specs]
        else:
            days = load_bars_csv(args.csv, args.symbol, args.spread_bps, seed)
        return [d for d in days if args.day_from <= d.label <= args.day_to]

    base = apply_overrides(Config(symbols=(args.symbol,)), args.set)
    variants = {v: replace(VARIANTS[v], **{k: getattr(base.guards, k) for k in _changed(base.guards)})
                for v in args.variants.split(",")}
    broker_kw = dict(latency_s=args.latency, toxicity=args.toxicity, fill_k=args.fill_k,
                     participation=args.participation, improve_participation=args.improve_participation)
    results: dict[tuple[str, str, str], list[DayResult]] = {}
    series_rows = []
    for seed in range(args.seeds):
        for day in days_for(seed):
            for variant, guards in variants.items():
                for fm in fill_models:
                    series = [] if args.out and seed == 0 else None
                    r = run_day(day, replace(base, guards=guards), fm, seed, series, **broker_kw)
                    results.setdefault((day.label, variant, fm), []).append(r)
                    for row in series or []:
                        series_rows.append((day.label, variant, fm, *row))

    print(f"\n{args.source} | {args.symbol} | {args.seeds} seed(s) | latency {args.latency}s | "
          f"toxicity {args.toxicity} | participation {args.participation}/{args.improve_participation} | "
          f"loss limit ${base.risk.daily_loss_limit:.0f} | overrides: {', '.join(args.set) or 'none'}\n")
    hdr = f"{'day':<11}{'variant':<17}{'fills':<14}{'mean P&L':>10}{'worst':>10}{'max DD':>9}" \
          f"{'max inv $':>11}{'shares':>8}{'quoting':>9}{'kills':>7}{'not flat':>9}"
    print(hdr)
    print("-" * len(hdr))
    totals: dict[tuple[str, str], list[float]] = {}
    last_day = None
    for (label, variant, fm), rs in results.items():
        if last_day and label != last_day:
            print()
        last_day = label
        pnls = [r.pnl for r in rs]
        totals.setdefault((variant, fm), []).append(statistics.mean(pnls))
        print(f"{label:<11}{variant:<17}{fm:<14}{statistics.mean(pnls):>10,.0f}{min(pnls):>10,.0f}"
              f"{max(r.max_drawdown for r in rs):>9,.0f}{max(r.max_inventory_usd for r in rs):>11,.0f}"
              f"{statistics.mean(r.volume for r in rs):>8,.0f}"
              f"{statistics.mean(r.quoting_frac for r in rs):>9.0%}"
              f"{sum(r.killed for r in rs):>4}/{len(rs)}{sum(r.end_pos != 0 for r in rs):>6}/{len(rs)}")

    print("\nwhole period (sum of per-day means):")
    for (variant, fm), daily in totals.items():
        print(f"  {variant:<17}{fm:<14}{sum(daily):>10,.0f}   worst day {min(daily):>8,.0f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["day", "variant", "fill_model", "t_s", "mid", "position", "pnl"])
            w.writerows(series_rows)
        print(f"\nseries written to {args.out}")


if __name__ == "__main__":
    main()
