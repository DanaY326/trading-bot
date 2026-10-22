"""Entry point.

    python run.py sim                      # one simulated trading day, no account needed
    python run.py sim --days 20 --seed 1   # many days, report distribution of daily P&L
    python run.py paper                    # Alpaca paper trading (needs APCA_API_KEY_ID / APCA_API_SECRET_KEY)
    python run.py paper --profile pilot --feed sip    # the small measurement run (see PILOT.md)

Every paper/live run writes an event log to logs/; grade it with `python report.py logs/<file>`.
"""
import argparse
import logging
import os
import statistics
import sys

from mm.broker.sim import SimBroker, SimSymbol
from mm.config import Config, pilot_config
from mm.engine import Engine

SIM_START_PRICES = {"GME": 25.0, "AMC": 4.0, "KOSS": 6.0, "BB": 3.5}


def run_sim_day(cfg: Config, seed: int) -> Engine:
    broker = SimBroker({s: SimSymbol(mid=SIM_START_PRICES.get(s, 10.0)) for s in cfg.symbols}, seed=seed)
    engine = Engine(broker, cfg)
    while broker.session().is_open:
        engine.step()
        broker.advance(cfg.loop_interval_s)
    engine.step()  # sees the close, cancels everything
    return engine


def cmd_sim(args) -> None:
    cfg = Config()
    daily = []
    for d in range(args.days):
        engine = run_sim_day(cfg, seed=args.seed + d)
        pnl = engine.portfolio.pnl(engine.marks)
        daily.append(pnl)
        if args.days == 1 or args.verbose:
            print(f"\nday {d + 1}: pnl ${pnl:,.2f}  killed={engine.killed}  orders={engine.broker.order_count}")
            for sym, p in engine.portfolio.positions.items():
                print(f"  {sym:5s} realized ${p.realized:9,.2f}  volume {p.volume:6d}  end pos {p.qty}")
            total = sum(engine.reasons.values())
            print("  time by state: " + ", ".join(
                f"{k} {v / total:.0%}" for k, v in engine.reasons.most_common()))
    if args.days > 1:
        wins = sum(p > 0 for p in daily)
        print(f"\n{args.days} days: mean ${statistics.mean(daily):,.2f}  "
              f"stdev ${statistics.stdev(daily):,.2f}  worst ${min(daily):,.2f}  "
              f"best ${max(daily):,.2f}  win rate {wins / args.days:.0%}")


def cmd_paper(args) -> None:
    from mm.broker.alpaca import AlpacaBroker

    key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("set APCA_API_KEY_ID and APCA_API_SECRET_KEY")
    live = args.live and args.i_understand_this_can_lose_money
    if args.live and not live:
        sys.exit("--live also requires --i-understand-this-can-lose-money")

    from mm.recorder import Recorder

    cfg = pilot_config() if args.profile == "pilot" else Config()
    broker = AlpacaBroker(key, secret, paper=not live, feed=args.feed)
    stray = {s: q for s, q in broker.positions().items() if s in cfg.symbols and q}
    if stray:
        broker.close()
        sys.exit(f"account already holds {stray}; flatten manually first so P&L tracking starts at zero")
    warnings = broker.account_warnings(touch_mode=cfg.strategy.touch_max_move_ticks > 0)
    for w in warnings:
        logging.warning(w)
    if warnings and live and not args.accept_warnings:
        broker.close()
        sys.exit("refusing to trade real money with the warnings above (override: --accept-warnings)")

    recorder = Recorder(args.log_dir)
    logging.info("%s run on %s; logging events to %s", "LIVE" if live else "paper",
                 ", ".join(cfg.symbols), recorder.path)
    try:
        Engine(broker, cfg, recorder).run()
    finally:
        broker.close()
        recorder.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--days", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("-v", "--verbose", action="store_true")
    p = sub.add_parser("paper")
    p.add_argument("--feed", default="iex", choices=["iex", "sip"])
    p.add_argument("--profile", default="default", choices=["default", "pilot"])
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--accept-warnings", action="store_true")
    p.add_argument("--live", action="store_true")
    p.add_argument("--i-understand-this-can-lose-money", action="store_true")
    args = ap.parse_args()
    if args.cmd == "sim":
        logging.getLogger("mm").setLevel(logging.WARNING)
        cmd_sim(args)
    else:
        cmd_paper(args)


if __name__ == "__main__":
    main()
