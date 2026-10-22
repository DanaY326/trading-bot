"""All tunable parameters in one place. Defaults are deliberately conservative."""
from dataclasses import dataclass, field

# Edit this list to change what the bot trades. These are the names that have
# historically traded as "meme stocks" (heavy retail flow, high short interest,
# violent squeezes). Verify each is still listed and liquid before going live.
MEME_UNIVERSE: tuple[str, ...] = ("GME", "AMC", "KOSS", "BB")


@dataclass(frozen=True)
class StrategyConfig:
    horizon_s: float = 30.0            # how far ahead we price inventory risk
    vol_mult: float = 1.5              # half-spread = vol_mult * sigma over horizon
    min_half_spread_bps: float = 15.0  # floor on half-spread, in bps of mid
    inventory_skew: float = 1.0        # quote shift (in half-spreads) at max inventory
    base_order_shares: int = 10
    max_position_shares: int = 200
    max_position_notional: float = 5_000.0
    long_only: bool = True             # meme names are often hard to borrow; never short
    ewma_halflife_s: float = 60.0
    min_vol_samples: int = 30          # warm-up before we quote anything
    requote_threshold_ticks: int = 1   # don't cancel/replace for moves smaller than this
    # Touch mode, for calm markets. A vol-scaled quote sits behind the best bid/offer
    # when the market is a tick or two wide, and nothing behind the touch gets
    # uninformed flow. When the expected move over the horizon is under this many
    # ticks, quote at the touch instead: the tick is then big relative to how far
    # price wanders, so the spread we earn outweighs getting picked off. That's true
    # of cheap names (AMC at $3: 1 tick = 33bps), not of GME at $25. 0 disables it.
    # 2.5 and 4 were tuned on Sep 23 - Oct 7 2026 minute bars and checked on Oct 8-21.
    touch_max_move_ticks: float = 2.5
    improve_min_spread_ticks: int = 3  # step one tick ahead of the touch when it's this wide
    touch_skew_ticks: int = 4          # ticks to lean both quotes at max inventory


@dataclass(frozen=True)
class GuardConfig:
    """Meme-stock specific circuit breakers. Any one of these pulls quotes."""
    max_horizon_vol_bps: float = 300.0  # stand aside if expected move over horizon > 3%
    jump_window_s: float = 10.0
    jump_threshold_bps: float = 150.0   # 1.5% move inside the window = squeeze/dump
    jump_cooldown_s: float = 60.0
    max_market_spread_bps: float = 100.0
    stale_quote_s: float = 5.0          # no fresh quote -> assume halt (LULD) or bad feed
    min_price: float = 2.0
    skip_open_s: float = 5 * 60         # opening auction aftermath is pure chaos
    flatten_before_close_s: float = 10 * 60  # never hold meme inventory overnight
    flatten_collar_bps: float = 50.0    # how far through the bid/ask flatten orders go
    exit_on_guard: bool = True          # while guarded with inventory, keep a reduce-only exit
    exit_after_s: float = 30.0          # ...and cross the spread if the guard lasts this long


@dataclass(frozen=True)
class RiskConfig:
    daily_loss_limit: float = 250.0     # kill switch: cancel everything and flatten
    max_gross_notional: float = 15_000.0
    max_orders_per_min: int = 60        # Alpaca allows ~200 API calls/min total


@dataclass(frozen=True)
class Config:
    symbols: tuple[str, ...] = MEME_UNIVERSE
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    loop_interval_s: float = 1.0
    # Live only: compare our position with the broker's this often. A mismatch that
    # persists across two checks (a dropped fill on the websocket) trips the kill switch.
    reconcile_every_s: float = 30.0


def pilot_config() -> Config:
    """A small first live run whose job is to measure real fills, not to make money.

    One cheap, tick-constrained name (where touch mode tested best), 10-share orders,
    at most ~$150 of inventory, and a $25 daily stop.
    """
    return Config(
        symbols=("AMC",),
        strategy=StrategyConfig(base_order_shares=10, max_position_shares=50,
                                max_position_notional=250.0),
        risk=RiskConfig(daily_loss_limit=25.0, max_gross_notional=300.0, max_orders_per_min=40),
    )
