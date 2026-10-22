"""Quote generation: volatility-scaled spread with inventory skew (Avellaneda-Stoikov flavoured)."""
import math

from .config import StrategyConfig
from .models import Book, Quote, round_to_tick, tick_size


def max_position(cfg: StrategyConfig, mid: float) -> int:
    return min(cfg.max_position_shares, int(cfg.max_position_notional // mid))


def compute_quote(book: Book, position: int, sigma_per_sec: float, cfg: StrategyConfig) -> Quote:
    mid = book.mid
    tick = tick_size(mid)

    sigma_h = sigma_per_sec * math.sqrt(cfg.horizon_s)
    half = max(cfg.min_half_spread_bps / 1e4, cfg.vol_mult * sigma_h) * mid

    max_pos = max_position(cfg, mid)
    inv = max(-1.0, min(1.0, position / max_pos)) if max_pos > 0 else 0.0

    if sigma_h * mid / tick < cfg.touch_max_move_ticks:
        bid_px, ask_px = _touch_prices(book, inv, tick, cfg)
    else:
        # Long inventory shifts both quotes down: we buy less eagerly and sell more eagerly.
        center = mid - inv * cfg.inventory_skew * half
        bid_px = round_to_tick(center - half, tick, "down")
        ask_px = round_to_tick(center + half, tick, "up")
    # Stay passive: never cross the market, even when heavily skewed.
    bid_px = min(bid_px, round(book.ask - tick, 4))
    ask_px = max(ask_px, round(book.bid + tick, 4))

    bid_qty = max(0, min(cfg.base_order_shares, max_pos - position))
    ask_room = position if cfg.long_only else max_pos + position
    ask_qty = max(0, min(cfg.base_order_shares, ask_room))

    return Quote(
        bid_px=bid_px if bid_qty else 0.0,
        bid_qty=bid_qty,
        ask_px=ask_px if ask_qty else 0.0,
        ask_qty=ask_qty,
    )


def _touch_prices(book: Book, inv: float, tick: float, cfg: StrategyConfig) -> tuple[float, float]:
    """Join the best bid/offer, or step one tick ahead of it when the market is wide.

    Inventory leans both quotes by whole ticks: when long, the bid backs off and the
    ask steps in. The caller clamps both so neither crosses the market.
    """
    spread_ticks = round((book.ask - book.bid) / tick)
    improve = 1 if spread_ticks >= cfg.improve_min_spread_ticks else 0
    shift = round(inv * cfg.touch_skew_ticks)
    return (round(book.bid + (improve - shift) * tick, 4),
            round(book.ask - (improve + shift) * tick, 4))


def exit_quote(book: Book, position: int) -> Quote:
    """Reduce-only passive order one tick inside the touch, used while a guard is active.

    Pulling every order when a guard trips leaves us holding inventory with no way out;
    this keeps an exit working without adding risk.
    """
    if position == 0:
        return Quote.none("flat")
    tick = tick_size(book.mid)
    if position > 0:
        px = max(round(book.ask - tick, 4), round(book.bid + tick, 4))
        return Quote(ask_px=px, ask_qty=position, reason="exit_passive")
    px = min(round(book.bid + tick, 4), round(book.ask - tick, 4))
    return Quote(bid_px=px, bid_qty=-position, reason="exit_passive")


def flatten_quote(book: Book, position: int, collar_bps: float) -> Quote:
    """Aggressive limit order that crosses the spread but caps slippage.

    We avoid market orders: during a halt they queue and fill at the reopen print,
    which on a meme stock can be 20%+ away.
    """
    if position == 0:
        return Quote.none("flat")
    tick = tick_size(book.mid)
    if position > 0:
        px = round_to_tick(book.bid * (1 - collar_bps / 1e4), tick, "down")
        return Quote(ask_px=max(px, tick), ask_qty=position, reason="flatten")
    px = round_to_tick(book.ask * (1 + collar_bps / 1e4), tick, "up")
    return Quote(bid_px=px, bid_qty=-position, reason="flatten")
