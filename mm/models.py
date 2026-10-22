import math
from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Book:
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    ts: float  # epoch seconds of the quote

    @property
    def valid(self) -> bool:
        return self.bid > 0 and self.ask > 0 and self.ask > self.bid

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_bps(self) -> float:
        return (self.ask - self.bid) / self.mid * 1e4


@dataclass(frozen=True)
class Quote:
    """Target two-sided quote. A side with qty 0 means 'no order on that side'."""
    bid_px: float = 0.0
    bid_qty: int = 0
    ask_px: float = 0.0
    ask_qty: int = 0
    reason: str = ""

    @staticmethod
    def none(reason: str) -> "Quote":
        return Quote(reason=reason)


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: Side
    qty: int
    px: float
    ts: float
    fee: float = 0.0


# US regulatory fees on sales. Commission-free brokers pass these through. The SEC
# rate is reset each fiscal year; check sec.gov/fast-answers for the current one.
SEC_FEE_PER_DOLLAR = 27.80e-6
FINRA_TAF_PER_SHARE = 0.000166
FINRA_TAF_MAX = 8.30


def regulatory_fee(side: Side, qty: int, px: float) -> float:
    if side is Side.BUY:
        return 0.0
    return SEC_FEE_PER_DOLLAR * qty * px + min(FINRA_TAF_PER_SHARE * qty, FINRA_TAF_MAX)


@dataclass(frozen=True)
class Session:
    is_open: bool
    secs_since_open: float
    secs_to_close: float


def tick_size(px: float) -> float:
    # Reg NMS: $0.01 at/above $1, $0.0001 below.
    return 0.01 if px >= 1.0 else 0.0001


def round_to_tick(px: float, tick: float, direction: str) -> float:
    n = px / tick
    # epsilon guards against 12.3 / 0.01 == 1229.9999999
    ticks = math.floor(n + 1e-9) if direction == "down" else math.ceil(n - 1e-9)
    return round(ticks * tick, 4)
