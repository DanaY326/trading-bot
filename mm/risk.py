from collections import deque
from dataclasses import dataclass

from .models import Fill, Side


@dataclass
class Position:
    qty: int = 0
    avg_px: float = 0.0
    realized: float = 0.0
    volume: int = 0

    def apply(self, side: Side, qty: int, px: float, fee: float = 0.0) -> None:
        signed = qty if side is Side.BUY else -qty
        self.volume += qty
        self.realized -= fee
        if self.qty == 0 or (self.qty > 0) == (signed > 0):
            self.avg_px = (self.avg_px * abs(self.qty) + px * qty) / (abs(self.qty) + qty)
            self.qty += signed
            return
        closing = min(qty, abs(self.qty))
        direction = 1 if self.qty > 0 else -1
        self.realized += closing * (px - self.avg_px) * direction
        self.qty += signed
        if self.qty == 0:
            self.avg_px = 0.0
        elif (self.qty > 0) != (direction > 0):  # flipped through zero
            self.avg_px = px

    def unrealized(self, mark: float) -> float:
        return self.qty * (mark - self.avg_px)


class Portfolio:
    def __init__(self, symbols):
        self.positions = {s: Position() for s in symbols}

    def on_fill(self, fill: Fill) -> None:
        self.positions[fill.symbol].apply(fill.side, fill.qty, fill.px, fill.fee)

    def pnl(self, marks: dict[str, float]) -> float:
        return sum(
            p.realized + (p.unrealized(marks[s]) if s in marks else 0.0)
            for s, p in self.positions.items()
        )

    def gross_notional(self, marks: dict[str, float]) -> float:
        return sum(abs(p.qty) * marks.get(s, p.avg_px) for s, p in self.positions.items())


class RateLimiter:
    def __init__(self, max_per_min: int):
        self.max_per_min = max_per_min
        self._events: deque[float] = deque()

    def allow(self, now: float) -> bool:
        while self._events and now - self._events[0] > 60:
            self._events.popleft()
        if len(self._events) >= self.max_per_min:
            return False
        self._events.append(now)
        return True
