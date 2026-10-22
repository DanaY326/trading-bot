import math
from collections import deque


class EwmaVol:
    """EWMA of squared log returns on irregularly spaced samples, per-second units."""

    def __init__(self, halflife_s: float):
        self.halflife_s = halflife_s
        self.var: float | None = None
        self.samples = 0
        self._last_t: float | None = None
        self._last_px: float | None = None

    def update(self, t: float, px: float) -> None:
        if self._last_t is None:
            self._last_t, self._last_px = t, px
            return
        dt = t - self._last_t
        if dt <= 0:
            return
        r = math.log(px / self._last_px)
        inst_var = r * r / dt
        alpha = 1 - 0.5 ** (dt / self.halflife_s)
        self.var = inst_var if self.var is None else (1 - alpha) * self.var + alpha * inst_var
        self.samples += 1
        self._last_t, self._last_px = t, px

    @property
    def sigma_per_sec(self) -> float | None:
        return None if self.var is None else math.sqrt(self.var)


class JumpDetector:
    """Flags a squeeze/dump when price moves more than a threshold inside a short window."""

    def __init__(self, window_s: float, threshold_bps: float, cooldown_s: float):
        self.window_s = window_s
        self.threshold = threshold_bps / 1e4
        self.cooldown_s = cooldown_s
        self._hist: deque[tuple[float, float]] = deque()
        self._cooldown_until = 0.0

    def update(self, t: float, px: float) -> None:
        self._hist.append((t, px))
        # Keep the newest sample at or before the window start, so sparse data
        # (e.g. one quote every 15s) still compares against something.
        while len(self._hist) > 1 and t - self._hist[1][0] >= self.window_s:
            self._hist.popleft()
        lo = min(p for _, p in self._hist)
        hi = max(p for _, p in self._hist)
        if math.log(hi / lo) > self.threshold:
            self._cooldown_until = t + self.cooldown_s

    def active(self, t: float) -> bool:
        return t < self._cooldown_until
