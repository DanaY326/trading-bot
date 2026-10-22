from abc import ABC, abstractmethod

from ..models import Book, Fill, Session, Side


class Broker(ABC):
    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    def session(self) -> Session: ...

    @abstractmethod
    def get_books(self, symbols) -> dict[str, Book]: ...

    @abstractmethod
    def place(self, symbol: str, side: Side, qty: int, px: float) -> str | None:
        """Submit a DAY limit order. Returns the order id, or None if rejected."""

    @abstractmethod
    def cancel(self, order_id: str) -> None: ...

    @abstractmethod
    def poll(self) -> tuple[list[Fill], set[str]]:
        """Fills since the last call, and the ids of orders still working."""

    def poll_acks(self) -> list[tuple[str, float, str]]:
        """(order id, time, "live" | "done") since the last call, where the broker reports it.

        "done" means the order left the book without a fill (cancel took effect, expired,
        rejected). Used only to measure how long orders were really resting.
        """
        return []
