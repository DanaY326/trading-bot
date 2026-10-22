"""Append-only JSONL log of everything needed to grade a live run afterwards.

One line per event: the book we saw each step, every order sent, cancel, and fill.
`report.py` turns these into fill rates, markouts, and calibrated replay settings.
"""
import json
import os
from datetime import datetime

from .models import Book, Fill, Side


class Recorder:
    def __init__(self, directory: str = "logs", path: str | None = None):
        if path is None:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{datetime.now():%Y-%m-%d_%H%M%S}.jsonl")
        self.path = path
        self._f = open(path, "a", buffering=1)  # line-buffered: a crash loses at most one event

    def _write(self, kind: str, ts: float, **fields) -> None:
        self._f.write(json.dumps({"k": kind, "ts": round(ts, 3), **fields}) + "\n")

    def book(self, b: Book) -> None:
        self._write("book", b.ts, sym=b.symbol, bid=b.bid, ask=b.ask, bsz=b.bid_size, asz=b.ask_size)

    def order(self, ts: float, sym: str, oid: str, side: Side, qty: int, px: float, book: Book | None,
              reason: str) -> None:
        self._write("order", ts, sym=sym, id=oid, side=side.value, qty=qty, px=px, reason=reason,
                    bid=book.bid if book else None, ask=book.ask if book else None)

    def ack(self, ts: float, oid: str, state: str) -> None:
        self._write("ack", ts, id=oid, state=state)

    def cancel(self, ts: float, oid: str) -> None:
        self._write("cancel", ts, id=oid)

    def fill(self, f: Fill) -> None:
        self._write("fill", f.ts, sym=f.symbol, id=f.order_id, side=f.side.value, qty=f.qty,
                    px=f.px, fee=round(f.fee, 6))

    def note(self, ts: float, msg: str) -> None:
        self._write("note", ts, msg=msg)

    def close(self) -> None:
        self._f.close()
