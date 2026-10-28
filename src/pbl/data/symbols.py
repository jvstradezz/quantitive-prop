"""Сопоставление символов FN ↔ Dukascopy и масштаб цены в .bi5."""

from __future__ import annotations

from typing import NamedTuple


class DukaSymbol(NamedTuple):
    duka: str  # имя папки в datafeed.dukascopy.com
    divisor: float  # цена в bi5 = целое; делим на divisor


# Масштаб проверен по живым тикам 2025-09-15 (US30 46610301 → 46610.301, EURUSD 116274 → 1.16274).
SYMBOLS: dict[str, DukaSymbol] = {
    "NAS100": DukaSymbol("USATECHIDXUSD", 1000.0),
    "US30": DukaSymbol("USA30IDXUSD", 1000.0),
    "SPX500": DukaSymbol("USA500IDXUSD", 1000.0),
    "GER30": DukaSymbol("DEUIDXEUR", 1000.0),
    "XAUUSD": DukaSymbol("XAUUSD", 1000.0),
    "XAGUSD": DukaSymbol("XAGUSD", 1000.0),
    "EURUSD": DukaSymbol("EURUSD", 100000.0),
    "GBPUSD": DukaSymbol("GBPUSD", 100000.0),
    "USDJPY": DukaSymbol("USDJPY", 1000.0),
}
