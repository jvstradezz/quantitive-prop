"""Проверки качества тиков: пропуски, дубли, перевёрнутые котировки, выбросы."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

MAX_JUMP_PCT = 2.0  # скачок mid между соседними тиками, %, считается выбросом
GAP_MINUTES = 30  # пауза между тиками внутри торгового дня


@dataclass(frozen=True)
class QualityReport:
    symbol: str
    ticks: int
    days_expected: int
    days_with_data: int
    missing_days: list[date]
    duplicates: int
    crossed: int  # ask < bid
    zero_spread_pct: float
    jumps: int
    gaps: pl.DataFrame  # start, end, minutes — паузы > GAP_MINUTES в будние дни, кроме выходных
    first_ts: str
    last_ts: str


def check_ticks(symbol: str, ticks: pl.DataFrame, expected_days: list[date]) -> QualityReport:
    """`expected_days` — будни периода; праздники тоже попадут в missing_days."""
    if ticks.is_empty():
        return QualityReport(
            symbol, 0, len(expected_days), 0, expected_days, 0, 0, 0.0, 0,
            pl.DataFrame({"start": [], "end": [], "minutes": []}), "", "",
        )  # fmt: skip
    t = ticks.sort("ts", maintain_order=True)
    days_with = set(t.select(pl.col("ts").dt.date().unique()).to_series().to_list())
    weekdays = [d for d in expected_days if d.weekday() < 5]
    missing = [d for d in weekdays if d not in days_with]
    dups = t.height - t.unique(subset=["ts", "bid", "ask"]).height
    crossed = t.filter(pl.col("ask") < pl.col("bid")).height
    zero = t.filter(pl.col("ask") == pl.col("bid")).height / t.height * 100
    mid = (pl.col("ask") + pl.col("bid")) / 2
    jumps = t.filter((mid / mid.shift(1) - 1).abs() * 100 > MAX_JUMP_PCT).height
    g = (
        t.select(
            pl.col("ts").shift(1).alias("start"),
            pl.col("ts").alias("end"),
            (pl.col("ts") - pl.col("ts").shift(1)).dt.total_minutes().alias("minutes"),
        )
        .filter(pl.col("minutes") > GAP_MINUTES)
        # паузы, которые не пересекают выходные/ночь пятница→воскресенье
        .filter(pl.col("end").dt.date() == pl.col("start").dt.date())
    )
    return QualityReport(
        symbol,
        t.height,
        len(weekdays),
        len(days_with),
        missing,
        dups,
        crossed,
        round(zero, 3),
        jumps,
        g,
        str(t["ts"].min()),
        str(t["ts"].max()),
    )
