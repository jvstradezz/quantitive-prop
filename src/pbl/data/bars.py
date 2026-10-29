"""Бары из тиков: bid/ask OHLC, число тиков, средний и максимальный спред."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import polars as pl

from pbl.data.candles import BARS_ROOT, bars_path
from pbl.data.servertime import to_server_naive


def build_bars(
    ticks: pl.DataFrame, every: str = "1m", clock: Literal["utc", "server"] = "utc"
) -> pl.DataFrame:
    """Агрегация тиков в бары.

    `clock="utc"` — ось `ts` в UTC; `clock="server"` — настенное время сервера FN (для D1 по
    торговому дню проп-правил). Окно левое-закрытое, метка — начало бара.
    """
    if ticks.is_empty():
        raise ValueError("нет тиков")
    t = ticks.sort("ts")
    if clock == "server":
        t = t.with_columns(to_server_naive(pl.col("ts")).alias("ts"))
    else:
        t = t.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
    t = t.with_columns((pl.col("ask") - pl.col("bid")).alias("spread"))
    return (
        t.group_by_dynamic("ts", every=every, closed="left", label="left")
        .agg(
            pl.col("bid").first().alias("bid_o"),
            pl.col("bid").max().alias("bid_h"),
            pl.col("bid").min().alias("bid_l"),
            pl.col("bid").last().alias("bid_c"),
            pl.col("ask").first().alias("ask_o"),
            pl.col("ask").max().alias("ask_h"),
            pl.col("ask").min().alias("ask_l"),
            pl.col("ask").last().alias("ask_c"),
            pl.len().alias("ticks"),
            pl.col("spread").mean().alias("spread_mean"),
            pl.col("spread").max().alias("spread_max"),
        )
        .sort("ts")
    )


def load_bars(symbol: str, start: dt.date, end: dt.date, *, root: Path = BARS_ROOT) -> pl.DataFrame:
    """M1-бары свечей из `data/bars/<symbol>/<год>/1m_utc_candles.parquet` за [start, end].

    Нет файла за год диапазона — ошибка со списком недостающих лет: данные не достраиваем.
    """
    files = {y: bars_path(symbol, y, root) for y in range(start.year, end.year + 1)}
    missing = [y for y, p in files.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{symbol}: нет баров M1 за {', '.join(map(str, missing))} в {root}; pbl data candles"
        )
    lo = dt.datetime.combine(start, dt.time(0), dt.UTC)
    hi = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0), dt.UTC)
    return (
        pl.concat([pl.read_parquet(p) for p in files.values()])
        .filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))
        .unique(subset="ts", keep="last")
        .sort("ts")
    )
