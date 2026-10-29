"""Бары из тиков: bid/ask OHLC, число тиков, средний и максимальный спред."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import polars as pl

from pbl.config import REPO_ROOT
from pbl.data.servertime import to_server_naive

BARS_ROOT = REPO_ROOT / "data" / "bars"


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


def load_bars(
    symbol: str, start: dt.date, end: dt.date, *, every: str = "1m", root: Path = BARS_ROOT
) -> pl.DataFrame:
    """Бары из `data/bars/<symbol>/<год>/<every>_utc.parquet` за [start, end] включительно.

    Нет файла за год диапазона — ошибка со списком недостающих лет: данные не достраиваем.
    """
    years = range(start.year, end.year + 1)
    files = {y: root / symbol / str(y) / f"{every}_utc.parquet" for y in years}
    missing = [y for y, p in files.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{symbol}: нет баров {every} за {', '.join(map(str, missing))} в {root}; "
            "pbl data download и pbl data bars"
        )
    lo = dt.datetime.combine(start, dt.time(0), dt.UTC)
    hi = dt.datetime.combine(end + dt.timedelta(days=1), dt.time(0), dt.UTC)
    frames = [pl.read_parquet(p) for p in files.values()]
    return (
        pl.concat(frames)
        .filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))
        .unique(subset="ts", keep="last")
        .sort("ts")
    )
