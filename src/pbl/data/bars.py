"""Бары из тиков: bid/ask OHLC, число тиков, средний и максимальный спред."""

from __future__ import annotations

from typing import Literal

import polars as pl

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
