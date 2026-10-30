"""Сделки `Trade` → таблица для гейтов и дневные матрицы на календаре Пн–Пт.

День сделки — по серверу фирмы по времени открытия, суббота и воскресенье — в пятницу (D-022).
Колонки: `day` (ordinal), `wd` (0 — пн), `minute` (минута дня сервера при входе), `symbol`,
`side` (+1/−1), `r_net`, `r_gross`, `cost_r` (Σ издержек, R), `hold_sec`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import polars as pl

from pbl.prop.servertime import to_server
from pbl.schemas import ServerTime, Trade

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

TRADE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "day": pl.Int64,
    "wd": pl.Int8,
    "minute": pl.Int16,
    "symbol": pl.String,
    "side": pl.Int8,
    "r_net": pl.Float64,
    "r_gross": pl.Float64,
    "cost_r": pl.Float64,
    "hold_sec": pl.Int64,
}


def trade_day(ts_utc: dt.datetime, server: ServerTime) -> tuple[dt.date, int]:
    """(торговый день, минута дня сервера); выходные → пятница."""
    local = to_server(ts_utc, server)
    day = local.date()
    day -= dt.timedelta(days=max(day.weekday() - 4, 0))
    return day, local.hour * 60 + local.minute


def trades_frame(trades: Sequence[Trade], server: ServerTime) -> pl.DataFrame:
    if not trades:
        return pl.DataFrame(schema=TRADE_SCHEMA)
    days, minutes = zip(*(trade_day(t.open_utc, server) for t in trades), strict=True)
    return pl.DataFrame(
        {
            "day": [d.toordinal() for d in days],
            "wd": [d.weekday() for d in days],
            "minute": list(minutes),
            "symbol": [t.symbol for t in trades],
            "side": [1 if t.side == "long" else -1 for t in trades],
            "r_net": [t.r_net for t in trades],
            "r_gross": [t.r_gross for t in trades],
            "cost_r": [
                t.cost_spread_r + t.cost_comm_r + t.cost_swap_r + t.cost_slip_r for t in trades
            ],
            "hold_sec": [t.hold_sec for t in trades],
        },
        schema=TRADE_SCHEMA,
    )


def window(t: pl.DataFrame, lo: int | None = None, hi: int | None = None) -> pl.DataFrame:
    """Сделки с `lo ≤ day < hi` (ordinal)."""
    if lo is not None:
        t = t.filter(pl.col("day") >= lo)
    if hi is not None:
        t = t.filter(pl.col("day") < hi)
    return t


def daily(t: pl.DataFrame, calendar: I64) -> tuple[F64, F64]:
    """Сумма `r_net` и число сделок по дням `calendar`; сделки вне календаря отбрасываются."""
    r = np.zeros(len(calendar))
    n = np.zeros(len(calendar))
    if t.is_empty():
        return r, n
    day = t["day"].to_numpy()
    pos = np.searchsorted(calendar, day)
    ok = pos < len(calendar)
    ok[ok] = calendar[pos[ok]] == day[ok]
    np.add.at(r, pos[ok], t["r_net"].to_numpy()[ok])
    np.add.at(n, pos[ok], 1.0)
    return r, n


def daily_matrix(
    tables: Mapping[str, pl.DataFrame], keys: Sequence[str], calendar: I64
) -> tuple[F64, F64]:
    """T × C: сумма `r_net` и число сделок по дням для каждого варианта."""
    r = np.zeros((len(calendar), len(keys)))
    n = np.zeros_like(r)
    for j, key in enumerate(keys):
        r[:, j], n[:, j] = daily(tables[key], calendar)
    return r, n
