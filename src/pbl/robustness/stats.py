"""Метрики таблицы сделок для гейтов: S1, Sharpe дня, годы, инструменты (VALIDATION §3–4).

Таблица сделок — polars с колонками `day` (ordinal торгового дня сервера), `r_net`, `r_gross`,
`cost_r` (Σ издержек в R), для годов и инструментов — ещё `symbol` (см. `robustness.trades`).
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import numpy.typing as npt
import polars as pl

from pbl.robustness.thresholds import (
    COST_MULT,
    MIN_YEAR_TRADES,
    YEAR_SHARE,
    YEARS_POSITIVE,
)

F64 = npt.NDArray[np.float64]
_EPOCH_ORD = dt.date(1970, 1, 1).toordinal()


def series_sharpe(x: F64) -> float:
    """Sharpe дневного ряда (ddof=1) без годовой нормировки; 0 при нулевом разбросе."""
    s = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return float(x.mean() / s) if s > 1e-12 else 0.0


def moments(x: F64) -> dict[str, float]:
    """Sharpe (ddof=1), асимметрия и эксцесс (нормальное = 3) дневного ряда — вход DSR."""
    sr = series_sharpe(x)
    sd = float(x.std(ddof=0)) if x.size else 0.0
    if sd <= 1e-12:
        return {"sharpe": sr, "skew": 0.0, "kurt": 3.0}
    z = (x - x.mean()) / sd
    return {"sharpe": sr, "skew": float(np.mean(z**3)), "kurt": float(np.mean(z**4))}


def s1_stats(t: pl.DataFrame) -> dict[str, float]:
    """Сделки → n, PF, mean R, E при издержках × 1.5, доля издержек от валовой (D-025)."""
    n = t.height
    if n == 0:
        return {"n": 0, "pf": 0.0, "mean_r": 0.0, "e15": 0.0, "cost_share": math.nan}
    r = t["r_net"].to_numpy()
    gross, cost = t["r_gross"].to_numpy(), t["cost_r"].to_numpy()
    wins, losses = float(r[r > 0].sum()), -float(r[r < 0].sum())
    g = float(gross.sum())
    return {
        "n": n,
        "pf": wins / losses if losses > 0 else math.inf,
        "mean_r": float(r.mean()),
        "e15": float((gross - COST_MULT * cost).mean()),
        "cost_share": float(cost.sum() / g) if g > 0 else math.nan,
    }


def edge_score(t: pl.DataFrame) -> float:
    """`mean R × √n` — сравнение варианта с фильтром и без него (VALIDATION §4.5 п. 2)."""
    if t.is_empty():
        return 0.0
    return float(t["r_net"].to_numpy().mean()) * math.sqrt(t.height)


def with_year(t: pl.DataFrame) -> pl.DataFrame:
    return t.with_columns(
        (pl.col("day") - _EPOCH_ORD).cast(pl.Int32).cast(pl.Date).dt.year().alias("year")
    )


def year_stability(t: pl.DataFrame) -> tuple[bool, pl.DataFrame]:
    """E > 0 минимум в 2/3 лет, где ≥ 20 сделок; ни один год не даёт > 50% чистой прибыли (P-05)."""
    if t.is_empty():
        return False, pl.DataFrame()
    by = (
        with_year(t)
        .group_by("year")
        .agg(pl.len().alias("n"), pl.col("r_net").sum().alias("sum_r"), pl.col("r_net").mean())
        .sort("year")
        .with_columns(pl.col("year").cast(pl.Int64))
    )
    counted = by.filter(pl.col("n") >= MIN_YEAR_TRADES)
    if counted.is_empty():
        return False, by
    share_ok = True
    total = float(by["sum_r"].sum())
    if total > 0:
        share_ok = float(by["sum_r"].to_numpy().max()) / total <= YEAR_SHARE
    pos = int((counted["r_net"] > 0).sum())
    return (pos >= math.ceil(YEARS_POSITIVE * counted.height) and share_ok), by


def instrument_edges(t: pl.DataFrame) -> dict[str, tuple[int, float]]:
    """Инструмент → (сделок, mean R)."""
    g = t.group_by("symbol").agg(pl.len().alias("n"), pl.col("r_net").mean().alias("m"))
    return {row["symbol"]: (row["n"], row["m"]) for row in g.iter_rows(named=True)}


def positive_instruments(t: pl.DataFrame, min_trades: int) -> int:
    """Сколько инструментов с ≥ `min_trades` сделок и чистым E > 0 (VALIDATION §4.6)."""
    return sum(1 for n, m in instrument_edges(t).values() if n >= min_trades and m > 0)
