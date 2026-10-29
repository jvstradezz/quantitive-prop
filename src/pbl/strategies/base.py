"""Интерфейс стратегий: сигналы по закрытым барам, без доступа к будущему (F-STRAT-1).

Стратегия получает кадр баров и возвращает `Signals`: решение по закрытию бара `bar` использует
только бары `≤ bar`, заявка исполняется движком не раньше бара `bar + 1`. Цены заявок, стопов и
тейков — как в MT5: buy-заявки и стоп шорта срабатывают по ask, sell-заявки и стоп лонга — по bid.

Кадр баров (`pbl.engine.market.strategy_frame`): `ts` (UTC), `bid_o…bid_c`, `ask_o…ask_c`,
`srv_day` (ordinal торгового дня сервера), `srv_min` (минута дня сервера), `srv_wd` (0 — пн).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
I8 = npt.NDArray[np.int8]

MARKET, STOP, LIMIT = 0, 1, 2
ORDER_TYPES = {"market": MARKET, "stop": STOP, "limit": LIMIT}
NAN = float("nan")


@dataclass(frozen=True)
class Signals:
    """Колонки сигналов (одна строка — одна заявка)."""

    bar: I64  # бар, по закрытию которого принято решение
    side: I8  # +1 лонг, −1 шорт
    order: I8  # MARKET / STOP / LIMIT
    level: F64  # цена stop/limit-заявки; NaN для market
    expire_s: I64  # заявка живёт, пока ts бара < expire_s (UTC, секунды); market — только bar + 1,
    # и тоже не позже срока (бар bar + 1 после разрыва данных может быть уже другим днём)
    sl: F64  # уровень стопа; NaN — стоп задан дистанцией `sl_dist` от цены исполнения
    sl_dist: F64
    tp: F64  # уровень тейка; NaN — тейк в R от фактического риска `tp_r` (NaN — без тейка)
    tp_r: F64
    oco: I64  # группа OCO: исполнение одной заявки отменяет остальные группы; −1 — нет

    def __len__(self) -> int:
        return int(self.bar.shape[0])

    def __post_init__(self) -> None:
        n = len(self)
        for f in fields(self):
            if getattr(self, f.name).shape != (n,):
                raise ValueError(f"колонка {f.name}: длина не совпадает с bar")
        if n and not np.isin(self.side, (-1, 1)).all():
            raise ValueError("side: только +1 или −1")
        if n and not np.isin(self.order, (MARKET, STOP, LIMIT)).all():
            raise ValueError("order: market, stop или limit")
        pending = self.order != MARKET
        if np.isnan(self.level[pending]).any():
            raise ValueError("у stop/limit-заявки нужен level")
        no_sl = np.isnan(self.sl) & ~(self.sl_dist > 0)
        if no_sl.any():
            raise ValueError("стоп обязателен (SPEC B3): sl или sl_dist > 0")

    @classmethod
    def empty(cls) -> Signals:
        return cls.build([], [], [])

    @classmethod
    def build(
        cls,
        bar: Sequence[int] | I64,
        side: Sequence[int] | npt.NDArray[np.integer],
        sl: Sequence[float] | F64 | None = None,
        *,
        order: Sequence[str] | str = "market",
        level: Sequence[float] | F64 | None = None,
        expire_s: Sequence[int] | I64 | None = None,
        sl_dist: Sequence[float] | F64 | None = None,
        tp: Sequence[float] | F64 | None = None,
        tp_r: Sequence[float] | F64 | float | None = None,
        oco: Sequence[int] | I64 | None = None,
    ) -> Signals:
        """Сборка из списков; пропущенные колонки — NaN / −1 / без срока."""
        b = np.asarray(bar, dtype=np.int64)
        n = b.shape[0]

        def f64(x: Sequence[float] | F64 | float | None) -> F64:
            if x is None:
                return np.full(n, np.nan)
            if isinstance(x, float | int):
                return np.full(n, float(x))
            return np.asarray(x, dtype=np.float64)

        kinds = [order] * n if isinstance(order, str) else list(order)
        return cls(
            bar=b,
            side=np.asarray(side, dtype=np.int8),
            order=np.asarray([ORDER_TYPES[k] for k in kinds], dtype=np.int8).reshape(n),
            level=f64(level),
            expire_s=(
                np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
                if expire_s is None
                else np.asarray(expire_s, dtype=np.int64)
            ),
            sl=f64(sl),
            sl_dist=f64(sl_dist),
            tp=f64(tp),
            tp_r=f64(tp_r),
            oco=np.full(n, -1, dtype=np.int64) if oco is None else np.asarray(oco, np.int64),
        )

    def take(self, mask: npt.NDArray[np.bool_]) -> Signals:
        return Signals(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    def sorted(self) -> Signals:
        order = np.argsort(self.bar, kind="stable")
        return Signals(**{f.name: getattr(self, f.name)[order] for f in fields(self)})


class Strategy(Protocol):
    """Стратегия: параметры — в самом объекте, данные — только через `signals`."""

    @property
    def name(self) -> str: ...

    def signals(self, bars: pl.DataFrame) -> Signals: ...


PRICE_COLS = tuple(f"{s}_{x}" for s in ("bid", "ask") for x in "ohlc")


def check_no_lookahead(
    strategy: Strategy, bars: pl.DataFrame, cuts: Sequence[int], seed: int = 0
) -> None:
    """Подмена будущих баров не меняет прошлых сигналов (ARCHITECTURE §9).

    Для каждого `k` из `cuts` цены баров `≥ k` случайно искажаются; сигналы с `bar < k`
    обязаны совпасть с исходными во всех колонках. Нарушение — AssertionError.
    Подглядывание на один бар видно только у самой границы `k`, поэтому точек нужно много.
    """
    rng = np.random.default_rng(seed)
    base = strategy.signals(bars)
    for k in cuts:
        if not 0 < k < bars.height:
            raise ValueError(f"cut {k} вне диапазона баров")
        shock = np.ones(bars.height)
        shock[k:] = rng.uniform(0.9, 1.1, bars.height - k)
        noisy = bars.with_columns(pl.col(c) * pl.Series(shock) for c in PRICE_COLS)
        got = strategy.signals(noisy)
        a, b = base.take(base.bar < k), got.take(got.bar < k)
        for f in fields(Signals):
            x, y = getattr(a, f.name), getattr(b, f.name)
            if x.shape != y.shape or not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                raise AssertionError(
                    f"{strategy.name}: сигналы до бара {k} зависят от будущего ({f.name})"
                )
