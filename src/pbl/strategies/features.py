"""Признаки для семейств автопоиска: бары старшего ТФ, ATR, EMA, дневные фильтры.

Всё считается по закрытым барам: значение в позиции `j` использует только данные `≤ j`
(накопленные суммы и рекурсии префикс-устойчивы), поэтому подмена будущих баров не меняет
прошлого (`check_no_lookahead`). Кадр — `pbl.engine.market.strategy_frame`: bid/ask M1 плюс
время сервера (`srv_day`, `srv_min`, `srv_wd`).

Бар старшего ТФ — `k` минут часов сервера (`srv_min // k`). Бар закончен, если в нём есть
последняя минута окна: без неё решение нельзя принять «на закрытии» так, как его примет бот.
Решение по бару ТФ — на его последнем M1-баре, заявка исполняется со следующего.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numba
import numpy as np
import numpy.typing as npt
import polars as pl

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
I8 = npt.NDArray[np.int8]
B8 = npt.NDArray[np.bool_]

NAN = float("nan")
ATR_N = 14
TREND_EMA_DAYS = 20
VOL_SHORT_DAYS, VOL_LONG_DAYS = 5, 60
FILTERS = ("none", "trend", "vol_hi", "vol_lo")
"""Фильтры сигналов (SPEC: «фильтры режима»): тренд дня и волатильность относительно нормы."""


def group_edges(key: I64) -> tuple[I64, I64]:
    """Границы подряд идущих групп одного ключа: [start, end)."""
    if key.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cut = np.flatnonzero(np.diff(key) != 0) + 1
    return (
        np.concatenate([[0], cut]).astype(np.int64),
        np.concatenate([cut, [key.size]]).astype(np.int64),
    )


def hhmm(value: str) -> int:
    h, m = value.split(":")
    return int(h) * 60 + int(m)


def rolling_mean(x: F64, n: int) -> F64:
    """Среднее последних `n` значений включая текущее; первые `n − 1` позиций — NaN."""
    out = np.full(x.shape, np.nan)
    if x.size >= n:
        c = np.concatenate([[0.0], np.cumsum(x)])
        out[n - 1 :] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x: F64, n: int) -> F64:
    """Выборочное стандартное отклонение (ddof=0) последних `n` значений включая текущее."""
    out = np.full(x.shape, np.nan)
    if x.size >= n:
        y = x - x[0]  # сдвиг по первому значению: меньше потеря точности, будущее не нужно
        c1 = np.concatenate([[0.0], np.cumsum(y)])
        c2 = np.concatenate([[0.0], np.cumsum(y * y)])
        mean = (c1[n:] - c1[:-n]) / n
        var = (c2[n:] - c2[:-n]) / n - mean * mean
        out[n - 1 :] = np.sqrt(np.maximum(var, 0.0))
    return out


def _ema(x: F64, span: int) -> F64:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    if x.size == 0:
        return out
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def _window_extreme(x: F64, n: int, is_max: bool) -> F64:
    """Максимум (минимум) `n` значений, предшествующих позиции (текущее не входит)."""
    out = np.full(x.shape, np.nan)
    for i in range(n, x.size):
        w = x[i - n : i]
        out[i] = w.max() if is_max else w.min()
    return out


ema = cast(Callable[[F64, int], F64], numba.njit(cache=True, nogil=True)(_ema))
window_extreme = cast(
    Callable[[F64, int, bool], F64], numba.njit(cache=True, nogil=True)(_window_extreme)
)


@dataclass(frozen=True)
class HtfBars:
    """Закрытые бары `k` минут по bid. `bar` — M1-индекс решения (последний M1-бар окна)."""

    k: int
    bar: I64
    day: I64
    minute: I64  # минута закрытия окна по часам сервера (конец окна)
    open: F64
    high: F64
    low: F64
    close: F64

    def __len__(self) -> int:
        return int(self.bar.shape[0])


@dataclass(frozen=True)
class DailyFeatures:
    """По строке на торговый день сервера; значения для дня `d` используют дни строго раньше."""

    day: I64
    trend: I8  # знак (закрытие прошлого дня − EMA20 закрытий до него): +1 / −1 / 0
    vol_ratio: F64  # средний размах за 5 прошлых дней / за 60; NaN, пока истории мало

    def lookup(self, days: I64) -> tuple[I8, F64]:
        i = np.searchsorted(self.day, days)
        ok = (i < self.day.size) & (self.day[np.minimum(i, self.day.size - 1)] == days)
        trend = np.where(ok, self.trend[np.minimum(i, self.day.size - 1)], 0).astype(np.int8)
        vol = np.where(ok, self.vol_ratio[np.minimum(i, self.day.size - 1)], np.nan)
        return trend, vol


class FeatureCache:
    """Кадр баров M1 и лениво считаемые признаки (по одному кэшу на инструмент)."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.day: I64 = frame["srv_day"].to_numpy().astype(np.int64)
        self.minute: I64 = frame["srv_min"].to_numpy().astype(np.int64)
        self.weekday: I64 = frame["srv_wd"].to_numpy().astype(np.int64)
        self.ts: I64 = frame["ts"].dt.epoch("s").to_numpy().astype(np.int64)
        self.bid_o: F64 = frame["bid_o"].to_numpy().astype(np.float64)
        self.bid_h: F64 = frame["bid_h"].to_numpy().astype(np.float64)
        self.bid_l: F64 = frame["bid_l"].to_numpy().astype(np.float64)
        self.bid_c: F64 = frame["bid_c"].to_numpy().astype(np.float64)
        self._htf: dict[int, HtfBars] = {}
        self._atr: dict[tuple[int, int], F64] = {}
        self._daily: DailyFeatures | None = None
        self.extra: dict[object, object] = {}
        """Кэш семейств (диапазоны сессий и т. п.): ключ → значение."""

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    # --- старший ТФ ---------------------------------------------------------------------

    def htf(self, k: int) -> HtfBars:
        if k not in self._htf:
            self._htf[k] = self._resample(k)
        return self._htf[k]

    def _resample(self, k: int) -> HtfBars:
        if k < 1 or 1440 % k:
            raise ValueError("ТФ в минутах должен делить сутки")
        key = self.day * 1440 + self.minute // k
        starts, ends = group_edges(key)
        last = ends - 1
        done = self.minute[last] % k == k - 1
        starts, ends, last = starts[done], ends[done], last[done]
        return HtfBars(
            k=k,
            bar=last,
            day=self.day[last],
            minute=self.minute[last] + 1,
            open=self.bid_o[starts],
            high=np.maximum.reduceat(self.bid_h, starts) if starts.size else np.zeros(0),
            low=np.minimum.reduceat(self.bid_l, starts) if starts.size else np.zeros(0),
            close=self.bid_c[last],
        )

    def atr(self, k: int, n: int = ATR_N) -> F64:
        """Средний истинный диапазон `n` баров ТФ `k` (значения в позициях `htf(k)`)."""
        if (k, n) not in self._atr:
            h = self.htf(k)
            prev = np.concatenate([[np.nan], h.close[:-1]])
            tr = np.fmax(h.high - h.low, np.fmax(np.abs(h.high - prev), np.abs(h.low - prev)))
            tr = np.where(np.isnan(prev), h.high - h.low, tr)
            self._atr[(k, n)] = rolling_mean(tr, n)
        return self._atr[(k, n)]

    def latest_closed(self, k: int, bars: I64) -> I64:
        """Индекс последнего бара ТФ `k`, решение по которому принято не позже M1-бара `bars`."""
        return np.searchsorted(self.htf(k).bar, bars, side="right").astype(np.int64) - 1

    # --- дневные признаки ---------------------------------------------------------------

    def daily(self) -> DailyFeatures:
        if self._daily is None:
            starts, ends = group_edges(self.day)
            hi = np.maximum.reduceat(self.bid_h, starts) if starts.size else np.zeros(0)
            lo = np.minimum.reduceat(self.bid_l, starts) if starts.size else np.zeros(0)
            close = self.bid_c[ends - 1]
            e = ema(close, TREND_EMA_DAYS)
            above = np.sign(close - e).astype(np.int8)
            trend = np.concatenate([[0], above[:-1]]).astype(np.int8)
            rng = hi - lo
            short = rolling_mean(rng, VOL_SHORT_DAYS)
            long_ = rolling_mean(rng, VOL_LONG_DAYS)
            ratio = np.concatenate([[np.nan], (short / long_)[:-1]])
            self._daily = DailyFeatures(self.day[starts], trend, ratio)
        return self._daily

    # --- фильтры сигналов -----------------------------------------------------------------

    def filter_mask(self, name: str, bar: I64, side: I8) -> B8:
        """Какие сигналы (бар решения, сторона) разрешает фильтр `name` (см. `FILTERS`)."""
        if name == "none":
            return np.ones(bar.shape, dtype=np.bool_)
        trend, vol = self.daily().lookup(self.day[bar])
        if name == "trend":
            return np.asarray(trend == side, dtype=np.bool_)
        if name == "vol_hi":
            return np.isfinite(vol) & (vol >= 1.0)
        if name == "vol_lo":
            return np.isfinite(vol) & (vol < 1.0)
        raise ValueError(f"неизвестный фильтр {name!r}")
