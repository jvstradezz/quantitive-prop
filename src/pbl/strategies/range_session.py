"""Диапазон сессии: пробой и возврат (IDEA-0001, «Range bot по мотивам Sniper Range Bot V2»).

Окно диапазона задаётся в часах сервера (у FN ET + 7 ч круглый год, D-019): high/low по bid за
окно. Две семьи входа:

- `breakout`: после окна — buy stop на `high + buffer · width` и sell stop на `low − buffer · width`
  до `until`; стоп — противоположная граница диапазона. Обе стороны — OCO (сработала одна — вторая
  снята); `sides="long"` или `"short"` оставляет одну заявку.
- `fade`: бар закрывается за границей, следующий закрывается обратно внутри — вход по рынку в
  сторону диапазона; стоп за экстремумом ложного пробоя `± buffer · width`. Не больше одной
  сделки на день и инструмент: берём первый сетап.

Тейк — `tp_r` от фактического риска (None — без тейка, выход по времени `Exits.close_at`).
`day_sides` — какие стороны разрешены в какой день недели (по серверу), фильтр идеи №1.
Решение на закрытии бара использует только бары `≤ bar`; фильтр ширины — только прошлые дни.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from pbl.strategies.base import Signals
from pbl.strategies.reference import hhmm

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B8 = npt.NDArray[np.bool_]

Side = Literal["long", "short", "both"]
DaySide = Literal["long", "short", "both", "none"]
DaySides = tuple[DaySide, DaySide, DaySide, DaySide, DaySide]
"""Разрешённые стороны по дням недели Пн…Пт."""

AUTHOR_DAY_SIDES: DaySides = ("long", "both", "long", "short", "both")
"""Фильтр дня из отчёта автора Sniper Range Bot V2: Пн лонг, Вт оба, Ср лонг, Чт шорт, Пт оба."""

ATR_DAYS = 14
MIN_COVER = 0.5
"""Окно считаем полным, если в нём есть бары хотя бы за половину минут и в последние 5 минут."""


@dataclass(frozen=True)
class SessionDays:
    """По строке на каждый торговый день сервера с полным окном диапазона."""

    day: I64  # ordinal дня сервера
    weekday: I64  # 0 — пн
    bar: I64  # индекс бара решения: последний бар окна
    high: F64
    low: F64
    expire_s: I64  # заявки живут, пока ts бара < expire_s (UTC, секунды)
    until_bar: I64  # индекс первого бара ≥ expire_s: сетапы ищем строго до него
    atr_prev: F64  # средний дневной размах за прошлые ATR_DAYS дней; NaN, пока истории мало

    def __len__(self) -> int:
        return int(self.bar.shape[0])

    @property
    def width(self) -> F64:
        return self.high - self.low


def _group_edges(key: I64) -> tuple[I64, I64]:
    """Границы подряд идущих групп одинакового ключа: [start, end)."""
    if key.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    cut = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate([[0], cut]).astype(np.int64)
    ends = np.concatenate([cut, [key.size]]).astype(np.int64)
    return starts, ends


def daily_range_prev(bars: pl.DataFrame, days: I64) -> F64:
    """Средний размах bid (high − low) за ATR_DAYS торговых дней перед каждым из `days`."""
    day = bars["srv_day"].to_numpy()
    starts, _ = _group_edges(day)
    if starts.size == 0:
        return np.full(days.shape, np.nan)
    hi = np.maximum.reduceat(bars["bid_h"].to_numpy(), starts)
    lo = np.minimum.reduceat(bars["bid_l"].to_numpy(), starts)
    rng = hi - lo
    ud = day[starts]
    csum = np.concatenate([[0.0], np.cumsum(rng)])
    k = np.searchsorted(ud, days)  # дней строго раньше `days`
    out = np.full(days.shape, np.nan)
    ok = k >= ATR_DAYS
    out[ok] = (csum[k[ok]] - csum[k[ok] - ATR_DAYS]) / ATR_DAYS
    return out


def session_days(bars: pl.DataFrame, start: str, end: str, until: str) -> SessionDays:
    """Диапазоны сессии по барам (кадр `strategy_frame`). Окно и `until` — часы сервера."""
    a, b, z = hhmm(start), hhmm(end), hhmm(until)
    if not 0 <= a < b <= z <= 1440:
        raise ValueError("нужно 0 ≤ range_start < range_end ≤ until ≤ 24:00")
    day = bars["srv_day"].to_numpy()
    minute = bars["srv_min"].to_numpy()
    wd = bars["srv_wd"].to_numpy()
    ts = bars["ts"].dt.epoch("s").to_numpy().astype(np.int64)
    hi, lo = bars["bid_h"].to_numpy(), bars["bid_l"].to_numpy()
    idx = np.flatnonzero((minute >= a) & (minute < b))
    starts, ends = _group_edges(day[idx])
    if starts.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return SessionDays(empty, empty, empty, np.zeros(0), np.zeros(0), empty, empty, np.zeros(0))
    r_hi = np.maximum.reduceat(hi[idx], starts)
    r_lo = np.minimum.reduceat(lo[idx], starts)
    last = idx[ends - 1]
    cover = (ends - starts) / (b - a)
    ok = (cover >= MIN_COVER) & (minute[last] >= b - 5) & (r_hi > r_lo)
    last, r_hi, r_lo = last[ok], r_hi[ok], r_lo[ok]
    expire = ts[last] + (z - minute[last]) * 60
    return SessionDays(
        day=day[last],
        weekday=wd[last],
        bar=last.astype(np.int64),
        high=r_hi,
        low=r_lo,
        expire_s=expire.astype(np.int64),
        until_bar=np.searchsorted(ts, expire).astype(np.int64),
        atr_prev=daily_range_prev(bars, day[last]),
    )


@dataclass(frozen=True)
class FadeEvents:
    """Ложные пробои: бар закрылся за границей, следующий — обратно внутри."""

    side: npt.NDArray[np.int8]  # +1 лонг (вынос вниз), −1 шорт (вынос вверх)
    bar: I64  # бар решения: бар, закрывшийся обратно внутри
    sd: I64  # индекс строки в SessionDays
    extreme: F64  # экстремум выноса (low для лонга, high для шорта)


def fade_events(bars: pl.DataFrame, sd: SessionDays) -> FadeEvents:
    """Первый ложный пробой вверх и первый вниз в каждый день (от бара после окна до `until`)."""
    c = bars["bid_c"].to_numpy()
    hi, lo = bars["bid_h"].to_numpy(), bars["bid_l"].to_numpy()
    ts = bars["ts"].dt.epoch("s").to_numpy().astype(np.int64)
    sides: list[int] = []
    bar: list[int] = []
    idx: list[int] = []
    ext: list[float] = []
    for k in range(len(sd)):
        s, e = int(sd.bar[k]) + 1, int(sd.until_bar[k])
        if e - s < 2:
            continue
        cc = c[s:e]
        for side, out in ((-1, cc > sd.high[k]), (1, cc < sd.low[k])):
            # соседние минуты: после разрыва данных «следующий бар» — уже другой момент
            adjacent = np.diff(ts[s:e]) == 60
            hit = np.flatnonzero(out[:-1] & ~out[1:] & adjacent)
            if hit.size == 0:
                continue
            h = int(hit[0])  # бар за границей; h + 1 закрылся внутри
            first = h
            while first > 0 and out[first - 1]:  # вынос мог начаться раньше
                first -= 1
            lo_i, hi_i = s + first, s + h + 2
            sides.append(side)
            bar.append(s + h + 1)
            idx.append(k)
            ext.append(float(hi[lo_i:hi_i].max() if side < 0 else lo[lo_i:hi_i].min()))
    return FadeEvents(
        side=np.asarray(sides, dtype=np.int8),
        bar=np.asarray(bar, dtype=np.int64),
        sd=np.asarray(idx, dtype=np.int64),
        extreme=np.asarray(ext, dtype=np.float64),
    )


def _allowed(sd: SessionDays, sides: Side, day_sides: DaySides | None) -> tuple[B8, B8]:
    """Маски по дням: можно ли лонг и можно ли шорт."""
    n = len(sd)
    long_ok = np.full(n, sides in ("long", "both"))
    short_ok = np.full(n, sides in ("short", "both"))
    if day_sides is not None:
        wd = np.minimum(sd.weekday, 4)
        weekend = sd.weekday > 4
        table_l = np.array([d in ("long", "both") for d in day_sides])
        table_s = np.array([d in ("short", "both") for d in day_sides])
        long_ok &= table_l[wd] & ~weekend
        short_ok &= table_s[wd] & ~weekend
    return long_ok, short_ok


def width_mask(sd: SessionDays, width_rel: tuple[float, float] | None) -> B8:
    """Фильтр ширины: `width / средний дневной размах за прошлые дни` в [lo, hi]."""
    if width_rel is None:
        return np.ones(len(sd), dtype=np.bool_)
    rel = sd.width / sd.atr_prev
    return np.isfinite(rel) & (rel >= width_rel[0]) & (rel <= width_rel[1])


def breakout_signals(
    sd: SessionDays,
    *,
    buffer: float = 0.0,
    tp_r: float | None = 1.0,
    sides: Side = "both",
    day_sides: DaySides | None = None,
    width_rel: tuple[float, float] | None = None,
) -> Signals:
    long_ok, short_ok = _allowed(sd, sides, day_sides)
    keep = width_mask(sd, width_rel)
    w = sd.width
    pieces: list[tuple[I64, int, F64, F64, I64, I64]] = []
    for side, ok in ((1, long_ok), (-1, short_ok)):
        m = ok & keep
        if side > 0:
            level, sl = sd.high[m] + buffer * w[m], sd.low[m]
        else:
            level, sl = sd.low[m] - buffer * w[m], sd.high[m]
        pieces.append((sd.bar[m], side, level, sl, sd.expire_s[m], np.flatnonzero(m)))
    bar = np.concatenate([p[0] for p in pieces])
    side_a = np.concatenate([np.full(p[0].shape, p[1], dtype=np.int8) for p in pieces])
    level_a = np.concatenate([p[2] for p in pieces])
    sl_a = np.concatenate([p[3] for p in pieces])
    exp_a = np.concatenate([p[4] for p in pieces])
    oco = np.concatenate([p[5] for p in pieces])
    # OCO нужна только там, где в день живут обе заявки
    both_days = np.intersect1d(pieces[0][5], pieces[1][5])
    group = np.where(np.isin(oco, both_days), oco, -1)
    return Signals.build(
        bar,
        side_a,
        sl_a,
        order="stop",
        level=level_a,
        expire_s=exp_a,
        tp_r=np.nan if tp_r is None else tp_r,
        oco=group.astype(np.int64),
    )


def fade_signals(
    ev: FadeEvents,
    sd: SessionDays,
    *,
    buffer: float = 0.0,
    tp_r: float | None = 1.0,
    sides: Side = "both",
    day_sides: DaySides | None = None,
    width_rel: tuple[float, float] | None = None,
) -> Signals:
    long_ok, short_ok = _allowed(sd, sides, day_sides)
    keep = width_mask(sd, width_rel)
    ok = np.where(ev.side > 0, long_ok[ev.sd], short_ok[ev.sd]) & keep[ev.sd]
    s, bar, k, ext = ev.side[ok], ev.bar[ok], ev.sd[ok], ev.extreme[ok]
    first = np.lexsort((bar, k))  # день, затем время
    s, bar, k, ext = s[first], bar[first], k[first], ext[first]
    if k.size:
        keep_first = np.concatenate([[True], k[1:] != k[:-1]])  # одна сделка в день
        s, bar, k, ext = s[keep_first], bar[keep_first], k[keep_first], ext[keep_first]
    sl = np.where(s > 0, ext - buffer * sd.width[k], ext + buffer * sd.width[k])
    return Signals.build(
        bar,
        s,
        sl,
        order="market",
        expire_s=sd.expire_s[k],
        tp_r=np.nan if tp_r is None else tp_r,
    )


@dataclass(frozen=True)
class RangeSession:
    """Стратегия идеи №1; параметры — в объекте, данные — только через `signals`."""

    mode: Literal["breakout", "fade"] = "breakout"
    range_start: str = "16:30"
    range_end: str = "17:00"
    until: str = "23:00"
    buffer: float = 0.0
    tp_r: float | None = 1.0
    sides: Side = "both"
    day_sides: DaySides | None = None
    width_rel: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.buffer < 0 or (self.tp_r is not None and self.tp_r <= 0):
            raise ValueError("buffer ≥ 0, tp_r > 0")

    @property
    def name(self) -> str:
        tag = "day" if self.day_sides is not None else "all"
        return f"range-{self.mode}-{self.range_start}-{self.range_end}-{self.sides}-{tag}"

    def signals(self, bars: pl.DataFrame) -> Signals:
        sd = session_days(bars, self.range_start, self.range_end, self.until)
        kw = {
            "buffer": self.buffer,
            "tp_r": self.tp_r,
            "sides": self.sides,
            "day_sides": self.day_sides,
            "width_rel": self.width_rel,
        }
        if self.mode == "breakout":
            return breakout_signals(sd, **kw)  # type: ignore[arg-type]
        return fade_signals(fade_events(bars, sd), sd, **kw)  # type: ignore[arg-type]
