"""Семейства стратегий для автопоиска (ROADMAP M4): пробой, возврат к среднему, тренд, сессии.

Каждое семейство — замороженный датакласс с параметрами и методом `signals(bars)` (протокол
`Strategy`, без доступа к будущему: проверяется `check_no_lookahead`). Общие правила:

- не больше одного входа в день на инструмент (первое срабатывание): сделки не пересекаются,
  а скан и событийный бэктест дают один и тот же результат;
- вход по рынку на следующем M1-баре после закрытия бара ТФ; стоп — `sl_atr × ATR(14)` ТФ
  от цены входа (стоп обязателен, SPEC B3); тейк в сигнале не задан — его добавляют выходы
  поиска (`Signals.tp_r`), как и закрытие позиции к концу сессии (`close_at`);
- вход только в окне сессии `[open, last_entry)` по часам сервера FN (ET + 7 ч, D-019).

Диапазонные семейства — обёртка над `range_session` (идея №1): пробой стоп-ордерами и возврат.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, cast

import numpy as np
import polars as pl

from pbl.schemas.strategy import ParamValue, Timeframe
from pbl.strategies.base import Signals
from pbl.strategies.features import (
    ATR_N,
    I8,
    I64,
    FeatureCache,
    group_edges,
    hhmm,
    rolling_mean,
    rolling_std,
    window_extreme,
)
from pbl.strategies.features import ema as ema_series
from pbl.strategies.range_session import (
    FadeEvents,
    SessionDays,
    breakout_signals,
    fade_events,
    fade_signals,
    session_days,
)

TIMEFRAMES: dict[int, Timeframe] = {1: "M1", 5: "M5", 15: "M15", 30: "M30", 60: "H1"}
UNTIL = "23:00"
"""Закрытие NY (16:00 ET) по серверу FN: в него закрываются позиции диапазонных семейств."""


@dataclass(frozen=True)
class TradeSession:
    """Окно входов и время закрытия позиций; часы сервера FN."""

    name: str
    open: str
    last_entry: str
    close: str


TRADE_SESSIONS: dict[str, TradeSession] = {
    s.name: s
    for s in (
        TradeSession("asia", "03:00", "09:00", "10:00"),  # 20:00–02:00 ET, закрытие 03:00 ET
        TradeSession("london", "10:00", "15:30", "16:30"),  # 03:00–08:30 ET, закрытие 09:30 ET
        TradeSession("ny", "16:30", "22:00", "23:00"),  # 09:30–15:00 ET, закрытие 16:00 ET
    )
}

RANGE_WINDOWS: dict[str, tuple[str, str]] = {
    "asia": ("03:00", "07:00"),
    "london": ("10:00", "12:00"),
    "ny15": ("16:30", "16:45"),
    "ny30": ("16:30", "17:00"),
    "ny60": ("16:30", "17:30"),
}
"""Окна диапазона, как в идее №1 (`range_sweep.SESSIONS`)."""


class FamilyBase:
    """Общее для семейств: ключ из параметров и `signals(bars)` через `signals_from`."""

    family: ClassVar[str]
    tf: int
    session: str

    def params(self) -> dict[str, ParamValue]:
        return {f.name: getattr(self, f.name) for f in fields(cast(Any, self))}

    @property
    def key(self) -> str:
        def fmt(v: ParamValue) -> str:
            return format(v, "g") if isinstance(v, float) else str(v)

        return self.family + "-" + "-".join(f"{k}{fmt(v)}" for k, v in self.params().items())

    @property
    def name(self) -> str:
        return self.key

    @property
    def timeframe(self) -> Timeframe:
        return TIMEFRAMES[self.tf]

    @property
    def close_at(self) -> str:
        return TRADE_SESSIONS[self.session].close

    def signals_from(self, fc: FeatureCache) -> Signals:
        raise NotImplementedError

    def signals(self, bars: pl.DataFrame) -> Signals:
        return self.signals_from(FeatureCache(bars))


def _entries(
    fc: FeatureCache, bar: I64, side: I8, atr: Any, sl_atr: float, in_session: Any
) -> Signals:
    """События бара ТФ → сигналы: рабочая сессия, годный ATR, первое событие дня."""
    ok = in_session & np.isfinite(atr) & (atr > 0)
    bar, side, atr = bar[ok], side[ok], atr[ok]
    order = np.argsort(bar, kind="stable")
    bar, side, atr = bar[order], side[order], atr[order]
    if bar.size:
        day = fc.day[bar]
        first = np.concatenate([[True], day[1:] != day[:-1]])
        bar, side, atr = bar[first], side[first], atr[first]
    return Signals.build(bar, side, sl_dist=sl_atr * atr, order="market")


def _session_mask(minute_end: I64, s: TradeSession) -> Any:
    return (minute_end > hhmm(s.open)) & (minute_end <= hhmm(s.last_entry))


def _merge(h_bar: I64, long_: Any, short: Any) -> tuple[I64, I8, Any]:
    """Булевы события «лонг» и «шорт» по барам ТФ → (индексы баров ТФ, стороны, маска)."""
    idx = np.concatenate([np.flatnonzero(long_), np.flatnonzero(short)]).astype(np.int64)
    side = np.concatenate([np.ones(int(long_.sum()), np.int8), -np.ones(int(short.sum()), np.int8)])
    return idx, side, h_bar[idx]


@dataclass(frozen=True)
class Donchian(FamilyBase):
    """Пробой канала: закрытие бара ТФ выше максимума (ниже минимума) `lookback` прошлых баров."""

    family: ClassVar[str] = "donchian"
    tf: int = 15
    lookback: int = 40
    sl_atr: float = 2.0
    session: str = "ny"

    def signals_from(self, fc: FeatureCache) -> Signals:
        h, atr = fc.htf(self.tf), fc.atr(self.tf, ATR_N)
        hh = window_extreme(h.high, self.lookback, True)
        ll = window_extreme(h.low, self.lookback, False)
        idx, side, bar = _merge(h.bar, h.close > hh, h.close < ll)
        s = TRADE_SESSIONS[self.session]
        return _entries(fc, bar, side, atr[idx], self.sl_atr, _session_mask(h.minute[idx], s))


@dataclass(frozen=True)
class MeanReversionZ(FamilyBase):
    """Возврат к среднему: закрытие бара ТФ дальше `z` стандартных отклонений от средней."""

    family: ClassVar[str] = "meanrev_z"
    tf: int = 15
    lookback: int = 20
    z: float = 2.0
    sl_atr: float = 2.0
    session: str = "ny"

    def signals_from(self, fc: FeatureCache) -> Signals:
        h, atr = fc.htf(self.tf), fc.atr(self.tf, ATR_N)
        mu, sd = rolling_mean(h.close, self.lookback), rolling_std(h.close, self.lookback)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(sd > 0, (h.close - mu) / sd, np.nan)
        idx, side, bar = _merge(h.bar, z <= -self.z, z >= self.z)
        s = TRADE_SESSIONS[self.session]
        return _entries(fc, bar, side, atr[idx], self.sl_atr, _session_mask(h.minute[idx], s))


@dataclass(frozen=True)
class TrendEma(FamilyBase):
    """Тренд: пересечение быстрой и медленной EMA закрытий бара ТФ."""

    family: ClassVar[str] = "trend_ema"
    tf: int = 15
    fast: int = 9
    slow: int = 21
    sl_atr: float = 2.0
    session: str = "ny"

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast < slow")

    def signals_from(self, fc: FeatureCache) -> Signals:
        h, atr = fc.htf(self.tf), fc.atr(self.tf, ATR_N)
        f, s_ = ema_series(h.close, self.fast), ema_series(h.close, self.slow)
        above = f > s_
        prev = np.concatenate([[above[0] if above.size else False], above[:-1]])
        warm = np.arange(len(h)) >= self.slow  # EMA прогрелась
        idx, side, bar = _merge(h.bar, above & ~prev & warm, ~above & prev & warm)
        s = TRADE_SESSIONS[self.session]
        return _entries(fc, bar, side, atr[idx], self.sl_atr, _session_mask(h.minute[idx], s))


@dataclass(frozen=True)
class SessionDrive(FamilyBase):
    """Первые `minutes` минут сессии задают направление: продолжение или разворот."""

    family: ClassVar[str] = "session_drive"
    tf: int = 1
    minutes: int = 30
    mode: str = "continue"  # continue | reverse
    sl_atr: float = 2.0
    session: str = "ny"

    def __post_init__(self) -> None:
        if self.mode not in ("continue", "reverse"):
            raise ValueError("mode: continue или reverse")

    def _drive(self, fc: FeatureCache) -> tuple[I64, Any, Any]:
        """(бары решения, ход открытие→решение, ATR(14) 15-минуток) по дням сессии."""
        key = ("drive", self.session, self.minutes)
        if key not in fc.extra:
            o, n = hhmm(TRADE_SESSIONS[self.session].open), self.minutes
            starts, ends = group_edges(fc.day)
            decision: list[int] = []
            move: list[float] = []
            for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
                m = fc.minute[a:b]
                i0 = a + int(np.searchsorted(m, o))
                i1 = a + int(np.searchsorted(m, o + n)) - 1
                if i0 >= b or fc.minute[i0] - o > 2 or i1 <= i0 or fc.minute[i1] < o + n - 3:
                    continue
                decision.append(i1)
                move.append(float(fc.bid_c[i1] - fc.bid_o[i0]))
            bar = np.asarray(decision, dtype=np.int64)
            j = fc.latest_closed(15, bar)
            atr = np.where(j >= 0, fc.atr(15, ATR_N)[np.maximum(j, 0)], np.nan)
            fc.extra[key] = (bar, np.asarray(move), atr)
        return cast("tuple[I64, Any, Any]", fc.extra[key])

    def signals_from(self, fc: FeatureCache) -> Signals:
        bar, move, atr = self._drive(fc)
        sign = np.sign(move).astype(np.int8)
        side = sign if self.mode == "continue" else -sign
        keep = sign != 0
        return _entries(
            fc, bar[keep], side[keep], atr[keep], self.sl_atr, np.ones(int(keep.sum()), bool)
        )


@dataclass(frozen=True)
class RangeBreakout(FamilyBase):
    """Пробой диапазона сессии стоп-ордерами OCO (идея №1); стоп — противоположная граница."""

    family: ClassVar[str] = "range_breakout"
    tf: int = 1
    session: str = "ny30"
    buffer: float = 0.0

    @property
    def close_at(self) -> str:
        return UNTIL

    def _days(self, fc: FeatureCache) -> Any:
        a, b = RANGE_WINDOWS[self.session]
        key = ("range", a, b)
        if key not in fc.extra:
            fc.extra[key] = session_days(fc.frame, a, b, UNTIL)
        return fc.extra[key]

    def signals_from(self, fc: FeatureCache) -> Signals:
        return breakout_signals(self._days(fc), buffer=self.buffer, tp_r=None)


@dataclass(frozen=True)
class RangeFade(FamilyBase):
    """Возврат в диапазон после ложного пробоя (идея №1); стоп — за экстремумом выноса."""

    family: ClassVar[str] = "range_fade"
    tf: int = 1
    session: str = "ny30"
    buffer: float = 0.0

    @property
    def close_at(self) -> str:
        return UNTIL

    def signals_from(self, fc: FeatureCache) -> Signals:
        a, b = RANGE_WINDOWS[self.session]
        key = ("range", a, b)
        if key not in fc.extra:
            fc.extra[key] = session_days(fc.frame, a, b, UNTIL)
        sd = cast(SessionDays, fc.extra[key])
        ekey = ("fade", a, b)
        if ekey not in fc.extra:
            fc.extra[ekey] = fade_events(fc.frame, sd)
        return fade_signals(cast(FadeEvents, fc.extra[ekey]), sd, buffer=self.buffer, tp_r=None)


Family = Donchian | MeanReversionZ | TrendEma | SessionDrive | RangeBreakout | RangeFade
FAMILY_CLASSES: dict[str, type[Family]] = {
    c.family: c
    for c in (Donchian, MeanReversionZ, TrendEma, SessionDrive, RangeBreakout, RangeFade)
}
