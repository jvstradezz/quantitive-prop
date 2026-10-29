"""Эталонные стратегии для проверки движка (не кандидаты в демо).

`OpeningRangeBreakout` — простой пробой диапазона: нужен тестам движка и прогону на реальных
барах. Варианты идеи №1 (Range bot) кодит тред идеи в `research/`, а не здесь.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from pbl.strategies.base import Signals


def hhmm(value: str) -> int:
    h, m = value.split(":")
    return int(h) * 60 + int(m)


@dataclass(frozen=True)
class OpeningRangeBreakout:
    """Диапазон [range_start, range_end) по серверу; на закрытии последнего бара диапазона —
    OCO buy stop на high и sell stop на low (по bid, как на графике MT5) до `until`.
    Стоп — противоположная граница диапазона, тейк — `tp_r` от фактического риска.

    По умолчанию — первые 30 минут NY: 9:30–10:00 ET = 16:30–17:00 сервера FN (ET + 7).
    """

    range_start: str = "16:30"
    range_end: str = "17:00"
    until: str = "23:00"
    tp_r: float = 1.0
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    name: str = "orb"

    def signals(self, bars: pl.DataFrame) -> Signals:
        a, b, z = hhmm(self.range_start), hhmm(self.range_end), hhmm(self.until)
        day = bars["srv_day"].to_numpy()
        minute = bars["srv_min"].to_numpy()
        wd = bars["srv_wd"].to_numpy()
        ts = bars["ts"].dt.epoch("s").to_numpy()
        hi, lo = bars["bid_h"].to_numpy(), bars["bid_l"].to_numpy()
        out_bar: list[int] = []
        out_hi: list[float] = []
        out_lo: list[float] = []
        out_exp: list[int] = []
        # бар решения: первый бар дня с минутой ≥ range_end − 1 (известен на своём закрытии)
        cur_day, r_hi, r_lo, done = -1, -np.inf, np.inf, True
        for i in range(bars.height):
            if day[i] != cur_day:
                cur_day, r_hi, r_lo, done = int(day[i]), -np.inf, np.inf, False
            if done or wd[i] not in self.weekdays:
                continue
            in_range = a <= minute[i] < b
            if in_range:
                r_hi, r_lo = max(r_hi, float(hi[i])), min(r_lo, float(lo[i]))
            if minute[i] >= b - 1:
                done = True
                if np.isfinite(r_hi) and r_hi > r_lo and minute[i] < z:
                    out_bar.append(i)
                    out_hi.append(r_hi)
                    out_lo.append(r_lo)
                    out_exp.append(int(ts[i]) + (z - int(minute[i])) * 60)
        n = len(out_bar)
        bar = np.repeat(np.asarray(out_bar, dtype=np.int64), 2)
        side = np.tile(np.asarray([1, -1], dtype=np.int8), n)
        level = np.column_stack([out_hi, out_lo]).ravel() if n else np.zeros(0)
        sl = np.column_stack([out_lo, out_hi]).ravel() if n else np.zeros(0)
        return Signals.build(
            bar,
            side,
            sl,
            order="stop",
            level=level,
            expire_s=np.repeat(np.asarray(out_exp, dtype=np.int64), 2),
            tp_r=self.tp_r,
            oco=np.repeat(np.arange(n, dtype=np.int64), 2),
        )
