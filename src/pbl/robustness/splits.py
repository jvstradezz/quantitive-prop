"""Хронологические разбиения: IS / holdout и окна walk-forward (VALIDATION §2, P-01, D-032)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

IS_FRACTION = 0.7
WF_TRAIN_MONTHS, WF_TEST_MONTHS, WF_STEP_MONTHS = 24, 6, 6


def add_months(day: dt.date, months: int) -> dt.date:
    """Сдвиг на календарные месяцы; день месяца обрезается по длине месяца."""
    y, m = divmod(day.year * 12 + day.month - 1 + months, 12)
    nxt = dt.date(y + (m + 1) // 12, (m + 1) % 12 + 1, 1)
    last = (nxt - dt.timedelta(days=1)).day
    return dt.date(y, m + 1, min(day.day, last))


def holdout_start(first: dt.date, last: dt.date, is_fraction: float = IS_FRACTION) -> dt.date:
    """Первый день holdout: IS — первые `is_fraction` календарных дней [first, last]."""
    if not 0 < is_fraction < 1 or last <= first:
        raise ValueError("нужны first < last и 0 < доля < 1")
    span = (last - first).days + 1
    return first + dt.timedelta(days=int(span * is_fraction))


@dataclass(frozen=True)
class WFWindow:
    """Окна полуоткрытые: [train_start, train_end) и [test_start, test_end)."""

    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date

    @property
    def train_years(self) -> float:
        return (self.train_end - self.train_start).days / 365.25

    @property
    def test_years(self) -> float:
        return (self.test_end - self.test_start).days / 365.25


def wf_windows(
    start: dt.date,
    end: dt.date,
    train_months: int = WF_TRAIN_MONTHS,
    test_months: int = WF_TEST_MONTHS,
    step_months: int = WF_STEP_MONTHS,
) -> list[WFWindow]:
    """Окна, целиком лежащие в [start, end). Последний неполный тест отбрасывается."""
    out: list[WFWindow] = []
    k = 0
    while True:
        a = add_months(start, k * step_months)
        b = add_months(a, train_months)
        c = add_months(b, test_months)
        if c > end:
            return out
        out.append(WFWindow(a, b, b, c))
        k += 1


def weekdays(start: dt.date, end: dt.date) -> npt.NDArray[np.int64]:
    """Дни Пн–Пт в [start, end) как ordinal (календарь дневных рядов)."""
    days = np.arange(start.toordinal(), end.toordinal(), dtype=np.int64)
    return days[(days - 1) % 7 < 5]


@dataclass(frozen=True)
class Splits:
    """Хронология: IS — [first, cut), holdout — [cut, last]; общая для всех инструментов (D-032)."""

    first: dt.date
    last: dt.date
    cut: dt.date

    @classmethod
    def from_range(cls, first: dt.date, last: dt.date, is_fraction: float = IS_FRACTION) -> Splits:
        return cls(first, last, holdout_start(first, last, is_fraction))

    @property
    def cut_ord(self) -> int:
        return self.cut.toordinal()

    def calendar(self, start: dt.date, end: dt.date) -> npt.NDArray[np.int64]:
        """Дни Пн–Пт серверного календаря в [start, end) как ordinal."""
        return weekdays(start, end)

    @property
    def is_calendar(self) -> npt.NDArray[np.int64]:
        return self.calendar(self.first, self.cut)

    @property
    def ho_calendar(self) -> npt.NDArray[np.int64]:
        return self.calendar(self.cut, self.last + dt.timedelta(days=1))
