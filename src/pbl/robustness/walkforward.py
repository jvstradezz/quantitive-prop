"""Walk-forward внутри IS и WFE (VALIDATION §2, §4; P-01).

В каждом окне выбираем лучший по Sharpe обучения вариант (с ≥ `WF_MIN_TRADES` сделок) и
смотрим его тест. Для стратегии без параметров (один вариант) — те же окна без переоптимизации.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pbl.robustness.splits import WFWindow
from pbl.robustness.stats import series_sharpe
from pbl.robustness.thresholds import WF_MIN_TRADES

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class WFPick:
    window: WFWindow
    key: str
    n_train: int
    train_r: float
    test_r: float
    n_test: int

    @property
    def train_per_year(self) -> float:
        return self.train_r / self.window.train_years

    @property
    def test_per_year(self) -> float:
        return self.test_r / self.window.test_years


def walk_forward(
    r: F64,
    n: F64,
    calendar: I64,
    keys: Sequence[str],
    windows: Sequence[WFWindow],
    min_trades: int | None = None,
) -> list[WFPick]:
    """`r`, `n` — T × C (сумма R и число сделок по дням `calendar`) для вариантов `keys`."""
    min_n = WF_MIN_TRADES if min_trades is None else min_trades
    picks: list[WFPick] = []
    for w in windows:
        tr = (calendar >= w.train_start.toordinal()) & (calendar < w.train_end.toordinal())
        te = (calendar >= w.test_start.toordinal()) & (calendar < w.test_end.toordinal())
        if not tr.any() or not te.any():
            continue
        best, best_sr = -1, -math.inf
        for j in range(len(keys)):
            if n[tr, j].sum() < min_n:
                continue
            sr = series_sharpe(r[tr, j])
            if sr > best_sr:
                best, best_sr = j, sr
        if best < 0:
            continue
        picks.append(
            WFPick(
                window=w,
                key=keys[best],
                n_train=int(n[tr, best].sum()),
                train_r=float(r[tr, best].sum()),
                test_r=float(r[te, best].sum()),
                n_test=int(n[te, best].sum()),
            )
        )
    return picks


def wfe(picks: Sequence[WFPick]) -> float:
    """WFE = годовая доходность тестов / годовая доходность обучений по склейке окон (§4)."""
    if not picks:
        return math.nan
    train = sum(p.train_r for p in picks) / sum(p.window.train_years for p in picks)
    test = sum(p.test_r for p in picks) / sum(p.window.test_years for p in picks)
    return test / train if train > 0 else math.nan
