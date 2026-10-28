"""Стрессы (MONTE_CARLO §6): преобразования списка сделок до построения дневных записей.

Исключение — «худшие годы»: это выбор подмножества дней для бутстрэпа.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from pbl.prop.days import DayTable, TradeArrays


def costs_x(ta: TradeArrays, m: float) -> TradeArrays:
    """Спред и проскальзывание × m: `r' = r_net − (m − 1)(spread + slip)`.

    Путь сделки сдвигается на ту же величину с момента входа.
    """
    extra = (m - 1.0) * (ta.cost_spread + ta.cost_slip)
    return ta.with_(
        r_net=ta.r_net - extra,
        cost_spread=ta.cost_spread * m,
        cost_slip=ta.cost_slip * m,
    )


def skip_trades(ta: TradeArrays, rng: np.random.Generator, p: float = 0.1) -> TradeArrays:
    """Каждая сделка удаляется с вероятностью p (хотя бы одна остаётся)."""
    keep = rng.random(len(ta)) >= p
    if not keep.any():
        keep[int(rng.integers(len(ta)))] = True
    return ta.take(keep)


def winrate_minus(ta: TradeArrays, rng: np.random.Generator, delta: float) -> TradeArrays:
    """`k = round(δ · n)` случайных прибыльных сделок становятся убыточными.

    Новый `r_net` — из эмпирического распределения убытков; путь —
    `min(исходный, линия от 0 до нового r_net)`: MFE пропадает, MAE не выше нового результата.
    """
    n = len(ta)
    winners = np.flatnonzero(ta.r_net > 0)
    losses = ta.r_net[ta.r_net < 0]
    k = min(round(delta * n), winners.shape[0])
    if k == 0:
        return ta
    pick = rng.choice(winners, size=k, replace=False)
    new_r = rng.choice(losses, size=k) if losses.size else np.full(k, -1.0)
    cost = ta.r_gross[pick] - ta.r_net[pick]
    r_net = ta.r_net.copy()
    r_gross = ta.r_gross.copy()
    mae = ta.mae.copy()
    mfe = ta.mfe.copy()
    r_net[pick] = new_r
    r_gross[pick] = new_r + cost
    mae[pick] = np.minimum(mae[pick], np.minimum(r_gross[pick], 0.0))
    mfe[pick] = 0.0
    return ta.with_(r_net=r_net, r_gross=r_gross, mae=mae, mfe=mfe)


def worst_period_days(table: DayTable) -> tuple[np.ndarray, str]:
    """Индексы дней «худших лет»: два худших календарных года по чистому R,
    или худшее 12-месячное окно, если полных лет меньше 4."""
    years = table.years()
    uniq = np.unique(years)
    if uniq.shape[0] >= 4:
        sums = np.array([table.db[years == y].sum() for y in uniq])
        worst = uniq[np.argsort(sums, kind="stable")[:2]]
        mask = np.isin(years, worst)
        return np.flatnonzero(mask), f"годы {', '.join(str(int(y)) for y in sorted(worst))}"
    dates = table.dates
    cum = np.concatenate([[0.0], np.cumsum(table.db)])
    best_i, best_j, best_sum = 0, table.n_days, np.inf
    j = 0
    for i in range(table.n_days):
        while j < table.n_days and dates[j] < dates[i] + 365:
            j += 1
        if dates[i] + 365 > dates[-1] + 1 and i > 0:
            break
        s = cum[j] - cum[i]
        if s < best_sum:
            best_i, best_j, best_sum = i, j, s
    a = dt.date.fromordinal(int(dates[best_i]))
    b = dt.date.fromordinal(int(dates[best_j - 1]))
    return np.arange(best_i, best_j), f"окно {a:%Y-%m-%d} … {b:%Y-%m-%d}"


@dataclass(frozen=True)
class StressSpec:
    id: str
    label: str
    cost_mult: float = 1.0
    skip: float = 0.0
    winrate_delta: float = 0.0
    worst_years: bool = False
    guard_slip_mult: float = 1.0


STRESSES: tuple[StressSpec, ...] = (
    StressSpec("base", "Базовый"),
    StressSpec("worst_years", "Худшие годы", worst_years=True),
    StressSpec("costs_1_5", "Издержки × 1.5", cost_mult=1.5, guard_slip_mult=3.0),
    StressSpec("costs_2", "Издержки × 2", cost_mult=2.0, guard_slip_mult=3.0),
    StressSpec("skip_10", "Пропуск 10% сделок", skip=0.1),
    StressSpec("wr_5", "Винрейт −5 п.п.", winrate_delta=0.05),
    StressSpec("wr_10", "Винрейт −10 п.п.", winrate_delta=0.10),
    StressSpec(
        "package",
        "Пакет: издержки × 1.5 + винрейт −5 + пропуск 10%",
        cost_mult=1.5,
        skip=0.1,
        winrate_delta=0.05,
        guard_slip_mult=3.0,
    ),
)


def apply_trade_stress(ta: TradeArrays, spec: StressSpec, rng: np.random.Generator) -> TradeArrays:
    out = ta
    if spec.cost_mult != 1.0:
        out = costs_x(out, spec.cost_mult)
    if spec.winrate_delta > 0:
        out = winrate_minus(out, rng, spec.winrate_delta)
    if spec.skip > 0:
        out = skip_trades(out, rng, spec.skip)
    return out
