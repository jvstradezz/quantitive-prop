"""Бутстрэп по дням: полоса MC для holdout (VALIDATION §4.1) и разность корзин (§4.5).

Дни — единица ресэмплинга (stationary bootstrap, длина блока Politis–White, P-18): сделки одного
дня и разных инструментов в этот день остаются вместе, поэтому зависимость между инструментами
не создаёт ложной точности. Дни без сделок входят нулями.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pbl.prop.bootstrap import block_length, stationary_indices

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

LOW_Q, HIGH_Q = 5.0, 95.0
METRICS = ("mean_r", "max_dd", "win_rate", "pf")
REQUIRED = ("mean_r", "max_dd")
"""Обязательные метрики полосы: `mean R` и `max DD` (P-02); win rate и PF — в отчёт."""
WORSE_IS_HIGHER = frozenset({"max_dd"})
"""У каких метрик плохая сторона — выше полосы; у остальных — ниже (D-029)."""


@dataclass(frozen=True)
class DayStats:
    """Дневные суммы по календарю торговых дней (нули в днях без сделок)."""

    r: F64  # Σ r_net
    n: F64  # число сделок
    wins: F64  # число прибыльных сделок
    win_r: F64  # Σ положительных r_net
    loss_r: F64  # Σ отрицательных r_net (≤ 0)

    def __len__(self) -> int:
        return int(self.r.shape[0])


def daystats(calendar: I64, day: I64, r_net: F64) -> DayStats:
    """Свёртка сделок (серверный день `day`, `r_net`) в календарь `calendar` (отсортирован)."""
    pos = np.searchsorted(calendar, day)
    if len(day) and (pos.max() >= len(calendar) or (calendar[pos] != day).any()):
        raise ValueError("день сделки вне календаря")
    m = len(calendar)

    def acc(w: F64) -> F64:
        return np.bincount(pos, weights=w, minlength=m).astype(np.float64)

    r = np.asarray(r_net, dtype=np.float64)
    return DayStats(
        r=acc(r),
        n=acc(np.ones_like(r)),
        wins=acc((r > 0).astype(np.float64)),
        win_r=acc(np.where(r > 0, r, 0.0)),
        loss_r=acc(np.where(r < 0, r, 0.0)),
    )


def max_drawdown(daily_r: F64) -> float:
    """Наибольшая просадка кумулятивной суммы R (положительное число); старт — 0."""
    cum = np.concatenate([[0.0], np.cumsum(daily_r)])
    return float((np.maximum.accumulate(cum) - cum).max())


def metrics(ds: DayStats) -> dict[str, float]:
    n = float(ds.n.sum())
    loss = -float(ds.loss_r.sum())
    return {
        "mean_r": float(ds.r.sum() / n) if n else float("nan"),
        "max_dd": max_drawdown(ds.r),
        "win_rate": float(ds.wins.sum() / n) if n else float("nan"),
        "pf": float(ds.win_r.sum() / loss) if loss > 0 else float("inf"),
        "n_trades": n,
    }


def _path_metrics(ds: DayStats, idx: npt.NDArray[np.int32]) -> dict[str, F64]:
    r = ds.r[idx]
    n = ds.n[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_r = r.sum(axis=1) / n
        win_rate = ds.wins[idx].sum(axis=1) / n
        loss = -ds.loss_r[idx].sum(axis=1)
        pf = np.where(loss > 0, ds.win_r[idx].sum(axis=1) / loss, np.inf)
    cum = np.concatenate([np.zeros((r.shape[0], 1)), np.cumsum(r, axis=1)], axis=1)
    max_dd = (np.maximum.accumulate(cum, axis=1) - cum).max(axis=1)
    return {"mean_r": mean_r, "max_dd": max_dd, "win_rate": win_rate, "pf": pf}


@dataclass(frozen=True)
class Band:
    observed: float
    low: float
    high: float
    worse_is_higher: bool = False

    @property
    def inside(self) -> bool:
        return self.low <= self.observed <= self.high

    @property
    def worse(self) -> bool:
        """За полосой с плохой стороны: holdout хуже 95% псевдо-историй."""
        return self.observed > self.high if self.worse_is_higher else self.observed < self.low

    @property
    def better(self) -> bool:
        """За полосой с хорошей стороны: не отказ, а повод проверить (инструкции §1, D-029)."""
        return not self.inside and not self.worse


@dataclass(frozen=True)
class BandResult:
    bands: dict[str, Band]
    block: float
    n_paths: int
    width: int

    @property
    def passed(self) -> bool:
        """Гейт: ни одна обязательная метрика не хуже полосы (D-029; раньше — строго внутри)."""
        return not any(self.bands[k].worse for k in REQUIRED)

    @property
    def better(self) -> list[str]:
        """Обязательные метрики лучше полосы — флаг аудита «holdout подозрительно хорош»."""
        return [k for k in REQUIRED if self.bands[k].better]


def holdout_band(
    source: DayStats,
    observed: DayStats,
    *,
    n_paths: int = 10_000,
    seed: int = 42,
    block: float | None = None,
    nested: bool = True,
) -> BandResult:
    """Псевдо-holdout той же длины (в днях), что `observed`, из дней `source` (WF-test).

    Проходит, если `mean R` не ниже 5% и `max DD` не выше 95% псевдо-историй (P-02, D-029).
    `nested` (D-030): каждый путь берётся из своей бутстрэп-копии источника, поэтому полоса
    учитывает и шум оценки по WF-test, а не только шум holdout. Синтетика `edge`, 100 seed:
    хорошая стратегия не проходила полосу в 13% случаев без вложенности и в 11% с ней
    (номинал двух односторонних 5% хвостов — ≤ 10%).
    """
    if len(source) < 20:
        raise ValueError("мало дней в источнике бутстрэпа")
    width = len(observed)
    rng = np.random.default_rng(seed)
    blk = block_length(source.r, block)
    idx = stationary_indices(rng, len(source), n_paths, width, blk)
    if nested:
        outer = stationary_indices(rng, len(source), n_paths, len(source), blk)
        idx = np.take_along_axis(outer, idx.astype(np.int64), axis=1).astype(np.int32)
    sim = _path_metrics(source, idx)
    obs = metrics(observed)
    bands: dict[str, Band] = {}
    for k in METRICS:
        v = sim[k][np.isfinite(sim[k])]
        lo, hi = np.percentile(v, [LOW_Q, HIGH_Q]) if v.size else (np.nan, np.nan)
        bands[k] = Band(obs[k], float(lo), float(hi), worse_is_higher=k in WORSE_IS_HIGHER)
    return BandResult(bands=bands, block=blk, n_paths=n_paths, width=width)


@dataclass(frozen=True)
class DiffResult:
    """Разность средних R двух корзин: `a − b`, 90% интервал бутстрэпа по дням."""

    diff: float
    low: float
    high: float
    p_nonpositive: float
    mean_a: float
    mean_b: float
    n_a: float
    n_b: float
    block: float

    @property
    def holds(self) -> bool:
        """Фильтр держится: интервал целиком выше нуля (VALIDATION §4.5, P-06)."""
        return self.low > 0


def bucket_diff(
    a: DayStats,
    b: DayStats,
    *,
    n_paths: int = 10_000,
    seed: int = 42,
    block: float | None = None,
    level: float = 90.0,
) -> DiffResult:
    """`mean R(a) − mean R(b)`: a, b — корзины на одном календаре (общие дни ресэмплируются)."""
    if len(a) != len(b):
        raise ValueError("корзины на разных календарях")
    n_a, n_b = float(a.n.sum()), float(b.n.sum())
    if n_a == 0 or n_b == 0:
        raise ValueError("пустая корзина")
    mean_a, mean_b = float(a.r.sum() / n_a), float(b.r.sum() / n_b)
    rng = np.random.default_rng(seed)
    blk = block_length(a.r + b.r, block)
    idx = stationary_indices(rng, len(a), n_paths, len(a), blk)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = a.r[idx].sum(axis=1) / a.n[idx].sum(axis=1) - b.r[idx].sum(axis=1) / b.n[idx].sum(
            axis=1
        )
    d = d[np.isfinite(d)]
    tail = (100.0 - level) / 2
    lo, hi = np.percentile(d, [tail, 100.0 - tail])
    return DiffResult(
        diff=mean_a - mean_b,
        low=float(lo),
        high=float(hi),
        p_nonpositive=float((d <= 0).mean()),
        mean_a=mean_a,
        mean_b=mean_b,
        n_a=n_a,
        n_b=n_b,
        block=blk,
    )
