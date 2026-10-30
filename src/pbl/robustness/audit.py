"""Флаги обязательного аудита [opus] (VALIDATION §10).

1. Средний убыток > 3 × среднего профита.
2. Серия убытков: p95 худшей серии подряд идущих убыточных сделок (в $ при оптимальном риске)
   > 50% общего лимита (P-11). Серия ищется на псевдо-историях блочного бутстрэпа по дням
   длиной в горизонт MC (P-13: 365 календарных дней ≈ 261 торговый день, P-38).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast

import numba
import numpy as np
import numpy.typing as npt

from pbl.prop.bootstrap import block_length, stationary_indices
from pbl.robustness.thresholds import AUDIT_LOSS_TO_WIN, AUDIT_STREAK_Q, AUDIT_STREAK_SHARE

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
HORIZON_DAYS = 261


@dataclass(frozen=True)
class AuditFlag:
    code: str
    name: str
    flagged: bool | None
    value: float
    limit: float
    detail: str


def loss_to_win(r_net: Sequence[float]) -> AuditFlag:
    r = np.asarray(r_net, dtype=np.float64)
    wins, losses = r[r > 0], r[r < 0]
    name = "средний убыток / средний профит"
    if not wins.size or not losses.size:
        return AuditFlag("loss_to_win", name, None, math.nan, AUDIT_LOSS_TO_WIN, "нет побед")
    ratio = float(-losses.mean() / wins.mean())
    return AuditFlag(
        "loss_to_win",
        name,
        ratio > AUDIT_LOSS_TO_WIN,
        ratio,
        AUDIT_LOSS_TO_WIN,
        f"{ratio:.2f} (порог {AUDIT_LOSS_TO_WIN:g})",
    )


def _worst_streaks_py(offsets: I64, r: F64, idx: npt.NDArray[np.int32]) -> F64:
    """Худшая (самая отрицательная) сумма подряд идущих убыточных сделок на каждом пути."""
    n_paths, width = idx.shape
    out = np.zeros(n_paths)
    for p in range(n_paths):
        cur = 0.0
        worst = 0.0
        for s in range(width):
            d = idx[p, s]
            for k in range(offsets[d], offsets[d + 1]):
                x = r[k]
                if x < 0:
                    cur += x
                    worst = min(worst, cur)
                else:
                    cur = 0.0
        out[p] = -worst
    return out


_worst_streaks = cast(Callable[..., F64], numba.njit(cache=True)(_worst_streaks_py))


def streak_quantile(
    day: npt.ArrayLike,
    r_net: npt.ArrayLike,
    calendar: I64,
    *,
    q: float = AUDIT_STREAK_Q,
    width: int = HORIZON_DAYS,
    n_paths: int = 10_000,
    seed: int = 42,
) -> float:
    """q-перцентиль худшей серии убытков (в R, положительное число) за `width` торговых дней.

    `day` — ordinal дня сделки (сделки в порядке времени), `calendar` — дни Пн–Пт выборки.
    """
    d = np.asarray(day, dtype=np.int64)
    r = np.asarray(r_net, dtype=np.float64)
    if not d.size:
        return 0.0
    order = np.argsort(d, kind="stable")
    d, r = d[order], r[order]
    pos = np.searchsorted(calendar, d)
    counts = np.bincount(pos, minlength=len(calendar))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    daily_r = np.bincount(pos, weights=r, minlength=len(calendar)).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = stationary_indices(rng, len(calendar), n_paths, width, block_length(daily_r))
    return float(np.percentile(_worst_streaks(offsets, r, idx), q))


def streak_flag(streak_r: float, risk: float, max_dist: float) -> AuditFlag:
    """`risk` — риск на сделку (доля B0), `max_dist` — общий лимит (доля B0)."""
    loss = streak_r * risk
    limit = AUDIT_STREAK_SHARE * max_dist
    return AuditFlag(
        "streak",
        f"p{AUDIT_STREAK_Q:g} серии убытков при оптимальном риске",
        loss > limit,
        loss,
        limit,
        f"{streak_r:.1f}R × {100 * risk:g}% = {100 * loss:.1f}% B0 "
        f"(порог {100 * limit:.1f}% B0 = {100 * AUDIT_STREAK_SHARE:g}% общего лимита)",
    )
