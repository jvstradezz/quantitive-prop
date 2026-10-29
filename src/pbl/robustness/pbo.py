"""Probability of Backtest Overfitting, CSCV (Bailey и др., 2017; VALIDATION §4.3).

Матрица T × N дневных доходностей режется на S блоков подряд; в каждом из C(S, S/2) разбиений
берём лучший по Sharpe вариант на «IS»-половине и смотрим его относительный ранг ω на второй
половине: λ = ln(ω / (1 − ω)). PBO — доля разбиений с λ ≤ 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
BATCH = 1500


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_combos: int
    n_blocks: int
    n_variants: int
    logits: F64

    @property
    def median_logit(self) -> float:
        return float(np.median(self.logits))


def _sharpe_from_sums(s1: F64, s2: F64, n: int) -> F64:
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 0.0) * n / (n - 1)
    out = np.zeros_like(mean)
    ok = var > 1e-18
    out[ok] = mean[ok] / np.sqrt(var[ok])
    return out


def pbo_cscv(returns: npt.ArrayLike, n_blocks: int = 16) -> PBOResult:
    """`returns` — T × N без NaN. Хвост, не делящийся на блоки, отбрасывается."""
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("нужна матрица T × N, N ≥ 2")
    if n_blocks < 2 or n_blocks % 2:
        raise ValueError("число блоков чётное")
    if np.isnan(x).any():
        raise ValueError("в матрице NaN: окна IS у вариантов должны совпадать")
    t, n_var = x.shape
    length = t // n_blocks
    if length < 2:
        raise ValueError(f"мало дней ({t}) для {n_blocks} блоков")
    x = x[: length * n_blocks]
    blocks = x.reshape(n_blocks, length, n_var)
    b1, b2 = blocks.sum(axis=1), (blocks**2).sum(axis=1)
    tot1, tot2 = b1.sum(axis=0), b2.sum(axis=0)
    half = n_blocks // 2
    n_half = length * half
    combos = np.asarray(list(combinations(range(n_blocks), half)), dtype=np.int64)
    logits = np.empty(len(combos))
    for a in range(0, len(combos), BATCH):
        idx = combos[a : a + BATCH]
        is1, is2 = b1[idx].sum(axis=1), b2[idx].sum(axis=1)
        sr_is = _sharpe_from_sums(is1, is2, n_half)
        sr_oos = _sharpe_from_sums(tot1 - is1, tot2 - is2, n_half)
        best = sr_is.argmax(axis=1)
        picked = sr_oos[np.arange(len(idx)), best]
        # ранг среди N по OOS: доля вариантов не лучше выбранного (ничьи — в пользу выбранного)
        rank = (sr_oos <= picked[:, None]).sum(axis=1)
        omega = rank / (n_var + 1)
        logits[a : a + BATCH] = np.log(omega / (1 - omega))
    return PBOResult(
        pbo=float((logits <= 0).mean()),
        n_combos=len(combos),
        n_blocks=n_blocks,
        n_variants=n_var,
        logits=logits,
    )


def n_combos(n_blocks: int) -> int:
    return math.comb(n_blocks, n_blocks // 2)
