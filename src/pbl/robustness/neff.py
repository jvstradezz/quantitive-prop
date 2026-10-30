"""Число независимых проб `N_eff` для DSR (VALIDATION §4.3, P-04).

Варианты одной идеи сильно похожи (соседние параметры дают почти те же сделки), поэтому сырое N
завышает планку SR₀. `N_eff` — число кластеров по корреляции дневных доходностей: вариант с
лучшим Sharpe открывает кластер, в него входят все ещё не распределённые варианты с ρ ≥ 0.5 к
лидеру; затем следующий по Sharpe и так далее (P-37). Детерминированно, O(N · K).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pbl.robustness.dsr import deflated_sharpe, expected_max_sharpe
from pbl.robustness.thresholds import NEFF_RHO

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Clusters:
    labels: I64  # номер кластера каждого варианта
    leaders: I64  # индекс лидера каждого кластера
    sharpe: F64  # Sharpe каждого варианта

    @property
    def n_eff(self) -> int:
        return int(self.leaders.size)

    def cluster_sharpes(self) -> F64:
        """Средний Sharpe внутри кластера — оценка одной независимой пробы."""
        return np.array([self.sharpe[self.labels == k].mean() for k in range(self.n_eff)])


def _sharpe_cols(x: F64) -> F64:
    sd = x.std(axis=0, ddof=1)
    out = np.zeros(x.shape[1])
    ok = sd > 1e-12
    out[ok] = x.mean(axis=0)[ok] / sd[ok]
    return out


def correlation_clusters(returns: npt.ArrayLike, rho: float = NEFF_RHO) -> Clusters:
    """`returns` — T × N дневных доходностей на общем календаре (нули в днях без сделок)."""
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("нужна матрица T × N, N ≥ 1")
    if x.shape[0] < 3:
        raise ValueError("мало дней для корреляции")
    n = x.shape[1]
    sr = _sharpe_cols(x)
    z = x - x.mean(axis=0)
    norm = np.sqrt((z * z).sum(axis=0))
    flat = norm < 1e-12
    norm[flat] = 1.0
    z = z / norm
    labels = np.full(n, -1, dtype=np.int64)
    leaders: list[int] = []
    for i in np.argsort(-sr, kind="stable"):
        if labels[i] >= 0:
            continue
        k = len(leaders)
        leaders.append(int(i))
        free = labels < 0
        corr = z[:, free].T @ z[:, i]
        idx = np.flatnonzero(free)[corr >= rho]
        labels[idx] = k
        labels[i] = k
    return Clusters(labels=labels, leaders=np.asarray(leaders, dtype=np.int64), sharpe=sr)


@dataclass(frozen=True)
class DSRResult:
    dsr: float  # гейт: N = N_eff (P-04)
    sr0: float
    n_eff: int
    dsr_raw: float  # справка: сырое N всех проб
    sr0_raw: float
    n_raw: int


def dsr_effective(
    sr: float,
    n_obs: int,
    skew: float,
    kurt: float,
    trial_sharpes: npt.ArrayLike,
    clusters: Clusters,
    extra_trials: int = 0,
) -> DSRResult:
    """DSR выбранной пробы при N = `N_eff` (+ пробы вне матрицы, `extra_trials`, сырыми).

    Дисперсия Sharpe для SR₀: при N_eff — по средним Sharpe кластеров; при сыром N — по всем.
    """
    all_sr = np.asarray(trial_sharpes, dtype=np.float64)
    n_raw = int(all_sr.size)
    var_raw = float(all_sr.var(ddof=1)) if n_raw > 1 else 0.0
    sr0_raw = expected_max_sharpe(n_raw, var_raw)
    cs = clusters.cluster_sharpes()
    var_eff = float(cs.var(ddof=1)) if cs.size > 1 else var_raw
    n_eff = clusters.n_eff + extra_trials
    sr0 = expected_max_sharpe(n_eff, var_eff)
    return DSRResult(
        dsr=deflated_sharpe(sr, n_obs, skew, kurt, sr0),
        sr0=sr0,
        n_eff=n_eff,
        dsr_raw=deflated_sharpe(sr, n_obs, skew, kurt, sr0_raw),
        sr0_raw=sr0_raw,
        n_raw=n_raw,
    )
