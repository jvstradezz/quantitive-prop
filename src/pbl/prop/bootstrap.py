"""Бутстрэп по дням и статистика долей (MONTE_CARLO §5, §7).

Stationary bootstrap (Politis, Romano 1994), циклический; длина блока — Politis, White (2004)
с поправкой Patton, Politis, White (2009), в пределах [3, 20] дней (P-18).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I32 = npt.NDArray[np.int32]

BLOCK_MIN, BLOCK_MAX = 3.0, 20.0


def politis_white_block(x: npt.ArrayLike) -> float:
    """Оптимальная средняя длина блока stationary bootstrap (без ограничений)."""
    v = np.asarray(x, dtype=np.float64)
    n = v.shape[0]
    if n < 8:
        return 1.0
    v = v - v.mean()
    var = float(v @ v) / n
    if var <= 0:
        return 1.0
    k_n = max(5, math.ceil(math.sqrt(math.log10(n))))
    m_max = min(math.ceil(math.sqrt(n)) + k_n, n - 1)
    acov = np.array([float(v[: n - k] @ v[k:]) / n for k in range(m_max + 1)])
    rho = acov / var
    thr = 2.0 * math.sqrt(math.log10(n) / n)
    # m̂ — наименьший лаг, после которого K_N автокорреляций подряд незначимы
    m_hat = m_max
    for m in range(0, m_max - k_n + 1):
        if bool(np.all(np.abs(rho[m + 1 : m + 1 + k_n]) < thr)):
            m_hat = m
            break
    big_m = min(2 * max(m_hat, 1), m_max)
    ks = np.arange(-big_m, big_m + 1)
    t = np.abs(ks) / big_m
    lam = np.where(t <= 0.5, 1.0, np.where(t <= 1.0, 2.0 * (1.0 - t), 0.0))
    r = acov[np.abs(ks)]
    g = float(np.sum(lam * np.abs(ks) * r))
    g0 = float(np.sum(lam * r))
    if g0 <= 0 or g == 0:
        return 1.0
    d_sb = 2.0 * g0**2
    return float((2.0 * g * g / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0))


def block_length(daily_r: npt.ArrayLike, override: float | None = None) -> float:
    """Длина блока для MC: `override` или Politis–White в [3, 20] (P-18)."""
    if override is not None:
        if override < 1:
            raise ValueError("длина блока ≥ 1")
        return float(override)
    return float(np.clip(politis_white_block(daily_r), BLOCK_MIN, BLOCK_MAX))


def stationary_indices(
    rng: np.random.Generator, n_days: int, n_paths: int, width: int, block: float
) -> I32:
    """Матрица индексов дней [путь, шаг]: старт — случайный день, дальше следующий день
    с вероятностью 1 − 1/L, иначе прыжок на новый случайный (циклически)."""
    if n_days <= 0:
        raise ValueError("нет дней для бутстрэпа")
    if block <= 1.0:
        return rng.integers(0, n_days, size=(n_paths, width), dtype=np.int32)
    starts = rng.integers(0, n_days, size=(n_paths, width), dtype=np.int64)
    jump = rng.random((n_paths, width)) < (1.0 / block)
    jump[:, 0] = True
    t = np.arange(width, dtype=np.int64)
    last = np.maximum.accumulate(np.where(jump, t[None, :], 0), axis=1)
    base = np.take_along_axis(starts, last, axis=1)
    return ((base + (t[None, :] - last)) % n_days).astype(np.int32)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Интервал Уилсона для доли (MONTE_CARLO §7)."""
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))
