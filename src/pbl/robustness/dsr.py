"""Deflated Sharpe Ratio (Bailey, López de Prado, 2014; VALIDATION §4.3).

Все Sharpe — на период наблюдения (дневной), без годовой нормировки. `kurt` — не избыточный
эксцесс (у нормального 3), как в `research.variants.daily_stats`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import NormalDist

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """SR₀: ожидаемый максимум Sharpe среди `n_trials` проб без истинного перевеса.

    `var_sr` — дисперсия оценок Sharpe по пробам. N ≤ 1 проб не даёт отбора: SR₀ = 0.
    """
    if n_trials < 2 or var_sr <= 0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(var_sr) * (
        (1 - g) * _N.inv_cdf(1 - 1 / n_trials) + g * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, sr0: float) -> float:
    """Вероятность, что истинный Sharpe выше SR₀, с поправкой на асимметрию и эксцесс."""
    if n_obs < 2:
        return math.nan
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0:
        return math.nan
    return _N.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))


def dsr_for_trials(
    sr: float, n_obs: int, skew: float, kurt: float, trial_sharpes: Sequence[float]
) -> tuple[float, float]:
    """(DSR, SR₀) выбранной пробы среди всех проб идеи: N и дисперсия — по их Sharpe."""
    n = len(trial_sharpes)
    if n < 2:
        sr0 = 0.0
    else:
        mean = sum(trial_sharpes) / n
        var = sum((x - mean) ** 2 for x in trial_sharpes) / (n - 1)
        sr0 = expected_max_sharpe(n, var)
    return deflated_sharpe(sr, n_obs, skew, kurt, sr0), sr0
