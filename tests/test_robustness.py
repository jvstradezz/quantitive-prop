"""Робастность: DSR, PBO, разбиения, полосы MC и разность корзин (VALIDATION §2–4)."""

from __future__ import annotations

import datetime as dt
import math
from statistics import NormalDist

import numpy as np
import pytest

from pbl.robustness.bands import (
    bucket_diff,
    daystats,
    holdout_band,
    max_drawdown,
    metrics,
)
from pbl.robustness.dsr import deflated_sharpe, dsr_for_trials, expected_max_sharpe
from pbl.robustness.pbo import n_combos, pbo_cscv
from pbl.robustness.splits import add_months, holdout_start, wf_windows

# --- DSR ----------------------------------------------------------------------------------


def test_expected_max_sharpe_matches_simulated_maximum() -> None:
    """E[max] из N независимых N(0, σ²): формула против симуляции (σ = 0.1)."""
    rng = np.random.default_rng(0)
    for n in (10, 100, 500):
        sim = rng.normal(0.0, 0.1, size=(4000, n)).max(axis=1).mean()
        assert expected_max_sharpe(n, 0.01) == pytest.approx(sim, abs=0.012)


def test_expected_max_sharpe_edge_cases() -> None:
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(100, 0.0) == 0.0
    assert expected_max_sharpe(1000, 1.0) > expected_max_sharpe(100, 1.0) > 0


def test_deflated_sharpe_hand_value_and_normal_case() -> None:
    # нормальные доходности: skew 0, kurt 3 → знаменатель 1 + SR² / 2
    sr, t = 0.1, 1001
    z = sr * math.sqrt(t - 1) / math.sqrt(1 + sr * sr / 2)
    assert deflated_sharpe(sr, t, 0.0, 3.0, 0.0) == pytest.approx(NormalDist().cdf(z))
    assert deflated_sharpe(sr, t, 0.0, 3.0, 0.05) < deflated_sharpe(sr, t, 0.0, 3.0, 0.0)
    # отрицательная асимметрия при положительном SR уменьшает DSR
    assert deflated_sharpe(sr, t, -1.0, 6.0, 0.0) < deflated_sharpe(sr, t, 0.0, 3.0, 0.0)
    assert math.isnan(deflated_sharpe(sr, 1, 0.0, 3.0, 0.0))
    assert math.isnan(deflated_sharpe(10.0, 100, 10.0, 3.0, 0.0))  # знаменатель ≤ 0


def test_dsr_penalizes_the_best_of_many_noise_strategies() -> None:
    rng = np.random.default_rng(1)
    t, n, reps = 500, 100, 60
    raw_ok = dsr_ok = 0
    for _ in range(reps):
        x = rng.normal(0.0, 1.0, size=(t, n))
        sr = x.mean(axis=0) / x.std(axis=0, ddof=1)
        best = int(sr.argmax())
        z = (x[:, best] - x[:, best].mean()) / x[:, best].std()
        skew, kurt = float(np.mean(z**3)), float(np.mean(z**4))
        raw_ok += deflated_sharpe(float(sr[best]), t, skew, kurt, 0.0) >= 0.95
        dsr, _ = dsr_for_trials(float(sr[best]), t, skew, kurt, list(sr))
        dsr_ok += dsr >= 0.95
    assert raw_ok / reps > 0.6, "без поправки лучший из шума выглядит значимым"
    assert dsr_ok / reps < 0.15, "с поправкой — нет"


def test_dsr_keeps_a_real_edge() -> None:
    rng = np.random.default_rng(2)
    t, n = 1000, 50
    x = rng.normal(0.0, 1.0, size=(t, n))
    x[:, 7] += 0.15  # Sharpe ≈ 0.15 на день при шуме остальных ~ 0.03
    sr = x.mean(axis=0) / x.std(axis=0, ddof=1)
    dsr, sr0 = dsr_for_trials(float(sr[7]), t, 0.0, 3.0, list(sr))
    assert sr0 < 0.1
    assert dsr > 0.95


# --- PBO ----------------------------------------------------------------------------------


def test_pbo_of_pure_noise_is_about_one_half_on_average() -> None:
    """Один набор шума даёт PBO с большим разбросом (комбинации зависимы); среднее — ≈ 0.5."""
    rng = np.random.default_rng(3)
    pbos = [pbo_cscv(rng.normal(size=(160, 20)), n_blocks=8).pbo for _ in range(200)]
    assert 0.4 < float(np.mean(pbos)) < 0.6
    assert n_combos(16) == 12870


def test_pbo_of_a_persistent_edge_is_near_zero() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(640, 40))
    x[:, 5] += 0.4
    res = pbo_cscv(x)
    assert res.n_combos == 12870
    assert res.pbo < 0.02
    assert res.median_logit > 2


def test_pbo_input_validation() -> None:
    with pytest.raises(ValueError, match="N ≥ 2"):
        pbo_cscv(np.zeros((100, 1)))
    with pytest.raises(ValueError, match="чётное"):
        pbo_cscv(np.zeros((100, 3)), n_blocks=7)
    with pytest.raises(ValueError, match="NaN"):
        pbo_cscv(np.full((100, 3), np.nan))
    with pytest.raises(ValueError, match="мало дней"):
        pbo_cscv(np.zeros((20, 3)))


def test_pbo_constant_columns_do_not_crash() -> None:
    x = np.zeros((160, 4))
    x[:, 0] = np.random.default_rng(6).normal(size=160)
    res = pbo_cscv(x, n_blocks=8)
    assert res.n_variants == 4
    assert np.isfinite(res.logits).all()


# --- разбиения ----------------------------------------------------------------------------


def test_add_months_clamps_day() -> None:
    assert add_months(dt.date(2023, 1, 31), 1) == dt.date(2023, 2, 28)
    assert add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)
    assert add_months(dt.date(2024, 11, 15), 3) == dt.date(2025, 2, 15)
    assert add_months(dt.date(2024, 3, 15), -3) == dt.date(2023, 12, 15)


def test_holdout_start_is_seventy_percent_of_calendar_days() -> None:
    first, last = dt.date(2021, 1, 1), dt.date(2025, 12, 31)
    cut = holdout_start(first, last)
    span = (last - first).days + 1
    assert (cut - first).days == int(span * 0.7)
    assert dt.date(2024, 6, 15) < cut < dt.date(2024, 7, 15)
    with pytest.raises(ValueError, match="first < last"):
        holdout_start(last, first)


def test_wf_windows_on_three_and_a_half_years() -> None:
    ws = wf_windows(dt.date(2021, 1, 4), dt.date(2024, 7, 5))
    assert [(w.train_start, w.test_start, w.test_end) for w in ws] == [
        (dt.date(2021, 1, 4), dt.date(2023, 1, 4), dt.date(2023, 7, 4)),
        (dt.date(2021, 7, 4), dt.date(2023, 7, 4), dt.date(2024, 1, 4)),
        (dt.date(2022, 1, 4), dt.date(2024, 1, 4), dt.date(2024, 7, 4)),
    ]
    assert ws[0].train_years == pytest.approx(2.0, abs=0.01)
    assert ws[0].test_years == pytest.approx(0.5, abs=0.01)
    assert wf_windows(dt.date(2021, 1, 4), dt.date(2022, 6, 1)) == []


# --- полосы и разность корзин -------------------------------------------------------------


def test_daystats_and_metrics_by_hand() -> None:
    cal = np.array([10, 11, 12, 13], dtype=np.int64)
    ds = daystats(cal, np.array([10, 10, 12], dtype=np.int64), np.array([1.0, -0.4, -1.0]))
    np.testing.assert_allclose(ds.r, [0.6, 0.0, -1.0, 0.0])
    np.testing.assert_allclose(ds.n, [2, 0, 1, 0])
    m = metrics(ds)
    assert m["mean_r"] == pytest.approx(-0.4 / 3)
    assert m["win_rate"] == pytest.approx(1 / 3)
    assert m["pf"] == pytest.approx(1.0 / 1.4)
    assert m["max_dd"] == pytest.approx(1.0)  # 0.6 → −0.4
    with pytest.raises(ValueError, match="вне календаря"):
        daystats(cal, np.array([99], dtype=np.int64), np.array([1.0]))


def test_max_drawdown() -> None:
    assert max_drawdown(np.array([1.0, -2.0, 0.5, -0.5, 3.0])) == pytest.approx(2.0)
    assert max_drawdown(np.array([-1.0, -1.0])) == pytest.approx(2.0)
    assert max_drawdown(np.array([1.0, 1.0])) == 0.0


def _random_days(n: int, mean: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    cal = np.arange(n, dtype=np.int64)
    has = rng.random(n) < 0.8
    day = cal[has]
    r = np.where(rng.random(day.size) < 0.5 + mean / 2, 1.0, -1.0)
    return day, r


def test_holdout_band_accepts_same_process_and_flags_a_much_better_one() -> None:
    cal = np.arange(600, dtype=np.int64)
    day, r = _random_days(600, 0.1, seed=1)
    src = daystats(cal, day, r)
    cal_h = np.arange(250, dtype=np.int64)
    same = daystats(cal_h, *_random_days(250, 0.1, seed=2))
    better = daystats(cal_h, *_random_days(250, 0.6, seed=3))
    ok = holdout_band(src, same, n_paths=2000, seed=7)
    assert ok.passed
    assert ok.width == 250
    good = holdout_band(src, better, n_paths=2000, seed=7)
    assert not good.bands["mean_r"].inside
    assert good.passed, "лучше полосы — не отказ (D-029)"
    assert "mean_r" in good.better
    worse = daystats(cal_h, *_random_days(250, -0.4, seed=4))
    bad = holdout_band(src, worse, n_paths=2000, seed=7)
    assert bad.bands["mean_r"].worse
    assert not bad.passed
    again = holdout_band(src, same, n_paths=2000, seed=7)
    assert again.bands["mean_r"].low == ok.bands["mean_r"].low, "фиксированный seed"


def test_holdout_band_needs_enough_source_days() -> None:
    cal = np.arange(10, dtype=np.int64)
    ds = daystats(cal, cal[:5], np.ones(5))
    with pytest.raises(ValueError, match="мало дней"):
        holdout_band(ds, ds)


def test_bucket_diff_detects_a_real_difference_and_not_noise() -> None:
    cal = np.arange(500, dtype=np.int64)
    a = daystats(cal, *_random_days(500, 0.3, seed=11))
    b = daystats(cal, *_random_days(500, 0.0, seed=12))
    res = bucket_diff(a, b, n_paths=3000, seed=1)
    assert res.diff == pytest.approx(res.mean_a - res.mean_b)
    assert res.holds
    assert res.low < res.diff < res.high
    noise_b = daystats(cal, *_random_days(500, 0.0, seed=13))
    noise_c = daystats(cal, *_random_days(500, 0.0, seed=14))
    same = bucket_diff(noise_b, noise_c, n_paths=3000, seed=1)
    assert not same.holds
    assert same.low < 0 < same.high


def test_bucket_diff_validation() -> None:
    cal = np.arange(50, dtype=np.int64)
    a = daystats(cal, cal[:10], np.ones(10))
    empty = daystats(cal, cal[:0], np.ones(0))
    with pytest.raises(ValueError, match="пустая"):
        bucket_diff(a, empty)
    with pytest.raises(ValueError, match="календарях"):
        bucket_diff(a, daystats(cal[:40], cal[:5], np.ones(5)))
