"""Детерминированные сценарии на каждое правило (MONTE_CARLO §12, V3 и V7).

Один путь по заданной последовательности дней. Риск f = 2⁻⁷, чтобы уровни сравнивались точно.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pbl.prop.days import DayTable, table_from_paths
from pbl.prop.metrics import summarize
from pbl.prop.rules import (
    DAILY_BASIS_MIN,
    DAILY_REF_MAX_BAL_EQ,
    MAX_BALANCE_EOD,
    MAX_BALANCE_RT,
    MAX_EQUITY_RT,
    MAX_STATIC,
    PayoutSpec,
    RuleSet,
)
from pbl.prop.sim import (
    DAILY,
    GUARD,
    INACTIVITY,
    MAX,
    OK,
    RULE,
    TIMEOUT,
    PathResults,
    RiskScheme,
    SimSettings,
    simulate,
)

F = 2.0**-7  # 0.78125%: двоичная дробь, уровни считаются без ошибки округления
LONG = SimSettings(horizon_cal=3650, ev_horizon_cal=0, pause_days=2)


def make_rules(**kw: Any) -> RuleSet:
    base: dict[str, Any] = {
        "key": "t",
        "label": "t",
        "size_usd": 10_000.0,
        "targets": (0.5,),
        "min_days": (0,),
        "max_days": (-1,),
        "daily_pct": None,
        "daily_ref": 0,
        "daily_basis": 0,
        "max_type": MAX_STATIC,
        "max_dist": 0.5,
        "lock_cap": math.inf,
        "lock_trigger": -1.0,
        "floor_after_payout": 0,
        "inactivity_days": -1,
        "payout": None,
    }
    base.update(kw)
    return RuleSet(**base)


def run(
    table: DayTable,
    rules: RuleSet,
    *,
    f: float = F,
    guards: bool = False,
    seq: list[int] | None = None,
    settings: SimSettings = LONG,
    jit: bool = True,
    **risk_kw: Any,
) -> PathResults:
    days = seq if seq is not None else list(range(table.n_days))
    idx = np.asarray([days], dtype=np.int32)
    risk = RiskScheme(
        f=tuple(f for _ in range(rules.n_slots)),
        guards=guards,
        target_buffer=risk_kw.pop("target_buffer", 0.0),
        **risk_kw,
    )
    return simulate(table, rules, risk, idx, settings, jit=jit)


# --- общий лимит --------------------------------------------------------------------------


def test_static_max_equality_is_not_breach() -> None:
    rules = make_rules(max_dist=4 * F)
    res = run(table_from_paths([[-4.0], [0.0]]), rules)
    assert res.fail[0] == TIMEOUT  # E = L — не пробой, дни кончились
    assert res.bal[0] == 1 - 4 * F
    res = run(table_from_paths([[-4.0], [-0.01]]), rules)
    assert res.fail[0] == MAX
    assert res.end_cal[0] == 1


def test_intraday_floating_breach_even_if_day_closes_up() -> None:
    rules = make_rules(max_dist=4 * F)
    res = run(table_from_paths([[-4.5, 2.0]]), rules)  # MAE ниже уровня, закрытие в плюс
    assert res.fail[0] == MAX


# --- дневной лимит ------------------------------------------------------------------------


def test_daily_from_start_of_day_balance() -> None:
    rules = make_rules(daily_pct=2 * F)
    # день 1: +3; день 2: просадка −2 от B_SOD = граница, не пробой; день 3: −2.5 → пробой,
    # хотя эквити выше начального баланса (уровень считается от баланса на начало дня)
    table = table_from_paths([[3.0], [-2.0, 0.0], [-2.5, 0.0]])
    res = run(table, rules)
    assert res.fail[0] == DAILY
    assert res.end_cal[0] == 2


def test_daily_basis_min_initial_reference_is_stricter_after_loss() -> None:
    table = table_from_paths([[-4.0], [-4.0], [-3.9]])
    base = make_rules(daily_pct=4 * F)
    assert run(table, base).fail[0] == TIMEOUT  # сумма лимита от B0: 4F, просели на 3.9F
    fp = make_rules(daily_pct=4 * F, daily_ref=DAILY_REF_MAX_BAL_EQ, daily_basis=DAILY_BASIS_MIN)
    assert run(table, fp).fail[0] == DAILY  # сумма от меньшей базы: 4F · 0.9375 < 3.9F


# --- цель и мин. дни ----------------------------------------------------------------------


def test_target_flatten_and_min_days() -> None:
    table = table_from_paths([[1.0], [1.0], [1.5], [1.0], [1.0], [1.0]])
    rules = make_rules(targets=(3 * F,), min_days=(0,))
    res = run(table, rules)
    assert res.fail[0] == OK
    assert res.pass_cal[0, 0] == 2  # третий день (ср), закрыто ровно по цели
    assert res.bal[0] == 1 + 3 * F

    rules5 = make_rules(targets=(3 * F,), min_days=(5,))
    res = run(table, rules5, f_after_target=F / 8)
    assert res.fail[0] == OK
    assert res.pass_cal[0, 0] == 4  # дотягиваем до 5 торговых дней мин. риском (P-14)
    assert res.tdays[0, 0] == 5
    assert res.bal[0] == 1 + 3 * F + 2 * F / 8


def test_target_buffer() -> None:
    table = table_from_paths([[3.0], [0.5]])
    rules = make_rules(targets=(3 * F,))
    assert run(table, rules, target_buffer=0.001).pass_cal[0, 0] == 1
    assert run(table, rules, target_buffer=0.0).pass_cal[0, 0] == 0


def test_phase_pause_and_second_phase() -> None:
    table = table_from_paths([[2.0], [2.0], [2.0], [2.0]])
    rules = make_rules(targets=(F, F), min_days=(0, 0), max_days=(-1, -1))
    res = run(table, rules)
    assert res.fail[0] == OK
    assert res.pass_cal[0, 0] == 0  # пн
    assert res.pass_cal[0, 1] == 3  # пауза 2 дня → старт в чт, цель в тот же день


# --- трейлинг -----------------------------------------------------------------------------


def test_eod_vs_intraday_trailing() -> None:
    # день 1: плавающий пик +3, закрытие +0.5; день 2: −3 от начала дня
    table = table_from_paths([[3.0, 0.5], [-3.0, 0.0]])
    eod = make_rules(max_type=MAX_BALANCE_EOD, max_dist=4 * F, lock_cap=1.0)
    eq = make_rules(max_type=MAX_EQUITY_RT, max_dist=4 * F, lock_cap=1.0)
    assert run(table, eod).fail[0] == TIMEOUT  # уровень от закрытия: 1 − 3.5F
    assert run(table, eq).fail[0] == MAX  # уровень от пика эквити: 1 − F


def test_balance_realtime_trailing_uses_intraday_balance_high() -> None:
    rules = make_rules(max_type=MAX_BALANCE_RT, max_dist=4 * F)
    table = table_from_paths([[2.0], [-4.5, 0.0]])
    assert run(table, rules).fail[0] == MAX  # H = 1 + 2F → уровень 1 − 2F
    assert run(table_from_paths([[2.0], [-3.9, 0.0]]), rules).fail[0] == TIMEOUT


def test_trailing_lock_at_initial_balance() -> None:
    rules = make_rules(max_type=MAX_BALANCE_EOD, max_dist=4 * F, lock_cap=1.0)
    ok = run(table_from_paths([[5.0], [-5.0], [0.0]]), rules)
    assert ok.fail[0] == TIMEOUT  # уровень зафиксирован на B0, E = B0 — не пробой
    bad = run(table_from_paths([[5.0], [-5.0], [-0.1]]), rules)
    assert bad.fail[0] == MAX
    no_lock = replace(rules, lock_cap=math.inf)
    assert run(table_from_paths([[5.0], [-4.5]]), no_lock).fail[0] == MAX  # уровень 1 + F


def test_lock_trigger_freezes_equity_trailing() -> None:
    rules = make_rules(max_type=MAX_EQUITY_RT, max_dist=4 * F, lock_cap=1.0, lock_trigger=4 * F)
    table = table_from_paths([[4.5, 1.0], [10.0], [-10.9, 0.0]])
    assert run(table, rules).fail[0] == TIMEOUT  # после +4F уровень стоит на B0
    no_trig = replace(rules, lock_trigger=-1.0, lock_cap=math.inf)
    assert run(table, no_trig).fail[0] == MAX


# --- предохранители бота (V7) --------------------------------------------------------------


def test_daily_guard_smooth_path_closes_at_guard_minus_slip() -> None:
    rules = make_rules(daily_pct=4 * F, max_dist=0.5)
    table = table_from_paths([[-1.5, -2.5, -3.0], [0.0]])
    res = run(table, rules, guards=True, gamma_d=0.5, guard_slip_r=0.125)
    assert res.fail[0] == TIMEOUT
    assert res.bal[0] == pytest.approx(1 - 2 * F - 0.125 * F)


def test_daily_guard_gap_through_firm_limit_is_breach() -> None:
    rules = make_rules(daily_pct=4 * F, max_dist=0.5)
    res = run(table_from_paths([[-1.0, -4.5]]), rules, guards=True, gamma_d=0.5)
    assert res.fail[0] == DAILY


def test_total_guard_stops_bot() -> None:
    rules = make_rules(max_dist=10 * F)
    res = run(table_from_paths([[-5.0], [-1.5]]), rules, guards=True, gamma_t=0.6)
    assert res.fail[0] == GUARD
    assert res.end_cal[0] == 1


def test_guard_for_program_without_daily_limit_uses_max_distance() -> None:
    # FN Instant: дневного лимита нет → дневной предохранитель = γ_D · дистанция общего (P-16)
    rules = make_rules(max_dist=6 * F, max_type=MAX_BALANCE_RT, lock_cap=1.0)
    res = run(table_from_paths([[-2.0, -3.5, 0.0], [0.0]]), rules, guards=True, gamma_d=0.5)
    assert res.fail[0] == TIMEOUT
    assert res.bal[0] == pytest.approx(1 - 3 * F - 0.1 * F)


# --- сроки и неактивность -----------------------------------------------------------------


def test_inactivity() -> None:
    rules = make_rules(inactivity_days=30)
    paths = [[0.5]] + [[] for _ in range(30)]
    res = run(table_from_paths(paths), rules)
    assert res.fail[0] == INACTIVITY


def test_horizon_timeout() -> None:
    rules = make_rules(targets=(10 * F,))
    table = table_from_paths([[0.1]] * 400)
    short = SimSettings(horizon_cal=30, ev_horizon_cal=0)
    res = run(table, rules, settings=short)
    assert res.fail[0] == TIMEOUT
    assert res.end_cal[0] == 30


# --- funded: выплаты и правила ------------------------------------------------------------


def funded_rules(payout: PayoutSpec, **kw: Any) -> RuleSet:
    return make_rules(targets=(), min_days=(), max_days=(), payout=payout, fee_usd=100.0, **kw)


def test_payout_first_after_then_cycle() -> None:
    po = PayoutSpec(id="std", label="std", split=0.8, first_after_days=21, cycle_days=14)
    rules = funded_rules(po)
    res = run(table_from_paths([[1.0]] * 40), rules)
    assert res.fail[0] == OK
    assert res.pay1_cal[0] == 21  # пн четвёртой недели
    assert res.pay1_amt[0] == pytest.approx(0.8 * 16 * F)  # 16 торговых дней по +1R


def test_payout_waits_for_next_cycle_if_profit_too_small() -> None:
    po = PayoutSpec(id="c", label="c", split=0.8, cycle_days=7, min_profit=0.05)
    rules = funded_rules(po)
    res = run(table_from_paths([[1.0]] * 40), rules)
    # 5% требует 7 дней по F (6.4 → 7); проверки только в дни цикла: cal 7, 14 → выплата на 14
    assert res.pay1_cal[0] == 14


def test_payout_consistency_best_day() -> None:
    po = PayoutSpec(id="od", label="od", split=0.9, consistency=0.4, min_profit=0.0)
    rules = funded_rules(po)
    res = run(table_from_paths([[4.0]] + [[1.0]] * 20), rules)
    assert res.pay1_cal[0] == 8  # лучший день 4F ≤ 40% только при сумме ≥ 10F: 7-й торговый день


def test_payout_profitable_days() -> None:
    po = PayoutSpec(id="3d", label="3d", split=0.6, profitable_days_n=3, profitable_days_min=0.01)
    rules = funded_rules(po)
    res = run(table_from_paths([[1.0], [2.0], [1.0], [2.0], [2.0]]), rules)
    assert res.pay1_cal[0] == 4  # дни с ≥ 1%: только +2R (1.56%) — 2-й, 4-й и 5-й


def test_payout_business_day_cycle_and_news_factor() -> None:
    po = PayoutSpec(id="b", label="b", split=0.8, cycle_business_days=5, news_counted=0.4)
    rules = funded_rules(po)
    table = table_from_paths([[2.0]] * 10, news_pos=[1.0] * 10)
    res = run(table, rules)
    assert res.pay1_cal[0] == 4
    assert res.pay1_amt[0] == pytest.approx(0.8 * (10 * F - 0.6 * 5 * F))


def test_ev_with_refund() -> None:
    po = PayoutSpec(id="std", label="std", split=0.8, first_after_days=0, cycle_days=7)
    rules = replace(funded_rules(po), refund_on_payout=1, addon_usd=5.0)
    res = run(table_from_paths([[2.0]] * 10), rules)
    st = summarize(res, rules)
    assert st.p1 == 1.0
    assert st.ev1 == pytest.approx(-105.0 + float(res.pay1_amt[0]) * 10_000 + 100.0)


def test_risk_rule_fails_funded() -> None:
    po = PayoutSpec(id="c", label="c", split=0.8, cycle_days=7)
    rules = funded_rules(po, risk_rule_max=0.03)
    table = table_from_paths([[1.0], [1.0]], risk_open_max=[1.0, 2.0])
    assert run(table, rules, f=0.02).fail[0] == RULE  # второй день: 2 позиции × 2% > 3%
    assert run(table, rules, f=0.01).fail[0] != RULE


def test_striking_counts_strikes() -> None:
    po = PayoutSpec(id="c", label="c", split=0.8, cycle_days=60)
    rules = funded_rules(po, striking_thr=0.012, strikes_to_close=2)
    table = table_from_paths([[0.5]] * 4, single_loss=[2.0, 0.0, 2.0, 0.0])
    assert run(table, rules, f=0.01).fail[0] == RULE  # два убытка идеи по 2% > 1.2%
    assert run(table, rules, f=0.005).fail[0] == TIMEOUT


# --- V5: numba и интерпретатор дают одно и то же ------------------------------------------


@pytest.mark.parametrize("guards", [True, False])
def test_jit_matches_python(guards: bool) -> None:
    rng = np.random.default_rng(3)
    paths = [
        list(np.round(rng.normal(0.1, 1.2, size=int(rng.integers(1, 4))), 2)) for _ in range(60)
    ]
    table = table_from_paths(paths, risk_open_max=[1.0] * 60)
    po = PayoutSpec(id="std", label="std", split=0.8, first_after_days=21, cycle_days=14)
    rules = make_rules(
        targets=(0.08, 0.05),
        min_days=(3, 3),
        max_days=(-1, -1),
        daily_pct=0.05,
        max_dist=0.10,
        payout=po,
        fee_usd=200.0,
        refund_on_payout=1,
        risk_rule_max=0.03,
    )
    idx = rng.integers(0, 60, size=(200, 270)).astype(np.int32)
    risk = RiskScheme(f=(0.01, 0.0075, 0.005), guards=guards, r1_x=0.3, r2=True)
    st = SimSettings()
    a = simulate(table, rules, risk, idx, st, jit=True)
    b = simulate(table, rules, risk, idx, st, jit=False)
    for name in ("fail", "fail_slot", "pass_cal", "tdays", "end_cal", "pay1_cal", "npay"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    for name in ("pay1_amt", "pay_sum", "bal"):
        assert np.allclose(getattr(a, name), getattr(b, name), rtol=0, atol=1e-12), name
