"""Аналитические проверки MC (MONTE_CARLO §12): V1 разорение, V2 длительность, V4 свойства,
V6 эталонные стратегии."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pbl.config import load_firms
from pbl.prop.bootstrap import stationary_indices, wilson
from pbl.prop.days import DayTable, day_table_from_trades
from pbl.prop.engine import MCConfig, MCEngine
from pbl.prop.rules import MAX_STATIC, RuleSet, compile_offer
from pbl.prop.sim import OK, RiskScheme, SimSettings, simulate
from pbl.prop.synthetic import coin_trades, synthetic_trades
from pbl.schemas import Firm, ServerTime

F = 2.0**-7
SERVER = ServerTime(winter_utc_offset=2, summer_utc_offset=3, dst_calendar="us")
Z99 = 2.576


def ruin_rules(a: int, b: int, **kw: Any) -> RuleSet:
    """Нижний барьер −a·f, цель +b·f; уровни сдвинуты на f/2, чтобы шаги ±f попадали точно."""
    base: dict[str, Any] = {
        "key": "ruin",
        "label": "ruin",
        "size_usd": 10_000.0,
        "targets": ((b - 0.5) * F,),
        "min_days": (0,),
        "max_days": (-1,),
        "daily_pct": None,
        "daily_ref": 0,
        "daily_basis": 0,
        "max_type": MAX_STATIC,
        "max_dist": (a - 0.5) * F,
        "lock_cap": math.inf,
        "lock_trigger": -1.0,
        "floor_after_payout": 0,
        "inactivity_days": -1,
        "payout": None,
    }
    base.update(kw)
    return RuleSet(**base)


def coin_table(p: float, n: int = 200) -> DayTable:
    return day_table_from_trades(coin_trades(n, p), SERVER)


def run_iid(table: DayTable, rules: RuleSet, n_paths: int, seed: int, width: int = 1500) -> Any:
    idx = stationary_indices(np.random.default_rng(seed), table.n_days, n_paths, width, 1.0)
    risk = RiskScheme(f=(F,) * rules.n_slots, guards=False, target_buffer=0.0)
    return simulate(table, rules, risk, idx, SimSettings(horizon_cal=2100, ev_horizon_cal=0))


def gambler(p: float, a: int, b: int) -> float:
    q = 1 - p
    if p == q:
        return a / (a + b)
    r = q / p
    return float((1 - r**a) / (1 - r ** (a + b)))


@pytest.mark.parametrize(("p", "a", "b"), [(0.55, 5, 8), (0.5, 6, 4), (0.45, 10, 5)])
def test_v1_gamblers_ruin(p: float, a: int, b: int) -> None:
    table = coin_table(p)
    assert table.n_trading_days == 200
    n = 20_000
    res = run_iid(table, ruin_rules(a, b), n, seed=11)
    wins = int((res.fail == OK).sum())
    lo, hi = wilson(wins, n, z=Z99)
    assert lo <= gambler(p, a, b) <= hi


def _trading_days_until(cal: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Число дней Пн–Пт в [0, cal] — сделок при одной сделке в день."""
    out: npt.NDArray[np.int64] = (cal // 7) * 5 + np.minimum(cal % 7, 4) + 1
    return out


def test_v2_duration_fair_coin() -> None:
    a, b = 5, 7
    res = run_iid(coin_table(0.5), ruin_rules(a, b), 20_000, seed=5)
    end = np.where(res.pass_cal[:, 0] >= 0, res.pass_cal[:, 0], res.end_cal)
    t = _trading_days_until(end.astype(np.int64)).astype(np.float64)
    assert (res.fail != 4).all()  # не упёрлись в горизонт
    half = Z99 * t.std(ddof=1) / math.sqrt(t.size)
    assert abs(t.mean() - a * b) <= half


# --- V4: свойства --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def edge_table() -> DayTable:
    return day_table_from_trades(
        synthetic_trades(n_days=300, win_rate=0.5, rr=1.3, cost_r=0.05, seed=21), SERVER
    )


def _p_phase1(table: DayTable, rules: RuleSet, f: float, seed: int = 1) -> float:
    idx = stationary_indices(np.random.default_rng(seed), table.n_days, 600, 270, 3.0)
    risk = RiskScheme(f=(f,) * rules.n_slots, guards=False)
    return float((simulate(table, rules, risk, idx, stop_slot=0).fail == OK).mean())


@given(
    t1=st.integers(2, 12),
    dt=st.integers(1, 6),
    d=st.integers(4, 12),
    f=st.sampled_from([0.005, 0.01, 0.02]),
)
def test_v4_higher_target_never_raises_p(
    edge_table: DayTable, t1: int, dt: int, d: int, f: float
) -> None:
    lo = ruin_rules(d, 1, targets=(t1 / 100,), max_dist=d / 100)
    hi = replace(lo, targets=((t1 + dt) / 100,))
    assert _p_phase1(edge_table, hi, f) <= _p_phase1(edge_table, lo, f)


@given(
    t=st.integers(2, 12),
    d1=st.integers(3, 10),
    dd=st.integers(1, 6),
    f=st.sampled_from([0.005, 0.01, 0.02]),
)
def test_v4_wider_limit_never_lowers_p(
    edge_table: DayTable, t: int, d1: int, dd: int, f: float
) -> None:
    narrow = ruin_rules(d1, 1, targets=(t / 100,), max_dist=d1 / 100, daily_pct=0.03)
    wide = replace(narrow, max_dist=(d1 + dd) / 100)
    assert _p_phase1(edge_table, wide, f) >= _p_phase1(edge_table, narrow, f)


@given(extra=st.floats(0.01, 0.3), f=st.sampled_from([0.005, 0.01]))
def test_v4_higher_costs_never_raise_p(extra: float, f: float) -> None:
    trades = synthetic_trades(n_days=200, win_rate=0.5, rr=1.3, cost_r=0.05, seed=4)
    costly = synthetic_trades(n_days=200, win_rate=0.5, rr=1.3, cost_r=0.05 + extra, seed=4)
    rules = ruin_rules(8, 1, targets=(0.06,), max_dist=0.08, daily_pct=0.04)
    base = _p_phase1(day_table_from_trades(trades, SERVER), rules, f)
    worse = _p_phase1(day_table_from_trades(costly, SERVER), rules, f)
    assert worse <= base


@pytest.fixture(scope="module")
def fundednext() -> Firm:
    return next(f for f in load_firms() if f.id == "fundednext")


def test_v4_probability_chain_and_seed(fundednext: Firm, edge_table: DayTable) -> None:
    prog = next(p for p in fundednext.programs if p.id == "fn-stellar-2step")
    offer = next(o for o in prog.iter_offers() if o.size.size_usd == 25_000)
    rules = compile_offer(fundednext.id, prog, offer)
    cfg = MCConfig(n_paths=2_000, opt_paths=500, seed=9, outer_boot=0)
    risk = RiskScheme(f=(0.01, 0.0075, 0.005))
    a = MCEngine(edge_table, cfg).run(rules, risk)
    b = MCEngine(edge_table, cfg).run(rules, risk)
    assert a == b  # одинаковый seed — одинаковый результат
    assert a.p1 <= a.p_funded <= a.p_phase[0]
    c = MCEngine(edge_table, replace(cfg, seed=10)).run(rules, risk)
    assert c != a


# --- V6: эталонные стратегии ---------------------------------------------------------------


def test_v6_coin_with_costs_has_negative_ev_everywhere(fundednext: Firm) -> None:
    trades = synthetic_trades(n_days=400, win_rate=0.5, rr=1.0, cost_r=0.08, seed=8)
    table = day_table_from_trades(trades, fundednext.server_time)
    eng = MCEngine(table, MCConfig(n_paths=2_000, seed=2, outer_boot=0))
    checked = 0
    for prog in fundednext.programs:
        for offer in prog.iter_offers():
            rules = compile_offer(fundednext.id, prog, offer)
            if rules.payout is None:
                continue
            for f in (0.005, 0.01, 0.02):
                st_ = eng.run(rules, RiskScheme(f=(f,) * rules.n_slots))
                assert st_.ev1 < 0, (rules.key, f)
                checked += 1
    assert checked > 50


def test_v6_probability_grows_with_edge(fundednext: Firm) -> None:
    prog = next(p for p in fundednext.programs if p.id == "fn-stellar-2step")
    offer = next(o for o in prog.iter_offers() if o.size.size_usd == 25_000)
    rules = compile_offer(fundednext.id, prog, offer)
    ps = []
    for wr in (0.40, 0.45, 0.50, 0.55):
        trades = synthetic_trades(n_days=500, win_rate=wr, rr=1.5, cost_r=0.06, seed=3)
        table = day_table_from_trades(trades, fundednext.server_time)
        eng = MCEngine(table, MCConfig(n_paths=2_000, seed=2, outer_boot=0))
        ps.append(eng.run(rules, RiskScheme(f=(0.0075, 0.0075, 0.005))).p1)
    assert ps == sorted(ps)
    assert ps[-1] - ps[0] > 0.3
