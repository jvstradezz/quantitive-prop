"""Оптимизатор риска по фазам (MONTE_CARLO §10).

Цель — максимум P(1-я выплата) при EV₁ > 0 (для шаблонов без выплат — P(funded)).
1. Раздельно по фазам: P(пройти фазу) по сетке риска, `opt_paths` путей, CRN.
2. Совместно: покоординатный спуск по (f₁, f₂, …, f_F, R1, R2) от точки шага 1, два прохода.
3. Переоценка топ-5 на основной матрице (другой seed, `n_paths` путей).
4. Устойчивый выбор: наименьший риск, чья P ниже максимума не больше чем на
   `max(1 п.п., полуширина CI шума MC)` (P-19).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from pbl.prop.engine import MCEngine
from pbl.prop.metrics import Stats
from pbl.prop.rules import RuleSet
from pbl.prop.sim import RiskScheme

GRID: tuple[float, ...] = (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03)
R1_OPTIONS: tuple[float | None, ...] = (None, 0.3, 0.5)
R2_OPTIONS: tuple[bool, ...] = (False, True)
NOISE_FLOOR = 0.01


@dataclass(frozen=True)
class Candidate:
    risk: RiskScheme
    stats: Stats

    @property
    def ev_ok(self) -> bool:
        ev = self.stats.ev1
        return math.isnan(ev) or ev > 0

    def score(self) -> tuple[bool, float]:
        return (self.ev_ok, self.stats.objective)


@dataclass(frozen=True)
class OptResult:
    best: Candidate
    top: tuple[Candidate, ...]
    curves: dict[int, list[tuple[float, float]]]  # слот → [(f, P фазы)]
    infeasible: dict[int, list[tuple[float, str]]]  # слот → [(f, почему нельзя)]
    ev_positive: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def feasible(engine: MCEngine, rules: RuleSet, slot: int, f: float) -> str | None:
    """Ограничения C1–C3 на риск слота по всем дням выборки; None — допустим."""
    table = engine.table
    k, share = table.kappa(f * rules.size_usd)
    if share > 0.05:
        return f"κ > 1.5 у {share:.0%} сделок (счёт мал для таких стопов)"
    if slot == rules.n_phases:
        traded = table.n_opened > 0
        if rules.risk_rule_max > 0:
            worst = float((f * k * table.risk_open_max)[traded].max(initial=0.0))
            if worst > rules.risk_rule_max + 1e-12:
                lim = 100 * rules.risk_rule_max
                return f"Risk Rule: открытый риск до {100 * worst:.2f}% > {lim:g}%"
        if rules.striking_thr > 0:
            worst = float((f * k * table.single_loss)[traded].max(initial=0.0))
            if worst >= rules.striking_thr:
                return (
                    f"Striking: убыток идеи до {100 * worst:.2f}% ≥ {100 * rules.striking_thr:g}%"
                )
    return None


def _robust_pick(cands: list[Candidate], size_key: Callable[[Candidate], float]) -> Candidate:
    pool = [c for c in cands if c.ev_ok] or cands
    top = max(pool, key=lambda c: c.stats.objective)
    lo, hi = top.stats.p1_ci if not math.isnan(top.stats.p1) else top.stats.p_funded_ci
    tol = max(NOISE_FLOOR, (hi - lo) / 2)
    near = [c for c in pool if c.stats.objective >= top.stats.objective - tol]
    return min(near, key=lambda c: (size_key(c), -c.stats.objective))


def optimize(engine: MCEngine, rules: RuleSet, base: RiskScheme | None = None) -> OptResult:
    base = base or RiskScheme(f=tuple(0.01 for _ in range(rules.n_slots)))
    if len(base.f) != rules.n_slots:
        raise ValueError("base.f должен покрывать все слоты")
    has_payout = rules.payout is not None
    slots = list(range(rules.n_phases)) + ([rules.n_phases] if has_payout else [])
    if not slots:
        raise ValueError("нечего оптимизировать: нет фаз и нет выплат")
    notes: list[str] = []

    # допустимая сетка по слотам
    grid: dict[int, list[float]] = {}
    infeasible: dict[int, list[tuple[float, str]]] = {}
    for s in slots:
        ok: list[float] = []
        bad: list[tuple[float, str]] = []
        for f in GRID:
            why = feasible(engine, rules, s, f)
            if why:
                bad.append((f, why))
            else:
                ok.append(f)
        grid[s] = ok
        infeasible[s] = bad
        if not ok:
            notes.append(f"слот {s}: ни один риск сетки не допустим ({bad[0][1]})")
    if any(not grid[s] for s in slots):
        best = Candidate(base, engine.run(rules, base))
        return OptResult(best, (best,), {}, infeasible, False, tuple(notes))

    # шаг 1: раздельно по фазам
    curves: dict[int, list[tuple[float, float]]] = {}
    f_start = list(base.f)
    for s in slots:
        cands = []
        for f in grid[s]:
            fs = list(base.f)
            fs[s] = f
            st = engine.run(rules, replace(base, f=tuple(fs)), opt=True, start_slot=s, stop_slot=s)
            cands.append((f, st))
        curves[s] = [(f, st.success) for f, st in cands]
        best_p = max(st.success for _, st in cands)
        tol = max(NOISE_FLOOR, max((st.success_ci[1] - st.success_ci[0]) / 2 for _, st in cands))
        f_start[s] = min(f for f, st in cands if st.success >= best_p - tol)
    if not has_payout:
        f_start[rules.n_phases] = f_start[rules.n_phases - 1]

    # шаг 2: покоординатный спуск на полном MC
    cache: dict[RiskScheme, Candidate] = {}

    def evaluate(risk: RiskScheme) -> Candidate:
        if risk not in cache:
            cache[risk] = Candidate(risk, engine.run(rules, risk, opt=True))
        return cache[risk]

    cur = evaluate(replace(base, f=tuple(f_start)))
    for _ in range(2):
        for s in slots:
            for f in grid[s]:
                fs = list(cur.risk.f)
                fs[s] = f
                c = evaluate(replace(cur.risk, f=tuple(fs)))
                if c.score() > cur.score():
                    cur = c
        for r1 in R1_OPTIONS:
            c = evaluate(replace(cur.risk, r1_x=r1))
            if c.score() > cur.score():
                cur = c
        if rules.n_phases:
            for r2 in R2_OPTIONS:
                c = evaluate(replace(cur.risk, r2=r2))
                if c.score() > cur.score():
                    cur = c

    # шаг 3: топ-5 на новом seed и полном числе путей
    ranked = sorted(cache.values(), key=lambda c: c.score(), reverse=True)[:5]
    top = [Candidate(c.risk, engine.run(rules, c.risk)) for c in ranked]

    # шаг 4: устойчивый выбор — наименьший риск в пределах шума
    def risk_size(c: Candidate) -> float:
        return float(np.sum(c.risk.f)) + (0.0 if c.risk.r1_x is None else -1e-6)

    best = _robust_pick(top, risk_size)
    if not best.ev_ok:
        notes.append("ни один вариант риска не дал EV₁ > 0")
    return OptResult(
        best=best,
        top=tuple(sorted(top, key=lambda c: c.score(), reverse=True)),
        curves=curves,
        infeasible=infeasible,
        ev_positive=best.ev_ok,
        notes=tuple(notes),
    )
