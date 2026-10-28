"""Сводка путей → метрики программы (MONTE_CARLO §7–9)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pbl.prop.bootstrap import wilson
from pbl.prop.rules import RuleSet
from pbl.prop.sim import OK, REASONS, PathResults


@dataclass(frozen=True)
class Stats:
    """Метрики одной программы при одной схеме риска. Доли — 0…1, деньги — $."""

    key: str
    n_paths: int
    success: float  # доля путей без провала (для полного прогона — P(1-я выплата) или P(funded))
    success_ci: tuple[float, float]
    p_phase: tuple[float, ...]  # P(пройти фазу k | дошли до k)
    p_funded: float
    p_funded_ci: tuple[float, float]
    p_payout_given_funded: float
    p1: float  # P(1-я выплата); nan — выплаты не моделируются
    p1_ci: tuple[float, float]
    days_funded_median: float
    days_funded_p90: float
    days_payout_median: float
    days_payout_p90: float
    fail_share: dict[str, float]  # доля всех путей по причине провала
    mean_payout1_usd: float  # E[s Π₁ | выплата]
    ev1: float
    ev180: float
    breakeven_p: float
    attempts: float  # 1 / P1
    expected_spend: float  # C / P1
    expected_days_with_retries: float
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def objective(self) -> float:
        """Целевая доля: P(1-я выплата), если выплаты есть, иначе P(funded)."""
        return self.p1 if not math.isnan(self.p1) else self.p_funded


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else math.nan


def summarize(res: PathResults, rules: RuleSet) -> Stats:
    n = res.n
    ok = res.fail == OK
    n_ok = int(ok.sum())
    full = res.start_slot == 0
    np_ = rules.n_phases
    first = res.start_slot

    # условные P по фазам
    p_phase: list[float] = []
    reached = np.ones(n, dtype=np.bool_)
    for k in range(first, min(res.stop_slot, np_ - 1) + 1):
        passed = res.pass_cal[:, k] >= 0
        p_phase.append(float(passed[reached].mean()) if reached.any() else math.nan)
        reached = passed
    if np_ == 0 or res.start_slot >= np_:
        funded_mask = np.ones(n, dtype=np.bool_)
        days_f = np.zeros(n)
    else:
        funded_mask = res.pass_cal[:, np_ - 1] >= 0
        days_f = res.pass_cal[funded_mask, np_ - 1].astype(np.float64) + 1
    n_f = int(funded_mask.sum())

    has_payout = rules.payout is not None and res.stop_slot >= np_
    paid = res.pay1_cal >= 0
    p1 = float(paid.mean()) if has_payout else math.nan
    p1_ci = wilson(int(paid.sum()), n) if has_payout else (math.nan, math.nan)
    days_p = res.pay1_cal[paid].astype(np.float64) + 1

    shares = {REASONS[r]: float((res.fail == r).mean()) for r in range(1, len(REASONS))}

    size = rules.size_usd
    cost = rules.cost_usd
    refund1 = rules.refund_usd if rules.refund_on_payout == 1 else 0.0
    mean_pay1 = float(res.pay1_amt[paid].mean() * size) if paid.any() else 0.0
    if has_payout and cost is not None and full:
        ev1 = -cost + float(res.pay1_amt.mean() * size) + p1 * refund1
        ev180 = -cost + float(res.pay_sum.mean() * size) + rules.refund_usd * float(
            res.refund.mean()
        )
        denom = mean_pay1 + refund1
        breakeven = cost / denom if denom > 0 else math.inf
        attempts = 1 / p1 if p1 > 0 else math.inf
        spend = cost / p1 if p1 > 0 else math.inf
    else:
        ev1 = ev180 = breakeven = math.nan
        attempts = spend = math.nan
    fail_days = res.end_cal[~ok].astype(np.float64)
    if has_payout and p1 > 0:
        t_ok = float(days_p.mean())
        t_fail = float(fail_days.mean()) if fail_days.size else 0.0
        exp_days = t_ok + (1 / p1 - 1) * t_fail
    else:
        exp_days = math.nan

    return Stats(
        key=rules.key,
        n_paths=n,
        success=n_ok / n,
        success_ci=wilson(n_ok, n),
        p_phase=tuple(p_phase),
        p_funded=n_f / n,
        p_funded_ci=wilson(n_f, n),
        p_payout_given_funded=float(paid[funded_mask].mean()) if has_payout and n_f else math.nan,
        p1=p1,
        p1_ci=p1_ci,
        days_funded_median=_pct(days_f, 50),
        days_funded_p90=_pct(days_f, 90),
        days_payout_median=_pct(days_p, 50),
        days_payout_p90=_pct(days_p, 90),
        fail_share=shares,
        mean_payout1_usd=mean_pay1,
        ev1=ev1,
        ev180=ev180,
        breakeven_p=breakeven,
        attempts=attempts,
        expected_spend=spend,
        expected_days_with_retries=exp_days,
        notes=rules.notes,
    )
