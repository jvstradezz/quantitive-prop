"""MC-движок для одной истории сделок: общие случайные числа, прогоны, стрессы, внешний бутстрэп.

CRN (MONTE_CARLO §5): одна матрица индексов дней на все программы, риски и стрессы при одном
seed, поэтому разница между вариантами — от правил, а не от шума. Потоки seed —
`SeedSequence(seed).spawn(...)` (§13).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from pbl.prop.bootstrap import block_length, stationary_indices
from pbl.prop.days import DayTable, TradeArrays, build_day_table
from pbl.prop.metrics import Stats, summarize
from pbl.prop.rules import RuleSet
from pbl.prop.sim import PathResults, RiskScheme, SimSettings, index_width, simulate
from pbl.prop.stress import STRESSES, StressSpec, apply_trade_stress, worst_period_days

SEED_MAIN, SEED_OPT, SEED_STRESS, SEED_OUTER = range(4)


@dataclass(frozen=True)
class MCConfig:
    n_paths: int = 10_000
    opt_paths: int = 2_000
    seed: int = 42
    block: float | None = None  # None — Politis–White в [3, 20]
    settings: SimSettings = field(default_factory=SimSettings)
    outer_boot: int = 200
    outer_paths: int = 2_000
    skip_repeats: int = 20


class MCEngine:
    """Таблица дней + матрицы индексов (основная и для оптимизатора) на одном seed."""

    def __init__(self, table: DayTable, cfg: MCConfig, trades: TradeArrays | None = None) -> None:
        self.table = table
        self.trades = trades
        self.cfg = cfg
        self.block = block_length(table.db, cfg.block)
        self.width = index_width(cfg.settings)
        self.seeds = np.random.SeedSequence(cfg.seed).spawn(4)
        self.idx_main = self._indices(self.seeds[SEED_MAIN], table, cfg.n_paths)
        self.idx_opt = self._indices(self.seeds[SEED_OPT], table, cfg.opt_paths)

    def _indices(
        self, seed: np.random.SeedSequence, table: DayTable, n_paths: int
    ) -> npt.NDArray[np.int32]:
        rng = np.random.default_rng(seed)
        return stationary_indices(rng, table.n_days, n_paths, self.width, self.block)

    def run(
        self,
        rules: RuleSet,
        risk: RiskScheme,
        *,
        opt: bool = False,
        start_slot: int = 0,
        stop_slot: int | None = None,
    ) -> Stats:
        idx = self.idx_opt if opt else self.idx_main
        res = simulate(
            self.table,
            rules,
            risk,
            idx,
            self.cfg.settings,
            start_slot=start_slot,
            stop_slot=stop_slot,
        )
        return summarize(res, rules)

    # --- стрессы (MONTE_CARLO §6) --------------------------------------------------------

    def run_stress(self, rules: RuleSet, risk: RiskScheme, spec: StressSpec) -> Stats:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.cfg.seed, SEED_STRESS, STRESSES.index(spec)])
        )
        risk_s = replace(risk, guard_slip_r=risk.guard_slip_r * spec.guard_slip_mult)
        if spec.worst_years:
            days, _ = worst_period_days(self.table)
            sub = self.table.take(days)
            idx = self._indices(np.random.SeedSequence([self.cfg.seed, 99]), sub, self.cfg.n_paths)
            return summarize(simulate(sub, rules, risk_s, idx, self.cfg.settings), rules)
        if spec.id == "base":
            return self.run(rules, risk)
        if self.trades is None:
            raise ValueError("для стрессов на сделках нужен список сделок")
        repeats = self.cfg.skip_repeats if spec.skip > 0 else 1
        per = max(1, self.cfg.n_paths // repeats)
        parts: list[PathResults] = []
        for _ in range(repeats):
            stressed = apply_trade_stress(self.trades, spec, rng)
            table = build_day_table(stressed, eps=self.table.eps)
            idx = self._indices(np.random.SeedSequence(int(rng.integers(2**31))), table, per)
            parts.append(simulate(table, rules, risk_s, idx, self.cfg.settings))
        return summarize(_concat(parts), rules)

    # --- неопределённость по выборке (MONTE_CARLO §7) -------------------------------------

    def outer_bootstrap(
        self, rules: RuleSet, risk: RiskScheme, n_boot: int | None = None
    ) -> tuple[float, float, float, float]:
        """5-й и 95-й перцентили целевой P и медианы дней до funded по псевдо-историям."""
        b = self.cfg.outer_boot if n_boot is None else n_boot
        rng = np.random.default_rng(self.seeds[SEED_OUTER])
        n = self.table.n_days
        ps: list[float] = []
        days: list[float] = []
        for _ in range(b):
            hist = stationary_indices(rng, n, 1, n, self.block)[0]
            pseudo = self.table.take(hist.astype(np.int64))
            idx = stationary_indices(rng, n, self.cfg.outer_paths, self.width, self.block)
            st = summarize(simulate(pseudo, rules, risk, idx, self.cfg.settings), rules)
            ps.append(st.objective)
            days.append(st.days_funded_median)
        arr = np.asarray(ps)
        d = np.asarray(days)
        d = d[np.isfinite(d)]
        lo, hi = (float(np.percentile(arr, 5)), float(np.percentile(arr, 95)))
        if d.size:
            return lo, hi, float(np.percentile(d, 5)), float(np.percentile(d, 95))
        return lo, hi, float("nan"), float("nan")


def _concat(parts: list[PathResults]) -> PathResults:
    first = parts[0]
    cols = {
        name: np.concatenate([getattr(p, name) for p in parts])
        for name in (
            "fail",
            "fail_slot",
            "pass_cal",
            "tdays",
            "end_cal",
            "pay1_cal",
            "pay1_amt",
            "pay_sum",
            "npay",
            "refund",
            "bal",
        )
    }
    return PathResults(**cols, start_slot=first.start_slot, stop_slot=first.stop_slot)
