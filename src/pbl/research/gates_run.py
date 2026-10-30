"""Конвейер гейтов для идеи из реестра: пробы — в счётчик вариантов, holdout — один раз на версию.

`robustness.gates` не знает о реестре (зависимости только «вправо»), поэтому здесь:
1. все пробы прогона (`Space.with_twins()`) пишутся в `research/registry/variants/<IDEA>.parquet`
   с дневным рядом IS (D-028); пробы идеи из прошлых прогонов входят в N для DSR сырыми;
2. holdout открывается, только если на этой версии он ещё не открыт (`RegistryGuard`).
Кодить новую идею: `Study` + функция «параметры → `VariantSpec`», затем `run_idea_gates`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import polars as pl

from pbl.config import REGISTRY
from pbl.research.variants import DailyR, VariantLog, VariantSpec
from pbl.robustness.gates import (
    GateConfig,
    GateReport,
    HoldoutReopened,
    Runner,
    run_holdout,
    run_is,
    run_s3,
)
from pbl.robustness.splits import Splits
from pbl.robustness.study import Params, Space, Study, params_key
from pbl.robustness.trades import window

SpecFn = Callable[[Params], VariantSpec]


def daily_r(t: pl.DataFrame, splits: Splits) -> DailyR:
    """Дневной ряд IS варианта для счётчика (дни Пн–Пт, сумма `r_net`)."""
    t = window(t, None, splits.cut_ord)
    end = splits.cut - dt.timedelta(days=1)
    if t.is_empty():
        return DailyR(splits.first, end, (), (), 0, 0.0, 0.0)
    g = t.group_by("day").agg(pl.col("r_net").sum()).sort("day")
    r = t["r_net"].to_numpy()
    return DailyR(
        start=splits.first,
        end=end,
        dates=tuple(dt.date.fromordinal(int(d)) for d in g["day"]),
        r=tuple(float(x) for x in g["r_net"]),
        n_trades=t.height,
        win_r=float(r[r > 0].sum()),
        loss_r=float(r[r < 0].sum()),
    )


@dataclass
class RegistryGuard:
    """Holdout версии открывается один раз: повтор на той же версии — `HoldoutReopened`."""

    opened_at: dt.date | None
    version: str
    seen_versions: frozenset[str]
    opened: dict[str, dt.date] = field(default_factory=dict)

    def open(self, key: str, on: dt.date) -> None:
        if self.opened or (self.opened_at is not None and self.version in self.seen_versions):
            raise HoldoutReopened(
                f"holdout уже открыт {self.opened_at} на версии {self.version}: поднимите версию"
            )
        self.opened[key] = on


@dataclass
class IdeaGates:
    report: GateReport
    log: VariantLog
    new_rows: int
    guard: RegistryGuard


def run_idea_gates(
    study: Study,
    space: Space,
    spec_fn: SpecFn,
    idea_id: str,
    cfg: GateConfig,
    *,
    version: str = "v1",
    holdout_opened_at: dt.date | None = None,
    registry: Path = REGISTRY,
    save: bool = False,
) -> IdeaGates:
    """S1 → S3 с записью проб в счётчик идеи. `save` — записать Parquet счётчика."""
    log = VariantLog.for_idea(idea_id, registry)
    seen = frozenset(log.frame()["version"].to_list())
    run = Runner(study)
    variants = space.with_twins()
    specs = {params_key(v): spec_fn(v) for v in variants}
    if any(s.version != version for s in specs.values()):
        raise ValueError("версия в VariantSpec не совпадает с версией прогона")
    ours = {s.variant_id for s in specs.values()}
    prior = log.frame().filter(~pl.col("variant_id").is_in(list(ours)))
    cfg = replace(
        cfg,
        extra_trials=prior.height,
        prior_sharpes=tuple(float(x) for x in prior["sharpe_daily"].to_list()),
    )
    rep = run_is(study, space, cfg, run)
    run_id = f"{idea_id.lower()}-{version}"
    new = sum(
        log.add(specs[params_key(v)], daily_r(run.frame(v), rep.splits), run_id=run_id, on=cfg.on)
        for v in variants
    )
    guard = RegistryGuard(holdout_opened_at, version, seen)
    if rep.finalist is not None and rep.picks:
        ho = run_holdout(study, space, rep, cfg, guard, run)
        if ho.passed and cfg.firms:
            run_s3(study, space, rep, cfg, run)
    if save:
        log.save()
    return IdeaGates(rep, log, new, guard)
