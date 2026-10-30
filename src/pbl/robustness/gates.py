"""Конвейер гейтов S1–S3 для любой `Study` (VALIDATION §2–6, §10).

Порядок защищает holdout:
1. IS (~70% истории): S1 по каждой пробе, walk-forward и WFE, PBO, `N_eff` и DSR, затем для
   топ-кандидатов по Sharpe IS — чувствительность ±10/20%, годы, инструменты, фильтры, автоотказ.
2. Holdout открывается один раз (`HoldoutGuard`) и только для финалиста — первого кандидата,
   прошедшего все гейты IS: полоса MC из тестов WF, E > 0, инструменты, фильтры.
3. S3: проп-MC на склейке тестов WF и holdout финалиста + флаги аудита.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from pbl.prop.engine import MCConfig
from pbl.prop.report import PropReport, Row, run_report
from pbl.robustness import thresholds as th
from pbl.robustness.audit import AuditFlag, loss_to_win, streak_flag, streak_quantile
from pbl.robustness.autoreject import AutoReject, AutoRejectInput, autoreject
from pbl.robustness.bands import BandResult, DayStats, DiffResult, bucket_diff, daystats
from pbl.robustness.bands import holdout_band as mc_band
from pbl.robustness.neff import Clusters, DSRResult, correlation_clusters, dsr_effective
from pbl.robustness.pbo import PBOResult, pbo_cscv
from pbl.robustness.splits import Splits, wf_windows
from pbl.robustness.stats import (
    edge_score,
    instrument_edges,
    moments,
    positive_instruments,
    s1_stats,
    series_sharpe,
    year_stability,
)
from pbl.robustness.study import FilterDef, Params, Space, Study, params_key
from pbl.robustness.trades import daily, daily_matrix, trade_day, window
from pbl.robustness.walkforward import WFPick, walk_forward, wfe
from pbl.schemas import Firm, Trade
from pbl.schemas.strategy import ParamValue

IS_GATES = (
    "S1",
    "WFE",
    "DSR/PBO",
    "чувствительность",
    "годы",
    "инструменты",
    "фильтры",
    "автоотказ",
)
HOLDOUT_GATES = ("полоса MC", "E > 0", "инструменты", "фильтры")


class HoldoutReopened(RuntimeError):
    """Holdout этой версии уже открыт: изменения → новая версия и новая запись в счётчике."""


class HoldoutGuard(Protocol):
    def open(self, key: str, on: dt.date) -> None:
        """Записать факт открытия; повторное открытие — `HoldoutReopened`."""
        ...


@dataclass
class OnceGuard:
    """Holdout открывается один раз за жизнь объекта (для синтетики и тестов)."""

    opened: dict[str, dt.date] = field(default_factory=dict)

    def open(self, key: str, on: dt.date) -> None:
        if self.opened:
            raise HoldoutReopened(f"holdout уже открыт {next(iter(self.opened.values()))}")
        self.opened[key] = on


@dataclass(frozen=True)
class GateConfig:
    is_fraction: float = 0.7
    wf: tuple[int, int, int] = (24, 6, 6)
    top: int = 5
    pbo_blocks: int = 16
    n_paths: int = 10_000
    seed: int = 42
    extra_trials: int = 0  # пробы идеи вне этого прогона (из счётчика), в N сырыми
    prior_sharpes: tuple[float, ...] = ()
    firms: tuple[Firm, ...] = ()
    mc: MCConfig | None = None
    program_ids: tuple[str, ...] | None = None
    on: dt.date = field(default_factory=lambda: dt.datetime.now(dt.UTC).date())


# --- кандидаты на IS ----------------------------------------------------------------------


@dataclass(frozen=True)
class SensRow:
    param: str
    factor: float
    value: ParamValue
    n: int
    pf: float


@dataclass(frozen=True)
class FilterCheck:
    name: str
    diff: DiffResult | None
    score: float  # mean R × √n варианта с фильтром
    twin_score: float  # то же без фильтра
    holds: bool


@dataclass
class Candidate:
    params: dict[str, ParamValue]
    key: str
    s1: dict[str, float]
    sharpe: float
    dsr: DSRResult
    sens: list[SensRow]
    years: pl.DataFrame
    instruments: dict[str, tuple[int, float]]
    filters: list[FilterCheck]
    reject: AutoReject | None
    gates: dict[str, bool | None]

    @property
    def passed(self) -> bool:
        return all(v is not False for v in self.gates.values())

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.gates.items() if v is False]


@dataclass
class HoldoutStage:
    opened_at: dt.date
    band: BandResult
    stats: dict[str, float]
    instruments: dict[str, tuple[int, float]]
    filters: list[FilterCheck]
    gates: dict[str, bool | None]

    @property
    def passed(self) -> bool:
        return all(v is not False for v in self.gates.values())


@dataclass
class S3Stage:
    trades: list[Trade]
    prop: PropReport
    passed: bool
    best: Row | None
    audit: list[AuditFlag]


@dataclass
class GateReport:
    strategy_id: str
    splits: Splits
    s1: pl.DataFrame
    n_trials: int
    picks: list[WFPick]
    wfe: float
    pbo: PBOResult | None
    clusters: Clusters
    candidates: list[Candidate]
    holdout: HoldoutStage | None = None
    s3: S3Stage | None = None

    @property
    def finalist(self) -> Candidate | None:
        return next((c for c in self.candidates if c.passed), None)

    @property
    def verdict(self) -> str:
        return verdict(self)


class Runner:
    """Кэш сделок вариантов: каждая проба считается один раз."""

    def __init__(self, study: Study) -> None:
        self.study = study
        self._frames: dict[str, pl.DataFrame] = {}
        self._trades: dict[str, list[Trade]] = {}

    def frame(self, params: Params) -> pl.DataFrame:
        k = params_key(params)
        if k not in self._frames:
            self._frames[k] = self.study.frame(params)
        return self._frames[k]

    def trades(self, params: Params) -> list[Trade]:
        k = params_key(params)
        if k not in self._trades:
            self._trades[k] = self.study.trades(params)
        return self._trades[k]

    def day(self, t: Trade) -> int:
        return trade_day(t.open_utc, self.study.server)[0].toordinal()


def perturb(params: Params, space: Space) -> list[tuple[str, float, dict[str, ParamValue]]]:
    """Соседи ±10% и ±20% по одному числовому параметру; целые — минимум на шаг ±1 (§4.2)."""
    out: list[tuple[str, float, dict[str, ParamValue]]] = []
    for name in space.numeric:
        v = params.get(name)
        if not isinstance(v, int | float) or isinstance(v, bool) or v == 0:
            continue
        for k in th.SENS_STEPS:
            new: ParamValue
            if name in space.integer or isinstance(v, int):
                new = round(v * k)
                if new == v:
                    new = int(v) + (1 if k > 1 else -1)
            else:
                new = float(v) * k
            out.append((name, k, {**params, name: new}))
    return out


def _bucket_check(
    f: FilterDef,
    cand: pl.DataFrame,
    twin: pl.DataFrame,
    calendar: np.ndarray,
    cfg: GateConfig,
    *,
    strict: bool,
) -> FilterCheck:
    """Разрешённые корзины лучше запрещённых и фильтр не хуже близнеца по mean R × √n (§4.5).

    `strict` (IS): 90% интервал разности целиком выше нуля (P-06); holdout — точечная разность.
    """
    mask = f.allowed(twin)
    a, b = twin.filter(mask), twin.filter(~mask)
    diff: DiffResult | None
    try:
        diff = bucket_diff(
            daystats(calendar, a["day"].to_numpy(), a["r_net"].to_numpy()),
            daystats(calendar, b["day"].to_numpy(), b["r_net"].to_numpy()),
            n_paths=cfg.n_paths,
            seed=cfg.seed,
        )
    except ValueError:
        diff = None
    score, twin_score = edge_score(cand), edge_score(twin)
    better = diff is not None and (diff.holds if strict else diff.diff > 0)
    return FilterCheck(f.name, diff, score, twin_score, bool(better and score >= twin_score))


def evaluate(
    params: dict[str, ParamValue],
    row: dict[str, object],
    run: Runner,
    space: Space,
    rep: GateReport,
    trial_sr: np.ndarray,
    cfg: GateConfig,
) -> Candidate:
    sp = rep.splits
    cal = sp.is_calendar
    key = params_key(params)
    t = window(run.frame(params), None, sp.cut_ord)
    m = moments(daily(t, cal)[0])
    dsr = dsr_effective(
        m["sharpe"], len(cal), m["skew"], m["kurt"], trial_sr, rep.clusters, cfg.extra_trials
    )
    sens = []
    for name, k, p in perturb(params, space):
        s = s1_stats(window(run.frame(p), None, sp.cut_ord))
        sens.append(SensRow(name, k, p[name], int(s["n"]), float(s["pf"])))
    years_ok, years = year_stability(t)
    filters = [
        _bucket_check(
            f,
            t,
            window(run.frame({**params, f.param: f.off}), None, sp.cut_ord),
            cal,
            cfg,
            strict=True,
        )
        for f in space.active_filters(params)
    ]
    is_trades = [x for x in run.trades(params) if run.day(x) < sp.cut_ord]
    rej = autoreject(AutoRejectInput(is_trades, run.study.server, family=run.study.family))
    pbo = rep.pbo
    gates: dict[str, bool | None] = {
        "S1": bool(row["s1"]),
        "WFE": bool(not math.isnan(rep.wfe) and rep.wfe >= th.MIN_WFE),
        "DSR/PBO": bool(
            (not math.isnan(dsr.dsr) and dsr.dsr >= th.MIN_DSR)
            or (pbo is not None and pbo.pbo <= th.MAX_PBO)
        ),
        "чувствительность": all(s.pf >= th.SENS_PF for s in sens) if sens else None,
        "годы": years_ok,
        "инструменты": positive_instruments(t, th.MIN_INSTRUMENT_TRADES) >= th.MIN_INSTRUMENTS,
        "фильтры": all(f.holds for f in filters) if filters else None,
        "автоотказ": not rej.rejected,
    }
    return Candidate(
        params=params,
        key=key,
        s1={k: float(v) for k, v in row.items() if isinstance(v, int | float)},
        sharpe=m["sharpe"],
        dsr=dsr,
        sens=sens,
        years=years,
        instruments=instrument_edges(t),
        filters=filters,
        reject=rej,
        gates=gates,
    )


# --- стадии ------------------------------------------------------------------------------


def run_is(study: Study, space: Space, cfg: GateConfig, run: Runner | None = None) -> GateReport:
    """Всё, что видно без holdout. Пробы — `space.with_twins()`; финалист — `report.finalist`."""
    run = run or Runner(study)
    first, last = study.period
    sp = Splits.from_range(first, last, cfg.is_fraction)
    variants = space.with_twins()
    keys = [params_key(v) for v in variants]
    frames = {k: run.frame(v) for k, v in zip(keys, variants, strict=True)}
    cal = sp.is_calendar
    r_mat, n_mat = daily_matrix(frames, keys, cal)

    rows = []
    for j, k in enumerate(keys):
        s = s1_stats(window(frames[k], None, sp.cut_ord))
        rows.append(
            {
                "key": k,
                **s,
                "sharpe_daily": series_sharpe(r_mat[:, j]),
                "ok_n": s["n"] >= th.MIN_TRADES,
                "ok_pf": s["pf"] >= th.MIN_PF,
                "ok_e15": s["e15"] > 0,
            }
        )
    s1 = pl.DataFrame(rows).with_columns(
        (pl.col("ok_n") & pl.col("ok_pf") & pl.col("ok_e15")).alias("s1")
    )
    picks = walk_forward(r_mat, n_mat, cal, keys, wf_windows(sp.first, sp.cut, *cfg.wf))
    pbo: PBOResult | None = None
    if len(keys) >= 2:
        try:
            pbo = pbo_cscv(r_mat, cfg.pbo_blocks)
        except ValueError:
            pbo = None
    clusters = correlation_clusters(r_mat)
    trial_sr = np.concatenate([clusters.sharpe, np.asarray(cfg.prior_sharpes, dtype=float)])
    rep = GateReport(
        strategy_id=study.strategy_id,
        splits=sp,
        s1=s1,
        n_trials=len(keys) + cfg.extra_trials,
        picks=picks,
        wfe=wfe(picks),
        pbo=pbo,
        clusters=clusters,
        candidates=[],
    )
    by_key = dict(zip(keys, variants, strict=True))
    ranked = s1.filter(pl.col("s1")).sort("sharpe_daily", descending=True).head(cfg.top)
    rep.candidates = [
        evaluate(by_key[r["key"]], r, run, space, rep, trial_sr, cfg)
        for r in ranked.iter_rows(named=True)
    ]
    return rep


def _wf_source(run: Runner, rep: GateReport, by_key: Mapping[str, Params]) -> DayStats:
    """Дневные статистики тестовых окон WF у выбранных в окне вариантов (источник полосы)."""
    parts: list[DayStats] = []
    for p in rep.picks:
        cal = rep.splits.calendar(p.window.test_start, p.window.test_end)
        t = window(run.frame(by_key[p.key]), int(cal[0]), int(cal[-1]) + 1)
        parts.append(daystats(cal, t["day"].to_numpy(), t["r_net"].to_numpy()))
    return DayStats(
        r=np.concatenate([x.r for x in parts]),
        n=np.concatenate([x.n for x in parts]),
        wins=np.concatenate([x.wins for x in parts]),
        win_r=np.concatenate([x.win_r for x in parts]),
        loss_r=np.concatenate([x.loss_r for x in parts]),
    )


def run_holdout(
    study: Study,
    space: Space,
    rep: GateReport,
    cfg: GateConfig,
    guard: HoldoutGuard,
    run: Runner | None = None,
) -> HoldoutStage:
    """Открыть holdout ровно для финалиста (один раз): полоса MC, E > 0, инструменты, фильтры."""
    fin = rep.finalist
    if fin is None:
        raise ValueError("нет финалиста: holdout не открываем")
    if not rep.picks:
        raise ValueError("нет окон walk-forward: полосу MC не из чего строить")
    run = run or Runner(study)
    guard.open(fin.key, cfg.on)
    sp = rep.splits
    cal = sp.ho_calendar
    by_key = {params_key(v): v for v in space.with_twins()}
    t = window(run.frame(fin.params), sp.cut_ord, None)
    obs = daystats(cal, t["day"].to_numpy(), t["r_net"].to_numpy())
    band = mc_band(_wf_source(run, rep, by_key), obs, n_paths=cfg.n_paths, seed=cfg.seed)
    stats = s1_stats(t)
    filters = [
        _bucket_check(
            f,
            t,
            window(run.frame({**fin.params, f.param: f.off}), sp.cut_ord, None),
            cal,
            cfg,
            strict=False,
        )
        for f in space.active_filters(fin.params)
    ]
    n_pos = positive_instruments(t, th.HO_MIN_INSTRUMENT_TRADES)
    gates: dict[str, bool | None] = {
        "полоса MC": band.passed,
        "E > 0": stats["mean_r"] > th.HO_MIN_MEAN_R,
        "инструменты": n_pos >= th.MIN_INSTRUMENTS,
        "фильтры": all(f.holds for f in filters) if filters else None,
    }
    rep.holdout = HoldoutStage(cfg.on, band, stats, instrument_edges(t), filters, gates)
    return rep.holdout


def oos_trades(run: Runner, space: Space, rep: GateReport) -> list[Trade]:
    """Вход проп-MC: тесты WF (вариант, выбранный в окне) + holdout финалиста (VALIDATION §2)."""
    fin = rep.finalist
    if fin is None:
        raise ValueError("нет финалиста")
    by_key = {params_key(v): v for v in space.with_twins()}
    segs: list[tuple[Params, int, int, str]] = [
        (by_key[p.key], p.window.test_start.toordinal(), p.window.test_end.toordinal(), "WF")
        for p in rep.picks
    ]
    segs.append((fin.params, rep.splits.cut_ord, rep.splits.last.toordinal() + 1, "HOLDOUT"))
    out: list[Trade] = []
    for params, lo, hi, sample in segs:
        out.extend(
            t.model_copy(update={"sample": sample})
            for t in run.trades(params)
            if lo <= run.day(t) < hi
        )
    out.sort(key=lambda t: t.open_utc)
    return out


def _best_row(prop: PropReport) -> Row | None:
    if prop.recommended is not None:
        return prop.recommended
    firm_rows = [r for r in prop.rows if not r.rules.is_template]
    return max(firm_rows, key=lambda r: r.stats.objective, default=None)


def run_s3(
    study: Study, space: Space, rep: GateReport, cfg: GateConfig, run: Runner | None = None
) -> S3Stage:
    """Проп-MC на OOS (P(1-я выплата) ≥ 30% и EV₁ > 0 хотя бы в одной программе) + аудит."""
    if not cfg.firms:
        raise ValueError("для S3 нужны фирмы (configs/firms)")
    run = run or Runner(study)
    trades = oos_trades(run, space, rep)
    mc = cfg.mc or MCConfig(n_paths=cfg.n_paths, seed=cfg.seed)
    prop = run_report(trades, cfg.firms, [], mc, program_ids=cfg.program_ids)
    passed = any(
        r.stats.p1 >= th.S3_MIN_P1 and r.stats.ev1 > th.S3_MIN_EV1
        for r in prop.rows
        if not r.rules.is_template and not math.isnan(r.stats.p1)
    )
    audit = [loss_to_win([t.r_net for t in trades])]
    if rep.holdout is not None:
        better = rep.holdout.band.better
        audit.append(
            AuditFlag(
                "holdout_better",
                "holdout лучше полосы MC",
                bool(better),
                float(len(better)),
                0.0,
                ", ".join(better) or "нет",
            )
        )
    best = _best_row(prop)
    if best is not None and trades:
        days = np.array([run.day(t) for t in trades], dtype=np.int64)
        cal = np.unique(
            np.concatenate(
                [rep.splits.calendar(p.window.test_start, p.window.test_end) for p in rep.picks]
                + [rep.splits.ho_calendar]
            )
        )
        q = streak_quantile(
            days[np.isin(days, cal)],
            np.array([t.r_net for t in trades])[np.isin(days, cal)],
            cal,
            n_paths=cfg.n_paths,
            seed=cfg.seed,
        )
        audit.append(streak_flag(q, max(best.opt.best.risk.f), best.rules.max_dist))
    rep.s3 = S3Stage(trades, prop, passed, best, audit)
    return rep.s3


def run_gates(
    study: Study, space: Space, cfg: GateConfig, guard: HoldoutGuard | None = None
) -> GateReport:
    """S1 → S2 (IS) → holdout финалиста → S3; следующая стадия — только если прошла прошлая."""
    run = Runner(study)
    rep = run_is(study, space, cfg, run)
    if rep.finalist is None or not rep.picks:
        return rep
    ho = run_holdout(study, space, rep, cfg, guard or OnceGuard(), run)
    if ho.passed and cfg.firms:
        run_s3(study, space, rep, cfg, run)
    return rep


def verdict(rep: GateReport) -> str:
    """Одна строка по IDEAS §4: `в демо` / `доработать: …` / `отклонена: …`."""
    if not rep.candidates:
        return "отклонена: ни один вариант не прошёл S1 (сделок, PF, E при издержках × 1.5)"
    fin = rep.finalist
    if fin is None:
        best = rep.candidates[0]
        why = ", ".join(best.failed) or "—"
        if best.gates.get("автоотказ") is False and best.reject is not None:
            why += " (" + "; ".join(best.reject.reasons) + ")"
        return f"отклонена: лучший по Sharpe вариант не прошёл гейты IS ({why}); holdout не открыт"
    if rep.holdout is None:
        return "доработать: нет окон walk-forward для полосы MC (мало истории)"
    if not rep.holdout.passed:
        bad = ", ".join(k for k, v in rep.holdout.gates.items() if v is False)
        return f"отклонена: финалист не прошёл holdout ({bad})"
    if rep.s3 is None:
        return "доработать: S1–S2 пройдены, проп-MC не запускался (нет фирм)"
    if rep.s3.passed:
        audit = [a.name for a in rep.s3.audit if a.flagged]
        tail = f"; нужен аудит: {', '.join(audit)}" if audit else ""
        return (
            "в демо: S1–S3 пройдены (P(1-я выплата) ≥ 30% и EV₁ > 0 хотя бы в одной программе)"
            + tail
        )
    return "доработать: S1–S2 пройдены, проп-MC не дал P(1-я выплата) ≥ 30% и EV₁ > 0"
