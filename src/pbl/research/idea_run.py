"""Прогон идеи №1 через гейты S1–S3 и ответ по IDEAS §4.

Две стадии. `run_is` видит только IS (до общей даты отсечения, P-35): S1 всей сетки, запись проб
в счётчик, walk-forward, DSR/PBO, проверки топ-кандидатов. `run_holdout` вызывается только для
финалиста, прошедшего всё на IS, — один раз; затем событийный бэктест OOS-сделок и проп-MC (S3).
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import polars as pl

from pbl.config import REGISTRY, load_firms
from pbl.engine.backtest import RunMeta, backtest
from pbl.prop.engine import MCConfig
from pbl.prop.report import PropReport, run_report
from pbl.prop.servertime import server_date
from pbl.research.idea_eval import (
    I64,
    MIN_WFE,
    SENS_PF,
    HoldoutResult,
    SensRow,
    Splits,
    Tables,
    WFPick,
    bucket_stats,
    build_tables,
    daily_matrix,
    filter_verdict,
    finalist_holdout,
    instrument_edges,
    pooled,
    pooled_daystats,
    s1_table,
    sensitivity,
    series_sharpe,
    walk_forward,
    wf_daystats,
    wfe,
    year_stability,
)
from pbl.research.range_sweep import (
    IDEA_ID,
    Bundle,
    Config,
    config_signals,
    daily_from_table,
    grid,
    variant_spec,
)
from pbl.research.variants import VariantLog
from pbl.robustness.bands import DayStats, DiffResult, metrics
from pbl.robustness.dsr import dsr_for_trials
from pbl.robustness.pbo import PBOResult, pbo_cscv
from pbl.robustness.splits import holdout_start, wf_windows
from pbl.robustness.thresholds import (
    HO_MIN_INSTRUMENT_TRADES,
    HO_MIN_MEAN_R,
    MAX_PBO,
    MIN_DSR,
    MIN_INSTRUMENT_TRADES,
    MIN_INSTRUMENTS,
    S3_MIN_P1,
)
from pbl.schemas import Trade

GATES = ("S1", "WFE", "DSR/PBO", "чувствительность", "годы", "инструменты", "фильтр дня")
HOLDOUT_GATES = ("полоса MC", "E > 0", "инструменты", "фильтр дня")


@dataclass(frozen=True)
class RunParams:
    start: dt.date
    end: dt.date
    version: str = "v1"
    on: dt.date = field(default_factory=lambda: dt.datetime.now(dt.UTC).date())
    n_paths: int = 10_000
    seed: int = 42
    wf: tuple[int, int, int] = (24, 6, 6)
    is_fraction: float = 0.7
    top: int = 5
    pbo_blocks: int = 16
    firm_ids: tuple[str, ...] = ("fundednext",)
    registry: Path = REGISTRY
    mc: MCConfig | None = None

    @property
    def run_id(self) -> str:
        return f"{IDEA_ID.lower()}-{self.version}"


@dataclass
class Candidate:
    cfg: Config
    row: dict[str, Any]
    dsr: float
    sr0: float
    sens: list[SensRow]
    years: pl.DataFrame
    instruments: dict[str, tuple[int, float]]
    filt: DiffResult | None
    twin_sharpe: float | None
    gates: dict[str, bool | None]

    @property
    def passed(self) -> bool:
        return all(v is not False for v in self.gates.values())

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.gates.items() if v is False]


@dataclass
class ISStage:
    splits: Splits
    configs: list[Config]
    tables: Tables
    s1: pl.DataFrame
    log: VariantLog
    new_rows: int
    picks: list[WFPick]
    wfe: float
    pbo: PBOResult | None
    n_trials: int
    ids: dict[str, str]
    sharpe: dict[str, float]
    candidates: list[Candidate] = field(default_factory=list)
    filter_grid: pl.DataFrame = field(default_factory=pl.DataFrame)

    @property
    def finalist(self) -> Candidate | None:
        return next((c for c in self.candidates if c.passed), None)


@dataclass
class HoldoutStage:
    result: HoldoutResult
    filt: DiffResult | None
    twin_sharpe: float | None
    sharpe: float
    scan_vs_backtest: float
    gates: dict[str, bool | None]
    trades: list[Trade]
    prop: PropReport | None
    s3_ok: bool | None

    @property
    def passed(self) -> bool:
        return all(v is not False for v in self.gates.values())


# --- стадия IS ----------------------------------------------------------------------------


def make_splits(bundles: dict[str, Bundle], is_fraction: float) -> Splits:
    """Общая хронология: первый день, где есть бары у всех, до последнего, где они ещё есть."""
    first = max(int(b.market.day.min()) for b in bundles.values())
    last = min(int(b.market.day.max()) for b in bundles.values())
    f, la = dt.date.fromordinal(first), dt.date.fromordinal(last)
    return Splits(f, la, holdout_start(f, la, is_fraction))


def basket_name(symbols: Sequence[str]) -> str:
    return "BASKET:" + "+".join(symbols)


def log_variants(
    tables: Tables, configs: Sequence[Config], splits: Splits, log: VariantLog, p: RunParams
) -> tuple[dict[str, str], int]:
    """Пробы IS в счётчик: инструменты без рядов, корзина с рядом. → (id корзин, новых строк)."""
    end = splits.cut - dt.timedelta(days=1)
    ids: dict[str, str] = {}
    new = 0
    for cfg in configs:
        per = tables.get(cfg.key)
        if per is None:
            continue
        for sym, df in per.items():
            daily = daily_from_table(df.filter(pl.col("day") < splits.cut_ord), splits.first, end)
            new += log.add(
                variant_spec(cfg, sym, p.version),
                daily,
                run_id=p.run_id,
                on=p.on,
                keep_series=False,
            )
        pool = pooled(tables, cfg.key, None, splits.cut_ord)
        spec = variant_spec(cfg, basket_name(list(per)), p.version)
        new += log.add(
            spec,
            daily_from_table(pool, splits.first, end),
            run_id=p.run_id,
            on=p.on,
            keep_series=True,
        )
        ids[cfg.key] = spec.variant_id
    return ids, new


def filter_grid(
    bundles: dict[str, Bundle], configs: Sequence[Config], calendar: I64
) -> pl.DataFrame:
    """По каждому скану без фильтра: mean R разрешённых и запрещённых корзин автора."""
    rows = []
    for cfg in configs:
        if cfg.filt != "none":
            continue
        a, f = bucket_stats(bundles, [cfg], calendar)
        ma, mf = metrics(a), metrics(f)
        rows.append(
            {
                "key": cfg.key,
                "n_allowed": int(ma["n_trades"]),
                "n_forbidden": int(mf["n_trades"]),
                "mean_r_allowed": ma["mean_r"],
                "mean_r_forbidden": mf["mean_r"],
                "allowed_better": bool(ma["mean_r"] > mf["mean_r"]),
            }
        )
    return pl.DataFrame(rows)


def evaluate(
    cfg: Config, row: dict[str, Any], bundles: dict[str, Bundle], stage: ISStage, p: RunParams
) -> Candidate:
    """Гейты кандидата, которые видны на IS (VALIDATION §3–4)."""
    splits, tables, log = stage.splits, stage.tables, stage.log
    vid = stage.ids[cfg.key]
    lrow = log.frame().filter(pl.col("variant_id") == vid).row(0, named=True)
    dsr, sr0 = dsr_for_trials(
        float(lrow["sharpe_daily"]),
        int(lrow["n_days"]),
        float(lrow["skew"]),
        float(lrow["kurt"]),
        log.frame()["sharpe_daily"].to_list(),
    )
    t = pooled(tables, cfg.key, None, splits.cut_ord)
    sens = sensitivity(bundles, cfg, splits)
    years_ok, years = year_stability(t)
    inst = instrument_edges(t)
    pos = sum(1 for n, m in inst.values() if n >= MIN_INSTRUMENT_TRADES and m > 0)

    diff: DiffResult | None = None
    twin_sharpe: float | None = None
    filt_ok: bool | None = None
    if cfg.filt == "author":
        try:
            diff = filter_verdict(
                bundles, [cfg], splits.is_calendar, n_paths=p.n_paths, seed=p.seed
            )
        except ValueError:
            diff = None
        twin_sharpe = stage.sharpe.get(replace(cfg, filt="none").key)
        filt_ok = bool(
            diff is not None
            and diff.holds
            and twin_sharpe is not None
            and row["sharpe_daily"] >= twin_sharpe
        )
    pbo = stage.pbo
    gates: dict[str, bool | None] = {
        "S1": bool(row["s1"]),
        "WFE": bool(not math.isnan(stage.wfe) and stage.wfe >= MIN_WFE),
        "DSR/PBO": bool(
            (not math.isnan(dsr) and dsr >= MIN_DSR) or (pbo is not None and pbo.pbo <= MAX_PBO)
        ),
        "чувствительность": bool(sens) and all(s.pf >= SENS_PF for s in sens),
        "годы": years_ok,
        "инструменты": pos >= MIN_INSTRUMENTS,
        "фильтр дня": filt_ok,
    }
    return Candidate(cfg, row, dsr, sr0, sens, years, inst, diff, twin_sharpe, gates)


def run_is(bundles: dict[str, Bundle], p: RunParams) -> ISStage:
    """Всё, что можно вычислить, не открывая holdout. Пишет пробы в `log` (файл — вызывающий)."""
    splits = make_splits(bundles, p.is_fraction)
    configs = grid()
    tables = build_tables(bundles, configs, splits)
    s1 = s1_table(tables, configs, splits)
    log = VariantLog.for_idea(IDEA_ID, p.registry)
    ids, new = log_variants(tables, configs, splits, log, p)

    keys = [c.key for c in configs if c.key in tables]
    cal = splits.is_calendar
    r_mat, n_mat = daily_matrix(tables, keys, cal)
    picks = walk_forward(r_mat, n_mat, cal, keys, wf_windows(splits.first, splits.cut, *p.wf))
    try:
        pbo: PBOResult | None = pbo_cscv(
            log.returns_matrix([ids[k] for k in keys]).values, p.pbo_blocks
        )
    except ValueError:
        pbo = None

    stage = ISStage(
        splits=splits,
        configs=configs,
        tables=tables,
        s1=s1,
        log=log,
        new_rows=new,
        picks=picks,
        wfe=wfe(picks),
        pbo=pbo,
        n_trials=len(log),
        ids=ids,
        sharpe=dict(zip(s1["key"].to_list(), s1["sharpe_daily"].to_list(), strict=True)),
    )
    by_key = {c.key: c for c in configs}
    ranked = s1.filter(pl.col("s1")).sort("sharpe_daily", descending=True).head(p.top)
    stage.candidates = [
        evaluate(by_key[r["key"]], r, bundles, stage, p) for r in ranked.iter_rows(named=True)
    ]
    stage.filter_grid = filter_grid(bundles, configs, cal)
    return stage


# --- стадия holdout и S3 ------------------------------------------------------------------


def oos_trades(
    bundles: dict[str, Bundle], stage: ISStage, finalist: Config, run_id: str
) -> list[Trade]:
    """Событийный бэктест: тестовые окна WF (конфигурация, выбранная в окне) и holdout финалиста."""
    by_key = {c.key: c for c in stage.configs}
    server = next(f for f in load_firms() if f.id == "fundednext").server_time
    segments = [
        (by_key[pk.key], pk.window.test_start.toordinal(), pk.window.test_end.toordinal(), "WF")
        for pk in stage.picks
    ]
    segments.append((finalist, stage.splits.cut_ord, stage.splits.last.toordinal() + 1, "HOLDOUT"))
    cache: dict[tuple[str, str], list[Trade]] = {}
    out: list[Trade] = []
    for cfg, lo, hi, sample in segments:
        for sym, b in bundles.items():
            if not b.live(cfg):
                continue
            if (cfg.key, sym) not in cache:
                meta = RunMeta(
                    strategy_id=IDEA_ID,
                    variant_id=f"{cfg.key}-{sym}",
                    profit_to_usd=b.profit_to_usd,
                )
                sig = config_signals(b, cfg, "both", cfg.day_sides)
                cache[(cfg.key, sym)] = backtest(
                    b.market, sig, cfg.exit.exits, meta, with_equity=False
                ).trades
            for t in cache[(cfg.key, sym)]:
                if lo <= server_date(t.open_utc, server).toordinal() < hi:
                    out.append(t.model_copy(update={"sample": sample}))
    out.sort(key=lambda t: t.open_utc)
    return out


def run_holdout(bundles: dict[str, Bundle], stage: ISStage, p: RunParams) -> HoldoutStage:
    """Открыть holdout ровно для финалиста: полоса MC, E, инструменты, фильтр дня, затем S3."""
    fin = stage.finalist
    if fin is None:
        raise ValueError("нет финалиста: holdout не открываем")
    cfg, sp = fin.cfg, stage.splits
    wf_source = wf_daystats(stage.tables, stage.picks, sp)
    res = finalist_holdout(
        stage.tables, cfg.key, wf_source, sp, opened_at=p.on, n_paths=p.n_paths, seed=p.seed
    )
    cal = sp.ho_calendar
    obs: DayStats = pooled_daystats(stage.tables, cfg.key, cal)
    sharpe = series_sharpe(obs.r)
    twin = replace(cfg, filt="none")
    twin_sharpe = None
    if twin.key in stage.tables:
        twin_sharpe = series_sharpe(pooled_daystats(stage.tables, twin.key, cal).r)
    try:
        diff: DiffResult | None = filter_verdict(
            bundles, [cfg], cal, n_paths=p.n_paths, seed=p.seed
        )
    except ValueError:
        diff = None

    pos = sum(1 for n, m in res.per_instrument.values() if n >= HO_MIN_INSTRUMENT_TRADES and m > 0)
    filt_ok: bool | None = None
    if cfg.filt == "author":
        filt_ok = bool(
            diff is not None and diff.diff > 0 and twin_sharpe is not None and sharpe >= twin_sharpe
        )
    gates: dict[str, bool | None] = {
        "полоса MC": res.band.passed,
        "E > 0": res.stats["mean_r"] > HO_MIN_MEAN_R,
        "инструменты": pos >= MIN_INSTRUMENTS,
        "фильтр дня": filt_ok,
    }
    trades = oos_trades(bundles, stage, cfg, p.run_id)
    ho = [t for t in trades if t.sample == "HOLDOUT"]
    scan_r = float(res.stats["mean_r"] * res.n_trades)
    bt_r = float(sum(t.r_net for t in ho))
    gap = abs(bt_r - scan_r) / max(abs(scan_r), 1.0)

    prop: PropReport | None = None
    s3_ok: bool | None = None
    if all(v is not False for v in gates.values()) and trades:
        firms = [f for f in load_firms() if f.id in p.firm_ids]
        prop = run_report(trades, firms, [], p.mc or MCConfig(n_paths=p.n_paths, seed=p.seed))
        s3_ok = any(
            r.stats.p1 >= S3_MIN_P1 and r.stats.ev1 > 0
            for r in prop.rows
            if not math.isnan(r.stats.p1)
        )
    return HoldoutStage(res, diff, twin_sharpe, sharpe, gap, gates, trades, prop, s3_ok)


# --- вердикт ------------------------------------------------------------------------------


def verdict(stage: ISStage, ho: HoldoutStage | None) -> str:
    """Одна строка по IDEAS §4: `в демо` / `доработать: …` / `отклонена: …`."""
    if not stage.candidates:
        return "отклонена: ни один вариант не прошёл S1 (сделок, PF, E при издержках × 1.5)"
    fin = stage.finalist
    if fin is None:
        best = stage.candidates[0]
        why = ", ".join(best.failed) or "—"
        return f"отклонена: лучший по Sharpe вариант не прошёл гейты IS ({why}); holdout не открыт"
    if ho is None:
        raise ValueError("финалист есть, а holdout не прогнан")
    if not ho.passed:
        bad = ", ".join(k for k, v in ho.gates.items() if v is False)
        return f"отклонена: финалист не прошёл holdout ({bad})"
    if ho.s3_ok:
        return "в демо: S1–S3 пройдены (P(1-я выплата) ≥ 30% и EV₁ > 0 хотя бы в одной программе)"
    return "доработать: S1–S2 пройдены, проп-MC не дал P(1-я выплата) ≥ 30% и EV₁ > 0"
