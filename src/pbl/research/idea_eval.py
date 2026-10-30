"""Оценка вариантов идеи по гейтам S1–S2 (VALIDATION §3–4) на таблицах сделок `range_sweep`.

Порядок защищает holdout: S1, walk-forward, DSR/PBO, годы, чувствительность и корзины фильтра
считаются только на IS; holdout открывается один раз, для одного финалиста, и только если всё
до него прошло (`finalist_holdout`).

Единица отбора — пул инструментов: конфигурация с одинаковыми параметрами на всех инструментах
корзины (`range_sweep.INSTRUMENTS`). Доходность дня — сумма `r_net` сделок всех инструментов.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
import polars as pl

from pbl.research.range_sweep import Bundle, Config, config_trades, scan_kind
from pbl.robustness.bands import (
    BandResult,
    DayStats,
    DiffResult,
    bucket_diff,
    daystats,
    holdout_band,
)
from pbl.robustness.splits import Splits as Splits
from pbl.robustness.stats import instrument_edges as instrument_edges
from pbl.robustness.stats import s1_stats as s1_stats
from pbl.robustness.stats import series_sharpe as series_sharpe
from pbl.robustness.stats import year_stability as year_stability
from pbl.robustness.thresholds import MIN_PF, MIN_TRADES
from pbl.robustness.thresholds import MIN_WFE as MIN_WFE
from pbl.robustness.thresholds import SENS_PF as SENS_PF
from pbl.robustness.thresholds import SENS_STEPS as SENS_STEPS
from pbl.robustness.walkforward import WFPick as WFPick
from pbl.robustness.walkforward import walk_forward as walk_forward
from pbl.robustness.walkforward import wfe as wfe
from pbl.strategies.range_session import Side

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

_ONE_SIDED: tuple[Side, ...] = ("long", "short")

ALLOWED_BUCKETS = frozenset({(0, 1), (1, 1), (1, -1), (2, 1), (3, -1), (4, 1), (4, -1)})
"""(день недели, сторона), разрешённые фильтром автора; остальные корзины — запрещённые."""


Tables = dict[str, dict[str, pl.DataFrame]]
"""config.key → инструмент → сделки (`range_sweep.TABLE_SCHEMA`) за всю историю."""


def build_tables(bundles: dict[str, Bundle], configs: Sequence[Config], splits: Splits) -> Tables:
    """Сделки каждой конфигурации на каждом инструменте, где окно торгуется у фирмы."""
    lo, hi = splits.first.toordinal(), splits.last.toordinal()
    out: Tables = {}
    for cfg in configs:
        per: dict[str, pl.DataFrame] = {}
        for sym, b in bundles.items():
            if b.live(cfg):
                per[sym] = config_trades(b, cfg).filter(pl.col("day").is_between(lo, hi))
        if per:
            out[cfg.key] = per
    return out


def pooled(tables: Tables, key: str, lo: int | None = None, hi: int | None = None) -> pl.DataFrame:
    """Сделки конфигурации по всем инструментам; `lo ≤ day < hi` (ordinal) — окно."""
    parts = []
    for sym, df in tables[key].items():
        t = df.with_columns(pl.lit(sym).alias("symbol"))
        if lo is not None:
            t = t.filter(pl.col("day") >= lo)
        if hi is not None:
            t = t.filter(pl.col("day") < hi)
        parts.append(t)
    return pl.concat(parts)


def daily_matrix(tables: Tables, keys: Sequence[str], calendar: I64) -> tuple[F64, F64]:
    """T × C: сумма `r_net` и число сделок по дням календаря для каждой конфигурации."""
    r = np.zeros((len(calendar), len(keys)))
    n = np.zeros_like(r)
    for j, key in enumerate(keys):
        for df in tables[key].values():
            if df.is_empty():
                continue
            day = df["day"].to_numpy()
            pos = np.searchsorted(calendar, day)
            ok = pos < len(calendar)
            ok[ok] = calendar[pos[ok]] == day[ok]
            np.add.at(r[:, j], pos[ok], df["r_net"].to_numpy()[ok])
            np.add.at(n[:, j], pos[ok], 1.0)
    return r, n


def oos_calendar(picks: Sequence[WFPick], splits: Splits, *, with_holdout: bool = False) -> I64:
    """Дни OOS (VALIDATION §4.5): тестовые окна WF и, если открыт, holdout."""
    parts = [splits.calendar(p.window.test_start, p.window.test_end) for p in picks]
    if with_holdout:
        parts.append(splits.ho_calendar)
    return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


# --- S1 -----------------------------------------------------------------------------------


def s1_table(tables: Tables, configs: Sequence[Config], splits: Splits) -> pl.DataFrame:
    """S1 по каждой конфигурации на IS пула инструментов (гейты в колонках `ok_*`)."""
    rows = []
    cal = splits.is_calendar
    keys = [c.key for c in configs if c.key in tables]
    r_mat, _ = daily_matrix(tables, keys, cal)
    sharpe = dict(zip(keys, (series_sharpe(r_mat[:, j]) for j in range(len(keys))), strict=True))
    for cfg in configs:
        if cfg.key not in tables:
            continue
        s = s1_stats(pooled(tables, cfg.key, None, splits.cut_ord))
        rows.append(
            {
                "key": cfg.key,
                "mode": cfg.mode,
                "session": cfg.session,
                "exit": cfg.exit.name,
                "filt": cfg.filt,
                **s,
                "sharpe_daily": sharpe[cfg.key],
                "ok_n": s["n"] >= MIN_TRADES,
                "ok_pf": s["pf"] >= MIN_PF,
                "ok_e15": s["e15"] > 0,
            }
        )
    df = pl.DataFrame(rows)
    return df.with_columns((pl.col("ok_n") & pl.col("ok_pf") & pl.col("ok_e15")).alias("s1"))


# --- walk-forward -------------------------------------------------------------------------


def pooled_daystats(tables: Tables, key: str, calendar: I64) -> DayStats:
    """Дневные суммы пула по дням `calendar` (отсортирован; дни могут идти с пропусками)."""
    t = pooled(tables, key, int(calendar[0]), int(calendar[-1]) + 1)
    t = t.filter(pl.col("day").is_in(pl.Series(calendar)))
    return daystats(calendar, t["day"].to_numpy(), t["r_net"].to_numpy())


def wf_daystats(tables: Tables, picks: Sequence[WFPick], splits: Splits) -> DayStats:
    """Склейка дневных статистик тестовых окон выбранных конфигураций (источник полосы MC)."""
    parts: list[DayStats] = []
    for p in picks:
        cal = splits.calendar(p.window.test_start, p.window.test_end)
        parts.append(pooled_daystats(tables, p.key, cal))
    return DayStats(
        r=np.concatenate([x.r for x in parts]),
        n=np.concatenate([x.n for x in parts]),
        wins=np.concatenate([x.wins for x in parts]),
        win_r=np.concatenate([x.win_r for x in parts]),
        loss_r=np.concatenate([x.loss_r for x in parts]),
    )


# --- проверки финалиста на IS -------------------------------------------------------------


def perturbations(cfg: Config) -> list[tuple[str, Config]]:
    """Соседи конфигурации ±10% и ±20% по одному параметру (§4.2): окно, выход, буфер."""
    out: list[tuple[str, Config]] = []
    a, b = (int(x[:2]) * 60 + int(x[3:]) for x in (cfg.start, cfg.end))
    for k in SENS_STEPS:
        length = round((b - a) * k)
        if length == b - a:
            length += 1 if k > 1 else -1
        if length >= 1:
            e = a + length
            out.append((f"окно ×{k:g}", replace(cfg, end=f"{e // 60:02d}:{e % 60:02d}")))
        out.append((f"выход ×{k:g}", replace(cfg, exit=cfg.exit.scaled(k))))
        if cfg.buffer > 0:
            out.append((f"буфер ×{k:g}", replace(cfg, buffer=cfg.buffer * k)))
    return out


@dataclass(frozen=True)
class SensRow:
    label: str
    n: int
    pf: float


def sensitivity(bundles: dict[str, Bundle], cfg: Config, splits: Splits) -> list[SensRow]:
    """PF на IS пула у каждого соседа; проходит, если PF ≥ 1.15 везде (P-03)."""
    rows = []
    for label, c in perturbations(cfg):
        parts = [
            config_trades(b, c).filter(pl.col("day") < splits.cut_ord)
            for b in bundles.values()
            if b.live(c)
        ]
        s = s1_stats(pl.concat(parts)) if parts else {"n": 0, "pf": 0.0}
        rows.append(SensRow(label, int(s["n"]), float(s["pf"])))
    return rows


# --- корзины «день × направление» ---------------------------------------------------------


def bucket_stats(
    bundles: dict[str, Bundle], cfgs: Sequence[Config], calendar: I64
) -> tuple[DayStats, DayStats]:
    """Независимые лонг- и шорт-сканы → (разрешённые, запрещённые) корзины на календаре.

    `cfgs` — конфигурации без фильтра (сканы не зависят от фильтра, повторов нет).
    """
    days_in = pl.Series(calendar)
    days: dict[bool, list[I64]] = {True: [], False: []}
    rs: dict[bool, list[F64]] = {True: [], False: []}
    for cfg in cfgs:
        for b in bundles.values():
            if not b.live(cfg):
                continue
            for kind in _ONE_SIDED:
                t = scan_kind(b, cfg, kind).filter(pl.col("day").is_in(days_in))
                wd, side = t["wd"].to_numpy(), t["side"].to_numpy()
                allowed = np.array(
                    [(int(w), int(s)) in ALLOWED_BUCKETS for w, s in zip(wd, side, strict=True)],
                    dtype=bool,
                )
                for flag in (True, False):
                    m = allowed if flag else ~allowed
                    days[flag].append(t["day"].to_numpy()[m])
                    rs[flag].append(t["r_net"].to_numpy()[m])
    a = daystats(calendar, np.concatenate(days[True]), np.concatenate(rs[True]))
    f = daystats(calendar, np.concatenate(days[False]), np.concatenate(rs[False]))
    return a, f


def edge_score(ds: DayStats) -> float:
    """mean R × √n = Σ R / √n: чем больше сделок при том же mean R, тем увереннее край."""
    n = float(ds.n.sum())
    return float(ds.r.sum() / math.sqrt(n)) if n > 0 else 0.0


@dataclass(frozen=True)
class FilterCheck:
    """§4.5 на заданном календаре OOS: корзины автора и вариант с фильтром против без него."""

    diff: DiffResult | None
    score_author: float
    score_none: float

    @property
    def ok(self) -> bool:
        """Разрешённые корзины лучше запрещённых и фильтр не хуже по mean R × √n."""
        return self.diff is not None and self.diff.diff > 0 and self.score_author >= self.score_none


def filter_check(
    bundles: dict[str, Bundle],
    tables: Tables,
    cfg: Config,
    calendar: I64,
    *,
    n_paths: int = 10_000,
    seed: int = 42,
) -> FilterCheck | None:
    """Проверка фильтра дня на параметрах `cfg`; пустой календарь OOS — проверки нет."""
    if len(calendar) == 0:
        return None
    none_cfg, author_cfg = replace(cfg, filt="none"), replace(cfg, filt="author")
    try:
        diff: DiffResult | None = filter_verdict(
            bundles, [none_cfg], calendar, n_paths=n_paths, seed=seed
        )
    except ValueError:
        diff = None
    return FilterCheck(
        diff,
        edge_score(pooled_daystats(tables, author_cfg.key, calendar)),
        edge_score(pooled_daystats(tables, none_cfg.key, calendar)),
    )


def filter_verdict(
    bundles: dict[str, Bundle],
    cfgs: Sequence[Config],
    calendar: I64,
    *,
    n_paths: int = 10_000,
    seed: int = 42,
) -> DiffResult:
    a, f = bucket_stats(bundles, cfgs, calendar)
    return bucket_diff(a, f, n_paths=n_paths, seed=seed)


# --- holdout (один раз, для финалиста) ----------------------------------------------------


@dataclass(frozen=True)
class HoldoutResult:
    opened_at: dt.date
    band: BandResult
    n_trades: int
    per_instrument: dict[str, tuple[int, float]]
    stats: dict[str, float]


def finalist_holdout(
    tables: Tables,
    key: str,
    wf_source: DayStats,
    splits: Splits,
    *,
    opened_at: dt.date,
    n_paths: int = 10_000,
    seed: int = 42,
) -> HoldoutResult:
    """Открыть holdout: метрики финалиста против полосы MC из тестов walk-forward (§4.1)."""
    cal = splits.ho_calendar
    obs = pooled_daystats(tables, key, cal)
    t = pooled(tables, key, splits.cut_ord, None)
    return HoldoutResult(
        opened_at=opened_at,
        band=holdout_band(wf_source, obs, n_paths=n_paths, seed=seed),
        n_trades=t.height,
        per_instrument=instrument_edges(t),
        stats=s1_stats(t),
    )
