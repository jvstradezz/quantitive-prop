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
from pbl.robustness.splits import WFWindow
from pbl.strategies.range_session import Side

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]

# S1 (VALIDATION §3) и S2 (§4)
MIN_TRADES, MIN_PF, COST_MULT = 200, 1.3, 1.5
MIN_WFE = 0.5
SENS_PF, SENS_STEPS = 1.15, (0.8, 0.9, 1.1, 1.2)
MIN_YEAR_TRADES, YEAR_SHARE = 20, 0.5
WF_MIN_TRADES = 100
"""Минимум сделок в окне обучения, чтобы конфигурацию вообще рассматривать при отборе."""

_ONE_SIDED: tuple[Side, ...] = ("long", "short")

ALLOWED_BUCKETS = frozenset({(0, 1), (1, 1), (1, -1), (2, 1), (3, -1), (4, 1), (4, -1)})
"""(день недели, сторона), разрешённые фильтром автора; остальные корзины — запрещённые."""


@dataclass(frozen=True)
class Splits:
    """Хронология: IS — [first, cut), holdout — [cut, last]; общая для всех инструментов (P-35)."""

    first: dt.date
    last: dt.date
    cut: dt.date

    @property
    def cut_ord(self) -> int:
        return self.cut.toordinal()

    def calendar(self, start: dt.date, end: dt.date) -> I64:
        """Дни Пн–Пт серверного календаря в [start, end) как ordinal."""
        days = np.arange(start.toordinal(), end.toordinal(), dtype=np.int64)
        return days[(days - 1) % 7 < 5]

    @property
    def is_calendar(self) -> I64:
        return self.calendar(self.first, self.cut)

    @property
    def ho_calendar(self) -> I64:
        return self.calendar(self.cut, self.last + dt.timedelta(days=1))


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


def series_sharpe(x: F64) -> float:
    s = float(x.std(ddof=1)) if x.size > 1 else 0.0
    return float(x.mean() / s) if s > 1e-12 else 0.0


# --- S1 -----------------------------------------------------------------------------------


def s1_stats(t: pl.DataFrame) -> dict[str, float]:
    """Сделки пула → n, PF, mean R, E при издержках × 1.5, доля издержек от валовой."""
    n = t.height
    if n == 0:
        return {"n": 0, "pf": 0.0, "mean_r": 0.0, "e15": 0.0, "cost_share": math.nan}
    r = t["r_net"].to_numpy()
    gross, cost = t["r_gross"].to_numpy(), t["cost_r"].to_numpy()
    wins, losses = float(r[r > 0].sum()), -float(r[r < 0].sum())
    g = float(gross.sum())
    return {
        "n": n,
        "pf": wins / losses if losses > 0 else math.inf,
        "mean_r": float(r.mean()),
        "e15": float((gross - COST_MULT * cost).mean()),
        "cost_share": float(cost.sum() / g) if g > 0 else math.nan,
    }


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


@dataclass(frozen=True)
class WFPick:
    window: WFWindow
    key: str
    n_train: int
    train_r: float
    test_r: float
    n_test: int

    @property
    def train_per_year(self) -> float:
        return self.train_r / self.window.train_years

    @property
    def test_per_year(self) -> float:
        return self.test_r / self.window.test_years


def walk_forward(
    r: F64, n: F64, calendar: I64, keys: Sequence[str], windows: Sequence[WFWindow]
) -> list[WFPick]:
    """В каждом окне — лучшая по Sharpe обучения конфигурация (n ≥ WF_MIN_TRADES) и её тест."""
    picks: list[WFPick] = []
    for w in windows:
        tr = (calendar >= w.train_start.toordinal()) & (calendar < w.train_end.toordinal())
        te = (calendar >= w.test_start.toordinal()) & (calendar < w.test_end.toordinal())
        if not tr.any() or not te.any():
            continue
        best, best_sr = -1, -math.inf
        for j in range(len(keys)):
            if n[tr, j].sum() < WF_MIN_TRADES:
                continue
            sr = series_sharpe(r[tr, j])
            if sr > best_sr:
                best, best_sr = j, sr
        if best < 0:
            continue
        picks.append(
            WFPick(
                window=w,
                key=keys[best],
                n_train=int(n[tr, best].sum()),
                train_r=float(r[tr, best].sum()),
                test_r=float(r[te, best].sum()),
                n_test=int(n[te, best].sum()),
            )
        )
    return picks


def wfe(picks: Sequence[WFPick]) -> float:
    """WFE = годовая доходность тестов / годовая доходность обучений по склейке окон (§4)."""
    if not picks:
        return math.nan
    train = sum(p.train_r for p in picks) / sum(p.window.train_years for p in picks)
    test = sum(p.test_r for p in picks) / sum(p.window.test_years for p in picks)
    return test / train if train > 0 else math.nan


def pooled_daystats(tables: Tables, key: str, calendar: I64) -> DayStats:
    t = pooled(tables, key, int(calendar[0]), int(calendar[-1]) + 1)
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


def year_stability(t: pl.DataFrame) -> tuple[bool, pl.DataFrame]:
    """E > 0 минимум в 2/3 лет, где ≥ 20 сделок; ни один год не даёт > 50% чистой прибыли (P-05)."""
    if t.is_empty():
        return False, pl.DataFrame()
    years = t.with_columns(
        pl.col("day")
        .map_elements(lambda d: dt.date.fromordinal(d).year, return_dtype=pl.Int64)
        .alias("year")
    )
    by = (
        years.group_by("year")
        .agg(pl.len().alias("n"), pl.col("r_net").sum().alias("sum_r"), pl.col("r_net").mean())
        .sort("year")
    )
    counted = by.filter(pl.col("n") >= MIN_YEAR_TRADES)
    if counted.is_empty():
        return False, by
    share_ok = True
    total = float(by["sum_r"].sum())
    if total > 0:
        share_ok = float(by["sum_r"].to_numpy().max()) / total <= YEAR_SHARE
    pos = int((counted["r_net"] > 0).sum())
    return (pos >= math.ceil(2 / 3 * counted.height) and share_ok), by


def instrument_edges(t: pl.DataFrame) -> dict[str, tuple[int, float]]:
    """Инструмент → (сделок, mean R)."""
    g = t.group_by("symbol").agg(pl.len().alias("n"), pl.col("r_net").mean().alias("m"))
    return {row["symbol"]: (row["n"], row["m"]) for row in g.iter_rows(named=True)}


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
    lo, hi = int(calendar[0]), int(calendar[-1]) + 1
    days: dict[bool, list[I64]] = {True: [], False: []}
    rs: dict[bool, list[F64]] = {True: [], False: []}
    for cfg in cfgs:
        for b in bundles.values():
            if not b.live(cfg):
                continue
            for kind in _ONE_SIDED:
                t = scan_kind(b, cfg, kind).filter((pl.col("day") >= lo) & (pl.col("day") < hi))
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
