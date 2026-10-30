"""Автопоиск стратегий (ROADMAP M4): перебор → S1 → счётчик DSR → событийный бэктест → проп-MC.

Порядок защищает holdout и счётчик (VALIDATION §2, §4.3):

1. Бары после даты отсечения IS/holdout в поиск не попадают вообще (`make_windows`): holdout
   открывают гейты S2 для финалистов, не поиск.
2. Векторный `scan` всех конфигураций на корзинах инструментов одного класса (окно выбора).
   Каждая проба — конфигурация × инструмент и конфигурация × корзина — пишется в счётчик
   `research/registry/variants/<campaign>.parquet`, даже если провалила S1.
3. S1 (сделок, PF, E при издержках × 1.5), затем DSR по счётчику / PBO по выжившим.
4. Топ по Sharpe → событийный бэктест (`backtest`, одна позиция) → экран OOS → проп-MC;
   финальный порядок — по P(1-я выплата) при EV₁ > 0.

Режимы (`SearchParams.mode`):

- `forward` — выбор по первым `select_months` месяцам IS, оценка и MC по остатку IS
  (тесты WF без переоптимизации, VALIDATION §2): MC на сделках, которых отбор не видел.
- `is` — выбор и MC на всём IS; отчёт помечается `IN-SAMPLE — справка, не для решений`.

Экран OOS в поиске — предварительный: в нём отбираются 200 из тысяч вариантов, поэтому он не
заменяет holdout. Остальные гейты S2/S3 подключаются через `SearchParams.extra_gates`
(`Gate`) и `eval_trades` (чужие OOS-сделки для MC вместо стандартных).
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from pbl.config import REGISTRY, load_firms
from pbl.engine.backtest import RunMeta, backtest, scan
from pbl.engine.market import session_mask
from pbl.prop.days import build_day_table, trades_to_arrays
from pbl.prop.engine import MCConfig, MCEngine
from pbl.prop.optimize import optimize
from pbl.prop.rules import compile_offer
from pbl.prop.servertime import server_date
from pbl.research.idea_eval import COST_MULT, MIN_PF, MIN_TRADES, s1_stats
from pbl.research.range_sweep import FIRM, Bundle, make_bundle, window_is_live
from pbl.research.search_space import (
    GEN_VERSION,
    SearchConfig,
    enumerate_configs,
    sample_configs,
    with_tp,
)
from pbl.research.variants import DailyR, VariantLog, VariantSpec
from pbl.robustness.dsr import deflated_sharpe, expected_max_sharpe
from pbl.robustness.pbo import PBOResult, pbo_cscv
from pbl.robustness.splits import add_months, holdout_start
from pbl.schemas import Firm, Trade
from pbl.strategies.base import Signals
from pbl.strategies.families import RANGE_WINDOWS, TRADE_SESSIONS, Family, RangeBreakout, RangeFade
from pbl.strategies.features import FeatureCache, hhmm

Mode = Literal["forward", "is"]
CAMPAIGN = "AUTO-0001"
MIN_LIVE_SHARE = 0.5
"""Доля окна входов семейства, торгуемая у фирмы, чтобы инструмент участвовал в пробе."""
FLUSH_UNITS = 400
"""Сколько единиц копится до пакетной записи в счётчик (память и скорость)."""

SCAN_SCHEMA = {
    "day": pl.Int64,
    "side": pl.Int8,
    "r_net": pl.Float64,
    "r_gross": pl.Float64,
    "cost_r": pl.Float64,
}


@dataclass(frozen=True)
class Budget:
    """Бюджет ночного запуска (D-034, P-22 из DECISIONS)."""

    max_scans: int = 50_000
    max_event_runs: int = 200
    mc_top: int = 20
    max_seconds: float = 6 * 3600.0


@dataclass(frozen=True)
class S1Gate:
    """Пороги S1 (VALIDATION §3); значения по умолчанию — из прогона идеи №1."""

    min_trades: int = MIN_TRADES
    min_pf: float = MIN_PF
    cost_mult: float = COST_MULT


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool | None  # None — гейт неприменим к кандидату
    detail: str = ""


@dataclass(frozen=True)
class Windows:
    """Хронология поиска: общая для всех инструментов (D-032).

    Выбор — `[first, sel_end)`, оценка — `[sel_end, cut)`; holdout `[cut, …]` поиск не видит.
    В режиме `is` оценка совпадает с выбором (`sel_end == cut`).
    """

    first: dt.date
    sel_end: dt.date
    cut: dt.date
    last: dt.date

    @property
    def forward(self) -> bool:
        return self.sel_end < self.cut

    @property
    def sel(self) -> tuple[int, int]:
        return self.first.toordinal(), self.sel_end.toordinal()

    @property
    def ev(self) -> tuple[int, int]:
        return (self.sel_end.toordinal(), self.cut.toordinal()) if self.forward else self.sel


@dataclass(frozen=True)
class SearchParams:
    start: dt.date
    end: dt.date
    campaign: str = CAMPAIGN
    mode: Mode = "forward"
    select_months: int = 24
    is_fraction: float = 0.7
    seed: int = 42
    n_paths: int = 10_000
    on: dt.date = field(default_factory=lambda: dt.datetime.now(dt.UTC).date())
    budget: Budget = field(default_factory=Budget)
    s1: S1Gate = field(default_factory=S1Gate)
    min_dsr: float = 0.95
    max_pbo: float = 0.2
    pbo_blocks: int = 16
    min_eval_trades: int = 30
    programs: tuple[str, ...] = ("fn-stellar-2step",)
    sizes_usd: tuple[int, ...] = (25_000,)
    firm_id: str = FIRM
    registry: Path = REGISTRY
    tag: str = ""
    """Добавка к версии проб: новая проба на тех же данных (например, после правки данных окна)."""
    explore: bool = False
    """Гейты только помечают кандидатов, но не отсеивают (разведка; в отчёте — предупреждение)."""
    families: Sequence[Family] | None = None
    exits: Sequence[str] | None = None
    filters: Sequence[str] | None = None
    mc: MCConfig | None = None
    extra_gates: tuple[Gate, ...] = ()
    eval_trades: EvalTrades | None = None

    @property
    def version(self) -> str:
        return f"{GEN_VERSION}{self.tag}"


# --- данные -------------------------------------------------------------------------------


def make_windows(bars: dict[str, pl.DataFrame], p: SearchParams) -> Windows:
    """Общий первый день, последний общий день, дата отсечения и конец окна выбора."""
    if not bars:
        raise ValueError("нет баров")
    firsts = [b["ts"].min() for b in bars.values()]
    lasts = [b["ts"].max() for b in bars.values()]
    first = max(x.date() for x in firsts)  # type: ignore[union-attr]
    last = min(x.date() for x in lasts)  # type: ignore[union-attr]
    cut = holdout_start(first, last, p.is_fraction)
    sel_end = cut if p.mode == "is" else add_months(first, p.select_months)
    if p.mode == "forward" and (cut - sel_end).days < 30:
        raise ValueError(
            f"мало данных для режима forward: выбор до {sel_end}, отсечение {cut}; "
            "уменьшите --select-months или используйте --mode is"
        )
    return Windows(first, sel_end, cut, last)


def restrict(bars: pl.DataFrame, w: Windows) -> pl.DataFrame:
    """Бары общего окна до даты отсечения: holdout в поиск не попадает."""
    lo = dt.datetime.combine(w.first, dt.time(0), dt.UTC)
    hi = dt.datetime.combine(w.cut, dt.time(0), dt.UTC)
    return bars.filter((pl.col("ts") >= lo) & (pl.col("ts") < hi))


@dataclass
class Instrument:
    """Рынок движка, кэш признаков и кэш сигналов одного инструмента."""

    symbol: str
    bundle: Bundle
    cache: FeatureCache
    _base: dict[str, Signals] = field(default_factory=dict)
    _filtered: dict[tuple[str, str], Signals] = field(default_factory=dict)

    @classmethod
    def build(cls, symbol: str, bars: pl.DataFrame) -> Instrument:
        b = make_bundle(symbol, bars)
        return cls(symbol, b, FeatureCache(b.frame))

    def signals(self, cfg: SearchConfig) -> Signals:
        """Сигналы конфигурации: семейство → фильтр → тейк выхода (первые два кэшируются)."""
        sk = cfg.signal_key
        if sk not in self._base:
            self._base[sk] = cfg.family.signals_from(self.cache)
        key = (sk, cfg.filt)
        if key not in self._filtered:
            base = self._base[sk]
            self._filtered[key] = (
                base
                if cfg.filt == "none"
                else base.take(self.cache.filter_mask(cfg.filt, base.bar, base.side))
            )
        return with_tp(self._filtered[key], cfg.exit.tp_r)


def family_live(inst: Instrument, fam: Family) -> bool:
    """Окно входов семейства торгуется у фирмы (в понедельник): диапазон — целиком, иначе ≥ 50%."""
    spec = inst.bundle.spec
    if isinstance(fam, RangeBreakout | RangeFade):
        a, b = RANGE_WINDOWS[fam.session]
        return window_is_live(spec, a, b)
    s = TRADE_SESSIONS[fam.session]
    minute = np.arange(hhmm(s.open), hhmm(s.last_entry), dtype=np.int64)
    live = session_mask(spec, np.zeros_like(minute), minute)
    return bool(live.mean() >= MIN_LIVE_SHARE)


def group_symbols(instruments: dict[str, Instrument]) -> dict[str, tuple[str, ...]]:
    """Корзины по классу актива (`indices`, `metals`, `fx`, …); порядок — как у инструментов."""
    out: dict[str, list[str]] = {}
    for sym, inst in instruments.items():
        out.setdefault(inst.bundle.spec.asset_class, []).append(sym)
    return {k: tuple(v) for k, v in out.items()}


# --- векторный скан -----------------------------------------------------------------------


def scan_table(inst: Instrument, cfg: SearchConfig, lo: int, hi: int) -> pl.DataFrame:
    """Сделки скана конфигурации на инструменте за серверные дни `[lo, hi)`."""
    sig = inst.signals(cfg)
    if len(sig):
        d = inst.cache.day[sig.bar]
        sig = sig.take((d >= lo) & (d < hi))
    if len(sig) == 0:
        return pl.DataFrame(schema=SCAN_SCHEMA)
    res = scan(inst.bundle.market, sig, cfg.exit.exits, profit_to_usd=inst.bundle.profit_to_usd)
    if res.n_trades == 0:
        return pl.DataFrame(schema=SCAN_SCHEMA)
    return pl.DataFrame(
        {
            "day": inst.bundle.market.day[res.entry_bar],
            "side": res.side.astype(np.int8),
            "r_net": res.r_net,
            "r_gross": res.r_gross,
            "cost_r": res.cost_r,
        },
        schema=SCAN_SCHEMA,
    )


def daily_r(table: pl.DataFrame, start: dt.date, end: dt.date) -> DailyR:
    """Дневной ряд `r_net` (D-022: суббота и воскресенье — в пятницу) для счётчика вариантов."""
    if table.is_empty():
        return DailyR(start, end, (), (), 0, 0.0, 0.0)
    day = table["day"].to_numpy()
    r = table["r_net"].to_numpy()
    first_bd = start.toordinal() + max(0, 7 - start.weekday()) % 7 * (start.weekday() >= 5)
    bd = np.maximum(day - np.maximum((day - 1) % 7 - 4, 0), first_bd)
    uniq, inv = np.unique(bd, return_inverse=True)
    sums = np.bincount(inv.ravel(), weights=r)
    return DailyR(
        start=start,
        end=end,
        dates=tuple(dt.date.fromordinal(int(d)) for d in uniq),
        r=tuple(sums.tolist()),
        n_trades=len(r),
        win_r=float(r[r > 0].sum()),
        loss_r=float(r[r <= 0].sum()),
    )


def variant_spec(cfg: SearchConfig, instrument: str, p: SearchParams, w: Windows) -> VariantSpec:
    """Проба для счётчика: новая выборка данных или версия генератора — новая проба."""
    fam = cfg.family
    return VariantSpec(
        idea_id=p.campaign,
        family=fam.family,
        version=f"{p.version}:{w.first:%Y%m%d}-{w.sel_end:%Y%m%d}",
        instrument=instrument,
        timeframe=fam.timeframe,
        session=fam.session,
        params=fam.params(),
        exits=cfg.exit_specs,
        filters=cfg.filter_specs,
        firm_profile=p.firm_id,
    )


def basket_name(symbols: Sequence[str]) -> str:
    return "BASKET:" + "+".join(symbols)


@dataclass
class Unit:
    """Конфигурация на корзине инструментов: единица отбора (как в идее №1, D-032)."""

    cfg: SearchConfig
    group: str
    symbols: tuple[str, ...]
    variant_id: str  # проба корзины в счётчике
    stats: dict[str, float]  # S1: n, pf, mean_r, e15, cost_share; sum_r — сумма r_net скана
    s1: bool

    @property
    def key(self) -> str:
        return f"{self.cfg.key}@{self.group}"


def scan_stage(
    instruments: dict[str, Instrument],
    configs: Sequence[SearchConfig],
    log: VariantLog,
    p: SearchParams,
    w: Windows,
    deadline: float,
) -> tuple[list[Unit], int, bool]:
    """Скан всех конфигураций, пакетная запись проб. → (единицы, сканов, прервано ли по времени)."""
    lo, hi = w.sel
    groups = group_symbols(instruments)
    end = w.sel_end - dt.timedelta(days=1)
    units: list[Unit] = []
    batch: list[tuple[VariantSpec, DailyR, bool]] = []
    n_scans = 0
    truncated = False

    def flush() -> None:
        log.add_many(batch, run_id=f"{p.campaign}-{w.first:%Y%m%d}-{w.sel_end:%Y%m%d}", on=p.on)
        batch.clear()

    for cfg in configs:
        if time.monotonic() > deadline:
            truncated = True
            break
        for group, symbols in groups.items():
            live = [s for s in symbols if family_live(instruments[s], cfg.family)]
            tables = {s: scan_table(instruments[s], cfg, lo, hi) for s in live}
            n_scans += len(live)
            tables = {s: t for s, t in tables.items() if not t.is_empty()}
            if not tables:
                continue  # стратегия ни разу не сработала: это не проба
            for s, t in tables.items():
                batch.append((variant_spec(cfg, s, p, w), daily_r(t, w.first, end), False))
            pool = pl.concat(tables.values())
            st = s1_stats(pool)
            gross, cost = pool["r_gross"].to_numpy(), pool["cost_r"].to_numpy()
            st["e15"] = float((gross - p.s1.cost_mult * cost).mean())
            ok = bool(st["n"] >= p.s1.min_trades and st["pf"] >= p.s1.min_pf and st["e15"] > 0)
            st["sum_r"] = float(pool["r_net"].sum())
            spec = variant_spec(cfg, basket_name(list(tables)), p, w)
            batch.append((spec, daily_r(pool, w.first, end), ok))
            units.append(Unit(cfg, group, tuple(tables), spec.variant_id, st, ok))
        if len(batch) >= FLUSH_UNITS:
            flush()
    flush()
    return units, n_scans, truncated


# --- DSR / PBO ----------------------------------------------------------------------------


@dataclass
class Deflation:
    """DSR/PBO среди выживших S1 (VALIDATION §4.3): N — все пробы в счётчике кампании."""

    n_trials: int
    sr0: float
    pbo: PBOResult | None
    sharpe: dict[str, float]
    dsr: dict[str, float]


def deflate(log: VariantLog, units: Sequence[Unit], p: SearchParams) -> Deflation:
    frame = log.frame()
    trials = frame["sharpe_daily"].to_numpy()
    var = float(trials.var(ddof=1)) if trials.size > 1 else 0.0
    sr0 = expected_max_sharpe(int(trials.size), var)
    ids = [u.variant_id for u in units]
    rows = {
        r["variant_id"]: r
        for r in frame.filter(pl.col("variant_id").is_in(ids)).iter_rows(named=True)
    }
    sharpe = {v: float(r["sharpe_daily"]) for v, r in rows.items()}
    dsr = {
        v: deflated_sharpe(
            float(r["sharpe_daily"]), int(r["n_days"]), float(r["skew"]), float(r["kurt"]), sr0
        )
        for v, r in rows.items()
    }
    pbo: PBOResult | None = None
    survivors = [u.variant_id for u in units if u.s1]
    if len(survivors) >= 2:
        try:
            pbo = pbo_cscv(log.returns_matrix(survivors).values, p.pbo_blocks)
        except ValueError:
            pbo = None
    return Deflation(int(trials.size), sr0, pbo, sharpe, dsr)


# --- событийный бэктест и MC --------------------------------------------------------------


@dataclass(frozen=True)
class McSummary:
    """Лучшая программа по P(1-я выплата) при EV₁ > 0 (иначе — просто лучшая по P)."""

    label: str
    key: str
    p1: float
    p1_ci: tuple[float, float]
    p_funded: float
    ev1: float
    days_funded_median: float
    risk: str
    ev_ok: bool
    n_trades: int


@dataclass
class Candidate:
    unit: Unit
    sharpe: float
    dsr: float
    gates: dict[str, bool | None] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    eval_trades: list[Trade] = field(default_factory=list)
    eval_sample: str = "IS"
    eval_stats: dict[str, float] = field(default_factory=dict)
    parity_gap: float = math.nan
    mc: McSummary | None = None

    @property
    def passed(self) -> bool:
        return all(v is not False for v in self.gates.values())

    @property
    def failed(self) -> list[str]:
        return [k for k, v in self.gates.items() if v is False]


@dataclass(frozen=True)
class SearchContext:
    params: SearchParams
    windows: Windows
    firm: Firm
    instruments: dict[str, Instrument]
    log: VariantLog
    deflation: Deflation


Gate = Callable[[Candidate, SearchContext], GateResult]
"""Гейт кандидата после событийного бэктеста: S2/S3 из `pbl.robustness` подключаются сюда."""
EvalTrades = Callable[[Candidate, SearchContext], tuple[Sequence[Trade], str] | None]
"""Свои OOS-сделки для MC (например, WF + holdout): (сделки, метка выборки) или None."""


def pf_of(r: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    lose = -float(r[r < 0].sum())
    return float(r[r > 0].sum()) / lose if lose > 0 else math.inf


QUICK_STRIKE_SEC, QUICK_STRIKE_SHARE = 30, 0.20
"""FN Quick Strike (SPEC B2): доля прибыли со сделок короче 30 с должна быть меньше 20%."""


def eval_stats(trades: Sequence[Trade]) -> dict[str, float]:
    r = np.asarray([t.r_net for t in trades], dtype=np.float64)
    if r.size == 0:
        return {"n": 0.0, "mean_r": math.nan, "pf": math.nan, "win_rate": math.nan, "qs": 0.0}
    hold = np.asarray([t.hold_sec for t in trades])
    profit = np.clip(r, 0.0, None)
    total = float(profit.sum())
    return {
        "n": float(r.size),
        "mean_r": float(r.mean()),
        "pf": pf_of(r),
        "win_rate": float((r > 0).mean()),
        "qs": float(profit[hold < QUICK_STRIKE_SEC].sum() / total) if total > 0 else 0.0,
    }


def run_events(
    unit: Unit, instruments: dict[str, Instrument], p: SearchParams, w: Windows, firm: Firm
) -> list[Trade]:
    """Событийный бэктест (одна позиция) по инструментам корзины; сделки IS в хронологии."""
    out: list[Trade] = []
    for sym in unit.symbols:
        inst = instruments[sym]
        meta = RunMeta(
            strategy_id=unit.variant_id,
            variant_id=variant_spec(unit.cfg, sym, p, w).variant_id,
            sample="IS",
            profit_to_usd=inst.bundle.profit_to_usd,
        )
        res = backtest(
            inst.bundle.market,
            inst.signals(unit.cfg),
            unit.cfg.exit.exits,
            meta,
            with_equity=False,
        )
        out += res.trades
    out.sort(key=lambda t: t.open_utc)
    return out


def split_samples(trades: Sequence[Trade], w: Windows, firm: Firm) -> tuple[list[Trade], str]:
    """Сделки оценки: `forward` — после конца окна выбора (метка WF), `is` — все (метка IS)."""
    if not w.forward:
        return list(trades), "IS"
    cut = w.sel_end.toordinal()
    out = [
        t.model_copy(update={"sample": "WF"})
        for t in trades
        if server_date(t.open_utc, firm.server_time).toordinal() >= cut
    ]
    return out, "WF"


def mc_summary(trades: Sequence[Trade], firm: Firm, p: SearchParams) -> McSummary | None:
    """Проп-MC на сделках оценки: оптимальный риск по фазам, лучшая программа из `p.programs`."""
    if len(trades) < p.min_eval_trades:
        return None
    cfg = p.mc or MCConfig(n_paths=p.n_paths, seed=p.seed, outer_boot=0)
    try:
        ta = trades_to_arrays(trades, firm.server_time, None)
        eng = MCEngine(build_day_table(ta), cfg, ta)
    except ValueError:
        return None
    best: tuple[tuple[bool, float], McSummary] | None = None
    for prog in firm.programs:
        if prog.id not in p.programs:
            continue
        for offer in prog.iter_offers(ea_only=True):
            if offer.size.size_usd not in p.sizes_usd:
                continue
            rules = compile_offer(firm.id, prog, offer, firm.name)
            opt = optimize(eng, rules)
            st = opt.best.stats
            if math.isnan(st.p1):
                continue
            f = opt.best.risk.f
            row = McSummary(
                label=rules.label,
                key=rules.key,
                p1=st.p1,
                p1_ci=st.p1_ci,
                p_funded=st.p_funded,
                ev1=st.ev1,
                days_funded_median=st.days_funded_median,
                risk=" / ".join(f"{100 * x:g}" for x in f) + "%",
                ev_ok=bool(st.ev1 > 0),
                n_trades=len(trades),
            )
            score = (row.ev_ok, row.p1)
            if best is None or score > best[0]:
                best = (score, row)
    return None if best is None else best[1]


# --- прогон -------------------------------------------------------------------------------


@dataclass
class SearchResult:
    params: SearchParams
    windows: Windows
    log: VariantLog
    deflation: Deflation
    units: list[Unit]
    candidates: list[Candidate]
    funnel: dict[str, int]
    by_family: dict[str, dict[str, int]]
    truncated: bool
    elapsed: float
    groups: dict[str, tuple[str, ...]]
    n_configs_total: int

    @property
    def basis(self) -> str:
        return "IN-SAMPLE" if not self.windows.forward else "WF (вне окна выбора)"


def rank_key(c: Candidate) -> tuple[int, int, float, float]:
    """MC с EV₁ > 0 — выше всех, затем по P(1-я выплата); без MC — по Sharpe выбора."""
    mc = c.mc
    has = mc is not None
    return (
        1 if has and mc is not None and mc.ev_ok else 0,
        1 if has else 0,
        mc.p1 if mc is not None else -1.0,
        c.sharpe,
    )


def run_search(bars: dict[str, pl.DataFrame], p: SearchParams) -> SearchResult:
    """Весь перебор на барах `{символ: M1 bid/ask}`; `log.save()` вызывает вызывающий."""
    t0 = time.monotonic()
    deadline = t0 + p.budget.max_seconds
    w = make_windows(bars, p)
    firm = next(f for f in load_firms() if f.id == p.firm_id)
    instruments = {s: Instrument.build(s, restrict(b, w)) for s, b in bars.items()}
    groups = group_symbols(instruments)

    all_configs = enumerate_configs(
        p.families,
        exits=p.exits,
        filters=p.filters if p.filters is not None else ("none", "trend", "vol_hi", "vol_lo"),
    )
    configs = sample_configs(
        all_configs,
        p.budget.max_scans,
        sum(len(v) for v in groups.values()),
        p.seed,
    )
    log = VariantLog.for_idea(p.campaign, p.registry)
    units, n_scans, truncated = scan_stage(instruments, configs, log, p, w, deadline)
    survivors = [u for u in units if u.s1]
    defl = deflate(log, units, p)
    ctx = SearchContext(p, w, firm, instruments, log, defl)

    # S1 → DSR/PBO → отбор в событийный бэктест по Sharpe окна выбора
    pbo_ok = defl.pbo is not None and defl.pbo.pbo <= p.max_pbo
    cands: list[Candidate] = []
    for u in sorted(survivors, key=lambda u: -defl.sharpe.get(u.variant_id, -math.inf)):
        dsr = defl.dsr.get(u.variant_id, math.nan)
        c = Candidate(u, defl.sharpe.get(u.variant_id, math.nan), dsr)
        c.gates["S1"] = True
        c.gates["DSR/PBO"] = bool((not math.isnan(dsr) and dsr >= p.min_dsr) or pbo_ok)
        cands.append(c)
    # одинаковые сделки при разных выходах или фильтрах — один исход: в бэктест идёт первый
    seen: set[tuple[float, float, float]] = set()
    unique: list[Candidate] = []
    for c in cands:
        fp = (c.unit.stats["n"], round(c.unit.stats["sum_r"], 8), round(c.sharpe, 10))
        if fp not in seen:
            seen.add(fp)
            unique.append(c)
    pool = [c for c in unique if p.explore or c.gates["DSR/PBO"]]
    event = pool[: p.budget.max_event_runs]
    for c in event:
        if time.monotonic() > deadline:
            truncated = True
            break
        c.trades = run_events(c.unit, instruments, p, w, firm)
        ev_trades, sample = split_samples(c.trades, w, firm)
        c.eval_trades, c.eval_sample = ev_trades, sample
        sel_r = sum(
            t.r_net
            for t in c.trades
            if server_date(t.open_utc, firm.server_time).toordinal() < w.sel[1]
        )
        c.parity_gap = abs(sel_r - c.unit.stats["sum_r"]) / max(abs(c.unit.stats["sum_r"]), 1.0)
        if p.eval_trades is not None:
            own = p.eval_trades(c, ctx)
            if own is not None:
                c.eval_trades, c.eval_sample = list(own[0]), own[1]
        c.eval_stats = eval_stats(c.eval_trades)
        n_ev, mean_ev = c.eval_stats["n"], c.eval_stats["mean_r"]
        c.gates["OOS E>0 (экран)"] = bool(n_ev >= p.min_eval_trades and mean_ev > 0)
        c.gates["Quick Strike"] = bool(c.eval_stats["qs"] < QUICK_STRIKE_SHARE)
        for gate in p.extra_gates:
            res = gate(c, ctx)
            c.gates[res.name] = res.passed
            if res.detail:
                c.details[res.name] = res.detail

    mc_pool = [c for c in event if p.explore or c.passed][: p.budget.mc_top]
    for c in mc_pool:
        if time.monotonic() > deadline:
            truncated = True
            break
        c.mc = mc_summary(c.eval_trades, firm, p)

    ranked = sorted(event, key=rank_key, reverse=True)
    funnel = {
        "конфигураций в сетке": len(all_configs),
        "конфигураций прогнано": len(configs),
        "сканов": n_scans,
        "проб в счётчике кампании": len(log),
        "единиц (конфигурация × корзина)": len(units),
        "прошли S1": len(survivors),
        "из них с уникальным исходом сделок": len(unique),
        "прошли DSR/PBO": sum(1 for c in unique if c.gates["DSR/PBO"]),
        "событийный бэктест": len(event),
        "прошли все гейты": sum(1 for c in event if c.passed),
        "проп-MC": sum(1 for c in event if c.mc is not None),
        "MC: EV₁ > 0": sum(1 for c in event if c.mc is not None and c.mc.ev_ok),
    }
    by_family: dict[str, dict[str, int]] = {}
    for u in units:
        d = by_family.setdefault(u.cfg.family.family, {"units": 0, "s1": 0})
        d["units"] += 1
        d["s1"] += int(u.s1)
    return SearchResult(
        params=p,
        windows=w,
        log=log,
        deflation=defl,
        units=units,
        candidates=ranked,
        funnel=funnel,
        by_family=by_family,
        truncated=truncated,
        elapsed=time.monotonic() - t0,
        groups=groups,
        n_configs_total=len(all_configs),
    )


def with_params(p: SearchParams, **kw: Any) -> SearchParams:
    return replace(p, **kw)
