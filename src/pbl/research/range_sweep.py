"""Сетка вариантов идеи №1 (Range bot) и массовый скан на движке (IDEAS §5, VALIDATION §3).

Вариант = инструмент × окно диапазона × семья входа × выход × фильтр дня. На каждый вариант
считаются три независимых скана — только лонг, только шорт, обе стороны (OCO для пробоя, первый
сетап для возврата); фильтр дня автора собирается из них по дням недели без повторного скана.
Те же независимые лонг/шорт-сделки дают корзины «день × направление» для проверки фильтра.

Деньги в R при номинальном риске 1R на сделку; `scan` не ограничивает число позиций (на одной
сделке в день на инструмент это не меняет результата). Правила фирмы, меняющие сделки (окно
00:00–02:00 сервера и т. п.), в `Market.entry_ok` учтены.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np
import polars as pl

from pbl.config import CONFIGS, load_firms, load_model
from pbl.engine.backtest import Exits, ScanResult, scan
from pbl.engine.market import FirmTradeProfile, Market, prepare_market, session_mask, strategy_frame
from pbl.instruments.costs import CostModel
from pbl.research.variants import DailyR, VariantSpec
from pbl.schemas import InstrumentSpec
from pbl.schemas.strategy import ExitSpec, FilterSpec
from pbl.strategies.base import Signals
from pbl.strategies.range_session import (
    AUTHOR_DAY_SIDES,
    DaySides,
    FadeEvents,
    SessionDays,
    Side,
    breakout_signals,
    fade_events,
    fade_signals,
    session_days,
)

IDEA_ID = "IDEA-0001"
FIRM = "fundednext"
PROGRAM = "fn-stellar-2step"
INSTRUMENTS = ("NAS100", "US30", "SPX500", "GER30")
UNTIL = "23:00"
"""Закрытие NY (16:00 ET) по серверу FN: до него живут заявки, в него закрываются позиции."""

SESSIONS: dict[str, tuple[str, str]] = {
    "asia": ("03:00", "07:00"),  # 20:00–00:00 ET
    "london": ("10:00", "12:00"),  # 03:00–05:00 ET
    "ny15": ("16:30", "16:45"),  # первые 15 мин NY
    "ny30": ("16:30", "17:00"),
    "ny60": ("16:30", "17:30"),
}
MODES = ("breakout", "fade")
BUFFERS = (0.0, 0.05, 0.1)
"""Отступ стоп-ордеров пробоя в долях ширины диапазона (карточка IDEA-0001)."""
FILTERS = ("none", "author")
KINDS: tuple[Side, ...] = ("long", "short", "both")

TABLE_SCHEMA = {
    "day": pl.Int64,
    "wd": pl.Int8,
    "side": pl.Int8,
    "r_net": pl.Float64,
    "r_gross": pl.Float64,
    "cost_r": pl.Float64,
    "hold_sec": pl.Int64,
}


@dataclass(frozen=True)
class ExitCfg:
    """Выход: тейк в R плюс правила движка. Уровни в R масштабируются целиком (чувствительность)."""

    name: str
    tp_r: float | None
    exits: Exits
    specs: tuple[ExitSpec, ...]

    def scaled(self, k: float) -> ExitCfg:
        e = self.exits
        return replace(
            self,
            tp_r=None if self.tp_r is None else self.tp_r * k,
            exits=replace(
                e,
                be_at_r=e.be_at_r * k,
                trail_start_r=e.trail_start_r * k,
                trail_dist_r=e.trail_dist_r * k,
                partial_r=e.partial_r * k,
            ),
            specs=tuple(
                ExitSpec(
                    kind=s.kind,
                    params={
                        key: (v * k if isinstance(v, float) and key != "frac" else v)
                        for key, v in s.params.items()
                    },
                )
                for s in self.specs
            ),
        )


def _close(until: str = UNTIL) -> ExitSpec:
    return ExitSpec(kind="session_close", params={"at": until})


def exit_grid(until: str = UNTIL) -> dict[str, ExitCfg]:
    """Семь выходов карточки: RR 1:1, 1:2, 1:3, трейлинг, частичный, безубыток, по времени.

    `until` — время сервера, к которому закрываются позиции (по умолчанию закрытие NY).
    """
    close = _close(until)
    ex = Exits(close_at=until)
    return {
        "rr_1_1": ExitCfg("rr_1_1", 1.0, ex, (ExitSpec(kind="rr", params={"rr": 1.0}), close)),
        "rr_1_2": ExitCfg("rr_1_2", 2.0, ex, (ExitSpec(kind="rr", params={"rr": 2.0}), close)),
        "rr_1_3": ExitCfg("rr_1_3", 3.0, ex, (ExitSpec(kind="rr", params={"rr": 3.0}), close)),
        "trailing": ExitCfg(
            "trailing",
            None,
            replace(ex, trail_start_r=1.0, trail_dist_r=1.0),
            (ExitSpec(kind="trailing", params={"start_r": 1.0, "dist_r": 1.0}), close),
        ),
        "partial": ExitCfg(
            "partial",
            2.0,
            replace(ex, partial_r=1.0, partial_frac=0.5),
            (
                ExitSpec(kind="rr", params={"rr": 2.0}),
                ExitSpec(kind="partial", params={"at_r": 1.0, "frac": 0.5}),
                close,
            ),
        ),
        "breakeven": ExitCfg(
            "breakeven",
            2.0,
            replace(ex, be_at_r=1.0),
            (
                ExitSpec(kind="rr", params={"rr": 2.0}),
                ExitSpec(kind="breakeven", params={"at_r": 1.0}),
                close,
            ),
        ),
        "time_exit": ExitCfg("time_exit", None, ex, (close,)),
    }


@dataclass(frozen=True)
class Config:
    """Стратегия без инструмента: инструменты складываются в корзину (пулинг)."""

    session: str
    mode: Literal["breakout", "fade"]
    exit: ExitCfg
    filt: Literal["none", "author"]
    start: str = ""
    end: str = ""
    buffer: float = 0.0

    def __post_init__(self) -> None:
        if not self.start:
            a, b = SESSIONS[self.session]
            object.__setattr__(self, "start", a)
            object.__setattr__(self, "end", b)

    @property
    def key(self) -> str:
        buf = f"-b{self.buffer:g}" if self.buffer else ""
        return f"{self.mode}-{self.session}-{self.exit.name}-{self.filt}{buf}"

    @property
    def day_sides(self) -> DaySides | None:
        return AUTHOR_DAY_SIDES if self.filt == "author" else None


def grid() -> list[Config]:
    """Сетка: 5 окон × 7 выходов × 2 фильтра × (3 буфера пробоя + 1 возврат) = 280."""
    exits = exit_grid()
    return [
        Config(session=s, mode=m, exit=exits[e], filt=f, buffer=b)  # type: ignore[arg-type]
        for s in SESSIONS
        for m in MODES
        for e in exits
        for f in FILTERS
        for b in (BUFFERS if m == "breakout" else (0.0,))
    ]


def window_is_live(spec: InstrumentSpec, start: str, end: str) -> bool:
    """Окно диапазона целиком внутри торговой сессии инструмента у фирмы (в понедельник)."""
    a, b = (int(x[:2]) * 60 + int(x[3:]) for x in (start, end))
    minute = np.arange(a, b, dtype=np.int64)
    return bool(session_mask(spec, np.zeros_like(minute), minute).all())


@dataclass
class Bundle:
    """Бары, рынок и кэши одного инструмента."""

    symbol: str
    spec: InstrumentSpec
    market: Market
    frame: pl.DataFrame
    profit_to_usd: float | None = None
    _days: dict[tuple[str, str, str], SessionDays] = field(default_factory=dict)
    _fade: dict[tuple[str, str, str], FadeEvents] = field(default_factory=dict)
    _scans: dict[tuple[object, ...], pl.DataFrame] = field(default_factory=dict)

    def sessions(self, start: str, end: str) -> SessionDays:
        key = (start, end, UNTIL)
        if key not in self._days:
            self._days[key] = session_days(self.frame, start, end, UNTIL)
        return self._days[key]

    def events(self, start: str, end: str) -> FadeEvents:
        key = (start, end, UNTIL)
        if key not in self._fade:
            self._fade[key] = fade_events(self.frame, self.sessions(start, end))
        return self._fade[key]

    def live(self, cfg: Config) -> bool:
        return window_is_live(self.spec, cfg.start, cfg.end)


def make_bundle(symbol: str, bars: pl.DataFrame, *, firm_id: str = FIRM) -> Bundle:
    """Бары M1 (UTC, bid/ask) → рынок движка с издержками программы FN Stellar 2-Step."""
    firm = next(f for f in load_firms() if f.id == firm_id)
    prog = next(p for p in firm.programs if p.id == PROGRAM)
    spec = load_model(CONFIGS / "instruments" / firm_id / f"{symbol}.yaml", InstrumentSpec)
    market = prepare_market(
        bars,
        spec,
        firm.server_time,
        CostModel(spread="conservative"),
        FirmTradeProfile.from_rules(prog.trade_rules, funded=False),
        prog.commission,
    )
    # GER30 считается в EUR (O-20): на R курс не влияет — комиссия по индексам 0, ночей нет
    to_usd = 1.0 if spec.profit_currency != "USD" else None
    return Bundle(symbol, spec, market, strategy_frame(bars, firm.server_time), to_usd)


def config_signals(
    b: Bundle, cfg: Config, kind: Side = "both", day_sides: DaySides | None = None
) -> Signals:
    """Сигналы конфигурации напрямую (для событийного бэктеста); фильтр дня — `day_sides`."""
    sd = b.sessions(cfg.start, cfg.end)
    kw = {"buffer": cfg.buffer, "tp_r": cfg.exit.tp_r, "sides": kind, "day_sides": day_sides}
    if cfg.mode == "breakout":
        return breakout_signals(sd, **kw)  # type: ignore[arg-type]
    return fade_signals(b.events(cfg.start, cfg.end), sd, **kw)  # type: ignore[arg-type]


def _table(b: Bundle, res: ScanResult) -> pl.DataFrame:
    if res.n_trades == 0:
        return pl.DataFrame(schema=TABLE_SCHEMA)
    return pl.DataFrame(
        {
            "day": b.market.day[res.entry_bar],
            "wd": b.market.weekday[res.entry_bar].astype(np.int8),
            "side": res.side.astype(np.int8),
            "r_net": res.r_net,
            "r_gross": res.r_gross,
            "cost_r": res.cost_r,
            "hold_sec": res.hold_sec,
        }
    )


def scan_kind(b: Bundle, cfg: Config, kind: Side) -> pl.DataFrame:
    """Сделки одного независимого скана (без фильтра дня); кэш по параметрам скана."""
    key = (cfg.mode, cfg.start, cfg.end, cfg.buffer, cfg.exit.tp_r, cfg.exit.exits, kind)
    if key not in b._scans:
        sig = config_signals(b, cfg, kind)
        res = scan(
            b.market,
            sig,
            cfg.exit.exits,
            profit_to_usd=b.profit_to_usd,
        )
        b._scans[key] = _table(b, res)
    return b._scans[key]


def config_trades(b: Bundle, cfg: Config) -> pl.DataFrame:
    """Сделки конфигурации на инструменте: `none` — обе стороны каждый день; `author` — по дням."""
    if cfg.filt == "none":
        return scan_kind(b, cfg, "both")
    parts: list[pl.DataFrame] = []
    for wd, rule in enumerate(AUTHOR_DAY_SIDES):
        if rule == "none":
            continue
        kind: Side = rule
        parts.append(scan_kind(b, cfg, kind).filter(pl.col("wd") == wd))
    return pl.concat(parts).sort("day", "side") if parts else pl.DataFrame(schema=TABLE_SCHEMA)


def variant_spec(cfg: Config, symbol: str, version: str = "v1") -> VariantSpec:
    filters = (
        [FilterSpec(kind="day_of_week", params=dict(zip(_DAYS, AUTHOR_DAY_SIDES, strict=True)))]
        if cfg.filt == "author"
        else []
    )
    return VariantSpec(
        idea_id=IDEA_ID,
        family=f"range_{cfg.mode}",
        version=version,
        instrument=symbol,
        timeframe="M1",
        session=cfg.session,
        params={
            "range_start": cfg.start,
            "range_end": cfg.end,
            "until": UNTIL,
            "buffer": cfg.buffer,
            "exit": cfg.exit.name,
        },
        exits=list(cfg.exit.specs),
        filters=filters,
        firm_profile=FIRM,
    )


_DAYS = ("mon", "tue", "wed", "thu", "fri")


def business_day(day_ordinal: int) -> dt.date:
    """Серверный день сделки; суббота и воскресенье (у индексов FN не бывает) — в пятницу."""
    d = dt.date.fromordinal(int(day_ordinal))
    return d - dt.timedelta(days=max(d.weekday() - 4, 0))


def daily_from_table(table: pl.DataFrame, start: dt.date, end: dt.date) -> DailyR:
    """Дневной ряд `r_net` для счётчика вариантов; сделки вне окна отвергаются."""
    if table.is_empty():
        return DailyR(start, end, (), (), 0, 0.0, 0.0)
    g = (
        table.with_columns(
            pl.col("day").map_elements(business_day, return_dtype=pl.Date).alias("date")
        )
        .group_by("date")
        .agg(pl.col("r_net").sum())
        .sort("date")
    )
    r = table["r_net"].to_numpy()
    return DailyR(
        start=start,
        end=end,
        dates=tuple(g["date"].to_list()),
        r=tuple(g["r_net"].to_list()),
        n_trades=table.height,
        win_r=float(r[r > 0].sum()),
        loss_r=float(r[r <= 0].sum()),
    )
