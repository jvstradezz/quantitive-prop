"""Пространство автопоиска: семейства × параметры × сессии × выходы × фильтры (ROADMAP M4).

Конфигурация поиска (`SearchConfig`) — семейство с параметрами, выход и фильтр; инструмент в неё
не входит: конфигурация прогоняется на корзине инструментов одного класса (как идея №1, D-032),
а каждая проба — конфигурация × инструмент и конфигурация × корзина — попадает в счётчик
вариантов для DSR (VALIDATION §4.3). Перечисление детерминировано; при нехватке бюджета
(P-22, D-034) берётся детерминированная подвыборка по seed.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from pbl.research.range_sweep import ExitCfg, exit_grid
from pbl.schemas.strategy import ExitSpec, FilterSpec
from pbl.strategies.base import Signals
from pbl.strategies.families import (
    RANGE_WINDOWS,
    TRADE_SESSIONS,
    Donchian,
    Family,
    MeanReversionZ,
    RangeBreakout,
    RangeFade,
    SessionDrive,
    TrendEma,
)
from pbl.strategies.features import FILTERS

GEN_VERSION = "g1"
"""Версия генератора: меняется вместе с кодом семейств, сеткой или правилами сканирования."""

BUFFERS = (0.0, 0.05, 0.1)
TF_SETS = {"donchian": (5, 15, 30), "meanrev": (5, 15), "trend": (5, 15, 30)}
SL_ATR = (1.5, 3.0)


def family_grid() -> list[Family]:
    """Все семейства с сеткой параметров (детерминированный порядок)."""
    out: list[Family] = []
    out += [RangeBreakout(session=s, buffer=b) for s in RANGE_WINDOWS for b in BUFFERS]
    out += [RangeFade(session=s) for s in RANGE_WINDOWS]
    sessions = tuple(TRADE_SESSIONS)
    out += [
        Donchian(tf=tf, lookback=lb, sl_atr=sl, session=s)
        for tf, lb, sl, s in itertools.product(TF_SETS["donchian"], (20, 40, 80), SL_ATR, sessions)
    ]
    out += [
        MeanReversionZ(tf=tf, lookback=lb, z=z, sl_atr=sl, session=s)
        for tf, lb, z, sl, s in itertools.product(
            TF_SETS["meanrev"], (20, 50), (2.0, 2.5), SL_ATR, sessions
        )
    ]
    out += [
        TrendEma(tf=tf, fast=f, slow=sw, sl_atr=sl, session=s)
        for tf, (f, sw), sl, s in itertools.product(
            TF_SETS["trend"], ((9, 21), (20, 50)), SL_ATR, sessions
        )
    ]
    out += [
        SessionDrive(minutes=m, mode=mode, sl_atr=sl, session=s)
        for m, mode, sl, s in itertools.product(
            (15, 30, 60), ("continue", "reverse"), SL_ATR, sessions
        )
    ]
    return out


@dataclass(frozen=True)
class SearchConfig:
    """Семейство с параметрами × выход × фильтр."""

    family: Family
    exit: ExitCfg
    filt: str = "none"

    def __post_init__(self) -> None:
        if self.filt not in FILTERS:
            raise ValueError(f"фильтр из {FILTERS}")

    @property
    def key(self) -> str:
        return f"{self.family.key}|{self.exit.name}|{self.filt}"

    @property
    def signal_key(self) -> str:
        """Сигналы зависят только от семейства: выходы и фильтры строятся поверх."""
        return self.family.key

    @property
    def exit_specs(self) -> list[ExitSpec]:
        return list(self.exit.specs)

    @property
    def filter_specs(self) -> list[FilterSpec]:
        return [] if self.filt == "none" else [FilterSpec(kind=self.filt, params={})]


def enumerate_configs(
    families: Sequence[Family] | None = None,
    *,
    exits: Sequence[str] | None = None,
    filters: Sequence[str] = FILTERS,
) -> list[SearchConfig]:
    """Семейства × выходы × фильтры; выходы закрываются ко времени закрытия сессии семейства."""
    fams = list(families) if families is not None else family_grid()
    out: list[SearchConfig] = []
    for fam in fams:
        grid = exit_grid(fam.close_at)
        for name in exits if exits is not None else grid:
            for f in filters:
                out.append(SearchConfig(fam, grid[name], f))
    return out


def sample_configs(
    configs: Sequence[SearchConfig], max_scans: int, scans_per_config: int, seed: int
) -> list[SearchConfig]:
    """Укладывает перебор в бюджет сканов: `len × scans_per_config ≤ max_scans`.

    Если сетка не помещается, берётся равномерная случайная подвыборка (seed фиксирован,
    порядок сохраняется); разные seed в разные ночи покрывают разные части сетки.
    """
    cap = max(max_scans // max(scans_per_config, 1), 0)
    if len(configs) <= cap:
        return list(configs)
    pick = np.sort(np.random.default_rng(seed).choice(len(configs), size=cap, replace=False))
    return [configs[i] for i in pick]


def with_tp(signals: Signals, tp_r: float | None) -> Signals:
    """Те же сигналы с тейком `tp_r` R от фактического риска (None — без тейка)."""
    n = len(signals)
    tp = np.full(n, np.nan if tp_r is None else tp_r)
    return replace(signals, tp_r=tp)
