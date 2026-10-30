"""Что конвейер гейтов знает о стратегии: пространство вариантов и способ получить их сделки.

`Study` — любая стратегия (идея из реестра, семейство генератора, синтетика): по параметрам она
отдаёт сделки за всю историю `period`. Конвейер сам режет историю на IS и holdout и решает, когда
holdout можно открыть, поэтому `Study` не знает о разбиениях.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import polars as pl

from pbl.robustness.trades import trades_frame
from pbl.schemas import ServerTime, Trade
from pbl.schemas.strategy import ParamValue

Params = Mapping[str, ParamValue]


def params_key(params: Params) -> str:
    """Канонический ключ варианта: JSON с сортировкой ключей."""
    return json.dumps(dict(params), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


class Study(Protocol):
    @property
    def strategy_id(self) -> str: ...

    @property
    def family(self) -> str | None: ...

    @property
    def server(self) -> ServerTime: ...

    @property
    def period(self) -> tuple[dt.date, dt.date]:
        """Первый и последний день данных (общие для всех инструментов, P-35)."""
        ...

    def trades(self, params: Params) -> list[Trade]:
        """Сделки варианта за весь `period` (событийный бэктест; `sample` любой)."""
        ...

    def frame(self, params: Params) -> pl.DataFrame:
        """Те же сделки таблицей `robustness.trades.TRADE_SCHEMA` (может быть быстрым сканом)."""
        ...


class TradesStudy:
    """Основа для `Study`, у которой таблица строится из сделок."""

    server: ServerTime

    def trades(self, params: Params) -> list[Trade]:
        raise NotImplementedError

    def frame(self, params: Params) -> pl.DataFrame:
        return trades_frame(self.trades(params), self.server)


TradeMask = Callable[[pl.DataFrame], pl.Series]


@dataclass(frozen=True)
class FilterDef:
    """Фильтр дня недели или времени, который проверяется на OOS (VALIDATION §4.5).

    Вариант с `params[param] != off` — «с фильтром»; его близнец с `off` — «без фильтра».
    `allowed` — маска разрешённых фильтром сделок по таблице близнеца (колонки wd, minute, side).
    """

    name: str
    param: str
    off: ParamValue
    allowed: TradeMask


@dataclass(frozen=True)
class Space:
    """Все пробы идеи (каждая — в счётчик для DSR) и параметры для чувствительности."""

    variants: Sequence[Params]
    numeric: tuple[str, ...] = ()
    filters: tuple[FilterDef, ...] = ()
    integer: frozenset[str] = field(default_factory=frozenset)

    def with_twins(self) -> list[dict[str, ParamValue]]:
        """Варианты плюс близнецы без фильтра (оба — в счётчик, §4.5 п. 3), без повторов."""
        out: dict[str, dict[str, ParamValue]] = {}
        for v in self.variants:
            out.setdefault(params_key(v), dict(v))
            for f in self.filters:
                if f.param in v and v[f.param] != f.off:
                    twin = {**v, f.param: f.off}
                    out.setdefault(params_key(twin), twin)
        return list(out.values())

    def active_filters(self, params: Params) -> list[FilterDef]:
        return [f for f in self.filters if f.param in params and params[f.param] != f.off]
