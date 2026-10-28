"""Спецификация инструмента: `configs/instruments/<firm>/<symbol>.yaml` (SPEC F-INST-1)."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from pbl.schemas.common import HHMM, AssetClass, OpenItemId, StrictModel


class SpreadProfile(StrictModel):
    """Спред в пунктах; ключ `hour` — час по серверу брокера (0–23)."""

    hour: int = Field(ge=0, le=23)
    typical: float = Field(ge=0)
    worst: float = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.worst < self.typical:
            raise ValueError("worst спред не может быть меньше typical")
        return self


class CommissionSpec(StrictModel):
    per_lot_usd: float | None = Field(default=None, ge=0)
    turnover_pct: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.per_lot_usd is not None and self.turnover_pct is not None:
            raise ValueError("комиссия задаётся либо $/лот, либо % оборота")
        return self


class TradingSession(StrictModel):
    """Окно торгов по серверу; сессии могут переходить через полночь."""

    day: Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    start: HHMM
    end: HHMM


class InstrumentSpec(StrictModel):
    symbol: str
    firm: str
    asset_class: AssetClass
    contract_size: float = Field(gt=0)
    point: float = Field(gt=0, description="размер пункта в единицах цены")
    point_value_usd: float = Field(gt=0, description="стоимость пункта на 1 лот, $")
    min_lot: float = Field(gt=0)
    lot_step: float = Field(gt=0)
    max_lot_per_order: float | None = Field(default=None, gt=0)
    commission: CommissionSpec
    spread_by_hour: list[SpreadProfile] = Field(
        default_factory=list,
        description="пусто, пока спред плавающий и не измерен; тогда обязателен open_items",
    )
    swap_long: float
    swap_short: float
    swap_unit: Literal["points", "usd_per_lot", "pct_annual"]
    triple_swap_day: Literal["mon", "tue", "wed", "thu", "fri"]
    sessions: list[TradingSession] = Field(min_length=1)
    server_tz: str = Field(description="правило времени сервера, ссылка на Firm.server_time")
    dukascopy_symbol: str | None = Field(
        default=None, description="тикер источника истории (M1); None — ещё не сопоставлен"
    )
    source: Literal["mt5_specification", "conservative_estimate"]
    open_items: list[OpenItemId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        hours = [s.hour for s in self.spread_by_hour]
        if len(hours) != len(set(hours)):
            raise ValueError("часы в spread_by_hour должны быть уникальны")
        if not self.spread_by_hour and not self.open_items:
            raise ValueError("пустой spread_by_hour требует ссылки на открытый вопрос (O-04)")
        if self.source == "conservative_estimate" and not self.open_items:
            raise ValueError("консервативная оценка требует ссылки на открытый вопрос (O-04)")
        return self
