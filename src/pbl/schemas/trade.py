"""Сделка и дневная запись — вход проп-симулятора (MONTE_CARLO §2.1–2.2)."""

from __future__ import annotations

import datetime as dt
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from pbl.schemas.common import StrictModel

ExitReason = Literal["sl", "tp", "trail", "be", "partial", "time", "guard", "friday_close", "eod"]
Sample = Literal["IS", "WF", "HOLDOUT", "DEMO"]


def _require_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() != dt.UTC.utcoffset(None):
        raise ValueError("время должно быть в UTC (tz-aware)")
    return value


class Trade(StrictModel):
    trade_id: str
    strategy_id: str
    variant_id: str
    symbol: str
    side: Literal["long", "short"]
    open_utc: dt.datetime
    close_utc: dt.datetime
    entry: float
    exit: float
    sl: float = Field(description="стоп обязателен (SPEC B3)")
    tp: float | None = None
    stop_dist: float = Field(gt=0, description="|entry − sl| в пунктах")
    point_value: float = Field(gt=0, description="$ за пункт на 1 лот на момент входа")
    r_gross: float
    cost_spread_r: float = Field(ge=0)
    cost_comm_r: float = Field(ge=0)
    cost_swap_r: float = Field(description="своп может быть и положительным")
    cost_slip_r: float = Field(ge=0)
    r_net: float
    mae_r: float = Field(le=0, description="худший плавающий результат, R")
    mfe_r: float = Field(ge=0, description="лучший плавающий результат, R")
    hold_sec: int = Field(ge=0)
    exit_reason: ExitReason
    news_flag: bool = False
    mods: int = Field(default=0, ge=0)
    sample: Sample

    _utc = field_validator("open_utc", "close_utc")(_require_utc)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.close_utc < self.open_utc:
            raise ValueError("close_utc раньше open_utc")
        expected = self.r_gross - (
            self.cost_spread_r + self.cost_comm_r + self.cost_swap_r + self.cost_slip_r
        )
        if abs(expected - self.r_net) > 1e-9:
            raise ValueError("r_net должен равняться r_gross − Σ cost_*")
        return self


class DayRecord(StrictModel):
    """Дневная запись под профиль фирмы. Всё в R при номинальном риске 1R на сделку."""

    firm: str
    date: dt.date
    n_opened: int = Field(ge=0)
    c_in: float
    c_out: float
    db: float
    b_max: float
    path: list[float] = Field(min_length=2, description="зигзаг пути эквити относительно E_SOD")
    risk_open_max: float = Field(ge=0)
    news_pos_r: float = 0.0
    cost_spread_r: float = 0.0
    cost_comm_r: float = 0.0
    cost_swap_r: float = 0.0
    cost_slip_r: float = 0.0
    trades: list[str] = Field(default_factory=list)

    @property
    def weekday(self) -> int:
        return self.date.weekday()

    @property
    def year(self) -> int:
        return self.date.year
