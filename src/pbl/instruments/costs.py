"""Издержки инструмента для движка: спред по часам, проскальзывание, комиссия, своп, цена пункта.

Источник — `configs/instruments/<firm>/<symbol>.yaml` (F-INST-1) и комиссия программы
`Program.commission` (ставки по классам активов, DECISIONS O-19). Все цены — в единицах цены
инструмента, деньги — в $ на 1 лот.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from pbl.schemas import InstrumentSpec
from pbl.schemas.rules import Commission

F64 = npt.NDArray[np.float64]

SpreadMode = Literal["data", "typical", "worst", "conservative"]
"""`data` — спред из баров; `typical`/`worst` — из спеки по часу сервера; `conservative` —
больший из спреда бара и typical спеки (DECISIONS P-33)."""

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class CostModel:
    """Настройки издержек прогона; стрессы — те же поля с другими значениями."""

    spread: SpreadMode = "conservative"
    slip_spread_mult: float = 0.5  # проскальзывание market/stop-исполнений, доля typical (P-32)
    slip_points: float | None = None  # фиксированное проскальзывание в пунктах вместо доли
    commission_mult: float = 1.0  # O-19: чувствительность ×2 (вход и выход)
    swap_mult: float = 1.0

    def __post_init__(self) -> None:
        if self.slip_spread_mult < 0 or (self.slip_points is not None and self.slip_points < 0):
            raise ValueError("проскальзывание не может быть отрицательным")
        if self.commission_mult < 0 or self.swap_mult < 0:
            raise ValueError("множители издержек не могут быть отрицательными")


@dataclass(frozen=True)
class InstrumentCosts:
    symbol: str
    asset_class: str
    point: float
    contract_size: float
    profit_currency: str
    point_value_usd: float | None  # $ за пункт на 1 лот; None — зависит от курса
    point_value_ccy: float  # за пункт на 1 лот в валюте прибыли
    typical: F64  # 24 часа сервера, в единицах цены; часы без данных — максимум по спеке
    worst: F64
    comm_per_lot_usd: float  # за одну сторону
    comm_turnover: float  # доля оборота за одну сторону (0.0016% → 1.6e-5)
    comm_sides: int  # 1 — только вход, 2 — вход и выход
    swap_long: float  # за ночь на 1 лот: в пунктах (`swap_in_points`) или в $
    swap_short: float
    swap_in_points: bool
    triple_weekday: int  # 0 — понедельник

    def to_usd(self, price: float, profit_to_usd: float | None = None) -> float:
        """Курс валюты прибыли к $ на момент сделки (O-20, O-21)."""
        if self.profit_currency == "USD":
            return 1.0
        if self.symbol.startswith("USD") and self.symbol[3:6] == self.profit_currency:
            return 1.0 / price  # USDJPY: прибыль в JPY, курс — сама цена
        if profit_to_usd is None:
            raise ValueError(
                f"{self.symbol}: прибыль в {self.profit_currency}, нужен курс к $ (profit_to_usd)"
            )
        return profit_to_usd

    def pv_usd(self, price: float, profit_to_usd: float | None = None) -> float:
        if self.point_value_usd is not None:
            return self.point_value_usd
        return self.point_value_ccy * self.to_usd(price, profit_to_usd)

    def commission_usd(self, price: float, profit_to_usd: float | None = None) -> float:
        """Комиссия за круг на 1 лот, $ (обе стороны, если берётся и на выходе)."""
        per_side = self.comm_per_lot_usd
        if self.comm_turnover:
            notional = self.contract_size * price * self.to_usd(price, profit_to_usd)
            per_side += self.comm_turnover * notional
        return per_side * self.comm_sides

    def swap_per_night(self, side: int) -> float:
        """Своп за одну ночь на 1 лот (знак как в MT5: минус — платим)."""
        return self.swap_long if side > 0 else self.swap_short


def _program_commission(asset_class: str, c: Commission) -> tuple[float | None, float | None]:
    """(per_lot_usd, turnover_pct) по классу актива из ставок программы; None — нет ставки."""
    if asset_class == "fx":
        return c.fx_per_lot_usd, None
    if asset_class == "energy":
        return c.energy_per_lot_usd, None
    if asset_class in ("metals", "commodities"):
        return c.metals_per_lot_usd, c.metals_pct
    if asset_class == "indices":
        return c.indices_usd, None
    return None, None


def _by_hour(spec: InstrumentSpec, field: Literal["typical", "worst"]) -> F64:
    out = np.full(24, np.nan)
    for row in spec.spread_by_hour:
        out[row.hour] = getattr(row, field) * spec.point
    if np.isfinite(out).any():
        out[~np.isfinite(out)] = np.nanmax(out)  # час без замера — худший из известных
    return out


def instrument_costs(
    spec: InstrumentSpec, program_commission: Commission | None = None
) -> InstrumentCosts:
    """Издержки инструмента; комиссия — по ставке программы, иначе как в спеке MT5 (O-19)."""
    per_lot, pct = spec.commission.per_lot_usd, spec.commission.turnover_pct
    if program_commission is not None:
        p_lot, p_pct = _program_commission(spec.asset_class, program_commission)
        if p_lot is not None or p_pct is not None:
            per_lot, pct = p_lot, p_pct
    if spec.swap_unit == "pct_annual":
        raise ValueError(f"{spec.symbol}: своп в % годовых движок пока не считает")
    pv_ccy = spec.point_value_profit_ccy or spec.point_value_usd
    assert pv_ccy is not None  # гарантирует валидатор InstrumentSpec
    return InstrumentCosts(
        symbol=spec.symbol,
        asset_class=spec.asset_class,
        point=spec.point,
        contract_size=spec.contract_size,
        profit_currency=spec.profit_currency,
        point_value_usd=spec.point_value_usd,
        point_value_ccy=pv_ccy,
        typical=_by_hour(spec, "typical"),
        worst=_by_hour(spec, "worst"),
        comm_per_lot_usd=per_lot or 0.0,
        comm_turnover=(pct or 0.0) / 100.0,
        # O-19: неясно, берётся ли комиссия на выходе → как в спеке, по умолчанию только вход
        comm_sides=2 if spec.commission_on == "entry_and_exit" else 1,
        swap_long=spec.swap_long,
        swap_short=spec.swap_short,
        swap_in_points=spec.swap_unit == "points",
        triple_weekday=WEEKDAYS.index(spec.triple_swap_day),
    )


def swap_nights(day_open: int, day_close: int, triple_weekday: int) -> int:
    """Число начислений свопа между серверными днями (ordinal), с тройным днём.

    Своп начисляется при переходе через полночь сервера после дня Пн–Пт; в тройной день — ×3,
    после субботы и воскресенья — 0 (их покрывает тройной день).
    """
    n = 0
    for d in range(day_open, day_close):
        wd = (d - 1) % 7  # ordinal 1 = понедельник
        if wd < 5:
            n += 3 if wd == triple_weekday else 1
    return n
