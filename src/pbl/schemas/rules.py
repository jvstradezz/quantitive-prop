"""Правила счёта: фазы, лимиты, выплаты, правила сделок.

Модель следует docs/MONTE_CARLO.md §2.3 и §3. Неизвестное — `None` плюс ссылка в
`open_items` на вопрос из docs/DECISIONS.md; значения не выдумываем.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from pbl.schemas.common import HHMM, AssetClass, OpenItemId, StrictModel

# --- фазы ---------------------------------------------------------------------------------


class Phase(StrictModel):
    name: str
    target_pct: float = Field(gt=0, description="цель фазы, % от B0")
    min_days: int = Field(ge=0, description="минимум торговых дней")
    max_days: int | None = Field(default=None, ge=1, description="срок фазы; None — без срока")
    leverage: dict[AssetClass, float] | None = Field(
        default=None, description="плечо по классам актива (1:N → N)"
    )


# --- лимиты -------------------------------------------------------------------------------


class DailyLoss(StrictModel):
    """Дневной лимит (MONTE_CARLO §3.3). Нет лимита — поле `daily_loss` = None у правил."""

    pct: float = Field(gt=0)
    alt_pct: float | None = Field(
        default=None, gt=0, description="альтернативный лимит, выбираемый при покупке (FP 3%)"
    )
    reference: Literal["sod_balance", "max_sod_balance_equity"] = Field(
        description="от чего отсчитывается уровень: баланс на начало дня или max(баланс, эквити)"
    )
    amount_basis: Literal["initial", "reference", "min_initial_reference"] = Field(
        description="от чего берётся сумма лимита; min_initial_reference — меньшее из двух"
    )
    alt_min_days: int | None = Field(
        default=None, ge=0, description="мин. торговых дней фазы, если выбран alt_pct (FP: 0)"
    )
    check: Literal["realtime_equity", "eod"] = "realtime_equity"
    includes_floating: bool = True
    includes_swaps_commissions: bool = True
    reset_time_server: HHMM = "00:00"


class MaxLoss(StrictModel):
    """Общий лимит (MONTE_CARLO §3.4). Ровно одно из `pct` и `abs_usd`."""

    type: Literal["static", "trailing"]
    pct: float | None = Field(default=None, gt=0, description="% от B0")
    abs_usd: float | None = Field(default=None, gt=0, description="сумма в $ (futures-стиль)")
    alt_pct: float | None = Field(default=None, gt=0, description="лимит с add-on (Lite: до 10%)")
    trail_source: Literal["balance_eod", "balance_realtime", "equity_realtime"] | None = None
    lock_level: Literal["initial_balance"] | None = Field(
        default=None, description="выше этого уровня трейлинг не поднимается"
    )
    lock_trigger_profit_pct: float | None = Field(
        default=None, gt=0, description="прибыль, после которой уровень фиксируется (FP Zero: 5)"
    )
    check: Literal["realtime_equity", "eod_balance"] = "realtime_equity"
    payout_floor_behavior: Literal["never_lowers", "resets"] | None = Field(
        default=None, description="что с уровнем после выплаты (MONTE_CARLO §3.4)"
    )
    open_items: list[OpenItemId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.pct is None) == (self.abs_usd is None):
            raise ValueError("нужно ровно одно из pct и abs_usd")
        if self.type == "trailing" and self.trail_source is None:
            raise ValueError("для trailing нужен trail_source")
        if self.type == "static" and (
            self.trail_source is not None
            or self.lock_level is not None
            or self.lock_trigger_profit_pct is not None
        ):
            raise ValueError("trail_source, lock_level и lock_trigger задаются только для trailing")
        return self


# --- выплаты ------------------------------------------------------------------------------


class SplitUpgrade(StrictModel):
    pct: float = Field(gt=0, le=100)
    condition: str


class PayoutOption(StrictModel):
    """Один вариант выплат. В MC каждый вариант — отдельная строка таблицы программ."""

    id: str
    label: str
    split_pct: float = Field(gt=0, le=100, description="доля трейдера, %")
    split_upgrade: SplitUpgrade | None = None
    first_after_days: int | None = Field(default=None, ge=0, description="календарных дней")
    cycle_days: int | None = Field(default=None, ge=1, description="календарных дней")
    cycle_business_days: int | None = Field(default=None, ge=1)
    min_profit_pct: float | None = Field(
        default=None, ge=0, description="мин. рост для выплаты, % от B0"
    )
    consistency_best_day_max_pct: float | None = Field(
        default=None, gt=0, description="лучший день ≤ x% прибыли (условие выплаты)"
    )
    profitable_days: tuple[int, float] | None = Field(
        default=None, description="(N прибыльных дней, мин. % B0 в каждом)"
    )
    min_days_override: int | None = Field(
        default=None,
        ge=0,
        description="мин. торговых дней на фазу при этом варианте, если отличается от Phase",
    )
    notes: str | None = None
    open_items: list[OpenItemId] = Field(default_factory=list)


# --- правила сделок и funded --------------------------------------------------------------


class NewsRule(StrictModel):
    window_min: int = Field(ge=1, description="± минут от high-impact новости")
    challenge: Literal["allowed", "prohibited"]
    funded: Literal["profit_factor", "deduct_profit", "prohibited"]
    funded_profit_counted_pct: float | None = Field(
        default=None, ge=0, le=100, description="для profit_factor: доля засчитываемой прибыли"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.funded == "profit_factor") != (self.funded_profit_counted_pct is not None):
            raise ValueError("funded_profit_counted_pct задаётся ровно при funded=profit_factor")
        return self


class QuickStrike(StrictModel):
    """Сделки короче `hold_sec` и доля их прибыли (FN)."""

    hold_sec: int = Field(ge=1)
    warn_profit_share_pct: float = Field(gt=0, le=100)
    breach_profit_share_pct: float = Field(gt=0, le=100)


class TradeRules(StrictModel):
    """Правила, меняющие сами сделки (ARCHITECTURE §6): применяются в движке."""

    quick_strike: QuickStrike | None = None
    max_trades_day: int | None = Field(default=None, ge=1)
    max_messages_day: int | None = Field(default=None, ge=1)
    account_disabled_messages: int | None = Field(default=None, ge=1)
    settlement_window_server: tuple[HHMM, HHMM] | None = Field(
        default=None, description="окно по серверу, в котором не торгуем"
    )
    news: NewsRule | None = None
    weekend_holding_challenge: bool | None = None
    weekend_holding_funded: bool | None = Field(
        default=None, description="False — позиции закрываются в пятницу"
    )
    max_lot_per_order: float | None = Field(default=None, gt=0)
    commission_notes: str | None = None
    open_items: list[OpenItemId] = Field(default_factory=list)


class RiskRule(StrictModel):
    """Risk Rule FN: суммарный риск открытых позиций, % от B0 (funded и Instant)."""

    max_open_risk_pct: float = Field(gt=0)
    sl_required: bool = True
    second_violation_limit_pct: float | None = Field(default=None, gt=0)


class Striking(StrictModel):
    """Striking System FP на funded."""

    single_idea_floating_loss_pct: float = Field(gt=0)
    applies_from_size_exclusive: int | None = Field(
        default=None, description="счета больше этого размера ($); None — все"
    )
    strikes_to_close: int = Field(ge=1)


class ProfitConcentration(StrictModel):
    single_idea_pct_of_target: float = Field(gt=0)
    profitable_days_required: int = Field(ge=1)
    min_day_pct: float = Field(gt=0)
    applies_from_size: int | None = None


class FundedRules(StrictModel):
    leverage: dict[AssetClass, float] | None = None
    risk_rule: RiskRule | None = None
    striking: Striking | None = None
    profit_concentration: ProfitConcentration | None = None
    open_items: list[OpenItemId] = Field(default_factory=list)


class Commission(StrictModel):
    """Комиссия: $/лот и/или % от оборота (металлы у FN)."""

    fx_per_lot_usd: float | None = Field(default=None, ge=0)
    energy_per_lot_usd: float | None = Field(default=None, ge=0)
    metals_per_lot_usd: float | None = Field(default=None, ge=0)
    metals_pct: float | None = Field(
        default=None, ge=0, description="% как указан в источнике (0.0016 = 0.0016%)"
    )
    indices_usd: float | None = Field(default=None, ge=0)
    note: str | None = None


# --- итоговый набор правил ----------------------------------------------------------------


class AccountRules(StrictModel):
    """Правила счёта без коммерции: общая основа шаблонов и программ фирм."""

    phases: list[Phase] = Field(
        description="фазы челленджа; пусто — instant. Funded описывается через `funded`."
    )
    daily_loss: DailyLoss | None = Field(description="None — дневного лимита нет")
    max_loss: MaxLoss
    inactivity_days: int | None = Field(default=None, ge=1)
    inactivity_open_items: list[OpenItemId] = Field(default_factory=list)
    trade_rules: TradeRules = Field(default_factory=TradeRules)
    funded: FundedRules = Field(default_factory=FundedRules)
    commission: Commission | None = None

    @model_validator(mode="after")
    def _check_rules(self) -> Self:
        names = [p.name for p in self.phases]
        if len(names) != len(set(names)):
            raise ValueError("имена фаз должны быть уникальны")
        return self
