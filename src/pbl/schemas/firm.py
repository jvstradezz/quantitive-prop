"""Фирмы, программы и шаблоны типов пропов (MONTE_CARLO §2.3, §4)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from typing import Literal, Self

from pydantic import Field, model_validator

from pbl.schemas.common import OpenItemId, StrictModel
from pbl.schemas.rules import AccountRules, PayoutOption

Platform = Literal["MT4", "MT5", "cTrader", "MatchTrader"]


class ServerTime(StrictModel):
    """Время сервера брокера: определяет границу торгового дня (ARCHITECTURE §5)."""

    winter_utc_offset: int = Field(ge=-12, le=14)
    summer_utc_offset: int = Field(ge=-12, le=14)
    dst_calendar: Literal["us", "eu"] | None = Field(
        default=None, description="по какому календарю переход; None — перехода нет (фикс.)"
    )
    day_reset: str = "00:00"
    verified: bool = False
    open_items: list[OpenItemId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        fixed = self.winter_utc_offset == self.summer_utc_offset
        if fixed != (self.dst_calendar is None):
            raise ValueError("dst_calendar задаётся, только если зимнее и летнее смещения различны")
        return self


class EaPolicy(StrictModel):
    allowed: Literal["yes", "own_ea_with_proof", "no"]
    platforms: list[Platform] = Field(default_factory=list)
    max_account_size_exclusive: int | None = Field(
        default=None, description="EA только на счетах меньше этого размера, $"
    )
    third_party_ea: Literal["allowed", "trade_manager_only", "prohibited"]
    vps: Literal["private_dedicated_ip", "prohibited"]
    unique_strategy_per_account: bool = False
    forbidden_in_ea: list[str] = Field(default_factory=list)
    notes: str | None = None
    open_items: list[OpenItemId] = Field(default_factory=list)


class SizeOption(StrictModel):
    size_usd: int = Field(gt=0)
    fee_usd: float = Field(gt=0)
    ea_allowed: bool
    open_items: list[OpenItemId] = Field(default_factory=list)


class Addon(StrictModel):
    id: str
    label: str
    fee_usd: float | None = Field(default=None, ge=0, description="None — цена неизвестна")
    refundable: bool = False
    applies_to_sizes: list[int] | None = Field(default=None, description="None — все размеры")
    open_items: list[OpenItemId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.fee_usd is None and not self.open_items:
            raise ValueError("неизвестная цена add-on требует ссылки на открытый вопрос")
        return self


class PropTypeTemplate(AccountRules):
    """Шаблон типа пропа (`configs/prop_types/`): правила без фирмы, цены и размеров."""

    id: str
    name: str
    style: Literal["cfd", "futures"]
    reference_balance_usd: int | None = Field(
        default=None, description="размер счёта, к которому привязаны суммы в $ (futures-стиль)"
    )
    values_source: Literal["firm_program", "idea_report", "placeholder"] = Field(
        description="откуда числа: реальная программа фирмы, отчёт идеи или условные значения"
    )
    values_source_ref: str | None = None
    notes: str | None = None


class Program(AccountRules):
    """Реальная программа фирмы (`configs/firms/`). Цены и правила — из PROP_RULES."""

    id: str
    name: str
    template: str | None = Field(
        default=None, description="ближайший шаблон из configs/prop_types/"
    )
    sizes: list[SizeOption]
    addons: list[Addon] = Field(default_factory=list)
    ea: EaPolicy
    payout_options: list[PayoutOption] = Field(default_factory=list)
    fee_refund_on_payout: int | None = Field(
        default=None, ge=1, description="взнос возвращается с выплатой №k; None — не возвращается"
    )
    data_source: Literal["official", "third_party_review"]
    verified: bool = Field(
        description="сверено с официальными правилами; False — нужна ссылка на открытый вопрос"
    )
    notes: str | None = None
    open_items: list[OpenItemId] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_program(self) -> Self:
        sizes = [s.size_usd for s in self.sizes]
        if len(sizes) != len(set(sizes)):
            raise ValueError("размеры счёта должны быть уникальны")
        ids = [o.id for o in self.payout_options]
        if len(ids) != len(set(ids)):
            raise ValueError("id вариантов выплат должны быть уникальны")
        if not self.payout_options and not self.open_items:
            raise ValueError("пустые payout_options требуют ссылки на открытый вопрос")
        if not self.verified and not self.open_items:
            raise ValueError("непроверенная программа должна ссылаться на открытый вопрос")
        if self.max_loss.pct is not None and self.max_loss.pct <= (
            self.daily_loss.pct if self.daily_loss else 0
        ):
            raise ValueError("общий лимит должен быть больше дневного")
        return self

    def iter_offers(self, *, ea_only: bool = True) -> Iterator[ProgramOffer]:
        """Строки таблицы программ: размер × вариант выплат (MONTE_CARLO §4.2)."""
        for size in self.sizes:
            if ea_only and not size.ea_allowed:
                continue
            for payout in self.payout_options:
                yield ProgramOffer(program_id=self.id, size=size, payout=payout)


class ProgramOffer(StrictModel):
    program_id: str
    size: SizeOption
    payout: PayoutOption


class Firm(StrictModel):
    id: str
    name: str
    role: Literal["primary", "backup"]
    asset_type: Literal["cfd"] = "cfd"
    server_time: ServerTime
    platforms: list[Platform]
    bot_platform: Platform | None = Field(
        default=None, description="платформа бота у этой фирмы (D-003)"
    )
    source_doc: str = Field(description="источник правил (D-005)")
    source_date: dt.date
    programs: list[Program]
    notes: str | None = None

    @model_validator(mode="after")
    def _check_firm(self) -> Self:
        ids = [p.id for p in self.programs]
        if len(ids) != len(set(ids)):
            raise ValueError("id программ должны быть уникальны")
        if self.bot_platform is not None and self.bot_platform not in self.platforms:
            raise ValueError("bot_platform должна входить в platforms")
        return self
