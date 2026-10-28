"""Описание стратегии: данные для реестра и движка (SPEC §5, VALIDATION §6)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pbl.schemas.common import StrictModel

ParamValue = float | int | str | bool

Timeframe = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

ExitKind = Literal["rr", "trailing", "partial", "breakeven", "time", "volatility", "session_close"]


class ExitSpec(StrictModel):
    kind: ExitKind
    params: dict[str, ParamValue] = Field(default_factory=dict)


class FilterSpec(StrictModel):
    kind: str
    params: dict[str, ParamValue | dict[str, str]] = Field(default_factory=dict)


class SessionWindow(StrictModel):
    """Окно сессии. Пишем в ET, МСК и времени сервера фирмы (инструкции §2)."""

    name: str
    et: str
    msk: str
    server_fn: str | None = Field(default=None, description="время сервера FundedNext")


class Strategy(StrictModel):
    id: str
    family: str
    params: dict[str, ParamValue] = Field(default_factory=dict)
    instruments: list[str] = Field(min_length=1)
    timeframe: Timeframe
    sessions: list[SessionWindow] = Field(default_factory=list)
    exits: list[ExitSpec] = Field(min_length=1)
    filters: list[FilterSpec] = Field(default_factory=list)
    firm_profile: str = Field(description="id фирмы, под чьи правила сделок собран вариант")
    idea_id: str | None = None
