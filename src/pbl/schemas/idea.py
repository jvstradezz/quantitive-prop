"""Карточка идеи (IDEAS §3) и запись истории."""

from __future__ import annotations

import datetime as dt
import re
from typing import Literal

from pydantic import Field, field_validator

from pbl.schemas.common import StrictModel
from pbl.schemas.strategy import SessionWindow, Timeframe

IdeaStatus = Literal["Новая", "Формализована", "Кодится", "Тест", "Результат", "Демо", "Отклонена"]

_IDEA_ID_RE = re.compile(r"^IDEA-\d{4}$")


class IdeaSource(StrictModel):
    channel: Literal["chat", "dashboard", "generator"]
    ref: str | None = None
    attachments: list[str] = Field(default_factory=list)


IdeaAssetClass = Literal[
    "fx_cfd", "metal_cfd", "index_cfd", "energy_cfd", "commodity_cfd", "crypto_cfd", "stock_cfd"
]
"""Словарь карточек, как в docs/IDEAS.md (`index_cfd`); не совпадает с `AssetClass` 1:1."""


class IdeaMarkets(StrictModel):
    asset_classes: list[IdeaAssetClass] = Field(min_length=1)
    instruments: list[str] = Field(min_length=1)


class HistoryEntry(StrictModel):
    date: dt.date
    status: IdeaStatus
    note: str


class IdeaCard(StrictModel):
    id: str
    title: str
    status: IdeaStatus
    created: dt.date
    author: str
    source: IdeaSource
    summary: str
    markets: IdeaMarkets
    timeframes: list[Timeframe] = Field(min_length=1)
    sessions: list[SessionWindow] = Field(default_factory=list)
    entry: str
    stop: str
    take: str
    exits: list[str] = Field(default_factory=list)
    filters: list[dict[str, dict[str, str]]] = Field(default_factory=list)
    firm_rules: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list, max_length=3)
    variants_tested: int = Field(default=0, ge=0)
    holdout_opened_at: dt.date | None = None
    results: list[str] = Field(default_factory=list)
    verdict: str | None = None
    history: list[HistoryEntry] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _IDEA_ID_RE.match(value):
            raise ValueError("id карточки вида IDEA-0001")
        return value
