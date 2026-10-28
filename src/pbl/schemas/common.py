"""Общие типы схем."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict

# Проценты везде в «человеческих» единицах: 5 значит 5%, как в PROP_RULES.
# Доли в трейдах и MC (f, R) — отдельные поля с суффиксом в имени.


class StrictModel(BaseModel):
    """База: опечатка в YAML — ошибка, а не молчаливо проигнорированное поле."""

    model_config = ConfigDict(extra="forbid")


AssetClass = Literal["fx", "indices", "metals", "energy", "commodities", "crypto", "stocks"]
"""`commodities` — «сырьё» у FundedNext одной группой (металлы и нефть вместе)."""

OPEN_ITEM_RE = re.compile(r"^O-\d{2}$")


def _check_open_item(value: str) -> str:
    if not OPEN_ITEM_RE.match(value):
        raise ValueError(f"ожидается ID открытого вопроса вида O-01, получено {value!r}")
    return value


OpenItemId = Annotated[str, AfterValidator(_check_open_item)]
"""Ссылка на открытый вопрос в docs/DECISIONS.md (раздел «Открытые вопросы»)."""

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _check_hhmm(value: str) -> str:
    if not _HHMM_RE.match(value):
        raise ValueError(f"ожидается время HH:MM, получено {value!r}")
    return value


HHMM = Annotated[str, AfterValidator(_check_hhmm)]
