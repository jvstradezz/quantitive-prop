"""Реестр идей: карточки `research/registry/IDEA-NNNN-<slug>.yaml`, переходы статусов, проверки.

Статусы и правила переходов — docs/IDEAS.md §2. «Доработать» — не статус, а действие:
возвращает идею в `Кодится` и пишет в `history` прежний вердикт.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Any, get_args

import yaml

from pbl.config import REGISTRY, load_model
from pbl.schemas.idea import HistoryEntry, IdeaCard, IdeaStatus

STATUSES: tuple[IdeaStatus, ...] = get_args(IdeaStatus)

_PAST_FORMALIZED: frozenset[IdeaStatus] = frozenset(STATUSES) - {"Новая"}

ALLOWED: dict[IdeaStatus, frozenset[IdeaStatus]] = {
    "Новая": frozenset({"Формализована", "Отклонена"}),
    "Формализована": frozenset({"Кодится", "Отклонена"}),
    "Кодится": frozenset({"Тест", "Отклонена"}),
    "Тест": frozenset({"Результат", "Кодится", "Отклонена"}),
    "Результат": frozenset({"Демо", "Кодится", "Отклонена"}),
    "Демо": frozenset({"Кодится", "Отклонена"}),
    "Отклонена": frozenset({"Кодится"}),
}
"""Переход `Тест/Результат/Демо/Отклонена → Кодится` — это «доработать»."""

VERDICT_PREFIX: dict[IdeaStatus, str] = {"Демо": "в демо", "Отклонена": "отклонена"}
"""Вердикт одной строкой (IDEAS §4): `в демо` / `доработать: …` / `отклонена: …`."""

_FILE_RE = re.compile(r"^(IDEA-\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.yaml$")


def card_filename(idea_id: str, slug: str) -> str:
    name = f"{idea_id}-{slug}.yaml"
    if not _FILE_RE.match(name):
        raise ValueError(f"имя карточки вида IDEA-0001-<slug>.yaml, получено {name!r}")
    return name


_WRAP = 100


class _Dumper(yaml.SafeDumper):
    """Многострочные строки — блоком `|`, длинные — свёрнутым `>-`: читаемо и ровный diff."""


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else (">" if len(data) > _WRAP - 10 else None)
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_Dumper.add_representer(str, _str_representer)


def dump_card(card: IdeaCard) -> str:
    """Каноничный YAML карточки: порядок полей схемы, кириллица как есть, даты без кавычек."""
    data: dict[str, Any] = card.model_dump(mode="python")
    return yaml.dump(
        data,
        Dumper=_Dumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=_WRAP,
    )


def save_card(card: IdeaCard, path: Path) -> None:
    path.write_text(dump_card(card), encoding="utf-8")


def card_paths(directory: Path | None = None) -> list[Path]:
    base = directory or REGISTRY
    return sorted(p for p in base.glob("IDEA-*.yaml") if p.is_file())


def load_card(path: Path) -> IdeaCard:
    return load_model(path, IdeaCard)


def find_card(idea_id: str, directory: Path | None = None) -> tuple[Path, IdeaCard]:
    matches = [p for p in card_paths(directory) if p.name.startswith(f"{idea_id}-")]
    if len(matches) != 1:
        raise LookupError(f"карточка {idea_id}: найдено файлов {len(matches)}")
    return matches[0], load_card(matches[0])


def advance(
    card: IdeaCard,
    to: IdeaStatus,
    *,
    note: str,
    on: dt.date,
    verdict: str | None = None,
) -> IdeaCard:
    """Новая карточка со статусом `to` и записью в `history`; исходная не меняется."""
    if to not in ALLOWED[card.status]:
        raise ValueError(f"{card.id}: переход {card.status} → {to} не предусмотрен (IDEAS §2)")
    if not note.strip():
        raise ValueError("note обязателен: что произошло и почему")
    new_verdict = verdict if verdict is not None else card.verdict
    if to in _PAST_FORMALIZED and to != "Отклонена" and card.questions:
        raise ValueError(f"{card.id}: вопросы не закрыты ({len(card.questions)})")
    if to in {"Кодится"}:
        # «доработать»: прежний вердикт остаётся в истории, в карточке его больше нет
        if card.verdict is not None:
            note = f"{note} (прежний вердикт: {card.verdict})"
        new_verdict = None
    if to == "Результат" and not (new_verdict or "").strip():
        raise ValueError(f"{card.id}: для статуса Результат нужен вердикт")
    prefix = VERDICT_PREFIX.get(to)
    if prefix is not None and not (new_verdict or "").lower().startswith(prefix):
        raise ValueError(f"{card.id}: вердикт для статуса {to} начинается с «{prefix}»")
    history = [*card.history, HistoryEntry(date=on, status=to, note=note)]
    return card.model_copy(update={"status": to, "verdict": new_verdict, "history": history})


def check_card(card: IdeaCard, path: Path | None = None) -> list[str]:
    """Согласованность карточки: статус, история, вердикт, вопросы, имя файла."""
    problems: list[str] = []
    if path is not None:
        m = _FILE_RE.match(path.name)
        if m is None or m.group(1) != card.id:
            problems.append(f"имя файла {path.name} не соответствует {card.id}-<slug>.yaml")
    if card.history[-1].status != card.status:
        problems.append(
            f"статус {card.status}, а последняя запись history — {card.history[-1].status}"
        )
    for prev, cur in zip(card.history, card.history[1:], strict=False):
        if cur.date < prev.date:
            problems.append(f"history: {cur.date} раньше {prev.date}")
        if cur.status not in ALLOWED[prev.status]:
            problems.append(f"history: переход {prev.status} → {cur.status} не предусмотрен")
    if card.status in _PAST_FORMALIZED and card.status != "Отклонена" and card.questions:
        problems.append(f"статус {card.status}, но есть открытые вопросы")
    if card.status in {"Результат", "Демо", "Отклонена"} and not (card.verdict or "").strip():
        problems.append(f"статус {card.status} без вердикта")
    prefix = VERDICT_PREFIX.get(card.status)
    if prefix is not None and card.verdict and not card.verdict.lower().startswith(prefix):
        problems.append(f"вердикт для статуса {card.status} начинается с «{prefix}»")
    if card.status in {"Новая", "Формализована", "Кодится"} and card.verdict is not None:
        problems.append(f"статус {card.status} при непустом вердикте")
    if card.holdout_opened_at is not None and card.status in {"Новая", "Формализована"}:
        problems.append("holdout открыт у карточки, не дошедшей до кода")
    return problems


def check_registry(directory: Path | None = None) -> list[str]:
    """Проблемы по всем карточкам; пусто — реестр согласован. Счётчик вариантов — отдельно."""
    from pbl.research.variants import VariantLog

    base = directory or REGISTRY
    problems: list[str] = []
    seen: dict[str, Path] = {}
    for path in card_paths(base):
        rel = path.name
        try:
            card = load_card(path)
        except Exception as exc:  # схема сама объяснит, что не так
            problems.append(f"{rel}: {exc}")
            continue
        if card.id in seen:
            problems.append(f"{rel}: id {card.id} уже в {seen[card.id].name}")
        seen[card.id] = path
        problems += [f"{rel}: {p}" for p in check_card(card, path)]
        counted = len(VariantLog.for_idea(card.id, base))
        if counted != card.variants_tested:
            problems.append(
                f"{rel}: variants_tested={card.variants_tested}, в счётчике вариантов {counted}"
            )
    return problems
