"""Загрузка и проверка YAML-конфигов (configs/, research/registry/)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from pbl.schemas import Firm, IdeaCard, InstrumentSpec, PropTypeTemplate

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
REGISTRY = REPO_ROOT / "research" / "registry"


def read_yaml(path: Path) -> Any:
    """Читает YAML; ключи с префиксом `_` — служебные (якоря), в схему не попадают."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = {k: v for k, v in data.items() if not str(k).startswith("_")}
    return data


def load_model[M: BaseModel](path: Path, model: type[M]) -> M:
    return model.model_validate(read_yaml(path))


def iter_yaml(directory: Path) -> Iterator[Path]:
    yield from sorted(p for p in directory.rglob("*.yaml") if p.is_file())


def load_firms(directory: Path | None = None) -> list[Firm]:
    return [load_model(p, Firm) for p in iter_yaml(directory or CONFIGS / "firms")]


def load_prop_types(directory: Path | None = None) -> list[PropTypeTemplate]:
    return [load_model(p, PropTypeTemplate) for p in iter_yaml(directory or CONFIGS / "prop_types")]


def load_instruments(directory: Path | None = None) -> list[InstrumentSpec]:
    return [load_model(p, InstrumentSpec) for p in iter_yaml(directory or CONFIGS / "instruments")]


def load_ideas(directory: Path | None = None) -> list[IdeaCard]:
    return [load_model(p, IdeaCard) for p in iter_yaml(directory or REGISTRY)]


def validate_all(root: Path | None = None) -> list[str]:
    """Проверяет все конфиги; возвращает список ошибок (пусто — всё валидно)."""
    base = root or REPO_ROOT
    jobs: list[tuple[Path, type[BaseModel]]] = [
        (base / "configs" / "firms", Firm),
        (base / "configs" / "prop_types", PropTypeTemplate),
        (base / "configs" / "instruments", InstrumentSpec),
        (base / "research" / "registry", IdeaCard),
    ]
    errors: list[str] = []
    for directory, model in jobs:
        if not directory.is_dir():
            continue
        for path in iter_yaml(directory):
            try:
                load_model(path, model)
            except (ValidationError, yaml.YAMLError) as exc:
                errors.append(f"{path.relative_to(base)}: {exc}")
    return errors
