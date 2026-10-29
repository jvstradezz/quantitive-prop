"""`pbl idea ...`: реестр идей (docs/IDEAS.md)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated, cast, get_args

import typer

from pbl.config import REGISTRY
from pbl.research import registry
from pbl.research.variants import VariantLog, refresh_card_counter
from pbl.schemas.idea import IdeaStatus

idea_app = typer.Typer(
    help="Реестр идей: карточки, статусы, счётчик вариантов", no_args_is_help=True
)

RegistryOpt = Annotated[Path, typer.Option(help="папка реестра")]


@idea_app.command("list")
def list_ideas(directory: RegistryOpt = REGISTRY) -> None:
    """Карточки: id, статус, число проб, заголовок."""
    for path in registry.card_paths(directory):
        card = registry.load_card(path)
        typer.echo(f"{card.id}  {card.status:<13}  проб {card.variants_tested:>5}  {card.title}")


@idea_app.command("check")
def check(directory: RegistryOpt = REGISTRY) -> None:
    """Согласованность карточек и счётчика вариантов (код возврата 1 при проблемах)."""
    problems = registry.check_registry(directory)
    for p in problems:
        typer.echo(p, err=True)
    if problems:
        raise typer.Exit(code=1)
    typer.echo("реестр согласован")


@idea_app.command("advance")
def advance(
    idea_id: Annotated[str, typer.Argument(help="IDEA-0001")],
    status: Annotated[str, typer.Argument(help="новый статус")],
    note: Annotated[str, typer.Option(help="что произошло, попадёт в history")],
    verdict: Annotated[
        str | None, typer.Option(help="в демо / доработать: … / отклонена: …")
    ] = None,
    on: Annotated[
        str | None, typer.Option(help="дата YYYY-MM-DD; по умолчанию сегодня UTC")
    ] = None,
    directory: RegistryOpt = REGISTRY,
) -> None:
    """Перевести идею в следующий статус (правила — IDEAS §2)."""
    if status not in get_args(IdeaStatus):
        raise typer.BadParameter(f"статус из {', '.join(get_args(IdeaStatus))}")
    path, card = registry.find_card(idea_id, directory)
    day = dt.date.fromisoformat(on) if on else dt.datetime.now(dt.UTC).date()
    try:
        new = registry.advance(card, cast(IdeaStatus, status), note=note, on=day, verdict=verdict)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    registry.save_card(new, path)
    typer.echo(f"{idea_id}: {card.status} → {new.status}")


@idea_app.command("sync-counter")
def sync_counter(idea_id: str, directory: RegistryOpt = REGISTRY) -> None:
    """Записать в карточку число проб из research/registry/variants/<id>.parquet."""
    path, card = registry.find_card(idea_id, directory)
    new = refresh_card_counter(card, VariantLog.for_idea(idea_id, directory))
    registry.save_card(new, path)
    typer.echo(f"{idea_id}: variants_tested = {new.variants_tested}")
