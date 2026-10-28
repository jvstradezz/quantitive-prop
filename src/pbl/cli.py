"""CLI проекта: `pbl ...`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pbl.config import REPO_ROOT, validate_all
from pbl.data.cli import data_app

app = typer.Typer(help="Quantitive prop: конвейер стратегий и пропов", no_args_is_help=True)

app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Корневая группа команд (нужна, чтобы подкоманды оставались именованными)."""


@app.command("validate-configs")
def validate_configs(
    root: Annotated[Path, typer.Option(help="корень репозитория")] = REPO_ROOT,
) -> None:
    """Проверить все YAML в configs/ и research/registry/ по pydantic-схемам."""
    errors = validate_all(root)
    if errors:
        for err in errors:
            typer.echo(err, err=True)
        raise typer.Exit(code=1)
    typer.echo("все конфиги валидны")


if __name__ == "__main__":
    app()
