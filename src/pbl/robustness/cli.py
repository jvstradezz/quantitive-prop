"""`pbl gates ...`: конвейер гейтов S1–S3 и автоотказ (docs/VALIDATION.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

gates_app = typer.Typer(help="Гейты S1–S3, автоотказ и аудит (VALIDATION)", no_args_is_help=True)


@gates_app.command("synth")
def synth(
    case: Annotated[str, typer.Argument(help="случай синтетики: edge, noise, decay, …")],
    out: Annotated[Path | None, typer.Option(help="папка отчёта")] = None,
    seed: Annotated[int, typer.Option(help="seed синтетики")] = 1,
    paths: Annotated[int, typer.Option(help="путей бутстрэпа и MC")] = 10_000,
    s3: Annotated[bool, typer.Option(help="считать проп-MC (S3) по FundedNext")] = True,
) -> None:
    """Прогнать гейты на синтетике с известным ответом и сравнить вердикт с ожидаемым."""
    from pbl.config import load_firms
    from pbl.prop.engine import MCConfig
    from pbl.robustness.gates import GateConfig, run_gates
    from pbl.robustness.report import write_gates_report
    from pbl.robustness.synthetic import CASES, synthetic_case

    if case not in CASES:
        raise typer.BadParameter(f"нет случая {case!r}; есть: {', '.join(CASES)}")
    study, space = synthetic_case(case, seed=seed)
    firms = tuple(f for f in load_firms() if f.id == "fundednext") if s3 else ()
    cfg = GateConfig(n_paths=paths, seed=seed, firms=firms, mc=MCConfig(n_paths=paths, seed=seed))
    rep = run_gates(study, space, cfg)
    what, expected = CASES[case]
    typer.echo(f"{case}: {what}; ожидаем — {expected}")
    typer.echo(f"вердикт: {rep.verdict}")
    if out is not None:
        typer.echo(f"отчёт: {write_gates_report(rep, out)}")


@gates_app.command("check-trades")
def check_trades(
    trades_csv: Annotated[Path, typer.Argument(help="CSV сделок (колонки схемы Trade)")],
    firm: Annotated[str, typer.Option(help="фирма, чей сервер задаёт торговый день")] = (
        "fundednext"
    ),
    family: Annotated[str | None, typer.Option(help="семейство стратегии")] = None,
) -> None:
    """Автоотказ и флаг «средний убыток > 3 × среднего профита» по готовому списку сделок."""
    from pbl.config import load_firms
    from pbl.prop.io import read_trades_csv
    from pbl.robustness.audit import loss_to_win
    from pbl.robustness.autoreject import AutoRejectInput, autoreject

    trades = read_trades_csv(trades_csv)
    server = next((f.server_time for f in load_firms() if f.id == firm), None)
    if server is None:
        raise typer.BadParameter(f"нет фирмы {firm!r} в configs/firms")
    rej = autoreject(AutoRejectInput(trades, server, family=family))
    for f in rej.findings:
        mark = "—" if f.triggered is None else ("ОТКАЗ" if f.triggered else "ok")
        typer.echo(f"{mark:>5}  {f.name}: {f.detail}")
    a = loss_to_win([t.r_net for t in trades])
    typer.echo(f"{'АУДИТ' if a.flagged else 'ok':>5}  {a.name}: {a.detail}")
    if rej.rejected:
        typer.echo("автоотказ: " + "; ".join(rej.reasons), err=True)
        raise typer.Exit(code=1)
