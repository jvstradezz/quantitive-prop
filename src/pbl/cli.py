"""CLI проекта: `pbl ...`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pbl.config import REPO_ROOT, validate_all
from pbl.data.cli import data_app
from pbl.research.cli import idea_app

app = typer.Typer(help="Quantitive prop: конвейер стратегий и пропов", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(idea_app, name="idea")


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


@app.command("site-data")
def site_data(
    out: Annotated[Path, typer.Option(help="папка для site.json")] = REPO_ROOT / "web/public/data",
    root: Annotated[Path, typer.Option(help="корень репозитория")] = REPO_ROOT,
    repo: Annotated[str | None, typer.Option(help="owner/name для ссылок на GitHub")] = None,
    sha: Annotated[str | None, typer.Option(help="коммит сборки")] = None,
) -> None:
    """Собрать JSON для сайта дашборда: вехи, идеи, отчёты проп-MC (только из git)."""
    import datetime as dt

    from pbl.api.site_data import build_site_data, write_site_data

    built_at = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()
    path = write_site_data(out, build_site_data(root, repo=repo, sha=sha, built_at=built_at))
    typer.echo(f"записано {path}")


prop_app = typer.Typer(
    help="Проп-симулятор Monte Carlo (docs/MONTE_CARLO.md)", no_args_is_help=True
)
app.add_typer(prop_app, name="prop")


@prop_app.command("synth")
def prop_synth(
    out: Annotated[Path, typer.Option(help="куда записать CSV сделок")],
    days: Annotated[int, typer.Option(help="дней Пн–Пт")] = 750,
    trade_prob: Annotated[float, typer.Option(help="доля дней со сделкой")] = 0.8,
    win_rate: Annotated[float, typer.Option(help="винрейт")] = 0.45,
    rr: Annotated[float, typer.Option(help="тейк в R")] = 1.5,
    cost_r: Annotated[float, typer.Option(help="издержки на сделку, R")] = 0.06,
    seed: Annotated[int, typer.Option()] = 7,
    strategy_id: Annotated[str, typer.Option()] = "synthetic",
) -> None:
    """Синтетические сделки с известным перевесом (проверка симулятора, не рынок)."""
    from pbl.prop.io import write_trades_csv
    from pbl.prop.synthetic import synthetic_trades

    trades = synthetic_trades(
        n_days=days,
        trade_prob=trade_prob,
        win_rate=win_rate,
        rr=rr,
        cost_r=cost_r,
        seed=seed,
        strategy_id=strategy_id,
    )
    write_trades_csv(trades, out)
    typer.echo(f"{len(trades)} сделок → {out}")


@prop_app.command("run")
def prop_run(
    trades_csv: Annotated[Path, typer.Argument(help="CSV сделок (колонки схемы Trade)")],
    out: Annotated[Path, typer.Option(help="папка отчёта, например reports/<run_id>")],
    firm: Annotated[
        list[str] | None, typer.Option(help="id фирмы; по умолчанию все из configs/firms")
    ] = None,
    program: Annotated[list[str] | None, typer.Option(help="только эти программы")] = None,
    templates: Annotated[bool, typer.Option(help="считать шаблоны configs/prop_types")] = True,
    paths: Annotated[int, typer.Option(help="путей MC (≥ 10 000 для решений)")] = 10_000,
    opt_paths: Annotated[int, typer.Option(help="путей на шаге оптимизатора")] = 2_000,
    outer_boot: Annotated[int, typer.Option(help="псевдо-историй внешнего бутстрэпа")] = 200,
    seed: Annotated[int, typer.Option()] = 42,
    block: Annotated[float | None, typer.Option(help="длина блока; по умолчанию авто")] = None,
    top: Annotated[int, typer.Option(help="сколько лучших программ стрессовать")] = 3,
    detail: Annotated[
        list[str] | None,
        typer.Option(help="ключ строки для стрессов, например fn-stellar-2step/25k/standard"),
    ] = None,
) -> None:
    """MC по всем программам фирм и шаблонам: таблица, стрессы, рекомендация → out/."""
    from pbl.config import load_firms, load_prop_types
    from pbl.prop.engine import MCConfig
    from pbl.prop.io import read_trades_csv
    from pbl.prop.report import PLAN_KEY, run_report, write_report

    trades = read_trades_csv(trades_csv)
    firms = load_firms()
    if firm:
        firms = [f for f in firms if f.id in firm]
        firms.sort(key=lambda f: firm.index(f.id))
    if not firms:
        raise typer.BadParameter("нет таких фирм в configs/firms")
    cfg = MCConfig(
        n_paths=paths, opt_paths=opt_paths, seed=seed, block=block, outer_boot=outer_boot
    )
    rep = run_report(
        trades,
        firms,
        load_prop_types() if templates else [],
        cfg,
        top_k=top,
        program_ids=program,
        detail_keys=detail or [PLAN_KEY],
    )
    path = write_report(rep, out)
    typer.echo(f"отчёт: {path}")
    if rep.recommended is not None:
        st = rep.recommended.stats
        typer.echo(
            f"лучше всего: {rep.recommended.rules.label} — P(1-я выплата) {100 * st.p1:.1f}%, "
            f"EV₁ {st.ev1:+.0f} $"
        )
    else:
        typer.echo("нет программы с EV₁ > 0")


if __name__ == "__main__":
    app()
