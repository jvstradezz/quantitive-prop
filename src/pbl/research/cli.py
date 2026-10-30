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


@idea_app.command("run")
def run(
    idea_id: Annotated[str, typer.Argument(help="IDEA-0001")],
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD включительно")],
    version: Annotated[str, typer.Option(help="версия кода; новая проба = новая версия")] = "v1",
    paths: Annotated[
        int, typer.Option(help="путей бутстрэпа и MC (≥ 10 000 для решений)")
    ] = 10_000,
    seed: Annotated[int, typer.Option()] = 42,
    top: Annotated[int, typer.Option(help="сколько лучших вариантов проверять")] = 5,
    wf_train: Annotated[int, typer.Option(help="месяцев обучения WF (P-01)")] = 24,
    wf_test: Annotated[int, typer.Option(help="месяцев теста WF")] = 6,
    wf_step: Annotated[int, typer.Option(help="шаг WF, месяцев")] = 6,
    is_fraction: Annotated[float, typer.Option(help="доля IS до holdout")] = 0.7,
    pbo_blocks: Annotated[int, typer.Option(help="блоков CSCV для PBO")] = 16,
    out: Annotated[
        Path | None, typer.Option(help="папка отчёта; по умолчанию reports/<run_id>")
    ] = None,
    bars_root: Annotated[Path | None, typer.Option(help="data/bars по умолчанию")] = None,
    update_card: Annotated[
        bool, typer.Option(help="записать пробы в реестр и перевести карточку по вердикту")
    ] = False,
    directory: RegistryOpt = REGISTRY,
) -> None:
    """S1–S3 для идеи №1 на барах из data/bars; holdout открывается один раз, для финалиста."""
    from pbl.config import REPO_ROOT
    from pbl.data.bars import load_bars
    from pbl.data.candles import BARS_ROOT
    from pbl.prop.engine import MCConfig
    from pbl.research.idea_report import card_after_run, chat_summary, write_idea_report
    from pbl.research.idea_run import RunParams, run_holdout, run_is, verdict
    from pbl.research.range_sweep import IDEA_ID, INSTRUMENTS, make_bundle

    if idea_id != IDEA_ID:
        raise typer.BadParameter(f"сетка и стратегия написаны только для {IDEA_ID}")
    day0, day1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    root = bars_root or BARS_ROOT
    try:
        bundles = {s: make_bundle(s, load_bars(s, day0, day1, root=root)) for s in INSTRUMENTS}
    except FileNotFoundError as exc:
        typer.echo(f"нет данных: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    path, card = registry.find_card(idea_id, directory)
    p = RunParams(
        day0,
        day1,
        version=version,
        on=dt.datetime.now(dt.UTC).date(),
        n_paths=paths,
        seed=seed,
        wf=(wf_train, wf_test, wf_step),
        is_fraction=is_fraction,
        top=top,
        pbo_blocks=pbo_blocks,
        registry=directory,
        mc=MCConfig(n_paths=paths, seed=seed),
    )
    seen = set(VariantLog.for_idea(idea_id, directory).frame()["version"].to_list())
    stage = run_is(bundles, p)
    ho = None
    if stage.finalist is not None:
        if card.holdout_opened_at is not None and version in seen:
            typer.echo(
                f"holdout уже открыт {card.holdout_opened_at} на версии {version}: "
                "поднимите --version (новая проба — новая запись в счётчике)",
                err=True,
            )
            raise typer.Exit(code=1)
        ho = run_holdout(bundles, stage, p)
    final = verdict(stage, ho)
    report = write_idea_report(stage, ho, p, out or REPO_ROOT / "reports" / p.run_id)
    if update_card:
        stage.log.save()
        registry.save_card(card_after_run(card, stage, ho, p, final), path)
    typer.echo(chat_summary(stage, ho, p, final))
    typer.echo(f"отчёт: {report}")


def search(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD включительно")],
    instruments: Annotated[
        str, typer.Option(help="символы через запятую (классы актива образуют корзины)")
    ] = "NAS100,US30,SPX500,GER30",
    campaign: Annotated[
        str, typer.Option(help="id кампании = имя файла счётчика проб")
    ] = "AUTO-0001",
    mode: Annotated[
        str, typer.Option(help="forward: выбор по первым месяцам IS, MC по остатку; is: всё на IS")
    ] = "forward",
    select_months: Annotated[int, typer.Option(help="месяцев окна выбора в режиме forward")] = 24,
    is_fraction: Annotated[float, typer.Option(help="доля IS до holdout")] = 0.7,
    paths: Annotated[int, typer.Option(help="путей MC (≥ 10 000 для решений)")] = 10_000,
    seed: Annotated[int, typer.Option()] = 42,
    max_scans: Annotated[int, typer.Option(help="бюджет векторных сканов за запуск")] = 50_000,
    event_runs: Annotated[int, typer.Option(help="бюджет событийных бэктестов")] = 200,
    mc_top: Annotated[int, typer.Option(help="сколько кандидатов считать в проп-MC")] = 20,
    max_hours: Annotated[float, typer.Option(help="бюджет времени, часов")] = 6.0,
    families: Annotated[
        str | None, typer.Option(help="только эти семейства через запятую; по умолчанию все")
    ] = None,
    programs: Annotated[str, typer.Option(help="программы FN через запятую")] = "fn-stellar-2step",
    sizes: Annotated[str, typer.Option(help="размеры счёта, $, через запятую")] = "25000",
    tag: Annotated[
        str, typer.Option(help="добавка к версии проб, если данные окна выбора изменились")
    ] = "",
    explore: Annotated[
        bool, typer.Option(help="гейты только помечают, не отсеивают (разведка)")
    ] = False,
    full_top: Annotated[int, typer.Option(help="полный отчёт проп-MC для лучших N")] = 3,
    dry_run: Annotated[bool, typer.Option(help="не записывать пробы в счётчик")] = False,
    out: Annotated[
        Path | None, typer.Option(help="папка отчёта; по умолчанию reports/search-…")
    ] = None,
    bars_root: Annotated[Path | None, typer.Option(help="data/bars по умолчанию")] = None,
    directory: RegistryOpt = REGISTRY,
) -> None:
    """Автопоиск: семейства × выходы × фильтры → S1 → DSR/PBO → бэктест → проп-MC (M4)."""
    from pbl.config import REPO_ROOT
    from pbl.data.bars import load_bars
    from pbl.data.candles import BARS_ROOT
    from pbl.prop.engine import MCConfig
    from pbl.research.search import Budget, SearchParams, run_search
    from pbl.research.search_report import chat_summary, write_search_report
    from pbl.research.search_space import family_grid
    from pbl.research.variants import VariantConflict

    if mode not in ("forward", "is"):
        raise typer.BadParameter("mode: forward или is")
    day0, day1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    symbols = [s.strip() for s in instruments.split(",") if s.strip()]
    root = bars_root or BARS_ROOT
    try:
        bars = {s: load_bars(s, day0, day1, root=root) for s in symbols}
    except FileNotFoundError as exc:
        typer.echo(f"нет данных: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    grid = family_grid()
    if families is not None:
        want = {f.strip() for f in families.split(",")}
        unknown = want - {f.family for f in grid}
        if unknown:
            raise typer.BadParameter(f"неизвестные семейства: {', '.join(sorted(unknown))}")
        grid = [f for f in grid if f.family in want]
    p = SearchParams(
        day0,
        day1,
        campaign=campaign,
        mode=mode,  # type: ignore[arg-type]
        select_months=select_months,
        is_fraction=is_fraction,
        seed=seed,
        n_paths=paths,
        budget=Budget(max_scans, event_runs, mc_top, max_hours * 3600),
        programs=tuple(x.strip() for x in programs.split(",")),
        sizes_usd=tuple(int(x) for x in sizes.split(",")),
        registry=directory,
        tag=tag,
        explore=explore,
        families=grid,
        mc=MCConfig(n_paths=paths, seed=seed),
    )
    try:
        res = run_search(bars, p)
    except VariantConflict as exc:
        typer.echo(f"{exc}; задайте --tag", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    report_dir = out or REPO_ROOT / "reports" / f"search-{campaign.lower()}-{p.on}"
    summary = write_search_report(res, report_dir, bars, full_top=full_top)
    if not dry_run:
        res.log.save()
    typer.echo(chat_summary(res, summary))
