"""`pbl data ...`: скачать тики, собрать отчёт покрытия и спредов, заполнить спеки."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import yaml

from pbl.config import REPO_ROOT
from pbl.data.dukascopy import RAW_ROOT, download_range, load_ticks, trading_days
from pbl.data.quality import check_ticks
from pbl.data.spread import patch_spec, render_yaml, spread_profile
from pbl.data.symbols import SYMBOLS

data_app = typer.Typer(help="Данные Dukascopy", no_args_is_help=True)
SPEC_DIR = REPO_ROOT / "configs" / "instruments" / "fundednext"


def _point(symbol: str) -> float:
    spec = yaml.safe_load((SPEC_DIR / f"{symbol}.yaml").read_text(encoding="utf-8"))
    return float(spec["point"])


@data_app.command("download")
def download(
    symbol: str,
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD включительно")],
    workers: int = 4,
) -> None:
    """Скачать тики bid/ask в data/raw/dukascopy/<symbol>/<year>/<date>.parquet."""
    if symbol not in SYMBOLS:
        raise typer.BadParameter(f"неизвестный символ, есть: {', '.join(SYMBOLS)}")
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    total = download_range(
        symbol,
        s,
        e,
        workers=workers,
        progress=lambda d, n: typer.echo(f"{symbol} {d} {n}") if d.day == 1 else None,
    )
    typer.echo(f"{symbol}: {total} тиков")


@data_app.command("report")
def report(
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    out: Annotated[Path, typer.Option()] = REPO_ROOT / "reports" / "dukascopy_coverage_spread.md",
    fill_specs: Annotated[bool, typer.Option(help="записать spread_by_hour в YAML")] = False,
) -> None:
    """Отчёт покрытия/качества и профиль спреда по скачанным данным."""
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    expected = [d for d in trading_days(s, e) if d.weekday() < 5]
    lines = [f"# Dukascopy: покрытие и спреды, {s} … {e}", ""]
    lines += [
        "| Символ | Тиков | Дней с данными / будней | Дубли | Перевёрнутые "
        "| Нулевой спред % | Скачки >2% | Паузы >30 мин |",
        "|---|---|---|---|---|---|---|---|",
    ]
    detail: list[str] = []
    for sym in symbols or list(SYMBOLS):
        ticks = load_ticks(sym, s, e, RAW_ROOT)
        q = check_ticks(sym, ticks, expected)
        lines.append(
            f"| {sym} | {q.ticks} | {q.days_with_data}/{q.days_expected} | {q.duplicates} | "
            f"{q.crossed} | {q.zero_spread_pct} | {q.jumps} | {q.gaps.height} |"
        )
        if ticks.is_empty():
            continue
        detail.append(f"## {sym}")
        detail.append(
            f"Будни без данных ({len(q.missing_days)}): "
            + ", ".join(str(d) for d in q.missing_days[:30])
            + "\n"
        )
        if q.gaps.height:
            top = q.gaps.sort("minutes", descending=True).head(5)
            detail.append(
                "Самые длинные паузы (UTC): "
                + "; ".join(
                    f"{r['start']:%m-%d %H:%M}→{r['end']:%H:%M} ({r['minutes']} мин)"
                    for r in top.iter_rows(named=True)
                )
                + "\n"
            )
        prof = spread_profile(ticks, _point(sym))
        detail.append("| Час сервера FN | Тиков | typical, пункты | worst (p99), пункты |")
        detail.append("|---|---|---|---|")
        for r in prof.iter_rows(named=True):
            detail.append(
                f"| {r['hour']:02d} | {r['ticks']} | {r['typical']:.1f} | {r['worst']:.1f} |"
            )
        detail.append("")
        if fill_specs:
            patch_spec(SPEC_DIR / f"{sym}.yaml", render_yaml(prof), SYMBOLS[sym].duka)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([*lines, "", *detail]), encoding="utf-8")
    typer.echo(f"отчёт: {out}")


@data_app.command("bars")
def bars_cmd(
    symbol: str,
    start: str,
    end: str,
    every: str = "1m",
    clock: str = "utc",
) -> None:
    """Собрать бары и сохранить в data/bars/<symbol>/<year>/<every>_<clock>.parquet."""
    from pbl.data.bars import build_bars

    ticks = load_ticks(symbol, date.fromisoformat(start), date.fromisoformat(end), RAW_ROOT)
    if clock not in ("utc", "server"):
        raise typer.BadParameter("clock: utc|server")
    b: pl.DataFrame = build_bars(ticks, every, clock)  # type: ignore[arg-type]
    for year, part in b.group_by(pl.col("ts").dt.year().alias("y")):
        y = year[0]
        p = REPO_ROOT / "data" / "bars" / symbol / f"{y}" / f"{every}_{clock}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        part.drop("y").write_parquet(p, compression="zstd")
    typer.echo(f"{symbol}: {b.height} баров {every}")
