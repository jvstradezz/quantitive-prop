"""`pbl engine run`: бэктест эталонной стратегии на скачанных барах Dukascopy."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Annotated

import typer

from pbl.config import CONFIGS, REPO_ROOT, load_firms, load_model
from pbl.engine.backtest import Exits, RunMeta, backtest
from pbl.engine.io import write_run
from pbl.engine.market import FirmTradeProfile, prepare_market, strategy_frame
from pbl.instruments.costs import CostModel, SpreadMode
from pbl.schemas import InstrumentSpec

engine_app = typer.Typer(help="Движок бэктеста (ARCHITECTURE §9)", no_args_is_help=True)


@engine_app.callback()
def main() -> None:
    """Группа команд движка."""


@engine_app.command("run")
def run(
    symbol: str,
    start: Annotated[str, typer.Option(help="YYYY-MM-DD")],
    end: Annotated[str, typer.Option(help="YYYY-MM-DD включительно")],
    firm_id: Annotated[str, typer.Option("--firm")] = "fundednext",
    program: Annotated[str, typer.Option(help="программа: комиссия и правила сделок")] = (
        "fn-stellar-2step"
    ),
    funded: bool = False,
    spread: Annotated[str, typer.Option(help="data|typical|worst|conservative")] = ("conservative"),
    range_start: str = "16:30",
    range_end: str = "17:00",
    until: str = "23:00",
    tp_r: float = 1.0,
    close_at: Annotated[str | None, typer.Option(help="HH:MM сервера")] = "23:30",
    risk_usd: Annotated[float, typer.Option(help="$ на 1R для таблицы издержек")] = 250.0,
    out: Annotated[Path | None, typer.Option(help="папка прогона")] = None,
) -> None:
    """Пробой диапазона (эталон движка) → data/runs/<run_id>/ и сводка издержек."""
    from pbl.data.bars import build_bars
    from pbl.data.dukascopy import RAW_ROOT, load_ticks
    from pbl.strategies.reference import OpeningRangeBreakout

    firm = next(f for f in load_firms() if f.id == firm_id)
    prog = next(p for p in firm.programs if p.id == program)
    spec = load_model(CONFIGS / "instruments" / firm_id / f"{symbol}.yaml", InstrumentSpec)
    ticks = load_ticks(symbol, dt.date.fromisoformat(start), dt.date.fromisoformat(end), RAW_ROOT)
    if ticks.is_empty():
        typer.echo(f"нет тиков {symbol} за {start}…{end}: pbl data download", err=True)
        raise typer.Exit(code=1)
    bars = build_bars(ticks, "1m", "utc")
    if spread not in ("data", "typical", "worst", "conservative"):
        raise typer.BadParameter("spread: data|typical|worst|conservative")
    mode: SpreadMode = spread  # type: ignore[assignment]
    market = prepare_market(
        bars,
        spec,
        firm.server_time,
        CostModel(spread=mode),
        FirmTradeProfile.from_rules(prog.trade_rules, funded=funded),
        prog.commission,
    )
    strat = OpeningRangeBreakout(
        range_start=range_start, range_end=range_end, until=until, tp_r=tp_r
    )
    sig = strat.signals(strategy_frame(bars, firm.server_time))
    res = backtest(
        market, sig, Exits(close_at=close_at), RunMeta(strategy_id=f"{strat.name}-{symbol}")
    )
    run_id = f"{strat.name}_{symbol}_{start}_{end}"
    out_dir = out or REPO_ROOT / "data" / "runs" / run_id
    write_run(res, out_dir)
    t = res.trades
    typer.echo(f"{symbol} {start}…{end}: баров {len(market)}, сделок {len(t)}, {res.signal_status}")
    if not t:
        return
    rows = {
        "Валовая": sum(x.r_gross for x in t),
        "Спред": -sum(x.cost_spread_r for x in t),
        "Комиссия": -sum(x.cost_comm_r for x in t),
        "Своп": -sum(x.cost_swap_r for x in t),
        "Проскальзывание": -sum(x.cost_slip_r for x in t),
        "Чистый": sum(x.r_net for x in t),
    }
    typer.echo(f"| Статья | R | $ при 1R = ${risk_usd:.0f} |\n|---|---|---|")
    for k, v in rows.items():
        typer.echo(f"| {k} | {v:+.2f} | {v * risk_usd:+.0f} |")
    typer.echo(
        f"WR {res.stats['win_rate']:.0%}, PF {res.stats['pf']:.2f}, "
        f"доля издержек от валовой {res.stats['cost_share_of_gross']:.0%}; файлы: {out_dir}"
    )
