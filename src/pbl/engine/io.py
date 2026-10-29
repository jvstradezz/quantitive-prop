"""Запись прогона: `trades.parquet`, `trades.csv` (формат `pbl prop run`) и `equity_R.parquet`.

Данные прогонов — в `data/runs/<run_id>/` (в git не попадают, ARCHITECTURE §8).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from pbl.engine.backtest import BacktestResult
from pbl.schemas import Trade

FIELDS = list(Trade.model_fields)


def trades_frame(trades: Sequence[Trade]) -> pl.DataFrame:
    rows = [t.model_dump() for t in trades]
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={k: pl.String for k in FIELDS})


def write_trades_csv(trades: Sequence[Trade], path: Path) -> None:
    """CSV с колонками `Trade` — тот же формат, что читает `pbl.prop.io.read_trades_csv`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for t in trades:
            row = t.model_dump(mode="json")
            w.writerow({k: "" if row[k] is None else row[k] for k in FIELDS})


def write_run(result: BacktestResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_frame(result.trades).write_parquet(out_dir / "trades.parquet", compression="zstd")
    write_trades_csv(result.trades, out_dir / "trades.csv")
    result.equity.write_parquet(out_dir / "equity_R.parquet", compression="zstd")
