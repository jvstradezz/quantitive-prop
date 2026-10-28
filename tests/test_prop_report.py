"""Оптимизатор, таблица программ и CLI `pbl prop` на маленьких прогонах."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pbl.cli import app
from pbl.config import load_firms, load_prop_types
from pbl.prop.days import day_table_from_trades
from pbl.prop.engine import MCConfig, MCEngine
from pbl.prop.optimize import GRID, feasible, optimize
from pbl.prop.report import recommend, run_report, to_markdown
from pbl.prop.rules import compile_offer
from pbl.prop.synthetic import synthetic_trades
from pbl.schemas import Firm

SMALL = MCConfig(
    n_paths=1_000, opt_paths=400, seed=3, outer_boot=4, outer_paths=200, skip_repeats=2
)


@pytest.fixture(scope="module")
def firms() -> list[Firm]:
    return load_firms()


@pytest.fixture(scope="module")
def fn(firms: list[Firm]) -> Firm:
    return next(f for f in firms if f.id == "fundednext")


def test_optimizer_respects_risk_rule_and_returns_curves(fn: Firm) -> None:
    trades = synthetic_trades(n_days=400, win_rate=0.5, rr=1.5, cost_r=0.05, seed=12)
    eng = MCEngine(day_table_from_trades(trades, fn.server_time), SMALL)
    prog = next(p for p in fn.programs if p.id == "fn-stellar-2step")
    offer = next(o for o in prog.iter_offers() if o.size.size_usd == 25_000)
    rules = compile_offer(fn.id, prog, offer)
    res = optimize(eng, rules)
    assert set(res.curves) == {0, 1, 2}
    assert all(f in GRID for f in res.best.risk.f)
    # Risk Rule 3%: одна позиция за раз → funded-риск выше 3% недопустим, а 3% — можно
    assert feasible(eng, rules, 2, 0.03) is None
    assert res.best.stats.p1 <= res.best.stats.p_funded
    assert res.best.risk.f[2] * eng.table.risk_open_max.max() <= 0.03 + 1e-12


def test_report_end_to_end(firms: list[Firm]) -> None:
    trades = synthetic_trades(n_days=400, win_rate=0.5, rr=1.5, cost_r=0.05, seed=12)
    rep = run_report(
        trades,
        firms,
        load_prop_types(),
        SMALL,
        top_k=1,
        program_ids=["fn-stellar-2step", "fn-stellar-instant", "fp-2step-flex"],
    )
    keys = {r.rules.key for r in rep.rows}
    assert "fn-stellar-2step/25k/standard" in keys
    assert "template/futures_eod_trailing" in keys
    assert any("2-Step Flex" in s for s in rep.skipped)  # P-24: выплат нет — не считаем
    assert rep.recommended is not None
    assert rep.recommended.rules.key == recommend(rep.rows).rules.key  # type: ignore[union-attr]
    assert rep.recommended.stresses
    assert rep.recommended.sample_ci is not None
    md = to_markdown(rep)
    for part in ("Лучше всего", "Стрессы", "Доля издержек"):
        assert part in md
    assert "COSTS_HEAVY" not in md  # издержки 0.05R при валовых 0.25R на сделку — 20%


def test_cli_synth_and_run(tmp_path: Path) -> None:
    runner = CliRunner()
    csv_path = tmp_path / "trades.csv"
    res = runner.invoke(app, ["prop", "synth", "--out", str(csv_path), "--days", "300"])
    assert res.exit_code == 0, res.output
    out = tmp_path / "rep"
    res = runner.invoke(
        app,
        [
            "prop",
            "run",
            str(csv_path),
            "--out",
            str(out),
            "--firm",
            "fundednext",
            "--program",
            "fn-stellar-2step",
            "--no-templates",
            "--paths",
            "500",
            "--opt-paths",
            "200",
            "--outer-boot",
            "2",
            "--top",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert (out / "summary.md").exists()
    assert (out / "prop_table.csv").read_text(encoding="utf-8").count("\n") == 10
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 42
    assert manifest["n_paths"] == 500
