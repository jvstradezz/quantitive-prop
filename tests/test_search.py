"""Автопоиск: перечисление и бюджет, пакетный счётчик, воронка, защита holdout, хуки гейтов, CLI.

Синтетические цены — случайное блуждание без края: гейты ослаблены явно (`S1Gate`, `min_dsr`),
проверяем механику и порядок стадий, а не рыночные выводы.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from pbl.cli import app
from pbl.config import load_firms
from pbl.prop.engine import MCConfig
from pbl.prop.servertime import server_date
from pbl.research import search, search_report
from pbl.research.range_sweep import exit_grid
from pbl.research.search import (
    Budget,
    Candidate,
    GateResult,
    S1Gate,
    SearchContext,
    SearchParams,
    SearchResult,
    make_windows,
    mc_summary,
    restrict,
    run_search,
)
from pbl.research.search_space import (
    GEN_VERSION,
    SearchConfig,
    enumerate_configs,
    family_grid,
    sample_configs,
)
from pbl.research.variants import DailyR, VariantConflict, VariantLog, VariantSpec
from pbl.robustness.autoreject import AutoRejectInput, autoreject
from pbl.schemas import Trade
from pbl.schemas.strategy import ExitSpec
from pbl.strategies.families import Donchian, Family, MeanReversionZ, RangeBreakout, TrendEma

BarsFactory = Callable[..., pl.DataFrame]
ON = dt.date(2026, 10, 30)
START, END = dt.date(2025, 4, 7), dt.date(2025, 10, 20)

FAMS: list[Family] = [
    Donchian(tf=15, lookback=20, session="ny"),
    Donchian(tf=5, lookback=20, session="london"),
    MeanReversionZ(tf=5, lookback=20, z=1.5, session="asia"),
    TrendEma(tf=5, fast=9, slow=21, session="london"),
    RangeBreakout(session="ny30"),
]


def params(registry: Path, **kw: object) -> SearchParams:
    base: dict[str, object] = {
        "campaign": "AUTO-T",
        "mode": "forward",
        "select_months": 3,
        "on": ON,
        "n_paths": 300,
        "s1": S1Gate(min_trades=1, min_pf=0.0, cost_mult=0.0),
        "min_dsr": 0.0,
        "min_eval_trades": 8,
        "pbo_blocks": 4,
        "families": FAMS,
        "exits": ("rr_1_1", "rr_1_2", "trailing"),
        "filters": ("none", "trend"),
        "budget": Budget(max_event_runs=12, mc_top=2),
        "registry": registry,
        "mc": MCConfig(n_paths=300, opt_paths=200, outer_boot=0, seed=1),
    }
    base.update(kw)
    return SearchParams(START, END, **base)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def bars(synthetic_bars: BarsFactory) -> dict[str, pl.DataFrame]:
    return {"NAS100": synthetic_bars(140, 1), "US30": synthetic_bars(140, 2)}


@pytest.fixture(scope="module")
def result(bars: dict[str, pl.DataFrame], tmp_path_factory: pytest.TempPathFactory) -> SearchResult:
    return run_search(bars, params(tmp_path_factory.mktemp("reg")))


# --- пространство и бюджет ----------------------------------------------------------------


def test_grid_size_is_stable_and_keys_are_unique() -> None:
    configs = enumerate_configs()
    assert len(family_grid()) == 194
    assert len(configs) == 194 * 7 * 4 == 5432
    assert len({c.key for c in configs}) == len(configs)
    # выходы закрываются ко времени закрытия сессии семейства, а не к общему 23:00
    asia = next(c for c in configs if c.family.session == "asia" and c.family.family == "donchian")
    assert asia.exit.exits.close_at == "10:00"
    ny = next(c for c in configs if c.family.family == "range_fade")
    assert ny.exit.exits.close_at == "23:00"


def test_budget_sampling_is_deterministic_and_keeps_order() -> None:
    configs = enumerate_configs()
    a = sample_configs(configs, max_scans=1000, scans_per_config=4, seed=7)
    b = sample_configs(configs, max_scans=1000, scans_per_config=4, seed=7)
    c = sample_configs(configs, max_scans=1000, scans_per_config=4, seed=8)
    assert len(a) == 250
    assert [x.key for x in a] == [x.key for x in b]
    assert [x.key for x in a] != [x.key for x in c]
    pos = {cfg.key: i for i, cfg in enumerate(configs)}
    idx = [pos[x.key] for x in a]
    assert idx == sorted(idx)
    assert sample_configs(configs, max_scans=10**9, scans_per_config=4, seed=7) == configs


def test_with_tp_replaces_only_the_take_profit() -> None:
    cfg = SearchConfig(FAMS[0], exit_grid("23:00")["rr_1_2"])
    assert cfg.exit.tp_r == 2.0
    assert cfg.filter_specs == []
    assert SearchConfig(FAMS[0], cfg.exit, "trend").filter_specs[0].kind == "trend"
    with pytest.raises(ValueError, match="фильтр из"):
        SearchConfig(FAMS[0], cfg.exit, "magic")


# --- счётчик проб -------------------------------------------------------------------------


def spec(vid: int = 1, idea: str = "AUTO-T") -> VariantSpec:
    return VariantSpec(
        idea_id=idea,
        family="donchian",
        instrument="NAS100",
        timeframe="M15",
        session="ny",
        params={"lookback": vid},
        exits=[ExitSpec(kind="rr", params={"rr": 1.0})],
        firm_profile="fundednext",
    )


def daily(r: float = 1.0) -> DailyR:
    return DailyR(
        dt.date(2025, 4, 7),
        dt.date(2025, 4, 18),
        (dt.date(2025, 4, 8),),
        (r,),
        1,
        max(r, 0),
        min(r, 0),
    )


def test_add_many_is_batched_idempotent_and_atomic(tmp_path: Path) -> None:
    log = VariantLog.for_idea("AUTO-T", tmp_path)
    items = [(spec(i), daily(), i % 2 == 0) for i in range(1, 6)]
    assert log.add_many(items, run_id="r", on=ON) == 5
    assert log.add_many(items, run_id="r", on=ON) == 0, "повтор с тем же результатом"
    assert len(log) == 5
    assert log.frame()["series_dates"].null_count() == 3

    clash = [(spec(9), daily(1.0), True), (spec(1), daily(2.0), True)]
    with pytest.raises(VariantConflict):
        log.add_many(clash, run_id="r", on=ON)
    assert len(log) == 5, "при конфликте лог не меняется"
    inner = [(spec(10), daily(1.0), True), (spec(10), daily(2.0), True)]
    with pytest.raises(VariantConflict):
        log.add_many(inner, run_id="r", on=ON)

    with pytest.raises(ValueError, match="вариант идеи"):
        log.add_many([(spec(1, "OTHER"), daily(), True)], run_id="r", on=ON)
    assert log.add(spec(20), daily(), run_id="r", on=ON) is True
    assert log.add(spec(20), daily(), run_id="r", on=ON) is False


# --- воронка и порядок стадий -------------------------------------------------------------


def test_every_probe_is_logged_and_funnel_only_narrows(result: SearchResult) -> None:
    f = result.funnel
    probes = sum(len(u.symbols) + 1 for u in result.units)
    assert f["проб в счётчике кампании"] == len(result.log) == probes
    assert f["единиц (конфигурация × корзина)"] == len(result.units)
    chain = [
        "единиц (конфигурация × корзина)",
        "прошли S1",
        "из них с уникальным исходом сделок",
        "прошли DSR/PBO",
        "событийный бэктест",
    ]
    vals = [f[k] for k in chain]
    assert vals == sorted(vals, reverse=True)
    assert f["прошли S1"] > 0, "с ослабленным S1 на синтетике кто-то выживает"
    assert f["проп-MC"] <= 2
    frame = result.log.frame()
    baskets = frame.filter(pl.col("instrument").str.starts_with("BASKET:"))
    assert baskets.height == len(result.units)
    survivors = {u.variant_id for u in result.units if u.s1}
    with_series = set(baskets.filter(pl.col("series_dates").is_not_null())["variant_id"])
    assert with_series == survivors, "ряды хранятся только у выживших S1"
    assert frame.filter(~pl.col("instrument").str.starts_with("BASKET:"))[
        "series_dates"
    ].null_count() == (frame.height - baskets.height)
    assert frame["version"].unique().len() == 1
    assert frame["version"][0].startswith(f"{GEN_VERSION}:")


def test_identical_outcomes_enter_the_event_stage_once(result: SearchResult) -> None:
    seen = {
        (c.unit.stats["n"], round(c.unit.stats["sum_r"], 8), round(c.sharpe, 10))
        for c in result.candidates
    }
    assert len(seen) == len(result.candidates)


def test_scan_and_event_backtest_agree(result: SearchResult) -> None:
    assert result.candidates
    for c in result.candidates:
        assert c.parity_gap < 1e-6, (c.unit.key, c.parity_gap)
        assert all(t.sample in ("IS", "WF") for t in c.trades)
        assert all(t.sl > 0 and t.stop_dist > 0 for t in c.trades), "стоп у каждой сделки"
        assert {t.strategy_id for t in c.trades} == {c.unit.variant_id}


def test_autoreject_from_robustness_is_wired_in_as_a_gate(result: SearchResult) -> None:
    server = load_firms()[0].server_time
    assert result.candidates
    for c in result.candidates:
        direct = autoreject(AutoRejectInput(c.trades, server))
        assert c.gates["Автоотказ"] is (not direct.rejected)
        assert ("Автоотказ" in c.details) is direct.rejected
        assert not direct.get("no_stop").triggered


def test_autoreject_blocks_a_candidate_that_trades_like_a_scalper(
    bars: dict[str, pl.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Каждая сделка — «удержание 0 с, прибыль» → Quick Strike: кандидат не идёт в MC."""
    real = search.run_events

    def fast(*a: object, **k: object) -> list[Trade]:
        trades = real(*a, **k)  # type: ignore[arg-type]
        return [t.model_copy(update={"hold_sec": 5, "r_gross": 1.0, "r_net": 0.9}) for t in trades]

    monkeypatch.setattr(search, "run_events", fast)
    res = run_search(bars, params(tmp_path, explore=True))
    assert res.candidates
    assert all(c.gates["Автоотказ"] is False for c in res.candidates)
    assert all("Quick Strike" in c.details["Автоотказ"] for c in res.candidates)
    assert all(not c.passed for c in res.candidates)


def test_forward_mode_evaluates_only_after_the_selection_window(result: SearchResult) -> None:
    w = result.windows
    assert w.forward
    assert w.first < w.sel_end < w.cut < w.last
    server = load_firms()[0].server_time
    assert all(c.eval_sample == "WF" for c in result.candidates)
    for c in result.candidates:
        for t in c.eval_trades:
            d = server_date(t.open_utc, server)
            assert w.sel_end <= d < w.cut
            assert t.sample == "WF"


def test_candidates_are_ranked_by_prop_probability(result: SearchResult) -> None:
    with_mc = [c for c in result.candidates if c.mc is not None]
    assert with_mc, "MC считается для лучших"
    keys = [search.rank_key(c) for c in result.candidates]
    assert keys == sorted(keys, reverse=True)
    for c in with_mc:
        assert c.mc is not None
        assert 0.0 <= c.mc.p1 <= 1.0
        assert c.mc.p1_ci[0] <= c.mc.p1 <= c.mc.p1_ci[1]
        assert c.mc.n_trades == len(c.eval_trades)
        assert "Stellar 2-Step $25K" in c.mc.label


def test_dsr_uses_the_whole_counter_for_n(result: SearchResult) -> None:
    d = result.deflation
    assert d.n_trials == len(result.log)
    assert d.sr0 >= 0
    assert set(d.dsr) >= {c.unit.variant_id for c in result.candidates}


def test_bars_after_the_cut_do_not_exist_for_the_search(
    bars: dict[str, pl.DataFrame], result: SearchResult, tmp_path: Path
) -> None:
    w = result.windows
    restricted = restrict(bars["NAS100"], w)
    ts_max, ts_min = restricted["ts"].max(), restricted["ts"].min()
    assert isinstance(ts_max, dt.datetime)
    assert isinstance(ts_min, dt.datetime)
    assert ts_max < dt.datetime.combine(w.cut, dt.time(0), dt.UTC)
    assert ts_min >= dt.datetime.combine(w.first, dt.time(0), dt.UTC)

    # цены holdout меняются — результат поиска тот же
    cut_ts = dt.datetime.combine(w.cut, dt.time(0), dt.UTC)
    price = [c for c in bars["NAS100"].columns if c.startswith(("bid_", "ask_"))]
    noisy = {
        s: b.with_columns(
            pl.when(pl.col("ts") >= cut_ts).then(pl.col(c) * 1.2).otherwise(pl.col(c)).alias(c)
            for c in price
        )
        for s, b in bars.items()
    }
    other = run_search(noisy, params(tmp_path))
    assert other.funnel == {**result.funnel}
    assert [c.unit.key for c in other.candidates] == [c.unit.key for c in result.candidates]
    assert [c.mc.p1 if c.mc else None for c in other.candidates] == [
        c.mc.p1 if c.mc else None for c in result.candidates
    ]


def test_rerun_after_save_adds_no_probes(bars: dict[str, pl.DataFrame], tmp_path: Path) -> None:
    p = params(tmp_path)
    first = run_search(bars, p)
    first.log.save()
    again = run_search(bars, p)
    assert len(again.log) == len(first.log)
    assert again.funnel == first.funnel
    # новое окно данных — новая версия пробы, а не конфликт со старым результатом
    shifted = dataclasses.replace(p, select_months=2)
    more = run_search(bars, shifted)
    assert len(more.log) > len(first.log)


def test_is_mode_is_marked_in_sample(bars: dict[str, pl.DataFrame], tmp_path: Path) -> None:
    res = run_search(bars, params(tmp_path, mode="is"))
    assert not res.windows.forward
    assert res.windows.sel_end == res.windows.cut
    assert res.basis == "IN-SAMPLE"
    text = search_report.summary_markdown(res)
    assert "IN-SAMPLE — справка, не для решений" in text
    assert all(c.eval_sample == "IS" for c in res.candidates)


def test_not_enough_data_for_forward_mode(bars: dict[str, pl.DataFrame], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="мало данных для режима forward"):
        make_windows(bars, params(tmp_path, select_months=12))


# --- гейты, хуки, разведка ----------------------------------------------------------------


def test_deflation_gate_stops_candidates_unless_exploring(
    bars: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    hard = params(tmp_path, min_dsr=1.1, max_pbo=-1.0)
    res = run_search(bars, hard)
    assert res.funnel["прошли S1"] > 0
    assert res.funnel["прошли DSR/PBO"] == 0
    assert res.candidates == []
    assert "DSR/PBO" in search_report.verdict(res)
    soft = run_search(bars, dataclasses.replace(hard, explore=True))
    assert soft.candidates
    assert all(c.gates["DSR/PBO"] is False for c in soft.candidates)
    assert "Разведка" in search_report.summary_markdown(soft)


def test_extra_gate_hook_blocks_mc_and_is_reported(
    bars: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    seen: list[str] = []

    def veto(c: Candidate, ctx: SearchContext) -> GateResult:
        seen.append(c.unit.key)
        assert ctx.windows.forward
        assert c.trades, "гейт видит сделки событийного бэктеста"
        return GateResult("S2-заглушка", False, "проверка хука")

    res = run_search(bars, params(tmp_path, extra_gates=(veto,)))
    assert seen
    assert res.funnel["проп-MC"] == 0
    assert all(c.gates["S2-заглушка"] is False and c.mc is None for c in res.candidates)
    assert all(c.details["S2-заглушка"] == "проверка хука" for c in res.candidates)
    assert "S2-заглушка ✗" in search_report.gates_text(res.candidates[0])


def test_eval_trades_hook_replaces_the_default_oos(
    bars: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    def own(c: Candidate, ctx: SearchContext) -> tuple[list[Trade], str]:
        return list(c.trades[:11]), "WF"

    res = run_search(bars, params(tmp_path, eval_trades=own))
    assert res.candidates
    assert all(c.eval_stats["n"] == 11 for c in res.candidates)


def test_zero_survivors_gives_an_honest_verdict(
    bars: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    res = run_search(bars, params(tmp_path, s1=S1Gate()))  # настоящие пороги S1
    assert res.funnel["прошли S1"] == 0
    assert res.candidates == []
    assert "edge в перебранном пространстве не найден" in search_report.verdict(res)
    assert len(res.log) > 0, "провалившие S1 пробы всё равно в счётчике"


def test_time_budget_truncates_and_says_so(bars: dict[str, pl.DataFrame], tmp_path: Path) -> None:
    res = run_search(bars, params(tmp_path, budget=Budget(max_seconds=-1.0)))
    assert res.truncated
    assert len(res.units) == 0
    assert "прервано по бюджету" in search_report.summary_markdown(res)


def test_scan_budget_limits_the_number_of_scans(
    bars: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    res = run_search(
        bars, params(tmp_path, budget=Budget(max_scans=20, max_event_runs=2, mc_top=1))
    )
    assert res.funnel["сканов"] <= 20
    assert res.funnel["конфигураций прогнано"] == 10  # 20 сканов / 2 инструмента


def test_mc_needs_enough_trades(result: SearchResult) -> None:
    firm = load_firms()[0]
    c = next(c for c in result.candidates if c.mc is not None)
    p = params(Path("."))
    assert mc_summary(c.eval_trades[:3], firm, p) is None
    assert mc_summary(c.eval_trades, firm, p) is not None


# --- отчёт и CLI --------------------------------------------------------------------------


def test_report_files_and_chat_summary(
    result: SearchResult, bars: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    out = tmp_path / "rep"
    summary = search_report.write_search_report(result, out, bars, full_top=1)
    text = summary.read_text(encoding="utf-8")
    assert "## Воронка" in text
    assert "COSTS_HEAVY" in text or "P(1-я выплата)" in text
    assert "holdout" in text
    board = pl.read_csv(out / "leaderboard.csv")
    assert board.height == len(result.candidates)
    assert board["rank"].to_list() == list(range(1, board.height + 1))
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["campaign"] == "AUTO-T"
    assert man["windows"]["cut"] == str(result.windows.cut)
    assert set(man["data"]) == {"NAS100", "US30"}
    assert (out / "top-1" / "summary.md").is_file()
    short = search_report.chat_summary(result, summary)
    assert len(short.splitlines()) <= 30


def write_bars(root: Path, bars: dict[str, pl.DataFrame]) -> None:
    for sym, b in bars.items():
        d = root / sym / "2025"
        d.mkdir(parents=True)
        b.write_parquet(d / "1m_utc_candles.parquet")


def test_cli_search_writes_report_and_counter(
    bars: dict[str, pl.DataFrame], tmp_path: Path
) -> None:
    write_bars(tmp_path / "bars", bars)
    reg = tmp_path / "reg"
    args = [
        "search", "--start", "2025-04-07", "--end", "2025-10-20",
        "--instruments", "NAS100,US30", "--select-months", "3", "--paths", "300",
        "--families", "donchian", "--max-scans", "40", "--event-runs", "2", "--mc-top", "1",
        "--full-top", "0", "--bars-root", str(tmp_path / "bars"), "--directory", str(reg),
        "--out", str(tmp_path / "out"), "--campaign", "AUTO-C",
    ]  # fmt: skip
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    assert "Автопоиск AUTO-C" in res.output
    assert len(res.output.strip().splitlines()) <= 30
    assert (tmp_path / "out" / "summary.md").is_file()
    log = VariantLog.for_idea("AUTO-C", reg)
    assert len(log) > 0, "пробы сохранены в счётчик"
    n_before = len(log)

    dry = CliRunner().invoke(app, [*args, "--dry-run", "--campaign", "AUTO-D"])
    assert dry.exit_code == 0, dry.output
    assert not VariantLog.shard_path("AUTO-D", reg).exists()
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 0, again.output
    assert len(VariantLog.for_idea("AUTO-C", reg)) == n_before


def test_cli_search_stops_without_data_and_on_bad_options(tmp_path: Path) -> None:
    base = ["search", "--start", "2025-01-01", "--end", "2025-12-31", "--bars-root", str(tmp_path)]
    res = CliRunner().invoke(app, base)
    assert res.exit_code == 1
    assert "нет данных" in res.output
    bad = CliRunner().invoke(app, [*base, "--mode", "x"])
    assert bad.exit_code != 0


def test_unknown_family_is_rejected(bars: dict[str, pl.DataFrame], tmp_path: Path) -> None:
    write_bars(tmp_path / "bars", bars)
    res = CliRunner().invoke(
        app,
        ["search", "--start", "2025-04-07", "--end", "2025-10-20", "--instruments", "NAS100",
         "--families", "magic", "--bars-root", str(tmp_path / "bars")],
    )  # fmt: skip
    assert res.exit_code != 0
    assert "неизвестные семейства" in res.output
