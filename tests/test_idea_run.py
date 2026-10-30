"""Прогон идеи №1 на синтетике: порядок стадий, счётчик проб, защита holdout, отчёт, CLI.

Синтетические цены — случайное блуждание без края. Гейты ослаблены `relaxed`: проверяем
механику и порядок расчётов, а не рыночные выводы.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from typer.testing import CliRunner

import pbl.robustness.bands as bands
import pbl.robustness.stats as rstats
import pbl.robustness.walkforward as rwf
from pbl.cli import app
from pbl.prop.engine import MCConfig
from pbl.research import idea_eval, idea_report, idea_run, range_sweep, registry
from pbl.research.idea_eval import pooled
from pbl.research.range_sweep import Bundle, make_bundle
from pbl.robustness.autoreject import AutoReject, AutoRejectInput

BarsFactory = Callable[..., pl.DataFrame]
ON = dt.date(2026, 10, 29)
START, END = dt.date(2025, 4, 7), dt.date(2025, 10, 20)


def small_grid() -> list[range_sweep.Config]:
    """Подсетка для тестов: два окна, три выхода, обе семьи и оба фильтра, буферы пробоя."""
    return [
        c
        for c in range_sweep.grid()
        if c.session in ("ny30", "london") and c.exit.name in ("rr_1_1", "rr_1_2", "trailing")
    ]


@pytest.fixture(autouse=True)
def fast_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idea_run, "grid", small_grid)


def _no_reject(inp: AutoRejectInput) -> AutoReject:
    return AutoReject(findings=(), n_trades=len(inp.trades))


@pytest.fixture
def relaxed(monkeypatch: pytest.MonkeyPatch) -> None:
    for mod, name, value in (
        (idea_eval, "MIN_TRADES", 1),
        (idea_eval, "MIN_PF", 0.0),
        (rstats, "COST_MULT", 0.0),
        (rwf, "WF_MIN_TRADES", 1),
        (rstats, "MIN_YEAR_TRADES", 1),
        (rstats, "YEAR_SHARE", 1.0),
        (idea_run, "wfe", lambda picks: 1.0),  # на случайном блуждании WFE часто NaN
        (idea_run, "year_stability", lambda t: (True, pl.DataFrame())),  # один год данных
        (idea_run, "SENS_PF", 0.0),
        (idea_run, "MIN_DSR", 0.0),
        (idea_run, "MIN_INSTRUMENTS", 0),
        (idea_run, "HO_MIN_MEAN_R", -1e9),
        (idea_run, "autoreject", _no_reject),  # случайные бары: стопы в баре входа, узкие окна
    ):
        monkeypatch.setattr(mod, name, value)


@pytest.fixture
def band_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """Полоса MC на случайном блуждании почти всегда не проходит: открываем её ради S3."""
    monkeypatch.setattr(bands, "REQUIRED", ())


@pytest.fixture(scope="module")
def bundles(synthetic_bars: BarsFactory) -> dict[str, Bundle]:
    return {
        "NAS100": make_bundle("NAS100", synthetic_bars(140, 1)),
        "US30": make_bundle("US30", synthetic_bars(140, 2)),
    }


def params(registry_dir: Path) -> idea_run.RunParams:
    return idea_run.RunParams(
        START,
        END,
        version="t",
        on=ON,
        n_paths=200,
        wf=(2, 1, 1),
        pbo_blocks=4,
        top=3,
        registry=registry_dir,
        mc=MCConfig(n_paths=300, opt_paths=200, outer_boot=0, seed=1),
    )


def test_every_probe_is_logged_and_the_run_is_repeatable(
    bundles: dict[str, Bundle], relaxed: None, tmp_path: Path
) -> None:
    p = params(tmp_path)
    st = idea_run.run_is(bundles, p)
    live = sum(len(per) + 1 for per in st.tables.values())  # инструменты + корзина
    assert len(st.log) == st.n_trials == st.new_rows == live
    frame = st.log.frame()
    baskets = frame.filter(pl.col("instrument").str.starts_with("BASKET:"))
    assert baskets.height == len(st.tables)
    assert baskets["series_dates"].null_count() == 0
    assert frame.filter(~pl.col("instrument").str.starts_with("BASKET:"))[
        "series_r"
    ].null_count() == (frame.height - baskets.height)
    assert len(st.candidates) <= p.top
    sharpes = [c.row["sharpe_daily"] for c in st.candidates]
    assert sharpes == sorted(sharpes, reverse=True)
    assert all(set(c.gates) == set(idea_run.GATES) for c in st.candidates)

    st.log.save()
    again = idea_run.run_is(bundles, p)
    assert again.new_rows == 0
    assert again.n_trials == st.n_trials


def test_bars_after_the_cut_cannot_change_the_is_stage(
    bundles: dict[str, Bundle], synthetic_bars: BarsFactory, relaxed: None, tmp_path: Path
) -> None:
    base = idea_run.run_is(bundles, params(tmp_path / "a"))
    cut = dt.datetime.combine(base.splits.cut, dt.time(0), dt.UTC)

    def spliced(symbol: str, seed: int, other: int) -> Bundle:
        a, b = synthetic_bars(140, seed), synthetic_bars(140, other)
        frame = pl.concat([a.filter(pl.col("ts") < cut), b.filter(pl.col("ts") >= cut)])
        return make_bundle(symbol, frame)

    altered = {"NAS100": spliced("NAS100", 1, 11), "US30": spliced("US30", 2, 12)}
    other = idea_run.run_is(altered, params(tmp_path / "b"))

    assert other.splits == base.splits
    assert other.s1.equals(base.s1)
    assert other.log.frame().equals(base.log.frame())
    assert [c.cfg.key for c in other.candidates] == [c.cfg.key for c in base.candidates]
    assert [c.gates for c in other.candidates] == [c.gates for c in base.candidates]
    assert idea_eval.wfe(other.picks) == pytest.approx(idea_eval.wfe(base.picks), nan_ok=True)
    assert [(x.key, x.test_r) for x in other.picks] == [(x.key, x.test_r) for x in base.picks]
    # проверка не вхолостую: после отсечения сделки действительно разные
    key = base.candidates[0].cfg.key
    ho_a = pooled(base.tables, key, base.splits.cut_ord)
    ho_b = pooled(other.tables, key, base.splits.cut_ord)
    assert not ho_a["r_net"].equals(ho_b["r_net"])


def test_holdout_is_not_opened_without_a_finalist(
    bundles: dict[str, Bundle], tmp_path: Path
) -> None:
    st = idea_run.run_is(bundles, params(tmp_path))  # строгие гейты: на синтетике S1 не пройти
    assert st.finalist is None
    with pytest.raises(ValueError, match="нет финалиста"):
        idea_run.run_holdout(bundles, st, params(tmp_path))
    assert idea_run.verdict(st, None).startswith("отклонена")


@pytest.fixture(scope="module")
def opened(
    bundles: dict[str, Bundle], tmp_path_factory: pytest.TempPathFactory
) -> tuple[idea_run.ISStage, idea_run.HoldoutStage, idea_run.RunParams]:
    mp = pytest.MonkeyPatch()
    try:
        for mod, name, value in (
            (idea_eval, "MIN_TRADES", 1),
            (idea_eval, "MIN_PF", 0.0),
            (rstats, "COST_MULT", 0.0),
            (rwf, "WF_MIN_TRADES", 1),
            (rstats, "MIN_YEAR_TRADES", 1),
            (rstats, "YEAR_SHARE", 1.0),
            (idea_run, "wfe", lambda picks: 1.0),
            (idea_run, "year_stability", lambda t: (True, pl.DataFrame())),
            (idea_run, "SENS_PF", 0.0),
            (idea_run, "MIN_DSR", 0.0),
            (idea_run, "MIN_INSTRUMENTS", 0),
            (idea_run, "HO_MIN_MEAN_R", -1e9),
            (idea_run, "autoreject", _no_reject),
            (bands, "REQUIRED", ()),
            (idea_run, "grid", small_grid),
        ):
            mp.setattr(mod, name, value)
        p = params(tmp_path_factory.mktemp("registry"))
        st = idea_run.run_is(bundles, p)
        return st, idea_run.run_holdout(bundles, st, p), p
    finally:
        mp.undo()


def test_oos_trades_come_from_wf_tests_and_the_holdout(
    opened: tuple[idea_run.ISStage, idea_run.HoldoutStage, idea_run.RunParams],
) -> None:
    st, ho, p = opened
    assert ho.result.opened_at == p.on
    assert {t.sample for t in ho.trades} == {"WF", "HOLDOUT"}
    assert len({t.trade_id for t in ho.trades}) == len(ho.trades)
    cut = dt.datetime.combine(st.splits.cut, dt.time(0), dt.UTC)
    first_test = min(pk.window.test_start for pk in st.picks)
    for t in ho.trades:
        day = t.open_utc.date()
        if t.sample == "HOLDOUT":
            assert t.open_utc >= cut - dt.timedelta(hours=3)
        else:
            assert first_test - dt.timedelta(days=1) <= day <= st.splits.cut
    assert ho.scan_vs_backtest < 0.05  # быстрый скан и событийный бэктест считают одно и то же
    assert ho.prop is not None
    assert ho.s3_ok in (True, False)


def test_verdict_follows_the_gates(
    opened: tuple[idea_run.ISStage, idea_run.HoldoutStage, idea_run.RunParams],
) -> None:
    st, ho, _ = opened
    win = dataclasses.replace(ho, s3_ok=True)
    assert idea_run.verdict(st, win).startswith("в демо")
    assert idea_run.verdict(st, dataclasses.replace(ho, s3_ok=False)).startswith("доработать")
    broken = dataclasses.replace(ho, gates={**ho.gates, "E > 0": False})
    assert idea_run.verdict(st, broken).startswith("отклонена: финалист не прошёл holdout")
    with pytest.raises(ValueError, match="holdout не прогнан"):
        idea_run.verdict(st, None)


def test_report_files_and_chat_summary(
    opened: tuple[idea_run.ISStage, idea_run.HoldoutStage, idea_run.RunParams], tmp_path: Path
) -> None:
    st, ho, p = opened
    path = idea_report.write_idea_report(st, ho, p, tmp_path / "out")
    text = path.read_text(encoding="utf-8")
    for needle in ("**Вердикт:**", "## 1. Лучшие варианты", "## 2. Фильтр дня", "## 3. Holdout"):
        assert needle in text
    assert "## 4. Издержки стратегии" in text
    assert "## 5. Проп-MC (S3)" in text
    out = tmp_path / "out"
    for name in (
        "s1_grid.csv",
        "candidates.csv",
        "wf_picks.csv",
        "trades_oos.csv",
        "manifest.json",
    ):
        assert (out / name).is_file(), name
    assert (out / "prop" / "summary.md").is_file()
    chat = (out / "chat.md").read_text(encoding="utf-8").splitlines()
    assert 0 < len(chat) <= 30


def test_card_walks_through_test_to_a_verdict(
    opened: tuple[idea_run.ISStage, idea_run.HoldoutStage, idea_run.RunParams],
) -> None:
    st, ho, p = opened
    _, card = registry.find_card("IDEA-0001")
    final = idea_run.verdict(st, ho)
    new = idea_report.card_after_run(card, st, ho, p, final)
    assert [h.status for h in new.history[-3:]] == ["Кодится", "Тест", new.status]
    assert new.status in {"Результат", "Отклонена"}
    assert new.variants_tested == st.n_trials
    assert new.holdout_opened_at == p.on
    assert new.verdict == final
    assert registry.check_card(new) == []
    with pytest.raises(ValueError, match="возможен из статуса Тест"):
        idea_report.card_after_run(new, st, ho, p, final)


def write_bars(root: Path, symbols: tuple[str, ...], synthetic_bars: BarsFactory) -> None:
    for k, sym in enumerate(symbols):
        d = root / sym / "2025"
        d.mkdir(parents=True)
        synthetic_bars(140, k + 1).write_parquet(d / "1m_utc_candles.parquet")


def test_cli_run_updates_the_registry_and_refuses_to_reopen_the_holdout(
    synthetic_bars: BarsFactory,
    relaxed: None,
    band_open: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(range_sweep, "INSTRUMENTS", ("NAS100", "US30"))
    bars = tmp_path / "bars"
    write_bars(bars, ("NAS100", "US30"), synthetic_bars)
    reg = tmp_path / "registry"
    reg.mkdir()
    src, card = registry.find_card("IDEA-0001")
    registry.save_card(card, reg / src.name)
    args = [
        "idea", "run", "IDEA-0001", "--start", str(START), "--end", str(END), "--version", "t",
        "--paths", "200", "--wf-train", "2", "--wf-test", "1", "--wf-step", "1",
        "--pbo-blocks", "4", "--top", "3", "--bars-root", str(bars), "--directory", str(reg),
        "--out", str(tmp_path / "out"), "--update-card",
    ]  # fmt: skip
    res = CliRunner().invoke(app, args)
    assert res.exit_code == 0, res.output
    assert (tmp_path / "out" / "summary.md").is_file()
    assert registry.check_registry(reg) == []  # счётчик в карточке равен числу проб в файле
    new = registry.load_card(reg / src.name)
    assert new.status in {"Результат", "Отклонена"}
    assert new.holdout_opened_at is not None

    again = CliRunner().invoke(app, args)
    assert again.exit_code == 1
    assert "holdout уже открыт" in again.output


def test_cli_stops_without_data(tmp_path: Path) -> None:
    res = CliRunner().invoke(
        app,
        ["idea", "run", "IDEA-0001", "--start", "2025-01-01", "--end", "2025-12-31",
         "--bars-root", str(tmp_path)],
    )  # fmt: skip
    assert res.exit_code == 1
    assert "нет данных" in res.output


def test_the_day_filter_is_judged_on_oos_days_only(
    bundles: dict[str, Bundle],
    relaxed: None,
    band_open: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§4.5: IS служит для отбора, вердикт по фильтру — только тесты WF и holdout."""
    seen: list[tuple[str, list[int]]] = []
    real = idea_eval.filter_check

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(("call", args[3].tolist()))
        return real(*args, **kwargs)

    monkeypatch.setattr(idea_run, "filter_check", spy)
    p = params(tmp_path)
    st = idea_run.run_is(bundles, p)
    assert seen, "кандидаты проверяются"
    is_days = set(st.splits.is_calendar.tolist())
    wf_days = set(idea_eval.oos_calendar(st.picks, st.splits).tolist())
    for _, days in seen:
        assert set(days) == wf_days
    assert wf_days < is_days  # не вся история IS, только тестовые окна
    before = len(seen)
    ho = idea_run.run_holdout(bundles, st, p)
    full = set(seen[before][1])
    assert full == wf_days | set(st.splits.ho_calendar.tolist())
    assert ho.filt is not None
    assert "OOS" in idea_report.filter_answer(st, ho)
    assert "holdout не открыт" in idea_report.filter_answer(st, None) or not st.candidates
