"""Оценка вариантов по гейтам S1–S2 на таблицах сделок: расчёты руками, без баров."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

import pbl.robustness.walkforward as rwf
from pbl.research import idea_eval as ev
from pbl.research.range_sweep import TABLE_SCHEMA, Config, exit_grid
from pbl.robustness.bands import DayStats
from pbl.robustness.splits import WFWindow

D = dt.date


def table(rows: list[tuple[int, int, int, float, float, float]]) -> pl.DataFrame:
    """(день-ordinal, день недели, сторона, r_net, r_gross, cost_r)."""
    return pl.DataFrame(
        {
            "day": [r[0] for r in rows],
            "wd": [r[1] for r in rows],
            "side": [r[2] for r in rows],
            "r_net": [r[3] for r in rows],
            "r_gross": [r[4] for r in rows],
            "cost_r": [r[5] for r in rows],
            "hold_sec": [600] * len(rows),
        },
        schema=TABLE_SCHEMA,
    )


def test_s1_stats_by_hand() -> None:
    t = table([(1, 0, 1, 2.0, 2.1, 0.1), (2, 1, 1, -1.0, -0.9, 0.1), (3, 2, -1, -1.0, -0.9, 0.1)])
    s = ev.s1_stats(t)
    assert s["n"] == 3
    assert s["pf"] == pytest.approx(2.0 / 2.0)
    assert s["mean_r"] == pytest.approx(0.0)
    # E при издержках × 1.5: валовая 0.3/3 = 0.1, издержки 0.1 → 0.1 − 0.15 = −0.05
    assert s["e15"] == pytest.approx((0.3 - 1.5 * 0.3) / 3)
    assert s["cost_share"] == pytest.approx(0.3 / 0.3)
    assert ev.s1_stats(t.clear())["n"] == 0
    only_wins = table([(1, 0, 1, 1.0, 1.1, 0.1)])
    assert math.isinf(ev.s1_stats(only_wins)["pf"])


def test_splits_calendar_keeps_weekdays_only() -> None:
    sp = ev.Splits(D(2025, 1, 6), D(2025, 1, 31), D(2025, 1, 20))
    cal = sp.is_calendar
    assert len(cal) == 10  # 6–17 января: две рабочие недели
    assert all((d - 1) % 7 < 5 for d in cal)
    assert sp.ho_calendar[0] == D(2025, 1, 20).toordinal()
    assert sp.ho_calendar[-1] == D(2025, 1, 31).toordinal()


def test_daily_matrix_sums_trades_per_day_and_ignores_days_off_the_calendar() -> None:
    mon = D(2025, 1, 6).toordinal()
    sat = D(2025, 1, 11).toordinal()
    tables: ev.Tables = {
        "c": {
            "A": table(
                [(mon, 0, 1, 1.0, 1, 0), (mon, 0, -1, 0.5, 1, 0), (mon + 1, 1, 1, -1.0, 1, 0)]
            ),
            "B": table([(mon + 1, 1, 1, 2.0, 1, 0), (sat, 5, 1, 9.0, 1, 0)]),
        }
    }
    cal = np.arange(mon, mon + 5, dtype=np.int64)
    r, n = ev.daily_matrix(tables, ["c"], cal)
    assert r[:, 0].tolist() == [1.5, 1.0, 0.0, 0.0, 0.0]
    assert n[:, 0].tolist() == [2, 2, 0, 0, 0]


def window(a: D, b: D, c: D) -> WFWindow:
    return WFWindow(a, b, b, c)


def test_walk_forward_takes_the_best_train_sharpe_and_reports_its_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rwf, "WF_MIN_TRADES", 5)
    cal = np.arange(D(2025, 1, 1).toordinal(), D(2025, 3, 1).toordinal(), dtype=np.int64)
    rng = np.random.default_rng(3)
    steady = 0.1 + 0.05 * rng.standard_normal(len(cal))
    noisy = 0.1 * rng.standard_normal(len(cal))
    r = np.column_stack([noisy, steady])
    n = np.ones_like(r)
    w = window(D(2025, 1, 1), D(2025, 2, 1), D(2025, 2, 15))
    (pick,) = ev.walk_forward(r, n, cal, ["noisy", "steady"], [w])
    assert pick.key == "steady"
    tr = (cal >= w.train_start.toordinal()) & (cal < w.train_end.toordinal())
    te = (cal >= w.test_start.toordinal()) & (cal < w.test_end.toordinal())
    assert pick.train_r == pytest.approx(steady[tr].sum())
    assert pick.test_r == pytest.approx(steady[te].sum())
    assert pick.n_train == int(tr.sum())


def test_walk_forward_skips_configs_with_too_few_trades(monkeypatch: pytest.MonkeyPatch) -> None:
    cal = np.arange(D(2025, 1, 1).toordinal(), D(2025, 3, 1).toordinal(), dtype=np.int64)
    r = np.column_stack([np.full(len(cal), 0.5), np.linspace(0, 1, len(cal))])
    n = np.column_stack([np.zeros(len(cal)), np.ones(len(cal))])
    w = window(D(2025, 1, 1), D(2025, 2, 1), D(2025, 2, 15))
    monkeypatch.setattr(rwf, "WF_MIN_TRADES", 5)
    (pick,) = ev.walk_forward(r, n, cal, ["empty", "busy"], [w])
    assert pick.key == "busy"
    n0 = np.zeros_like(n)
    assert ev.walk_forward(r, n0, cal, ["empty", "busy"], [w]) == []


def test_wfe_is_annualised_test_over_annualised_train() -> None:
    w = window(D(2023, 1, 1), D(2025, 1, 1), D(2025, 7, 1))
    p = ev.WFPick(w, "k", 100, train_r=40.0, test_r=10.0, n_test=20)
    expected = (10.0 / w.test_years) / (40.0 / w.train_years)
    assert ev.wfe([p]) == pytest.approx(expected)
    assert math.isnan(ev.wfe([]))
    assert math.isnan(ev.wfe([dataclasses.replace(p, train_r=-1.0)]))


def yearly(rows: dict[int, list[float]]) -> pl.DataFrame:
    out = []
    for year, rs in rows.items():
        day0 = D(year, 3, 3).toordinal()
        out += [(day0 + k, 0, 1, r, r, 0.0) for k, r in enumerate(rs)]
    return table(out)


def test_year_stability_needs_two_thirds_of_years_and_no_dominant_year() -> None:
    ok = yearly({2021: [1.0] * 20 + [-0.5] * 20, 2022: [1.0] * 20 + [-0.5] * 20, 2023: [0.5] * 40})
    passed, by = ev.year_stability(ok)
    assert passed
    assert by["year"].to_list() == [2021, 2022, 2023]

    two_bad = yearly({2021: [-1.0] * 30, 2022: [-1.0] * 30, 2023: [1.0] * 30})
    assert not ev.year_stability(two_bad)[0]

    # прибыль даёт один год: 2021 приносит 90% чистого результата
    one_year = yearly({2021: [1.0] * 30, 2022: [0.05] * 30, 2023: [0.05] * 30})
    assert not ev.year_stability(one_year)[0]

    sparse = yearly({2021: [1.0] * 5, 2022: [1.0] * 5})
    assert not ev.year_stability(sparse)[0]  # ни одного года с ≥ 20 сделок
    assert not ev.year_stability(sparse.clear())[0]


def test_perturbations_move_window_exit_and_buffer_by_ten_and_twenty_percent() -> None:
    cfg = Config("ny30", "breakout", exit_grid()["rr_1_2"], "none", buffer=0.05)
    got = dict(ev.perturbations(cfg))
    assert got["окно ×0.8"].end == "16:54"  # 30 мин × 0.8 = 24
    assert got["окно ×1.2"].end == "17:06"
    assert got["выход ×1.1"].exit.tp_r == pytest.approx(2.2)
    assert got["буфер ×0.9"].buffer == pytest.approx(0.045)
    assert len(got) == 3 * len(ev.SENS_STEPS)
    no_buffer = ev.perturbations(Config("ny30", "fade", exit_grid()["rr_1_1"], "none"))
    assert len(no_buffer) == 2 * len(ev.SENS_STEPS)
    short = dict(ev.perturbations(Config("ny15", "fade", exit_grid()["rr_1_1"], "none")))
    assert all(c.end != "16:45" for k, c in short.items() if k.startswith("окно"))


def test_buckets_split_trades_by_the_authors_weekday_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    mon = D(2025, 1, 6).toordinal()
    cal = np.arange(mon, mon + 5, dtype=np.int64)
    long_t = table(
        [(mon, 0, 1, 1.0, 1, 0), (mon + 1, 1, 1, 2.0, 1, 0), (mon + 3, 3, 1, -3.0, 1, 0)]
    )
    short_t = table(
        [(mon, 0, -1, -4.0, 1, 0), (mon + 3, 3, -1, 5.0, 1, 0), (mon + 4, 4, -1, 6.0, 1, 0)]
    )
    monkeypatch.setattr(ev, "scan_kind", lambda b, cfg, kind: long_t if kind == "long" else short_t)
    bundle = cast(Any, SimpleNamespace(live=lambda cfg: True))
    cfg = Config("ny30", "breakout", exit_grid()["rr_1_1"], "none")
    allowed, forbidden = ev.bucket_stats({"X": bundle}, [cfg], cal)
    # разрешены: пн лонг, вт лонг, чт шорт, пт шорт; запрещены: чт лонг, пн шорт
    assert allowed.r.sum() == pytest.approx(1.0 + 2.0 + 5.0 + 6.0)
    assert forbidden.r.sum() == pytest.approx(-3.0 - 4.0)
    assert allowed.n.sum() == 4
    assert forbidden.n.sum() == 2


def test_finalist_holdout_reports_the_band_instruments_and_opening_date() -> None:
    sp = ev.Splits(D(2025, 1, 6), D(2025, 3, 28), D(2025, 3, 3))
    ho_days = [int(d) for d in sp.ho_calendar]
    rows = [(d, 0, 1, 0.5 if k % 3 else -1.0, 1.0, 0.1) for k, d in enumerate(ho_days)]
    tables: ev.Tables = {"c": {"A": table(rows), "B": table(rows[:5])}}
    rng = np.random.default_rng(1)
    n = 120
    r = rng.choice([0.5, -1.0], size=n).astype(np.float64)
    source = DayStats(
        r,
        np.ones(n),
        (r > 0).astype(np.float64),
        np.where(r > 0, r, 0.0).astype(np.float64),
        np.where(r < 0, r, 0.0).astype(np.float64),
    )
    res = ev.finalist_holdout(tables, "c", source, sp, opened_at=D(2026, 10, 29), n_paths=200)
    assert res.opened_at == D(2026, 10, 29)
    assert res.n_trades == len(rows) + 5
    assert set(res.per_instrument) == {"A", "B"}
    assert res.per_instrument["B"][0] == 5
    assert set(res.band.bands) == {"mean_r", "max_dd", "win_rate", "pf"}
    assert res.stats["n"] == res.n_trades
