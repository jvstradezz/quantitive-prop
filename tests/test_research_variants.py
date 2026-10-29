"""Счётчик вариантов для DSR: id, дневные ряды, Parquet, конфликты (D-028)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from pbl.config import load_firms
from pbl.research.variants import (
    DailyR,
    VariantConflict,
    VariantLog,
    VariantSpec,
    daily_r_from_trades,
    daily_stats,
    refresh_card_counter,
    trading_day,
)
from pbl.schemas import IdeaCard, Trade
from pbl.schemas.firm import ServerTime
from pbl.schemas.strategy import ExitSpec

ON = dt.date(2026, 10, 29)
START, END = dt.date(2025, 1, 6), dt.date(2025, 1, 10)  # Пн–Пт, 5 торговых дней
IDEA = "IDEA-0001"


def _server() -> ServerTime:
    (fn,) = [f for f in load_firms() if f.id == "fundednext"]
    return fn.server_time


def _spec(**over: object) -> VariantSpec:
    base: dict[str, object] = {
        "idea_id": IDEA,
        "family": "range_breakout",
        "instrument": "NAS100",
        "timeframe": "M5",
        "session": "ny_first_30",
        "params": {"buffer": 0.05, "rr": 1.0},
        "exits": [ExitSpec(kind="rr", params={"rr": 1.0})],
        "firm_profile": "fundednext",
    }
    base.update(over)
    return VariantSpec.model_validate(base)


def _trade(day: dt.date, r_net: float, *, hour: int = 15, sample: str = "IS", n: int = 0) -> Trade:
    open_utc = dt.datetime.combine(day, dt.time(hour), dt.UTC)
    return Trade(
        trade_id=f"{day}-{n}",
        strategy_id="s",
        variant_id="v",
        symbol="NAS100",
        side="long",
        open_utc=open_utc,
        close_utc=open_utc + dt.timedelta(minutes=10),
        entry=100.0,
        exit=100.0 + r_net,
        sl=99.0,
        stop_dist=1.0,
        point_value=1.0,
        r_gross=r_net,
        cost_spread_r=0.0,
        cost_comm_r=0.0,
        cost_swap_r=0.0,
        cost_slip_r=0.0,
        r_net=r_net,
        mae_r=-1.0,
        mfe_r=1.0,
        hold_sec=600,
        exit_reason="tp" if r_net > 0 else "sl",
        sample=sample,  # type: ignore[arg-type]
    )


def _daily(r: list[float], start: dt.date = START, end: dt.date = END) -> DailyR:
    days = list(np.busday_offset(np.datetime64(start, "D"), np.arange(len(r)), roll="forward"))
    dates = tuple(d.astype(dt.date) for d in days)
    return DailyR(
        start=start,
        end=end,
        dates=dates,
        r=tuple(r),
        n_trades=len(r),
        win_r=sum(x for x in r if x > 0),
        loss_r=sum(x for x in r if x <= 0),
    )


def test_variant_id_is_stable_and_sensitive() -> None:
    a = _spec()
    assert a.variant_id == _spec(params={"rr": 1.0, "buffer": 0.05}).variant_id
    assert a.variant_id.startswith("V-")
    assert len(a.variant_id) == 18
    others = {
        _spec(version="v2").variant_id,
        _spec(instrument="US30").variant_id,
        _spec(params={"buffer": 0.1, "rr": 1.0}).variant_id,
        _spec(exits=[ExitSpec(kind="rr", params={"rr": 2.0})]).variant_id,
        _spec(session=None).variant_id,
        _spec(firm_profile="fundingpips").variant_id,
    }
    assert a.variant_id not in others
    assert len(others) == 6


def test_trading_day_follows_fn_server_and_weekends_go_to_friday() -> None:
    server = _server()
    # 22:30 UTC в пятницу 2025-01-10 = 00:30 субботы по серверу (GMT+2) → пятница
    assert trading_day(dt.datetime(2025, 1, 10, 22, 30, tzinfo=dt.UTC), server) == dt.date(
        2025, 1, 10
    )
    # 22:30 UTC в воскресенье = 00:30 понедельника по серверу
    assert trading_day(dt.datetime(2025, 1, 12, 22, 30, tzinfo=dt.UTC), server) == dt.date(
        2025, 1, 13
    )
    assert trading_day(dt.datetime(2025, 1, 11, 12, 0, tzinfo=dt.UTC), server) == dt.date(
        2025, 1, 10
    )


def test_daily_r_from_trades_sums_days_and_tracks_profit_factor() -> None:
    trades = [
        _trade(dt.date(2025, 1, 6), 1.0),
        _trade(dt.date(2025, 1, 6), -0.4, n=1),
        _trade(dt.date(2025, 1, 8), -1.0),
        _trade(dt.date(2025, 1, 10), 2.0, sample="WF"),
    ]
    daily = daily_r_from_trades(trades, start=START, end=END, server=_server())
    assert daily.dates == (dt.date(2025, 1, 6), dt.date(2025, 1, 8), dt.date(2025, 1, 10))
    assert daily.r == pytest.approx((0.6, -1.0, 2.0))
    assert (daily.n_trades, daily.win_r) == (4, 3.0)
    assert daily.loss_r == pytest.approx(-1.4)
    assert daily.n_days == 5


def test_holdout_demo_and_out_of_window_trades_are_rejected() -> None:
    server = _server()
    for sample in ("HOLDOUT", "DEMO"):
        with pytest.raises(ValueError, match="только IS и WF"):
            daily_r_from_trades(
                [_trade(dt.date(2025, 1, 6), 1.0, sample=sample)],
                start=START,
                end=END,
                server=server,
            )
    with pytest.raises(ValueError, match="вне окна"):
        daily_r_from_trades(
            [_trade(dt.date(2025, 1, 13), 1.0)], start=START, end=END, server=server
        )


def test_daily_validation() -> None:
    with pytest.raises(ValueError, match="end раньше"):
        DailyR(END, START, (), (), 0, 0.0, 0.0)
    with pytest.raises(ValueError, match="Пн–Пт"):
        DailyR(START, dt.date(2025, 1, 12), (dt.date(2025, 1, 11),), (1.0,), 1, 1.0, 0.0)
    with pytest.raises(ValueError, match="вне окна"):
        DailyR(START, END, (dt.date(2025, 1, 13),), (1.0,), 1, 1.0, 0.0)


def test_daily_stats_match_hand_calculation() -> None:
    stats = daily_stats(_daily([1.0, -1.0, 0.0, 0.0, 2.0]))
    # mean 0.4; std(ddof=1) = sqrt(5.2 / 4)
    assert stats["sharpe_daily"] == pytest.approx(0.4 / np.sqrt(5.2 / 4))
    z = (np.array([1, -1, 0, 0, 2]) - 0.4) / np.sqrt(5.2 / 5)
    assert stats["skew"] == pytest.approx(float(np.mean(z**3)))
    assert stats["kurt"] == pytest.approx(float(np.mean(z**4)))
    flat = daily_stats(_daily([]))
    assert flat == {"sharpe_daily": 0.0, "skew": 0.0, "kurt": 3.0}


def test_daily_stats_zero_days_are_in_the_calendar() -> None:
    """Дни без сделок — нули: Sharpe считается по календарю, а не по дням со сделками."""
    sparse = daily_stats(_daily([1.0], END, END))  # одна сделка в окне из одного дня
    assert sparse["sharpe_daily"] == 0.0
    long = daily_stats(_daily([1.0], START, dt.date(2025, 1, 31)))
    dense = daily_stats(_daily([1.0], START, START))
    assert long["sharpe_daily"] < 1.0
    assert dense["sharpe_daily"] == 0.0


def test_normal_returns_have_kurtosis_three() -> None:
    end = START + dt.timedelta(weeks=1000)
    days = int(np.busday_count(np.datetime64(START, "D"), np.datetime64(end, "D")))
    r = np.random.default_rng(1).normal(0.0, 1.0, size=days)
    dates = tuple(
        d.astype(dt.date)
        for d in np.busday_offset(np.datetime64(START, "D"), np.arange(days), roll="forward")
    )
    daily = DailyR(START, end - dt.timedelta(days=3), dates, tuple(map(float, r)), days, 1.0, -1.0)
    stats = daily_stats(daily)
    assert stats["kurt"] == pytest.approx(3.0, abs=0.35)
    assert stats["skew"] == pytest.approx(0.0, abs=0.15)


def test_log_roundtrip_counts_and_deterministic_bytes(tmp_path: Path) -> None:
    log = VariantLog.for_idea(IDEA, tmp_path)
    assert len(log) == 0
    assert not log.path.exists()
    assert log.add(_spec(), _daily([1.0, -1.0, 0.0, 0.0, 2.0]), run_id="r1", on=ON)
    assert log.add(_spec(instrument="US30"), _daily([0.5, 0.5]), run_id="r1", on=ON)
    assert log.add(_spec(family="range_fade"), _daily([1.0]), run_id="r1", on=ON, keep_series=False)
    log.save()
    assert log.path == tmp_path / "variants" / f"{IDEA}.parquet"

    again = VariantLog.for_idea(IDEA, tmp_path)
    assert len(again) == 3
    assert again.count(family="range_breakout") == 2
    assert again.count(instrument="US30") == 1
    assert again.count(family="range_fade", instrument="NAS100") == 1
    assert again.ids() == sorted(again.ids()), "строки отсортированы по id"
    row = again.frame().filter(again.frame()["variant_id"] == _spec().variant_id).row(0, named=True)
    assert (row["n_trades"], row["n_days"]) == (5, 5)
    assert row["net_r"] == pytest.approx(2.0)
    assert row["pf"] == pytest.approx(3.0 / 1.0)
    assert row["is_start"] == START
    assert row["logged_on"] == ON

    first = log.path.read_bytes()
    again.save()
    assert log.path.read_bytes() == first, "повторная запись даёт тот же файл"
    assert not list(log.path.parent.glob("*.tmp"))


def test_same_variant_is_idempotent_and_changed_result_conflicts(tmp_path: Path) -> None:
    log = VariantLog.for_idea(IDEA, tmp_path)
    assert log.add(_spec(), _daily([1.0, -1.0]), run_id="r1", on=ON)
    assert not log.add(_spec(), _daily([1.0, -1.0]), run_id="r2", on=ON)
    assert len(log) == 1
    with pytest.raises(VariantConflict, match="version"):
        log.add(_spec(), _daily([1.0, -2.0]), run_id="r3", on=ON)
    assert log.add(_spec(version="v2"), _daily([1.0, -2.0]), run_id="r3", on=ON)
    assert len(log) == 2


def test_foreign_idea_is_rejected(tmp_path: Path) -> None:
    log = VariantLog.for_idea(IDEA, tmp_path)
    with pytest.raises(ValueError, match="IDEA-0002"):
        log.add(_spec(idea_id="IDEA-0002"), _daily([1.0]), run_id="r", on=ON)


def test_returns_matrix_aligns_windows_and_marks_outside_as_nan(tmp_path: Path) -> None:
    log = VariantLog.for_idea(IDEA, tmp_path)
    wide = dt.date(2025, 1, 17)
    a, b = _spec(), _spec(instrument="US30")
    log.add(a, _daily([1.0, -1.0, 0.0, 0.0, 2.0]), run_id="r", on=ON)
    log.add(b, _daily([0.5, 0.0, 0.0, 3.0], START, wide), run_id="r", on=ON)
    log.save()
    m = VariantLog.for_idea(IDEA, tmp_path).returns_matrix([a.variant_id, b.variant_id])
    assert m.ids == [a.variant_id, b.variant_id]
    assert len(m.dates) == 10
    assert (m.dates[0], m.dates[-1]) == (START, wide)
    assert m.values.shape == (10, 2)
    np.testing.assert_allclose(m.values[:5, 0], [1.0, -1.0, 0.0, 0.0, 2.0])
    assert np.isnan(m.values[5:, 0]).all()
    np.testing.assert_allclose(m.values[:4, 1], [0.5, 0.0, 0.0, 3.0])
    assert (m.values[4:, 1] == 0.0).all()


def test_returns_matrix_needs_series_and_known_ids(tmp_path: Path) -> None:
    log = VariantLog.for_idea(IDEA, tmp_path)
    log.add(_spec(), _daily([1.0]), run_id="r", on=ON, keep_series=False)
    with pytest.raises(ValueError, match="keep_series"):
        log.returns_matrix()
    with pytest.raises(KeyError):
        log.returns_matrix(["V-nope"])
    with pytest.raises(ValueError, match="нет вариантов"):
        VariantLog.for_idea("IDEA-0002", tmp_path).returns_matrix()


def test_refresh_card_counter(tmp_path: Path) -> None:
    from pbl.research import registry

    card: IdeaCard = registry.load_card(next(iter(registry.card_paths())))
    log = VariantLog.for_idea(card.id, tmp_path)
    log.add(_spec(), _daily([1.0]), run_id="r", on=ON)
    assert refresh_card_counter(card, log).variants_tested == 1
    assert card.variants_tested == 0
