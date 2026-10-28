"""Время сервера, сделки → дни, бутстрэп, стрессы, CSV (MONTE_CARLO §2, §5, §6)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from pbl.config import load_firms
from pbl.prop.bootstrap import block_length, politis_white_block, stationary_indices, wilson
from pbl.prop.days import DayTable, LotSpec, build_day_table, trades_to_arrays
from pbl.prop.io import read_trades_csv, write_trades_csv
from pbl.prop.servertime import server_date, to_server, utc_offset_hours
from pbl.prop.stress import costs_x, skip_trades, winrate_minus, worst_period_days
from pbl.prop.synthetic import synthetic_trades
from pbl.schemas import ServerTime, Trade

FN = next(f for f in load_firms() if f.id == "fundednext").server_time
EU = ServerTime(winter_utc_offset=2, summer_utc_offset=3, dst_calendar="eu")


def utc(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> dt.datetime:
    return dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.UTC)


# --- время сервера (D-006, D-019) ---------------------------------------------------------


def test_fn_server_matches_mt5_observation_2026_10_28() -> None:
    # O-10: «Обзор рынка» FN показал 20:15:41 при 17:15 UTC
    assert to_server(utc(2026, 10, 28, 17, 15, 41), FN) == dt.datetime(2026, 10, 28, 20, 15, 41)
    assert utc_offset_hours(utc(2026, 10, 28, 17, 15), EU) == 2  # ЕС уже перешёл 25.10


def test_us_dst_switch_moves_day_boundary() -> None:
    # переход США: 1 ноября 2026, 06:00 UTC
    assert utc_offset_hours(utc(2026, 11, 1, 5, 59), FN) == 3
    assert utc_offset_hours(utc(2026, 11, 1, 6, 0), FN) == 2
    # летом сутки сервера начинаются в 21:00 UTC, зимой — в 22:00 UTC
    assert server_date(utc(2026, 10, 28, 21, 30), FN) == dt.date(2026, 10, 29)
    assert server_date(utc(2026, 11, 3, 21, 30), FN) == dt.date(2026, 11, 3)
    assert server_date(utc(2026, 11, 3, 22, 0), FN) == dt.date(2026, 11, 4)
    # весна: второе воскресенье марта, 07:00 UTC
    assert utc_offset_hours(utc(2026, 3, 8, 6, 59), FN) == 2
    assert utc_offset_hours(utc(2026, 3, 8, 7, 0), FN) == 3


def test_fixed_offset_server() -> None:
    fp = ServerTime(winter_utc_offset=3, summer_utc_offset=3, dst_calendar=None)
    assert server_date(utc(2026, 7, 1, 21, 0), fp) == dt.date(2026, 7, 2)


# --- сделки → дни -------------------------------------------------------------------------


def trade(
    i: int,
    open_utc: dt.datetime,
    minutes: int,
    r_gross: float,
    *,
    mae: float,
    mfe: float,
    cost: float = 0.0,
    exit_reason: str = "sl",
    stop_dist: float = 50.0,
    news: bool = False,
) -> Trade:
    return Trade.model_validate(
        {
            "trade_id": f"t{i}",
            "strategy_id": "s",
            "variant_id": "v",
            "symbol": "NAS100",
            "side": "long",
            "open_utc": open_utc,
            "close_utc": open_utc + dt.timedelta(minutes=minutes),
            "entry": 100.0,
            "exit": 101.0,
            "sl": 99.0,
            "stop_dist": stop_dist,
            "point_value": 10.0,
            "r_gross": r_gross,
            "cost_spread_r": cost,
            "cost_comm_r": 0.0,
            "cost_swap_r": 0.0,
            "cost_slip_r": 0.0,
            "r_net": r_gross - cost,
            "mae_r": mae,
            "mfe_r": mfe,
            "hold_sec": minutes * 60,
            "exit_reason": exit_reason,
            "news_flag": news,
            "sample": "WF",
        }
    )


def day_path(t: DayTable, d: int) -> list[float]:
    lo, hi = t.path_off[d], t.path_off[d + 1]
    return [round(float(x), 6) for x in t.path_val[lo:hi]]


def test_trade_paths_follow_exit_type() -> None:
    mon = utc(2026, 10, 26, 14, 0)
    trades = [
        trade(0, mon, 30, -1.0, mae=-1.0, mfe=0.6, cost=0.1),  # стоп: MFE раньше стопа
        trade(1, mon + dt.timedelta(days=1), 30, 1.5, mae=-0.4, mfe=1.5, exit_reason="tp"),
        trade(2, mon + dt.timedelta(days=2), 30, 0.3, mae=-0.7, mfe=1.2, exit_reason="time"),
    ]
    t = build_day_table(trades_to_arrays(trades, FN))
    assert day_path(t, 0) == [0.0, 0.5, -1.1]
    assert day_path(t, 1) == [0.0, -0.4, 1.5]
    assert day_path(t, 2) == [0.0, 1.2, -0.7, 0.3]
    assert t.db.tolist() == pytest.approx([-1.1, 1.5, 0.3])
    assert t.n_opened.tolist() == [1, 1, 1]


def test_overlapping_trades_sum_worst_points() -> None:
    mon = utc(2026, 10, 26, 14, 0)
    trades = [
        trade(0, mon, 60, -1.0, mae=-1.0, mfe=0.2),
        trade(1, mon + dt.timedelta(minutes=30), 60, 1.0, mae=-0.5, mfe=1.0, exit_reason="tp"),
        trade(2, mon + dt.timedelta(hours=3), 30, 1.0, mae=-0.2, mfe=1.0, exit_reason="tp"),
    ]
    t = build_day_table(trades_to_arrays(trades, FN))
    # группа 1: низ −1.5 (оба MAE разом), верх — лучший MFE одной сделки (1.0 > итог 0)
    assert day_path(t, 0) == [0.0, 1.0, -1.5, 0.0, -0.2, 1.0]
    assert t.risk_open_max[0] == 2.0
    assert t.b_max[0] == pytest.approx(1.0)
    assert t.single_loss[0] == pytest.approx(1.0)


def test_calendar_has_empty_weekdays_and_folds_weekend() -> None:
    trades = [
        trade(0, utc(2026, 10, 26, 14), 30, 1.0, mae=0.0, mfe=1.0, exit_reason="tp"),
        trade(1, utc(2026, 10, 31, 12), 30, -1.0, mae=-1.0, mfe=0.0),  # суббота
        trade(2, utc(2026, 11, 4, 14), 30, 1.0, mae=0.0, mfe=1.0, exit_reason="tp"),
    ]
    t = build_day_table(trades_to_arrays(trades, FN))
    assert t.n_days == 8  # пн 26.10 … ср 04.11, только Пн–Пт
    assert t.n_opened.tolist() == [1, 0, 0, 0, 1, 0, 0, 1]
    assert t.weekend_folded == 1
    assert any("выходного" in w for w in t.warnings)


def test_kappa_rounds_lots_down_with_min_lot() -> None:
    trades = [trade(0, utc(2026, 10, 26, 14), 30, 1.0, mae=0.0, mfe=1.0, stop_dist=50.0)]
    spec = {"NAS100": LotSpec(min_lot=0.1, lot_step=0.1)}
    t = build_day_table(trades_to_arrays(trades, FN, spec))
    # $ на 1 лот при стопе 50 пунктов: 500; риск $260 → 0.52 лота → 0.5 → κ = 250/260
    k, share = t.kappa(260.0)
    assert k[0] == pytest.approx(250 / 260)
    assert share == 0.0
    # риск $20 → 0.04 лота → мин. лот 0.1 → κ = 50/20 = 2.5 > 1.5
    k, share = t.kappa(20.0)
    assert k[0] == pytest.approx(2.5)
    assert share == 1.0
    k, _ = build_day_table(trades_to_arrays(trades, FN)).kappa(20.0)
    assert k[0] == 1.0  # без спеки лот не округляем


def test_take_rebuilds_csr_and_trades() -> None:
    trades = synthetic_trades(n_days=30, seed=2)
    t = build_day_table(trades_to_arrays(trades, FN))
    idx = np.asarray([3, 3, 0, 10], dtype=np.int64)
    s = t.take(idx)
    for j, d in enumerate(idx):
        assert day_path(s, j) == day_path(t, int(d))
    assert s.trade_day.shape[0] == int(t.n_opened[idx].sum())


# --- бутстрэп -----------------------------------------------------------------------------


def test_stationary_indices_runs_have_mean_length_l() -> None:
    idx = stationary_indices(np.random.default_rng(0), 1000, 400, 300, 5.0)
    cont = (np.diff(idx, axis=1) % 1000) == 1
    p_continue = cont.mean()
    assert p_continue == pytest.approx(1 - 1 / 5 + 1 / 5 / 1000, abs=0.01)
    assert idx.min() >= 0
    assert idx.max() < 1000


def test_block_length_reacts_to_autocorrelation() -> None:
    rng = np.random.default_rng(1)
    iid = rng.normal(size=1000)
    ar = np.zeros(1000)
    for i in range(1, 1000):
        ar[i] = 0.7 * ar[i - 1] + rng.normal()
    assert politis_white_block(ar) > politis_white_block(iid)
    assert block_length(iid) == 3.0
    assert 3.0 <= block_length(ar) <= 20.0
    assert block_length(iid, override=7) == 7.0


def test_wilson_interval() -> None:
    lo, hi = wilson(5000, 10_000)
    assert lo < 0.5 < hi
    assert hi - lo == pytest.approx(0.0196, abs=5e-4)  # ≈ ±1 п.п. при N = 10 000


# --- стрессы ------------------------------------------------------------------------------


def test_stress_transforms() -> None:
    ta = trades_to_arrays(synthetic_trades(n_days=400, win_rate=0.5, seed=5), FN)
    c = costs_x(ta, 1.5)
    assert np.allclose(ta.r_net - c.r_net, 0.5 * (ta.cost_spread + ta.cost_slip))
    rng = np.random.default_rng(0)
    w = winrate_minus(ta, rng, 0.05)
    assert (ta.r_net > 0).sum() - (w.r_net > 0).sum() == round(0.05 * len(ta))
    assert (w.mfe[w.r_net < 0] <= ta.mfe[w.r_net < 0]).all()
    s = skip_trades(ta, rng, 0.1)
    assert 0.8 * len(ta) < len(s) < len(ta)


def test_worst_period() -> None:
    t = build_day_table(trades_to_arrays(synthetic_trades(n_days=1300, seed=6), FN))
    _, label = worst_period_days(t)
    assert label.startswith("годы")  # ≥ 4 календарных лет → два худших года
    t2 = build_day_table(trades_to_arrays(synthetic_trades(n_days=500, seed=6), FN))
    days2, label2 = worst_period_days(t2)
    assert label2.startswith("окно")
    span = t2.dates[days2[-1]] - t2.dates[days2[0]]
    assert 360 <= span <= 365
    assert t2.db[days2].sum() <= t2.db[: days2.shape[0]].sum()


# --- CSV ----------------------------------------------------------------------------------


def test_csv_roundtrip(tmp_path: Path) -> None:
    trades = synthetic_trades(n_days=20, news_prob=0.3, seed=1)
    path = tmp_path / "t.csv"
    write_trades_csv(trades, path)
    assert read_trades_csv(path) == trades
