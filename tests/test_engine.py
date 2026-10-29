"""Движок: ручные примеры исполнения, издержки, правила фирмы, эквити, скан (ARCHITECTURE §9).

Цены в примерах — mid; спред бара 1.0 (bid = mid − 0.5, ask = mid + 0.5), проскальзывание 0,
если не сказано иное. Ожидаемые R посчитаны руками в комментариях.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest

from pbl.config import load_firms, load_instruments
from pbl.engine.backtest import BacktestResult, Exits, RunMeta, backtest, run_kernel, scan
from pbl.engine.market import FirmTradeProfile, Market, prepare_market
from pbl.instruments.costs import CostModel, instrument_costs, swap_nights
from pbl.instruments.servertime import server_clock, server_date
from pbl.schemas import InstrumentSpec, Trade
from pbl.strategies.base import Signals

FN = next(f for f in load_firms() if f.id == "fundednext")
SERVER = FN.server_time
SPECS = {s.symbol: s for s in load_instruments() if s.firm == "fundednext"}
NAS = SPECS["NAS100"]
EXACT = CostModel(spread="data", slip_points=0.0)
# вторник 16.09.2025, 14:00 UTC = 17:00 сервера FN (лето, ET + 7)
T0 = dt.datetime(2025, 9, 16, 14, 0, tzinfo=dt.UTC)

Row = tuple[float, float, float, float]


def make_bars(
    rows: Sequence[Row],
    spread: float = 1.0,
    start: dt.datetime = T0,
    times: Sequence[dt.datetime] | None = None,
) -> pl.DataFrame:
    a = np.asarray(rows, dtype=np.float64)
    ts = (
        list(times)
        if times is not None
        else [start + dt.timedelta(minutes=i) for i in range(len(rows))]
    )
    cols: dict[str, object] = {"ts": ts}
    for k, x in enumerate("ohlc"):
        cols[f"bid_{x}"] = a[:, k] - spread / 2
        cols[f"ask_{x}"] = a[:, k] + spread / 2
    cols["spread_mean"] = np.full(len(rows), spread)
    return pl.DataFrame(cols).with_columns(pl.col("ts").cast(pl.Datetime("ms", "UTC")))


def mkt(
    rows: Sequence[Row],
    spec: InstrumentSpec = NAS,
    cost: CostModel = EXACT,
    firm: FirmTradeProfile | None = None,
    **kw: object,
) -> Market:
    return prepare_market(make_bars(rows, **kw), spec, SERVER, cost, firm)  # type: ignore[arg-type]


def run(m: Market, sig: Signals, exits: Exits | None = None, **kw: object) -> BacktestResult:
    return backtest(m, sig, exits or Exits(), RunMeta("t"), **kw)  # type: ignore[arg-type]


def one(res: BacktestResult) -> Trade:
    assert len(res.trades) == 1, res.signal_status
    return res.trades[0]


LONG10 = Signals.build([0], [1], sl_dist=[10.0], tp_r=1.0)


# --- исполнение и выходы ------------------------------------------------------------------


def test_long_market_take_profit_hand_numbers() -> None:
    # вход: mid 100 → ask 100.5; стоп 90.5, тейк 110.5 (bid) = mid 111
    m = mkt([(100, 101, 99, 100), (100, 102, 99.6, 101), (101, 111.5, 100.5, 111)])
    t = one(run(m, LONG10))
    assert (t.entry, t.sl, t.tp, t.exit) == (100.5, 90.5, 110.5, 110.5)
    assert t.exit_reason == "tp"
    assert t.r_gross == pytest.approx(1.1)  # (111 − 100) / 10
    assert t.cost_spread_r == pytest.approx(0.1)  # (0.5 + 0.5) / 10
    assert t.r_net == pytest.approx(1.0)
    assert t.mae_r == pytest.approx(-0.04)  # mid 99.6
    assert t.mfe_r == pytest.approx(1.1)
    assert t.stop_dist == pytest.approx(1000)  # 10 / 0.01
    assert t.point_value == 0.1
    assert t.open_utc == T0 + dt.timedelta(minutes=1)


def test_stop_and_take_in_one_bar_is_stop() -> None:
    m = mkt([(100, 101, 99, 100), (101, 112, 89, 100)])
    t = one(run(m, LONG10))
    assert t.exit_reason == "sl"
    assert t.r_gross == pytest.approx(-0.9)  # стоп по bid 90.5 = mid 91
    assert t.r_net == pytest.approx(-1.0)
    assert t.hold_sec == 0


def test_gap_through_stop_fills_at_open() -> None:
    m = mkt([(100, 101, 99, 100), (100, 101, 99.6, 100), (85, 86, 84, 85)])
    t = one(run(m, LONG10))
    assert t.exit == pytest.approx(84.5)
    assert t.r_net == pytest.approx(-1.6)  # (85 − 100) / 10 − 0.1


def test_gap_through_take_fills_at_open_without_slippage() -> None:
    slip = CostModel(spread="data", slip_points=10)  # 0.1 цены
    m = mkt([(100, 101, 99, 100), (100, 101, 99.6, 100), (115, 116, 114, 115)], cost=slip)
    t = one(run(m, LONG10))
    assert t.entry == pytest.approx(100.6)
    assert t.exit_reason == "tp"
    assert t.cost_slip_r == pytest.approx(0.01)  # только вход
    assert t.r_gross == pytest.approx(1.5)


def test_short_stop_triggers_on_ask() -> None:
    sig = Signals.build([0], [-1], sl_dist=[10.0])
    # вход: bid 99.5, стоп 109.5 по ask = mid 109
    near = mkt([(100, 101, 99, 100), (100, 108.9, 99, 108)])
    assert run(near, sig).trades[0].exit_reason == "time"  # данные кончились, стоп цел
    hit = mkt([(100, 101, 99, 100), (100, 109.2, 99, 108)])
    t = one(run(hit, sig))
    assert (t.side, t.exit_reason) == ("short", "sl")
    assert t.r_gross == pytest.approx(-0.9)
    assert t.r_net == pytest.approx(-1.0)


def test_buy_stop_entry_and_worst_case_entry_bar() -> None:
    sig = Signals.build([0], [1], [95.0], order="stop", level=[105.0])
    # buy stop по ask 105 = mid 104.5; стоп 95 по bid = mid 95.5
    m = mkt([(100, 101, 99, 100), (100, 106, 99, 105.5)])
    t = one(run(m, sig))
    assert t.entry == pytest.approx(105.0)
    assert t.stop_dist == pytest.approx(1000)
    # минимум бара входа ниже стопа: порядок неизвестен → стоп в баре входа
    t2 = one(run(mkt([(100, 101, 99, 100), (100, 106, 95, 105)]), sig))
    assert (t2.exit_reason, t2.hold_sec) == ("sl", 0)
    assert t2.r_net == pytest.approx(-1.0)
    # гэп выше уровня: исполнение по открытию
    t3 = one(run(mkt([(100, 101, 99, 100), (107, 108, 106.5, 107)]), sig))
    assert t3.entry == pytest.approx(107.5)


def test_limit_entry_has_no_slippage() -> None:
    sig = Signals.build([0], [1], [90.0], order="limit", level=[98.0])
    m = mkt([(100, 101, 99, 100), (100, 100.5, 97, 98)], cost=CostModel("data", slip_points=10))
    t = one(run(m, sig))
    assert t.entry == pytest.approx(98.0)
    assert t.cost_slip_r == pytest.approx(0.1 / 8)  # только выход по концу данных


def test_pending_order_expires() -> None:
    expire = int((T0 + dt.timedelta(minutes=2)).timestamp())
    sig = Signals.build([0], [1], [90.0], order="stop", level=[105.0], expire_s=[expire])
    m = mkt([(100, 101, 99, 100), (100, 101, 99, 100), (100, 106, 99, 105)])
    assert run(m, sig).signal_status["no_fill"] == 1


def test_market_order_respects_expiry_after_data_gap() -> None:
    times = [T0, T0 + dt.timedelta(hours=3)]
    m = mkt([(100, 101, 99, 100)] * 2, times=times)
    late = Signals.build([0], [1], sl_dist=[10.0], expire_s=[int(T0.timestamp()) + 120])
    assert run(m, late).signal_status["no_fill"] == 1
    assert len(run(m, LONG10).trades) == 1  # без срока — исполнится после разрыва


def test_oco_cancels_other_leg() -> None:
    sig = Signals.build(
        [0, 0], [1, -1], [95.0, 105.0], order="stop", level=[104.0, 96.0], oco=[0, 0]
    )
    rows = [(100, 101, 99, 100), (100, 104.6, 99, 104), (104, 104, 95.2, 96), (96, 97, 95, 96)]
    res = run(mkt(rows), sig, max_positions=2)
    assert res.signal_status["oco_cancelled"] == 1
    assert [t.side for t in res.trades] == ["long"]


def test_breakeven_moves_stop_from_next_bar() -> None:
    sig = Signals.build([0], [1], sl_dist=[10.0])
    rows = [(100, 101, 99, 100), (100, 106.5, 99.8, 106), (106, 106.2, 100.9, 101)]
    t = one(run(mkt(rows), sig, Exits(be_at_r=0.5)))
    # лучший bid 106 → +0.55R; стоп → 100.5 (= вход по ask), срабатывает в баре 2
    assert (t.exit_reason, t.mods) == ("be", 1)
    assert t.r_net == pytest.approx(0.0)


def test_trailing_stop_hand_example() -> None:
    sig = Signals.build([0], [1], sl_dist=[10.0])
    rows = [
        (100, 101, 99, 100),
        (100, 112, 100, 111),  # лучший bid 111.5 → +1.1R, стоп → 101.5
        (111, 115, 110, 114),  # лучший bid 114.5, стоп → 104.5
        (114, 114.5, 104, 105),  # bid 103.5 ≤ 104.5 → выход по трейлингу, mid 105
    ]
    t = one(run(mkt(rows), sig, Exits(trail_start_r=1.0, trail_dist_r=1.0)))
    assert (t.exit_reason, t.mods) == ("trail", 2)
    assert t.r_gross == pytest.approx(0.5)
    assert t.r_net == pytest.approx(0.4)


def test_trailing_step_throttles_modifications() -> None:
    sig = Signals.build([0], [1], sl_dist=[10.0])
    rows: list[Row] = [(100, 101, 99, 100), (100, 106, 100, 106)]
    rows += [(106 + i * 0.2, 107 + i * 0.2, 105.5, 107 + i * 0.2) for i in range(1, 11)]
    t = one(run(mkt(rows), sig, Exits(trail_start_r=0.5, trail_dist_r=1.0, trail_step_r=0.5)))
    # бар 1: +0.5R → стоп 95.5; дальше рост на 0.02R за бар — меньше шага 0.5R
    assert t.mods == 1


def test_partial_exit_then_take_and_then_stop() -> None:
    sig = Signals.build([0], [1], sl_dist=[10.0], tp_r=2.0)
    ex = Exits(partial_r=1.0, partial_frac=0.5)
    up = [(100, 101, 99, 100), (100, 111.2, 99.5, 111), (111, 121.5, 110, 121)]
    t = one(run(mkt(up), sig, ex))
    assert t.exit_reason == "tp"
    assert t.r_net == pytest.approx(1.5)  # 0.5 · 1R + 0.5 · 2R
    assert t.mods == 1
    down = [(100, 101, 99, 100), (100, 111.2, 99.5, 111), (111, 111, 89, 90)]
    t2 = one(run(mkt(down), sig, ex))
    assert t2.r_net == pytest.approx(0.0)  # 0.5 · 1R − 0.5 · 1R


def test_time_exit_and_session_close() -> None:
    rows = [(100, 101, 99, 100)] * 6
    sig = Signals.build([0], [1], sl_dist=[10.0])
    t = one(run(mkt(rows), sig, Exits(max_hold_min=2)))
    assert (t.exit_reason, t.hold_sec) == ("time", 120)
    # 14:00 UTC = 17:00 сервера: закрыть в 17:03
    t2 = one(run(mkt(rows), sig, Exits(close_at="17:03")))
    assert t2.exit_reason == "eod"
    assert t2.close_utc == T0 + dt.timedelta(minutes=3)


def test_no_stop_outside_instrument_session() -> None:
    # NAS100 торгуется с 01:15 сервера; у Dukascopy тики есть и ночью
    tue = dt.datetime(2025, 9, 16, 20, 58, tzinfo=dt.UTC)  # 23:58 сервера
    times = [tue, tue + dt.timedelta(minutes=1), tue + dt.timedelta(minutes=32)]
    times.append(dt.datetime(2025, 9, 16, 22, 15, tzinfo=dt.UTC))  # ср 01:15 сервера
    rows = [(100, 101, 99, 100), (100, 101, 99, 100), (100, 100, 80, 85), (99, 100, 98, 99)]
    m = mkt(rows, times=times)
    assert not m.live[2]
    t = one(run(m, LONG10))
    assert t.exit_reason == "time"  # ночной провал ниже стопа не исполнил стоп
    assert t.mae_r == pytest.approx(-0.2)  # по бару 01:15: mid 98
    # провал продолжился к открытию сессии — стоп по цене открытия
    rows[3] = (85, 86, 84, 85)
    gap = one(run(mkt(rows, times=times), LONG10))
    assert gap.exit_reason == "sl"
    assert gap.exit == pytest.approx(84.5)


# --- правила фирмы ------------------------------------------------------------------------


def test_settlement_window_and_session_block_entries() -> None:
    # EURUSD: сессия 00:15–23:55 сервера; окно FN 00:00–02:00 — без входов
    start = dt.datetime(2025, 9, 15, 22, 0, tzinfo=dt.UTC)  # 01:00 сервера вт
    times = [start, start + dt.timedelta(minutes=1), start + dt.timedelta(hours=1, minutes=1)]
    rows = [(1.17, 1.1701, 1.1699, 1.17)] * 3
    firm = FirmTradeProfile.from_rules(next(iter(FN.programs)).trade_rules, funded=False)
    m = mkt(rows, SPECS["EURUSD"], firm=firm, times=times, spread=0.0001)
    assert not m.entry_ok[1]
    assert m.entry_ok[2]
    sig = Signals.build([0, 1], [1, 1], sl_dist=[0.002, 0.002])
    assert run(m, sig).signal_status["no_fill"] == 1
    # NAS100: торговля с 01:15 сервера
    nas = mkt(rows, firm=firm, times=times)
    assert not nas.entry_ok[1]
    assert nas.entry_ok[2]


def test_max_positions_and_trades_per_day() -> None:
    rows = [(100, 101, 99, 100)] * 8
    firm = FirmTradeProfile(max_trades_day=2)
    sig = Signals.build([0, 1, 2], [1, 1, 1], sl_dist=[10.0] * 3)
    res = run(mkt(rows, firm=firm), sig, Exits(max_hold_min=0.5), max_positions=1)
    # выход в открытии следующего бара: бар выхода занят → второй сигнал «busy»
    assert res.signal_status["busy"] == 1
    res3 = run(mkt(rows, firm=firm), sig, max_positions=5)
    assert (len(res3.trades), res3.signal_status["day_cap"]) == (2, 1)


def test_weekend_close_for_funded_without_weekend_holding() -> None:
    fri = dt.datetime(2025, 9, 19, 20, 40, tzinfo=dt.UTC)  # пт 23:40 сервера
    times = [fri + dt.timedelta(minutes=i) for i in range(8)]
    firm = FirmTradeProfile(weekend_close=True)
    m = mkt([(100, 101, 99, 100)] * 8, firm=firm, times=times)
    sig = Signals.build([0, 6], [1, 1], sl_dist=[10.0, 10.0])
    res = run(m, sig, max_positions=2)
    assert [t.exit_reason for t in res.trades] == ["friday_close"]
    assert res.trades[0].close_utc == fri + dt.timedelta(minutes=5)  # 23:45
    assert res.signal_status["no_fill"] == 1


# --- издержки -----------------------------------------------------------------------------


def test_swap_with_triple_day_hand_numbers() -> None:
    # NAS100: лонг −372.912 п. за ночь, тройной — пятница. Чт → пн: 1 + 3 = 4 ночи
    thu = dt.datetime(2025, 9, 18, 14, 0, tzinfo=dt.UTC)
    times = [thu, thu + dt.timedelta(minutes=1), dt.datetime(2025, 9, 22, 14, 0, tzinfo=dt.UTC)]
    rows = [(100, 101, 99, 100)] * 3
    m = mkt(rows, times=times)
    long_t = one(run(m, Signals.build([0], [1], sl_dist=[10.0])))
    assert long_t.cost_swap_r == pytest.approx(372.912 * 4 / 1000)
    short_t = one(run(m, Signals.build([0], [-1], sl_dist=[10.0])))
    assert short_t.cost_swap_r == pytest.approx(57.6 * 4 / 1000)


def test_swap_nights_calendar() -> None:
    wed = dt.date(2025, 9, 17).toordinal()
    assert swap_nights(wed, wed + 1, triple_weekday=2) == 3  # ночь со среды — тройная (FX)
    assert swap_nights(wed + 2, wed + 5, triple_weekday=2) == 1  # пт → пн: только пятница
    assert swap_nights(wed + 2, wed + 5, triple_weekday=4) == 3  # индексы: пятница ×3
    assert swap_nights(wed, wed, triple_weekday=4) == 0


def test_commission_program_rate_and_turnover() -> None:
    prog = next(p for p in FN.programs if p.id == "fn-stellar-2step")
    eur = SPECS["EURUSD"]
    sig = Signals.build([0], [1], sl_dist=[0.002])  # 200 пунктов по $1
    rows = [(1.17, 1.1701, 1.1699, 1.17)] * 3
    spec_rate = one(run(mkt(rows, eur, spread=0.0), sig))
    assert spec_rate.cost_comm_r == pytest.approx(7 / 200)
    m = prepare_market(make_bars(rows, spread=0.0), eur, SERVER, EXACT, None, prog.commission)
    assert one(run(m, sig)).cost_comm_r == pytest.approx(5 / 200)
    # XAUUSD: 0.0016% оборота: 100 унций × 2000 × 1.6e-5 = $3.2 на лот; стоп 10.0 = 1000 п. по $1
    xau = SPECS["XAUUSD"]
    mx = prepare_market(
        make_bars([(2000, 2001, 1999, 2000)] * 3, spread=0.0),
        xau,
        SERVER,
        EXACT,
        None,
        prog.commission,
    )
    t = one(run(mx, Signals.build([0], [1], sl_dist=[10.0])))
    assert t.cost_comm_r == pytest.approx(3.2 / 1000)
    c = instrument_costs(xau)
    assert c.comm_sides == 1  # O-19: по спеке — только вход


def test_non_usd_profit_currency() -> None:
    jpy = SPECS["USDJPY"]
    rows = [(150.0, 150.1, 149.9, 150.0)] * 3
    t = one(run(mkt(rows, jpy, spread=0.0), Signals.build([0], [1], sl_dist=[0.2])))
    assert t.point_value == pytest.approx(100 / 150)
    ger = SPECS["GER30"]
    m = mkt([(23000.0, 23001.0, 22999.0, 23000.0)] * 3, ger, spread=1.0)
    with pytest.raises(ValueError, match="курс"):
        run(m, Signals.build([0], [1], sl_dist=[20.0]))
    t2 = backtest(
        m, Signals.build([0], [1], sl_dist=[20.0]), Exits(), RunMeta("t", profit_to_usd=1.1)
    )
    assert t2.trades[0].point_value == pytest.approx(0.12 * 1.1)


def test_spread_modes_from_spec() -> None:
    rows = [(100, 101, 99, 100)] * 3  # 14:00 UTC = 17:00 сервера: typical 138.4 п., worst 155.9 п.
    typ = mkt(rows, cost=CostModel("typical", slip_points=0))
    assert typ.spread[0] == pytest.approx(1.384)
    worst = mkt(rows, cost=CostModel("worst", slip_points=0))
    assert worst.spread[0] == pytest.approx(1.559)
    cons = mkt(rows, cost=CostModel(), spread=2.0)
    assert cons.spread[0] == pytest.approx(2.0)  # данные хуже typical → данные
    assert cons.slip[0] == pytest.approx(0.5 * 1.384)  # P-32


# --- эквити, скан, паритет ----------------------------------------------------------------


def random_market(seed: int, n: int = 600, spread: float = 0.5) -> Market:
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.6, n))
    o = np.r_[mid[0], mid[:-1]] + rng.normal(0, 0.2, n) * (rng.random(n) < 0.05) * 10
    c = mid
    h = np.maximum(o, c) + rng.exponential(0.4, n)
    lo = np.minimum(o, c) - rng.exponential(0.4, n)
    rows = list(zip(o, h, lo, c, strict=True))
    return mkt(rows, cost=CostModel("data", slip_points=5), spread=spread)


def random_signals(seed: int, m: Market, k: int = 40) -> Signals:
    rng = np.random.default_rng(seed + 1)
    n = len(m)
    bar = np.sort(rng.integers(0, n - 2, k))
    side = rng.choice([-1, 1], k)
    kind = rng.choice(["market", "stop", "limit"], k)
    off = rng.uniform(0.5, 3, k)
    level = m.mid_c[bar] + np.where(kind == "stop", side, -side) * off
    expire = m.ts[np.minimum(bar + 30, n - 1)]
    return Signals.build(
        bar,
        side,
        order=list(kind),
        level=np.where(kind == "market", np.nan, level),
        expire_s=expire,
        sl_dist=rng.uniform(1, 5, k),
        tp_r=np.asarray(rng.choice([1.0, 2.0, np.nan], k), dtype=np.float64),
        oco=np.where(rng.random(k) < 0.3, bar, -1),
    )


@pytest.mark.parametrize("seed", range(6))
def test_jit_matches_python_and_invariants(seed: int) -> None:
    m = random_market(seed)
    sig = random_signals(seed, m)
    ex = Exits(be_at_r=0.7, trail_start_r=1.0, trail_dist_r=0.8, partial_r=0.8, max_hold_min=60)
    a = run_kernel(m, sig, ex, max_positions=2, max_trades_day=30, jit=True)
    b = run_kernel(m, sig, ex, max_positions=2, max_trades_day=30, jit=False)
    np.testing.assert_array_equal(a.ints, b.ints)
    np.testing.assert_allclose(a.floats, b.floats, equal_nan=True)
    res = run(m, sig, ex, max_positions=2)
    assert res.trades
    for t in res.trades:
        assert t.mae_r <= min(0.0, t.r_gross) + 1e-12
        assert t.mfe_r >= max(0.0, t.r_gross) - 1e-12
        if t.exit_reason == "sl" and t.mods == 0:
            assert t.r_net <= -1.0 + 1e-9  # стоп без переноса: не лучше −1R
    eq = res.equity
    for t in res.trades:
        e = eq.filter(pl.col("trade_id") == t.trade_id)
        assert e.height == (t.close_utc - t.open_utc).total_seconds() / 60 + 1
        assert e["r_close"][-1] == pytest.approx(t.r_net)
        assert (e["r_worst"] <= e["r_close"] + 1e-12).all()


def test_scan_matches_backtest_without_overlap() -> None:
    m = random_market(11)
    bars = np.arange(0, len(m) - 60, 60)
    sig = Signals.build(bars, np.where(bars % 120 == 0, 1, -1), sl_dist=[2.0] * len(bars), tp_r=1.5)
    ex = Exits(max_hold_min=50)
    s = scan(m, sig, ex)
    bt = run(m, sig, ex, max_positions=1)
    np.testing.assert_allclose(s.r_net, [t.r_net for t in bt.trades])
    assert s.summary()["n"] == len(bt.trades)


def test_server_clock_matches_scalar_server_date() -> None:
    pts = [
        dt.datetime(2026, 3, 8, 6, 59, tzinfo=dt.UTC),
        dt.datetime(2026, 3, 8, 7, 0, tzinfo=dt.UTC),
        dt.datetime(2026, 10, 28, 21, 30, tzinfo=dt.UTC),
        dt.datetime(2026, 11, 3, 21, 59, tzinfo=dt.UTC),
        dt.datetime(2026, 11, 3, 22, 0, tzinfo=dt.UTC),
        dt.datetime(2025, 12, 31, 23, 0, tzinfo=dt.UTC),
    ]
    day, minute, wd = server_clock(np.asarray([int(p.timestamp()) for p in pts], np.int64), SERVER)
    for p, d, _mi, w in zip(pts, day, minute, wd, strict=True):
        sd = server_date(p, SERVER)
        assert d == sd.toordinal()
        assert w == sd.weekday()
    assert minute[2] == 30  # 21:30 UTC = 00:30 сервера (GMT+3)
