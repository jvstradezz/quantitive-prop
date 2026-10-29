"""Диапазон сессии (IDEA-0001): окна, пробой, возврат, фильтр дня, отсутствие подглядывания.

Цены в примерах — mid, спред 0 (bid = ask), поэтому уровни совпадают с ценами. Окно 16:30–17:00
сервера: минуты 990–1019. Диапазон: high 110, low 90, ширина 20. Среда 17.09.2025, лето:
сервер = UTC + 3.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from pbl.config import load_firms, load_instruments
from pbl.engine.backtest import Exits, RunMeta, backtest, scan
from pbl.engine.market import prepare_market, strategy_frame
from pbl.instruments.costs import CostModel
from pbl.strategies.base import MARKET, STOP, check_no_lookahead
from pbl.strategies.range_session import (
    AUTHOR_DAY_SIDES,
    RangeSession,
    fade_events,
    session_days,
)

SERVER = next(f for f in load_firms() if f.id == "fundednext").server_time
NAS = next(s for s in load_instruments() if s.firm == "fundednext" and s.symbol == "NAS100")
EXACT = CostModel(spread="data", slip_points=0.0)
MON = dt.date(2025, 9, 15)  # лето, сервер = UTC + 3

Row = tuple[float, float, float, float]


def day_rows(
    overrides: dict[int, Row] | None = None, window: dict[int, Row] | None = None, base: float = 100
) -> dict[int, Row]:
    """Минуты 16:00–23:00 сервера: плоско на `base`, окно с high 110 / low 90, затем overrides."""
    rows: dict[int, Row] = {m: (base, base, base, base) for m in range(960, 1381)}
    rows[990] = (100, 110, 100, 105)
    rows[1000] = (105, 105, 90, 95)
    rows.update(window or {})
    rows.update(overrides or {})
    return rows


def make_frame(days: list[dict[int, Row]], first: dt.date = MON) -> pl.DataFrame:
    ts: list[dt.datetime] = []
    ohlc: list[Row] = []
    for k, rows in enumerate(days):
        midnight = dt.datetime.combine(first + dt.timedelta(days=k), dt.time(0), dt.UTC)
        for m in sorted(rows):
            ts.append(midnight + dt.timedelta(minutes=m - 180))
            ohlc.append(rows[m])
    return bars_from(ts, ohlc)


def bars_from(ts: list[dt.datetime], ohlc: list[Row]) -> pl.DataFrame:
    a = np.asarray(ohlc, dtype=np.float64)
    cols: dict[str, object] = {"ts": ts}
    for k, x in enumerate("ohlc"):
        cols[f"bid_{x}"] = a[:, k]
        cols[f"ask_{x}"] = a[:, k]
    cols["spread_mean"] = np.zeros(len(ts))
    return pl.DataFrame(cols).with_columns(pl.col("ts").cast(pl.Datetime("ms", "UTC")))


def frame_of(bars: pl.DataFrame) -> pl.DataFrame:
    return strategy_frame(bars, SERVER)


def one_day(**kw: object) -> pl.DataFrame:
    return frame_of(make_frame([day_rows(**kw)]))  # type: ignore[arg-type]


# --- окно диапазона ---------------------------------------------------------------------


def test_session_days_range_bar_and_expiry() -> None:
    f = one_day()
    sd = session_days(f, "16:30", "17:00", "23:00")
    assert len(sd) == 1
    assert (sd.high[0], sd.low[0], sd.width[0]) == (110, 90, 20)
    assert sd.weekday[0] == 0
    # последний бар окна — 16:59 (индекс: 30 баров 16:00–16:29 + 29)
    assert sd.bar[0] == 30 + 29
    assert f["srv_min"][int(sd.bar[0])] == 1019
    assert sd.expire_s[0] == f["ts"].dt.epoch("s")[int(sd.bar[0])] + (1380 - 1019) * 60
    assert np.isnan(sd.atr_prev[0])


def test_incomplete_window_is_skipped() -> None:
    rows = day_rows()
    for m in range(1000, 1020):
        del rows[m]  # остались 10 минут из 30 и нет баров в последних 5 минутах
    assert len(session_days(frame_of(make_frame([rows])), "16:30", "17:00", "23:00")) == 0
    flat = {m: (100.0, 100.0, 100.0, 100.0) for m in range(960, 1381)}
    assert len(session_days(frame_of(make_frame([flat])), "16:30", "17:00", "23:00")) == 0, (
        "нулевой диапазон"
    )


def test_bad_window_parameters() -> None:
    with pytest.raises(ValueError, match="range_start"):
        session_days(one_day(), "17:00", "16:30", "23:00")


# --- пробой -------------------------------------------------------------------------------


def test_breakout_orders_levels_and_oco() -> None:
    sig = RangeSession(buffer=0.1).signals(one_day())
    assert len(sig) == 2
    assert list(sig.order) == [STOP, STOP]
    by_side = {int(s): i for i, s in enumerate(sig.side)}
    long_i, short_i = by_side[1], by_side[-1]
    assert sig.level[long_i] == pytest.approx(112.0)  # high + 0.1 · 20
    assert sig.sl[long_i] == 90.0
    assert sig.level[short_i] == pytest.approx(88.0)
    assert sig.sl[short_i] == 110.0
    assert sig.oco[long_i] == sig.oco[short_i] >= 0
    assert (sig.bar == 59).all()
    assert (sig.tp_r == 1.0).all()


def test_breakout_single_side_has_no_oco() -> None:
    sig = RangeSession(sides="short").signals(one_day())
    assert list(sig.side) == [-1]
    assert list(sig.oco) == [-1]
    assert np.isnan(RangeSession(tp_r=None).signals(one_day()).tp_r).all()


def test_day_filter_by_weekday() -> None:
    days = [day_rows() for _ in range(5)]  # пн … пт 15–19.09.2025
    sig = RangeSession(day_sides=AUTHOR_DAY_SIDES).signals(frame_of(make_frame(days)))
    sides_per_day: dict[int, list[int]] = {}
    f = frame_of(make_frame(days))
    for b, s in zip(sig.bar, sig.side, strict=True):
        sides_per_day.setdefault(int(f["srv_wd"][int(b)]), []).append(int(s))
    assert {k: sorted(v) for k, v in sides_per_day.items()} == {
        0: [1],
        1: [-1, 1],
        2: [1],
        3: [-1],
        4: [-1, 1],
    }
    # OCO только там, где живут обе заявки (вт, пт)
    paired = {int(f["srv_wd"][int(b)]) for b, o in zip(sig.bar, sig.oco, strict=True) if o >= 0}
    assert paired == {1, 4}


def test_width_filter_uses_only_previous_days() -> None:
    days = [day_rows() for _ in range(25)]  # с 1 сентября: 19 торговых дней и выходные
    f = frame_of(make_frame(days, first=dt.date(2025, 9, 1)))
    f = f.filter(pl.col("srv_wd") < 5)
    sd = session_days(f, "16:30", "17:00", "23:00")
    assert np.isnan(sd.atr_prev[:14]).all()
    assert np.isfinite(sd.atr_prev[14:]).all()
    # дневной размах каждого дня — 20 (110 − 90), то есть width / atr = 1
    assert sd.atr_prev[14:] == pytest.approx(20.0)
    on = RangeSession(width_rel=(0.9, 1.1)).signals(f)
    off = RangeSession(width_rel=(1.5, 3.0)).signals(f)
    assert len(sd) == 19
    assert len(on) == 2 * (len(sd) - 14)
    assert len(off) == 0


# --- возврат ------------------------------------------------------------------------------


UP_FADE: dict[int, Row] = {
    1025: (100, 112, 100, 112),  # закрылся выше high 110
    1026: (112, 113, 107, 108),  # следующий закрылся внутри; экстремум выноса 113
    1027: (108, 108, 108, 108),  # вход шорт по рынку по 108
    1028: (108, 108, 108, 108),
    1029: (108, 108, 108, 108),
    1030: (108, 108, 102, 103),  # тейк 1R: риск 113 − 108 = 5 → 103
}


def test_fade_short_after_false_break_up() -> None:
    f = one_day(overrides=UP_FADE)
    sd = session_days(f, "16:30", "17:00", "23:00")
    ev = fade_events(f, sd)
    assert list(ev.side) == [-1]
    assert f["srv_min"][int(ev.bar[0])] == 1026
    assert ev.extreme[0] == 113
    sig = RangeSession(mode="fade").signals(f)
    assert list(sig.side) == [-1]
    assert list(sig.order) == [MARKET]
    assert sig.sl[0] == 113
    sig_buf = RangeSession(mode="fade", buffer=0.1).signals(f)
    assert sig_buf.sl[0] == pytest.approx(115.0)  # 113 + 0.1 · 20


def test_fade_extreme_covers_the_whole_excursion() -> None:
    two_bars_out = {
        1025: (100, 114, 100, 112),
        1026: (112, 113, 111, 112),  # ещё снаружи
        1027: (112, 112, 107, 108),  # обратно внутри
    }
    f = one_day(overrides=two_bars_out)
    sig = RangeSession(mode="fade").signals(f)
    assert f["srv_min"][int(sig.bar[0])] == 1027
    assert sig.sl[0] == 114, "экстремум — по всему выносу, а не по последним двум барам"


def test_fade_requires_adjacent_bar_back_inside() -> None:
    rows = day_rows(overrides={1025: (100, 112, 100, 112)})
    del rows[1026]  # следующий бар после выноса пропал: вернувшийся бар не соседний по времени
    assert len(RangeSession(mode="fade").signals(frame_of(make_frame([rows])))) == 0
    stays_out = {1025: (100, 112, 100, 112), 1026: (112, 113, 111, 112)}
    stays = {m: (112.0, 112.0, 112.0, 112.0) for m in range(1027, 1381)}
    assert len(RangeSession(mode="fade").signals(one_day(overrides={**stays_out, **stays}))) == 0


def test_fade_long_and_first_event_per_day() -> None:
    down = {
        1025: (100, 100, 88, 88),  # закрылся ниже low 90
        1026: (88, 93, 87, 92),  # обратно внутри; экстремум 87
        1040: (100, 112, 100, 112),  # позже ложный вынос вверх
        1041: (112, 113, 107, 108),
    }
    f = one_day(overrides=down)
    ev = fade_events(f, session_days(f, "16:30", "17:00", "23:00"))
    assert sorted(ev.side.tolist()) == [-1, 1]
    sig = RangeSession(mode="fade").signals(f)
    assert list(sig.side) == [1], "одна сделка в день: берём более ранний сетап"
    assert sig.sl[0] == 87
    assert list(RangeSession(mode="fade", sides="short").signals(f).side) == [-1]


# --- через движок -------------------------------------------------------------------------


def run(frame: pl.DataFrame, strat: RangeSession, exits: Exits | None = None):  # type: ignore[no-untyped-def]
    bars = frame.select(["ts", *[c for c in frame.columns if c.startswith(("bid_", "ask_"))]])
    bars = bars.with_columns(pl.lit(0.0).alias("spread_mean"))
    market = prepare_market(bars, NAS, SERVER, EXACT)
    return backtest(market, strat.signals(frame), exits or Exits(), RunMeta("rs"))


def test_breakout_long_take_profit_through_engine() -> None:
    f = one_day(
        overrides={
            1025: (100, 125, 100, 125),  # buy stop 110 исполнен; риск 110 − 90 = 20
            1030: (125, 131, 125, 130),  # тейк 130
        }
    )
    (t,) = run(f, RangeSession()).trades
    assert (t.side, t.entry, t.sl, t.exit, t.exit_reason) == ("long", 110, 90, 130, "tp")
    assert t.r_net == pytest.approx(1.0)


def test_breakout_short_stop_loss_through_engine() -> None:
    f = one_day(
        overrides={
            1025: (100, 100, 85, 85),  # sell stop 90 исполнен, риск 110 − 90 = 20
            1030: (85, 111, 85, 110),  # стоп 110
        }
    )
    (t,) = run(f, RangeSession()).trades
    assert (t.side, t.exit_reason) == ("short", "sl")
    assert t.r_net == pytest.approx(-1.0)


def test_fade_short_take_profit_through_engine() -> None:
    (t,) = run(one_day(overrides=UP_FADE), RangeSession(mode="fade")).trades
    assert (t.side, t.entry, t.sl, t.exit, t.exit_reason) == ("short", 108, 113, 103, "tp")
    assert t.r_net == pytest.approx(1.0)


def test_scan_and_backtest_agree_on_non_overlapping_signals() -> None:
    days = [day_rows(overrides=UP_FADE) for _ in range(5)]
    f = frame_of(make_frame(days))
    bars = f.select(["ts", *[c for c in f.columns if c.startswith(("bid_", "ask_"))]])
    market = prepare_market(bars.with_columns(pl.lit(0.0).alias("spread_mean")), NAS, SERVER, EXACT)
    sig = RangeSession(mode="fade").signals(f)
    res = backtest(market, sig, Exits(), RunMeta("rs"))
    sc = scan(market, sig, Exits())
    assert len(res.trades) == sc.n_trades == 5
    np.testing.assert_allclose(sc.r_net, [t.r_net for t in res.trades])


# --- без доступа к будущему ---------------------------------------------------------------


def random_walk_frame(n_days: int = 22, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days: list[dict[int, Row]] = []
    price = 100.0
    for _ in range(n_days):
        rows: dict[int, Row] = {}
        for m in range(960, 1381):
            step = rng.normal(0, 1.2)
            o, c = price, price + step
            rows[m] = (
                o,
                max(o, c) + abs(rng.normal(0, 0.4)),
                min(o, c) - abs(rng.normal(0, 0.4)),
                c,
            )
            price = c
        days.append(rows)
    f = frame_of(make_frame(days, first=dt.date(2025, 9, 1)))
    return f.filter(pl.col("srv_wd") < 5)


@pytest.mark.parametrize(
    "strat",
    [
        RangeSession(),
        RangeSession(mode="breakout", buffer=0.1, day_sides=AUTHOR_DAY_SIDES),
        RangeSession(mode="fade", range_start="16:30", range_end="17:00"),
        RangeSession(mode="fade", sides="long", width_rel=(0.2, 3.0)),
    ],
)
def test_no_lookahead(strat: RangeSession) -> None:
    frame = random_walk_frame()
    assert len(strat.signals(frame)) > 0
    check_no_lookahead(strat, frame, cuts=list(range(40, frame.height - 1, 37)))
