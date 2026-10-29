"""Стратегия → движок → проп-MC: без доступа к будущему, CSV для `pbl prop run`, прогон MC."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pbl.config import load_firms, load_instruments
from pbl.engine.backtest import Exits, RunMeta, backtest
from pbl.engine.io import write_run
from pbl.engine.market import FirmTradeProfile, prepare_market, strategy_frame
from pbl.instruments.costs import CostModel
from pbl.prop.days import LotSpec, day_table_from_trades
from pbl.prop.engine import MCConfig, MCEngine
from pbl.prop.io import read_trades_csv
from pbl.prop.rules import compile_offer
from pbl.prop.sim import RiskScheme
from pbl.strategies.base import Signals, check_no_lookahead
from pbl.strategies.reference import OpeningRangeBreakout

FN = next(f for f in load_firms() if f.id == "fundednext")
NAS = next(s for s in load_instruments() if s.firm == "fundednext" and s.symbol == "NAS100")
PROG = next(p for p in FN.programs if p.id == "fn-stellar-2step")


def synthetic_bars(n_days: int = 60, seed: int = 5) -> pl.DataFrame:
    """M1-бары NAS100-подобного ряда: будни лета 2025, 12:00–19:59 UTC (15:00–22:59 сервера)."""
    rng = np.random.default_rng(seed)
    days = [dt.date(2025, 6, 2) + dt.timedelta(days=i) for i in range(n_days * 7 // 5 + 7)]
    days = [d for d in days if d.weekday() < 5][:n_days]
    ts = [
        dt.datetime.combine(d, dt.time(12), dt.UTC) + dt.timedelta(minutes=i)
        for d in days
        for i in range(480)
    ]
    n = len(ts)
    mid = 20_000 + np.cumsum(rng.normal(0, 4.0, n))
    o = np.r_[mid[0], mid[:-1]]
    h = np.maximum(o, mid) + rng.exponential(2.0, n)
    lo = np.minimum(o, mid) - rng.exponential(2.0, n)
    s = 1.4
    cols: dict[str, object] = {"ts": ts}
    for x, v in zip("ohlc", (o, h, lo, mid), strict=True):
        cols[f"bid_{x}"] = v - s / 2
        cols[f"ask_{x}"] = v + s / 2
    cols["spread_mean"] = np.full(n, s)
    return pl.DataFrame(cols).with_columns(pl.col("ts").cast(pl.Datetime("ms", "UTC")))


@pytest.fixture(scope="module")
def bars() -> pl.DataFrame:
    return synthetic_bars()


@dataclass(frozen=True)
class Leaky:
    """Подглядывает в следующий бар — тест обязан это поймать."""

    name: str = "leaky"

    def signals(self, bars: pl.DataFrame) -> Signals:
        c = bars["bid_c"].to_numpy()
        idx = np.flatnonzero(c[1:] > c[:-1])
        return Signals.build(idx, np.ones(idx.shape[0], dtype=np.int8), sl_dist=[10.0] * len(idx))


@given(cut=st.integers(min_value=100, max_value=28_000))
def test_orb_has_no_lookahead(cut: int) -> None:
    frame = strategy_frame(synthetic_bars(n_days=60), FN.server_time)
    check_no_lookahead(OpeningRangeBreakout(), frame, [cut])


def test_lookahead_check_catches_leaky_strategy(bars: pl.DataFrame) -> None:
    frame = strategy_frame(bars, FN.server_time)
    with pytest.raises(AssertionError, match="будущего"):
        check_no_lookahead(Leaky(), frame, list(range(5_000, 25_000, 2_000)))


def test_orb_signals_are_oco_pairs_at_range_end(bars: pl.DataFrame) -> None:
    frame = strategy_frame(bars, FN.server_time)
    sig = OpeningRangeBreakout().signals(frame)
    assert len(sig) == 2 * 60
    # бар решения — 16:59 сервера (13:59 UTC летом), уровни — high/low диапазона по bid
    assert set(frame["srv_min"].to_numpy()[sig.bar]) == {16 * 60 + 59}
    assert (sig.level[sig.side == 1] > sig.level[sig.side == -1]).all()
    np.testing.assert_array_equal(sig.sl[sig.side == 1], sig.level[sig.side == -1])


def test_engine_trades_feed_prop_mc(bars: pl.DataFrame, tmp_path: Path) -> None:
    firm = FirmTradeProfile.from_rules(PROG.trade_rules, funded=False)
    market = prepare_market(bars, NAS, FN.server_time, CostModel(), firm, PROG.commission)
    sig = OpeningRangeBreakout(tp_r=1.5).signals(strategy_frame(bars, FN.server_time))
    res = backtest(market, sig, Exits(close_at="22:30"), RunMeta("orb-test", sample="WF"))
    assert len(res.trades) >= 40
    assert res.signal_status["oco_cancelled"] >= 40
    assert {t.exit_reason for t in res.trades} <= {"sl", "tp", "eod"}
    assert res.stats["trades_max_day"] == 1

    write_run(res, tmp_path)
    back = read_trades_csv(tmp_path / "trades.csv")
    assert [t.model_dump() for t in back] == [t.model_dump() for t in res.trades]
    assert pl.read_parquet(tmp_path / "equity_R.parquet").height == res.equity.height

    table = day_table_from_trades(res.trades, FN.server_time, {"NAS100": LotSpec(0.01, 0.01)})
    assert table.n_trading_days == len(res.trades)
    eng = MCEngine(table, MCConfig(n_paths=400, opt_paths=100, seed=1, outer_boot=0))
    offer = next(o for o in PROG.iter_offers() if o.size.size_usd == 25_000)
    stats = eng.run(compile_offer(FN.id, PROG, offer), RiskScheme(f=(0.005, 0.005, 0.005)))
    assert 0.0 <= stats.p1 <= stats.p_funded <= 1.0
    assert sum(stats.fail_share.values()) <= 1.0 + 1e-9
