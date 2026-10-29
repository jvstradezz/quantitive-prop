"""Сетка идеи №1 и массовый скан: размер сетки, окна у FN, сборка фильтра дня из сканов."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from pbl.config import CONFIGS, load_model
from pbl.engine.backtest import scan
from pbl.research.range_sweep import (
    Bundle,
    Config,
    business_day,
    config_trades,
    daily_from_table,
    exit_grid,
    grid,
    make_bundle,
    scan_kind,
    variant_spec,
    window_is_live,
)
from pbl.research.variants import VariantLog
from pbl.schemas import InstrumentSpec
from pbl.strategies.range_session import AUTHOR_DAY_SIDES

BarsFactory = Callable[..., pl.DataFrame]


def spec(symbol: str) -> InstrumentSpec:
    return load_model(CONFIGS / "instruments" / "fundednext" / f"{symbol}.yaml", InstrumentSpec)


@pytest.fixture(scope="module")
def nas(synthetic_bars: BarsFactory) -> Bundle:
    return make_bundle("NAS100", synthetic_bars())


def test_grid_size_and_names() -> None:
    g = grid()
    assert len(g) == 5 * 7 * 2 * (3 + 1) == 280
    assert len({c.key for c in g}) == 280
    assert set(exit_grid()) == {
        "rr_1_1",
        "rr_1_2",
        "rr_1_3",
        "trailing",
        "partial",
        "breakeven",
        "time_exit",
    }


def test_windows_are_checked_against_firm_trading_hours() -> None:
    nas, ger = spec("NAS100"), spec("GER30")
    for a, b in (("03:00", "07:00"), ("10:00", "12:00"), ("16:30", "17:30")):
        assert window_is_live(nas, a, b)
    assert not window_is_live(ger, "03:00", "07:00"), "DAX у FN торгуется с 10:15"
    assert not window_is_live(ger, "10:00", "12:00")
    assert window_is_live(ger, "10:15", "12:00")
    assert window_is_live(ger, "16:30", "17:30")


def test_exit_scaling_scales_r_levels_but_not_fractions() -> None:
    e = exit_grid()["partial"].scaled(1.2)
    assert e.tp_r == pytest.approx(2.4)
    assert e.exits.partial_r == pytest.approx(1.2)
    assert e.exits.partial_frac == 0.5
    part = next(s for s in e.specs if s.kind == "partial")
    assert part.params == {"at_r": pytest.approx(1.2), "frac": 0.5}
    assert exit_grid()["time_exit"].scaled(1.2).tp_r is None


def test_scan_produces_trades_with_days_and_costs(nas: Bundle) -> None:
    cfg = Config("ny30", "breakout", exit_grid()["rr_1_1"], "none")
    t = scan_kind(nas, cfg, "both")
    assert t.height > 30
    wd = t["wd"].to_numpy()
    assert wd.min() >= 0
    assert wd.max() <= 4
    assert (t["cost_r"] > 0).all(), "спред и проскальзывание всегда стоят денег"
    assert (t["r_net"] < t["r_gross"]).all()
    assert scan_kind(nas, cfg, "both") is t, "кэш скана"
    one_side = scan_kind(nas, cfg, "long")
    assert set(one_side["side"].to_list()) == {1}


@pytest.mark.parametrize("mode", ["breakout", "fade"])
@pytest.mark.parametrize("exit_name", ["rr_1_2", "trailing"])
def test_author_filter_composed_from_scans_equals_direct_run(
    nas: Bundle, mode: str, exit_name: str
) -> None:
    cfg = Config("ny30", mode, exit_grid()[exit_name], "author")  # type: ignore[arg-type]
    composed = config_trades(nas, cfg).sort("day", "side")
    from pbl.strategies.range_session import RangeSession

    strat = RangeSession(
        mode=mode,  # type: ignore[arg-type]
        range_start=cfg.start,
        range_end=cfg.end,
        tp_r=cfg.exit.tp_r,
        day_sides=AUTHOR_DAY_SIDES,
    )
    res = scan(nas.market, strat.signals(nas.frame), cfg.exit.exits, profit_to_usd=None)
    assert composed.height == res.n_trades > 20
    direct = np.sort(res.r_net)
    np.testing.assert_allclose(np.sort(composed["r_net"].to_numpy()), direct)
    days = nas.market.day[res.entry_bar]
    assert sorted(composed["day"].to_list()) == sorted(days.tolist())


def test_author_filter_only_trades_allowed_sides(nas: Bundle) -> None:
    cfg = Config("ny30", "breakout", exit_grid()["rr_1_1"], "author")
    t = config_trades(nas, cfg)
    for wd, rule in enumerate(AUTHOR_DAY_SIDES):
        sides = set(t.filter(pl.col("wd") == wd)["side"].to_list())
        allowed = {"long": {1}, "short": {-1}, "both": {1, -1}}[rule]
        assert sides <= allowed, (wd, rule)


def test_variant_specs_are_unique_per_grid_and_instrument() -> None:
    ids = {variant_spec(c, s).variant_id for c in grid() for s in ("NAS100", "US30")}
    assert len(ids) == 560
    a = variant_spec(Config("ny30", "fade", exit_grid()["rr_1_1"], "author"), "NAS100")
    assert a.filters[0].params["mon"] == "long"
    assert a.family == "range_fade"


def test_probe_count_matches_the_card() -> None:
    """Строки счётчика до мутаций: по инструментам окна и корзина на каждую конфигурацию."""
    specs = {s: spec(s) for s in ("NAS100", "US30", "SPX500", "GER30")}
    rows = sum(sum(window_is_live(sp, c.start, c.end) for sp in specs.values()) + 1 for c in grid())
    assert rows == 1288


def test_daily_series_feed_the_variant_log(nas: Bundle, tmp_path: pytest.TempPathFactory) -> None:
    cfg = Config("ny30", "breakout", exit_grid()["rr_1_1"], "none")
    t = config_trades(nas, cfg)
    days = t["day"].to_numpy()
    first, last = business_day(int(days.min())), business_day(int(days.max()))
    daily = daily_from_table(t, first, last)
    assert daily.n_trades == t.height
    assert sum(daily.r) == pytest.approx(float(t["r_net"].sum()))
    log = VariantLog.for_idea("IDEA-0001", tmp_path)  # type: ignore[arg-type]
    assert log.add(variant_spec(cfg, "NAS100"), daily, run_id="t", on=dt.date(2026, 10, 29))
    log.save()
    row = VariantLog.for_idea("IDEA-0001", tmp_path).frame().row(0, named=True)  # type: ignore[arg-type]
    assert row["n_trades"] == t.height
    assert row["net_r"] == pytest.approx(float(t["r_net"].sum()))


def test_business_day_moves_weekend_to_friday() -> None:
    sat = dt.date(2025, 4, 12)
    assert business_day(sat.toordinal()) == dt.date(2025, 4, 11)
    assert business_day(dt.date(2025, 4, 14).toordinal()) == dt.date(2025, 4, 14)
