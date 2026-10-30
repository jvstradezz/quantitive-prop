"""Признаки и семейства автопоиска: ТФ, скользящие, отсутствие доступа к будущему, 1 вход в день."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
import pytest

from pbl.research.range_sweep import make_bundle
from pbl.strategies.base import Signals, check_no_lookahead
from pbl.strategies.families import (
    Donchian,
    Family,
    MeanReversionZ,
    RangeBreakout,
    RangeFade,
    SessionDrive,
    TrendEma,
)
from pbl.strategies.features import (
    FILTERS,
    FeatureCache,
    ema,
    rolling_mean,
    rolling_std,
    window_extreme,
)

BarsFactory = Callable[..., pl.DataFrame]

FAMILIES: list[Family] = [
    Donchian(tf=15, lookback=20, session="ny"),
    Donchian(tf=5, lookback=20, session="london"),
    MeanReversionZ(tf=5, lookback=20, z=1.5, session="asia"),
    MeanReversionZ(tf=15, lookback=20, z=2.0, session="ny"),
    TrendEma(tf=15, fast=9, slow=21, session="ny"),
    TrendEma(tf=30, fast=9, slow=21, session="london"),
    SessionDrive(minutes=30, mode="continue", session="ny"),
    SessionDrive(minutes=15, mode="reverse", session="london"),
    RangeBreakout(session="ny30", buffer=0.1),
    RangeFade(session="ny60"),
]


@pytest.fixture(scope="module")
def frame(synthetic_bars: BarsFactory) -> pl.DataFrame:
    return make_bundle("NAS100", synthetic_bars(40, 3)).frame


def test_higher_timeframe_bars_are_whole_and_aligned(frame: pl.DataFrame) -> None:
    fc = FeatureCache(frame)
    h = fc.htf(15)
    days = len(np.unique(fc.day))
    assert len(h) == 91 * days, "бары 01:15–24:00 сервера → 91 полный бар M15 в сутки"
    assert (h.minute % 15 == 0).all()
    assert (np.diff(h.bar) > 0).all()
    assert (h.high >= h.low).all()
    np.testing.assert_array_equal(h.close, fc.bid_c[h.bar])
    # внутри первого бара: open — первого M1, high/low — по 15 барам
    i0 = int(h.bar[0])
    first = slice(i0 - 14, i0 + 1)
    assert h.open[0] == fc.bid_o[first][0]
    assert h.high[0] == fc.bid_h[first].max()
    assert h.low[0] == fc.bid_l[first].min()
    with pytest.raises(ValueError, match="делить сутки"):
        fc.htf(7)


def test_incomplete_higher_timeframe_bars_are_dropped(frame: pl.DataFrame) -> None:
    cut = frame.filter(pl.col("srv_min") % 60 != 29)  # в каждом часу нет минуты :29
    h = FeatureCache(cut).htf(30)
    assert (h.minute % 30 == 0).all()
    assert len(h) < len(FeatureCache(frame).htf(30)), "бары без последней минуты не закрыты"


def test_rolling_helpers_match_numpy() -> None:
    rng = np.random.default_rng(1)
    x = np.cumsum(rng.normal(size=200)) + 20_000
    mean, std = rolling_mean(x, 20), rolling_std(x, 20)
    assert np.isnan(mean[:19]).all()
    assert np.isnan(std[:19]).all()
    for j in (19, 57, 199):
        w = x[j - 19 : j + 1]
        assert mean[j] == pytest.approx(w.mean())
        assert std[j] == pytest.approx(w.std(), rel=1e-6)
    hi = window_extreme(x, 10, True)
    lo = window_extreme(x, 10, False)
    assert np.isnan(hi[:10]).all()
    assert hi[50] == x[40:50].max()
    assert lo[50] == x[40:50].min()
    e = ema(x, 9)
    assert e[0] == x[0]
    assert e[5] == pytest.approx(e[4] + 2 / 10 * (x[5] - e[4]))


def decision_cuts(sig: Signals, n_bars: int, k: int = 8) -> list[int]:
    """Точки отсечения сразу после баров решения: подглядывание на один бар видно только там."""
    pick = np.linspace(0, len(sig) - 1, k).astype(int)
    cuts = {int(sig.bar[i]) + 1 for i in pick}
    cuts |= set(np.random.default_rng(0).integers(1500, n_bars - 5, 3).tolist())
    return sorted(c for c in cuts if 0 < c < n_bars)


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.key)
def test_no_lookahead(fam: Family, frame: pl.DataFrame) -> None:
    check_no_lookahead(fam, frame, decision_cuts(fam.signals(frame), frame.height))


@dataclass(frozen=True)
class PeekNext:
    """Канарейка: решение по бару смотрит на закрытие следующего бара."""

    name: str = "peek-next"

    def signals(self, bars: pl.DataFrame) -> Signals:
        c = bars["bid_c"].to_numpy()
        idx = np.arange(1000, len(c) - 1, 400)
        side = np.where(c[idx + 1] > c[idx], 1, -1)
        return Signals.build(idx, side, sl_dist=np.full(len(idx), 5.0))


def test_the_cut_points_catch_a_one_bar_peek(frame: pl.DataFrame) -> None:
    leaky = PeekNext()
    cuts = decision_cuts(leaky.signals(frame), frame.height)
    with pytest.raises(AssertionError, match="зависят от будущего"):
        check_no_lookahead(leaky, frame, cuts)


@pytest.mark.parametrize("fam", FAMILIES, ids=lambda f: f.key)
def test_signals_have_stops_and_at_most_one_entry_per_day(fam: Family, frame: pl.DataFrame) -> None:
    sig = fam.signals(frame)
    assert len(sig) > 5, "на 40 днях семейство должно срабатывать"
    has_stop = ~np.isnan(sig.sl) | (sig.sl_dist > 0)
    assert has_stop.all(), "стоп обязателен (SPEC B3)"
    assert np.isnan(sig.tp_r).all(), "тейк добавляют выходы поиска"
    day = frame["srv_day"].to_numpy()[sig.bar]
    per_day = np.bincount(np.unique(day, return_inverse=True)[1])
    limit = 2 if isinstance(fam, RangeBreakout) else 1  # OCO: две заявки на один вход
    assert per_day.max() <= limit
    assert fam.close_at > "00:00"
    assert fam.timeframe in ("M1", "M5", "M15", "M30", "H1")


def test_entries_stay_inside_the_session_window(frame: pl.DataFrame) -> None:
    fc = FeatureCache(frame)
    for fam, lo, hi in (
        (Donchian(tf=5, lookback=20, session="london"), 10 * 60, 15 * 60 + 30),
        (TrendEma(tf=15, session="ny"), 16 * 60 + 30, 22 * 60),
    ):
        sig = fam.signals_from(fc)
        end_minute = fc.minute[sig.bar] + 1  # закрытие бара ТФ
        assert ((end_minute > lo) & (end_minute <= hi)).all(), fam.key


def test_session_drive_modes_are_mirror_images(frame: pl.DataFrame) -> None:
    a = SessionDrive(minutes=30, mode="continue", session="ny").signals(frame)
    b = SessionDrive(minutes=30, mode="reverse", session="ny").signals(frame)
    np.testing.assert_array_equal(a.bar, b.bar)
    np.testing.assert_array_equal(a.side, -b.side)


def test_filters_only_remove_signals(frame: pl.DataFrame) -> None:
    fc = FeatureCache(frame)
    sig = Donchian(tf=5, lookback=20, session="london").signals_from(fc)
    sizes = {}
    for name in FILTERS:
        mask = fc.filter_mask(name, sig.bar, sig.side)
        assert mask.shape == (len(sig),)
        sizes[name] = int(mask.sum())
    assert sizes["none"] == len(sig)
    assert all(v <= len(sig) for v in sizes.values())
    # объём разбит на «выше нормы» и «ниже нормы»: вместе не больше всех сигналов
    assert sizes["vol_hi"] + sizes["vol_lo"] <= len(sig)
    with pytest.raises(ValueError, match="неизвестный фильтр"):
        fc.filter_mask("magic", sig.bar, sig.side)


def test_trend_filter_keeps_only_the_daily_trend_side(frame: pl.DataFrame) -> None:
    fc = FeatureCache(frame)
    sig = TrendEma(tf=5, session="london").signals_from(fc)
    kept = sig.take(fc.filter_mask("trend", sig.bar, sig.side))
    trend, _ = fc.daily().lookup(fc.day[kept.bar])
    np.testing.assert_array_equal(trend, kept.side)


def test_family_params_roundtrip_into_unique_keys() -> None:
    a, b = Donchian(lookback=20), Donchian(lookback=40)
    assert a.key != b.key
    assert a.params()["lookback"] == 20
    assert a.name == a.key
    with pytest.raises(ValueError, match="fast < slow"):
        TrendEma(fast=21, slow=9)
    with pytest.raises(ValueError, match="continue или reverse"):
        SessionDrive(mode="x")


def test_signals_type(frame: pl.DataFrame) -> None:
    assert isinstance(Donchian().signals(frame), Signals)
