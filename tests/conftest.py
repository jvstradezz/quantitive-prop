import datetime as dt
import os
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, settings

UTC = dt.UTC
BarsFactory = Callable[..., pl.DataFrame]

# CI детерминирован: фиксированный seed генератора и конечное число примеров (ARCHITECTURE §11).
settings.register_profile(
    "ci",
    derandomize=True,
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("dev", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


def make_synthetic_bars(
    n_days: int = 70, seed: int = 5, drift: float = 0.0, first: dt.date = dt.date(2025, 4, 7)
) -> pl.DataFrame:
    """M1 bid/ask: случайное блуждание по часам сервера FN 01:15–24:00, летнее время (UTC + 3).

    Спред 0.8 пункта цены. Нужно только для механики; выводов о рынке не даёт.
    """
    rng = np.random.default_rng(seed)
    # first — понедельник; летом США сервер = UTC + 3, значит n_days ≲ 140 до перевода часов
    ts: list[dt.datetime] = []
    mid_o: list[float] = []
    price = 20_000.0
    for k in range(n_days * 7 // 5):
        day = first + dt.timedelta(days=k)
        if day.weekday() >= 5:
            continue
        midnight = dt.datetime.combine(day, dt.time(0), UTC)
        for minute in range(75, 1440):
            ts.append(midnight + dt.timedelta(minutes=minute - 180))
            mid_o.append(price)
            price += rng.normal(drift, 4.0)
    o = np.asarray(mid_o)
    c = np.append(o[1:], price)
    noise = np.abs(rng.normal(0, 2.0, size=(2, o.size)))
    h, lo = np.maximum(o, c) + noise[0], np.minimum(o, c) - noise[1]
    cols: dict[str, object] = {"ts": ts}
    for name, v in (("o", o), ("h", h), ("l", lo), ("c", c)):
        cols[f"bid_{name}"] = v - 0.4
        cols[f"ask_{name}"] = v + 0.4
    cols["spread_mean"] = np.full(o.size, 0.8)
    return pl.DataFrame(cols).with_columns(pl.col("ts").cast(pl.Datetime("ms", "UTC")))


@pytest.fixture(scope="session")
def synthetic_bars() -> BarsFactory:
    return make_synthetic_bars
