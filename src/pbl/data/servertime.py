"""Время сервера FundedNext: ET + 7 ч (GMT+3 летом, GMT+2 зимой; DECISIONS O-10)."""

from __future__ import annotations

import polars as pl

NY = "America/New_York"
FN_OFFSET_OVER_ET_HOURS = 7


def to_server_naive(ts: pl.Expr) -> pl.Expr:
    """UTC-время (Datetime[UTC]) → настенное время сервера FN без tz-метки."""
    ny = ts.dt.convert_time_zone(NY)
    return ny.dt.replace_time_zone(None) + pl.duration(hours=FN_OFFSET_OVER_ET_HOURS)
