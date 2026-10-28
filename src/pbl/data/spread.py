"""Профиль спреда по часам сервера FN и запись его в YAML спеков."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from pbl.data.servertime import to_server_naive

WORST_QUANTILE = 0.99


def spread_profile(ticks: pl.DataFrame, point: float) -> pl.DataFrame:
    """По часу сервера (0–23): typical = медиана, worst = 99-й перцентиль, в пунктах MT5."""
    t = ticks.with_columns(
        to_server_naive(pl.col("ts")).dt.hour().alias("hour"),
        ((pl.col("ask") - pl.col("bid")) / point).alias("pts"),
    )
    return (
        t.group_by("hour")
        .agg(
            pl.col("pts").median().alias("typical"),
            pl.col("pts").quantile(WORST_QUANTILE).alias("worst"),
            pl.len().alias("ticks"),
        )
        .sort("hour")
    )


def _fmt(x: float) -> str:
    return f"{x:.1f}"


def render_yaml(profile: pl.DataFrame, min_ticks: int = 200) -> str:
    """Блок `spread_by_hour` (часы с малым числом тиков пропускаем — рынок закрыт)."""
    rows = profile.filter(pl.col("ticks") >= min_ticks).iter_rows(named=True)
    lines = [
        "spread_by_hour:  # пункты MT5, час сервера FN; typical=медиана, worst=p99 (Dukascopy)"
    ]
    for r in rows:
        typical = _fmt(r["typical"])
        worst = _fmt(max(r["worst"], r["typical"]))
        lines.append(f"  - {{hour: {r['hour']}, typical: {typical}, worst: {worst}}}")
    return "\n".join(lines) + "\n"


def patch_spec(path: Path, block: str, duka_symbol: str) -> None:
    """Подставить spread_by_hour и dukascopy_symbol, не трогая остальные комментарии."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r"(?:# Спред в MT5[^\n]*\n)?spread_by_hour:[^\n]*\n(?:  - [^\n]*\n)*",
        "# Спред в MT5 «Плавающий», числа в спеке нет: профиль по Dukascopy (T5, O-04).\n" + block,
        text,
        count=1,
    )
    text = re.sub(r"dukascopy_symbol:[^\n]*", f"dukascopy_symbol: {duka_symbol}", text, count=1)
    path.write_text(text, encoding="utf-8")
