"""Тесты data: парсер bi5, бары, серверное время, качество, спред. Без сети."""

from __future__ import annotations

import lzma
import struct
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from pbl.data.bars import build_bars
from pbl.data.dukascopy import day_path, download_day, hour_url, parse_bi5, trading_days
from pbl.data.quality import check_ticks
from pbl.data.servertime import to_server_naive
from pbl.data.spread import patch_spec, render_yaml, spread_profile


def _bi5(records: list[tuple[int, int, int]]) -> bytes:
    return lzma.compress(
        b"".join(struct.pack(">IIIff", ms, a, b, 1.0, 2.0) for ms, a, b in records)
    )


HOUR = datetime(2025, 9, 15, 14, tzinfo=UTC)


def test_hour_url_month_is_zero_based() -> None:
    assert hour_url("EURUSD", HOUR).endswith("/EURUSD/2025/08/15/14h_ticks.bi5")


def test_parse_bi5() -> None:
    df = parse_bi5(_bi5([(440, 46610301, 46608699), (900, 46610400, 46608800)]), HOUR, 1000.0)
    assert df.height == 2
    assert df["ask"][0] == 46610.301
    assert df["bid"][0] == 46608.699
    assert df["ts"][0] == datetime(2025, 9, 15, 14, 0, 0, 440000, tzinfo=UTC)
    assert parse_bi5(b"", HOUR, 1000.0).is_empty()


def test_download_day_idempotent(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetch(url: str) -> bytes | None:
        calls.append(url)
        return _bi5([(10, 1002, 1000)]) if url.endswith("/03h_ticks.bi5") else None

    d = date(2025, 9, 15)
    assert download_day("US30", d, tmp_path, fetch) == 1
    assert len(calls) == 24
    assert day_path("US30", d, tmp_path).exists()
    assert download_day("US30", d, tmp_path, fetch) == 1
    assert len(calls) == 24  # второй вызов — из кэша


def test_trading_days_skip_saturday() -> None:
    days = trading_days(date(2025, 9, 12), date(2025, 9, 16))
    assert date(2025, 9, 13) not in days
    assert len(days) == 4


def test_server_time_follows_us_dst() -> None:
    ts = pl.DataFrame(
        {"ts": [datetime(2025, 7, 1, 12, tzinfo=UTC), datetime(2025, 12, 1, 12, tzinfo=UTC)]}
    )
    out = ts.select(to_server_naive(pl.col("ts")).alias("s"))["s"].to_list()
    assert out[0].hour == 15  # лето GMT+3
    assert out[1].hour == 14  # зима GMT+2


def _ticks() -> pl.DataFrame:
    base = datetime(2025, 7, 1, 12, tzinfo=UTC)
    rows = [
        (base.replace(minute=0, second=1), 100.0, 100.2),
        (base.replace(minute=0, second=30), 101.0, 101.2),
        (base.replace(minute=1, second=5), 99.0, 99.4),
    ]
    return pl.DataFrame(
        {
            "ts": [r[0] for r in rows],
            "bid": [r[1] for r in rows],
            "ask": [r[2] for r in rows],
            "bid_vol": [1.0] * 3,
            "ask_vol": [1.0] * 3,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )


def test_build_bars_ohlc() -> None:
    b = build_bars(_ticks(), "1m")
    assert b.height == 2
    first = b.row(0, named=True)
    assert (first["bid_o"], first["bid_h"], first["bid_l"], first["bid_c"]) == (
        100.0,
        101.0,
        100.0,
        101.0,
    )
    assert first["ticks"] == 2
    assert abs(first["spread_max"] - 0.2) < 1e-9


def test_build_bars_server_clock() -> None:
    b = build_bars(_ticks(), "1h", clock="server")
    assert b["ts"][0].hour == 15


def test_quality_flags() -> None:
    t = pl.concat(
        [_ticks(), _ticks().head(1), _ticks().head(1).with_columns(pl.lit(200.0).alias("ask"))]
    )
    q = check_ticks("X", t, [date(2025, 7, 1), date(2025, 7, 2)])
    assert q.days_with_data == 1
    assert q.missing_days == [date(2025, 7, 2)]
    assert q.duplicates == 1
    assert q.crossed == 0


def test_spread_profile_and_patch(tmp_path: Path) -> None:
    prof = spread_profile(_ticks(), point=0.1)
    row = prof.row(0, named=True)
    assert row["hour"] == 15
    assert abs(row["typical"] - 2.0) < 1e-6
    spec = tmp_path / "X.yaml"
    spec.write_text(
        "a: 1\n# Спред в MT5 «Плавающий»\nspread_by_hour: []\nb: 2\ndukascopy_symbol: null  # x\n",
        encoding="utf-8",
    )
    patch_spec(spec, render_yaml(prof, min_ticks=1), "XSYM")
    text = spec.read_text(encoding="utf-8")
    assert "{hour: 15, typical: 2.0, worst: 4.0}" in text
    assert "dukascopy_symbol: XSYM" in text
    assert "b: 2" in text
