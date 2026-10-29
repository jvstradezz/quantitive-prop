"""Минутные свечи bid/ask Dukascopy (.bi5) → M1-бары в Parquet.

URL: datafeed.dukascopy.com/datafeed/<SYM>/<YYYY>/<MM 0-based>/<DD>/<BID|ASK>_candles_min_1.bi5.
Запись — 24 байта big-endian: секунды от 00:00 UTC, open, close, low, high (целые), volume float32.
Файл — сутки по 1440 минут, минуты без сделок забиты плоскими свечами с volume 0 — их выкидываем.
"""

from __future__ import annotations

import lzma
import struct
import urllib.error
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from pbl.config import REPO_ROOT
from pbl.data.dukascopy import BASE_URL, Fetcher, http_fetch, trading_days
from pbl.data.symbols import SYMBOLS

CANDLE_ROOT = REPO_ROOT / "data" / "raw" / "dukascopy_candles"
BARS_ROOT = REPO_ROOT / "data" / "bars"
_REC = struct.Struct(">IIIIIf")
SIDES = ("BID", "ASK")

CANDLE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ts": pl.Datetime("ms", "UTC"),
    "o": pl.Float64,
    "h": pl.Float64,
    "l": pl.Float64,
    "c": pl.Float64,
    "vol": pl.Float32,
}


def candle_url(duka_symbol: str, day: date, side: str) -> str:
    return (
        f"{BASE_URL}/{duka_symbol}/{day.year}/{day.month - 1:02d}/{day.day:02d}/"
        f"{side}_candles_min_1.bi5"
    )


def candle_cache_path(symbol: str, day: date, side: str, root: Path = CANDLE_ROOT) -> Path:
    return root / symbol / f"{day.year}" / f"{day.isoformat()}_{side}.bi5"


def parse_candles(raw: bytes, day: date, divisor: float) -> pl.DataFrame:
    """Распаковать сутки свечей; плоские свечи без объёма удаляются, битая свеча — ValueError."""
    data = lzma.decompress(raw) if raw else b""
    rows = [_REC.unpack_from(data, i * _REC.size) for i in range(len(data) // _REC.size)]
    day0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
    kept = [r for r in rows if r[5] > 0 or not (r[1] == r[2] == r[3] == r[4])]
    for t, o, c, lo, hi, _ in kept:
        if not (lo <= min(o, c) and hi >= max(o, c)):
            raise ValueError(f"{day} +{t}s: low/high не охватывают open/close")
    return pl.DataFrame(
        {
            "ts": [day0 + timedelta(seconds=r[0]) for r in kept],
            "o": [r[1] / divisor for r in kept],
            "h": [r[4] / divisor for r in kept],
            "l": [r[3] / divisor for r in kept],
            "c": [r[2] / divisor for r in kept],
            "vol": [r[5] for r in kept],
        },
        schema=CANDLE_SCHEMA,
    )


def fetch_candles(symbol: str, day: date, side: str, root: Path, fetch: Fetcher) -> bytes:
    """Сырой .bi5 из кэша или сети; нет данных (404) — пустые байты, тоже кэшируем."""
    path = candle_cache_path(symbol, day, side, root)
    if path.exists():
        return path.read_bytes()
    raw = fetch(candle_url(SYMBOLS[symbol].duka, day, side)) or b""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return raw


def day_bars(symbol: str, day: date, root: Path = CANDLE_ROOT) -> pl.DataFrame:
    """M1-бары суток из кэша: bid_*/ask_*, объёмы; только минуты, где есть обе стороны."""
    divisor = SYMBOLS[symbol].divisor
    sides = {
        s: parse_candles(candle_cache_path(symbol, day, s, root).read_bytes(), day, divisor)
        for s in SIDES
    }
    bid, ask = sides["BID"], sides["ASK"]
    ren = {"o": "o", "h": "h", "l": "l", "c": "c", "vol": "vol"}
    bid = bid.rename({k: f"bid_{v}" for k, v in ren.items()})
    ask = ask.rename({k: f"ask_{v}" for k, v in ren.items()})
    return bid.join(ask, on="ts", how="inner").sort("ts")


def bars_path(symbol: str, year: int, root: Path = BARS_ROOT) -> Path:
    return root / symbol / f"{year}" / "1m_utc_candles.parquet"


def download_candles(
    symbol: str,
    start: date,
    end: date,
    root: Path = CANDLE_ROOT,
    workers: int = 4,
    fetch: Fetcher = http_fetch,
    rounds: int = 20,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Докачать свечи за start..end (без суббот). Возвращает число скачанных сутки×сторона."""
    jobs = [(d, s) for d in trading_days(start, end) for s in SIDES]

    def missing() -> list[tuple[date, str]]:
        return [(d, s) for d, s in jobs if not candle_cache_path(symbol, d, s, root).exists()]

    def safe(job: tuple[date, str]) -> bool:
        try:
            fetch_candles(symbol, job[0], job[1], root, fetch)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False
        return True

    n0 = len(missing())
    for _ in range(rounds):
        pending = missing()
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, _ok in enumerate(ex.map(safe, pending), 1):
                if progress and i % 50 == 0:
                    progress(i, len(pending))
    left = missing()
    if left:
        raise RuntimeError(f"{symbol}: {len(left)} файлов не докачаны, перезапусти команду")
    return n0


def build_year_bars(
    symbol: str, start: date, end: date, root: Path = CANDLE_ROOT, bars_root: Path = BARS_ROOT
) -> int:
    """Собрать кэш свечей в Parquet по годам; возвращает число M1-баров."""
    frames: dict[int, list[pl.DataFrame]] = {}
    for d in trading_days(start, end):
        if not all(candle_cache_path(symbol, d, s, root).read_bytes() for s in SIDES):
            continue  # нет суток на одной из сторон (выходной/праздник)
        df = day_bars(symbol, d, root)
        if not df.is_empty():
            frames.setdefault(d.year, []).append(df)
    total = 0
    for year, parts in frames.items():
        out = pl.concat(parts).sort("ts")
        p = bars_path(symbol, year, bars_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(p, compression="zstd")
        total += out.height
    return total
