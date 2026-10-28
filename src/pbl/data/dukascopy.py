"""Загрузчик тиков bid/ask Dukascopy (.bi5) в Parquet по дням.

URL: datafeed.dukascopy.com/datafeed/<SYM>/<YYYY>/<MM 0-based>/<DD>/<HH>h_ticks.bi5.
Запись — 20 байт big-endian: ms от начала часа, ask, bid (целые), ask_vol, bid_vol (float32).
"""

from __future__ import annotations

import lzma
import random
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from pbl.config import REPO_ROOT
from pbl.data.symbols import SYMBOLS

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
RAW_ROOT = REPO_ROOT / "data" / "raw" / "dukascopy"
_REC = struct.Struct(">IIIff")

TICK_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ts": pl.Datetime("ms", "UTC"),
    "bid": pl.Float64,
    "ask": pl.Float64,
    "bid_vol": pl.Float32,
    "ask_vol": pl.Float32,
}

Fetcher = Callable[[str], bytes | None]


def hour_url(duka_symbol: str, hour: datetime) -> str:
    return (
        f"{BASE_URL}/{duka_symbol}/{hour.year}/{hour.month - 1:02d}/{hour.day:02d}/"
        f"{hour.hour:02d}h_ticks.bi5"
    )


def parse_bi5(raw: bytes, hour: datetime, divisor: float) -> pl.DataFrame:
    """Распаковать .bi5 одного часа в тики; пустой ответ → пустой кадр."""
    data = lzma.decompress(raw) if raw else b""
    n = len(data) // _REC.size
    rows = [_REC.unpack_from(data, i * _REC.size) for i in range(n)]
    hour_utc = hour.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return pl.DataFrame(
        {
            "ts": [hour_utc + timedelta(milliseconds=r[0]) for r in rows],
            "bid": [r[2] / divisor for r in rows],
            "ask": [r[1] / divisor for r in rows],
            "bid_vol": [r[4] for r in rows],
            "ask_vol": [r[3] for r in rows],
        },
        schema=TICK_SCHEMA,
    )


def http_fetch(url: str, retries: int = 12, retries_429: int = 60) -> bytes | None:
    """GET с бэкоффом на 429/5xx/обрывы. 404 → None (нет данных за час).

    Dukascopy душит по IP: 429 — норма, ждём дольше (до `retries_429` попыток).
    """
    delay = 2.0
    for attempt in range(max(retries, retries_429)):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                body: bytes = resp.read()
                return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            limit = retries_429 if e.code == 429 else retries
            if e.code not in (429, 500, 502, 503, 504) or attempt >= limit - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt >= retries - 1:
                raise
        time.sleep(delay * (0.5 + random.random()))
        delay = min(delay * 2, 60.0)
    return None  # недостижимо


def day_path(symbol: str, day: date, root: Path = RAW_ROOT) -> Path:
    return root / symbol / f"{day.year}" / f"{day.isoformat()}.parquet"


def hour_cache_path(symbol: str, hour: datetime, root: Path = RAW_ROOT) -> Path:
    return root / "_bi5" / symbol / f"{hour:%Y-%m-%d}" / f"{hour:%H}.bi5"


def fetch_hour(symbol: str, hour: datetime, root: Path, fetch: Fetcher) -> bytes:
    """Сырой .bi5 часа из кэша на диске или из сети; нет данных — пустые байты (тоже кэшируем)."""
    path = hour_cache_path(symbol, hour, root)
    if path.exists():
        return path.read_bytes()
    raw = fetch(hour_url(SYMBOLS[symbol].duka, hour)) or b""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return raw


def download_day(symbol: str, day: date, root: Path = RAW_ROOT, fetch: Fetcher = http_fetch) -> int:
    """Скачать сутки (24 ч UTC) в Parquet; готовый файл пропускается. Возвращает число тиков."""
    path = day_path(symbol, day, root)
    if path.exists():
        return int(pl.scan_parquet(path).select(pl.len()).collect().item())
    divisor = SYMBOLS[symbol].divisor
    frames = []
    for h in range(24):
        hour = datetime(day.year, day.month, day.day, h, tzinfo=UTC)
        raw = fetch_hour(symbol, hour, root, fetch)
        if raw:
            frames.append(parse_bi5(raw, hour, divisor))
    df = (pl.concat(frames) if frames else pl.DataFrame(schema=TICK_SCHEMA)).sort(
        "ts", maintain_order=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)
    return df.height


def trading_days(start: date, end: date) -> list[date]:
    """Дни start..end включительно без суббот (у Dukascopy в субботу тиков нет)."""
    n = (end - start).days + 1
    return [d for d in (start + timedelta(days=i) for i in range(n)) if d.weekday() != 5]


def download_range(
    symbol: str,
    start: date,
    end: date,
    root: Path = RAW_ROOT,
    workers: int = 4,
    fetch: Fetcher = http_fetch,
    rounds: int = 20,
    progress: Callable[[date, int], None] | None = None,
) -> int:
    days = [d for d in trading_days(start, end) if not day_path(symbol, d, root).exists()]
    pending = [datetime(d.year, d.month, d.day, h, tzinfo=UTC) for d in days for h in range(24)]

    def safe(hr: datetime) -> bool:
        try:
            fetch_hour(symbol, hr, root, fetch)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return False
        return True

    # общий пул по часам: сеть душит запросы, отказ одного часа не должен останавливать остальные
    for _ in range(rounds):
        pending = [hr for hr in pending if not hour_cache_path(symbol, hr, root).exists()]
        if not pending:
            break
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(safe, pending))
    total = 0
    incomplete = 0
    for d in trading_days(start, end):
        hours = (datetime(d.year, d.month, d.day, h, tzinfo=UTC) for h in range(24))
        if not day_path(symbol, d, root).exists() and not all(
            hour_cache_path(symbol, hr, root).exists() for hr in hours
        ):
            incomplete += 1
            continue
        n = download_day(symbol, d, root, fetch)
        total += n
        if progress:
            progress(d, n)
    if incomplete:
        raise RuntimeError(f"{symbol}: {incomplete} дн. не докачаны, перезапусти команду")
    return total


def load_ticks(symbol: str, start: date, end: date, root: Path = RAW_ROOT) -> pl.DataFrame:
    files = [p for d in trading_days(start, end) if (p := day_path(symbol, d, root)).exists()]
    if not files:
        return pl.DataFrame(schema=TICK_SCHEMA)
    return pl.concat([pl.read_parquet(p) for p in files]).sort("ts", maintain_order=True)
